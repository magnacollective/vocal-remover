from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
import tempfile
import os
import shutil
import subprocess
import librosa
import numpy as np
from scipy import signal
import soundfile as sf

app = FastAPI(title="Vocal Remover & Audio Analysis API")

# CORS configuration (env-driven with safe defaults)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
print(f"[CORS] Allowed origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add explicit headers and handle preflight for edge cases
@app.middleware("http")
async def cors_handler(request: Request, call_next):
    if request.method == "OPTIONS":
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*" if "*" in ALLOWED_ORIGINS else ",".join(ALLOWED_ORIGINS),
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age": "86400",
            },
        )

    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*" if "*" in ALLOWED_ORIGINS else ",".join(ALLOWED_ORIGINS)
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    return response

@app.get("/")
def root():
    return {"status": "ok", "service": "vocal-remover-api"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_audio(audio: UploadFile = File(...)):
    """Analyze uploaded audio for BPM, key, duration, and sample rate.
    Returns: {"bpm": float, "key": string, "duration": string, "sample_rate": string}
    """
    print(f"[analyze] Received file: {audio.filename}")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Save uploaded file
            input_path = os.path.join(tmpdir, audio.filename or "audio")
            with open(input_path, "wb") as f:
                shutil.copyfileobj(audio.file, f)

            # Convert to WAV if needed
            wav_path = _ensure_wav(input_path, tmpdir)
            print(f"[analyze] Processing: {wav_path}")

            # Load audio with librosa
            y, sr = librosa.load(wav_path, sr=None, mono=True)
            duration_seconds = len(y) / sr

            # Separate harmonic/percussive for improved analysis
            y_harm, y_perc = librosa.effects.hpss(y)

            # Detect BPM from percussive/onset envelope; resolve half/double
            bpm = _detect_bpm(y_perc, sr)

            # Detect key from harmonic content with tuning + beat sync
            key = _detect_key(y_harm, sr)

            analysis_result = {
                "bpm": round(float(bpm), 1),
                "key": key,
                "duration": _format_duration(duration_seconds),
                "sample_rate": f"{sr} Hz",
            }

            print(f"[analyze] Analysis complete: {analysis_result}")
            return analysis_result

        except Exception as e:
            print(f"[analyze] Error: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/separate")
async def separate_stems(
    audio: UploadFile = File(...),
    stem_type: str = "vocals"
):
    """Separate audio into stems. 
    stem_type options: 'vocals', 'instrumental', 'drums', 'bass', 'all'
    Returns: WAV file of the requested stem
    """
    print(f"[separate] Received file: {audio.filename}, stem_type: {stem_type}")
    
    # Use a persistent temp dir for the duration of the response and clean up after sending
    tmpdir = tempfile.mkdtemp(prefix="vr-")
    try:
        # Save uploaded file
        input_path = os.path.join(tmpdir, audio.filename or "audio")
        with open(input_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        # Convert to WAV if needed
        wav_path = _ensure_wav(input_path, tmpdir)
        print(f"[separate] Processing: {wav_path}")

        # For now, use simple vocal removal technique
        # TODO: Integrate HTDemucs or other advanced models
        output_path = _simple_vocal_removal(wav_path, tmpdir, stem_type)

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Stem separation failed: output not created")

        # Return the separated audio file and clean up tmpdir after response is sent
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"{stem_type}_separated.wav",
            background=BackgroundTask(shutil.rmtree, tmpdir, True),
        )

    except Exception as e:
        print(f"[separate] Error: {e}")
        # Ensure cleanup on error as well
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Stem separation failed: {str(e)}")

def _ensure_wav(input_path: str, tmpdir: str) -> str:
    """Convert input audio to WAV format using FFmpeg."""
    if input_path.lower().endswith('.wav'):
        return input_path
    
    output_path = os.path.join(tmpdir, "converted.wav")
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "warning",
        "-i", input_path,
        "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
        output_path
    ]
    
    try:
        subprocess.check_call(cmd)
        return output_path
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {e}")

def _detect_bpm(y_perc: np.ndarray, sr: int) -> float:
    """Detect BPM with onset envelope and resolve half/double tempo."""
    try:
        # Onset envelope from percussive signal
        onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr)

        # Multiple tempo candidates
        tempos = librosa.beat.tempo(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=512,
            max_tempo=220,
            aggregate=None,
        )
        if tempos is None or len(tempos) == 0:
            raise ValueError("no tempo candidates")

        # Use strongest candidate
        tempo = float(tempos[0])

        # Resolve half/double ambiguity with simple heuristics
        # Prefer 90-190 BPM range typical for many genres
        def score(t: float) -> float:
            # Closer to mid-tempo gets a slight boost
            target = 120.0
            return -abs(t - target)

        candidates = [tempo]
        if tempo < 90:
            candidates.append(tempo * 2)
        if tempo > 180:
            candidates.append(tempo / 2)

        best = max(candidates, key=score)
        return float(best)
    except Exception:
        # Fallback: beat_track directly
        tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
        if tempo < 90:
            tempo *= 2
        if tempo > 180:
            tempo /= 2
        return float(max(60.0, min(220.0, tempo)))

def _detect_key(y_harm: np.ndarray, sr: int) -> str:
    """Detect musical key using tuned, beat-synchronous harmonic chroma and KS profiles."""
    try:
        # Estimate tuning and compute constant-Q chroma on harmonic signal
        tuning = librosa.estimate_tuning(y=y_harm, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=tuning, bins_per_octave=12, n_chroma=12)

        # Beat-synchronous average to reduce percussive/leak noise
        tempo, beats = librosa.beat.beat_track(y=y_harm, sr=sr)
        if beats is not None and len(beats) > 1:
            chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.median)
        else:
            chroma_sync = chroma

        chroma_mean = np.mean(chroma_sync, axis=1)
        chroma_sum = np.sum(chroma_mean)
        if chroma_sum > 0:
            chroma_mean = chroma_mean / chroma_sum

        # Krumhansl-Schmuckler profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)

        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        best_key = "C Major"
        best_corr = -1.0

        for i in range(12):
            maj = np.roll(major_profile, i)
            minr = np.roll(minor_profile, i)
            cmaj = np.corrcoef(chroma_mean, maj)[0, 1]
            cmin = np.corrcoef(chroma_mean, minr)[0, 1]
            if cmaj > best_corr:
                best_corr = cmaj
                best_key = f"{keys[i]} Major"
            if cmin > best_corr:
                best_corr = cmin
                best_key = f"{keys[i]} Minor"

        return best_key
    except Exception as e:
        print(f"Key detection error: {e}")
        return "C Major"

def _simple_vocal_removal(wav_path: str, tmpdir: str, stem_type: str) -> str:
    """Simple vocal removal using center channel extraction."""
    try:
        # Load stereo audio
        y, sr = librosa.load(wav_path, sr=None, mono=False)
        
        if y.ndim == 1:
            # Mono audio - can't do vocal removal
            output_path = os.path.join(tmpdir, f"{stem_type}.wav")
            sf.write(output_path, y, sr)
            return output_path
        
        if stem_type == "vocals":
            # Extract center channel (vocals are usually centered)
            vocals = (y[0] + y[1]) / 2
            output = np.array([vocals, vocals])
        elif stem_type == "instrumental":
            # Remove center channel (subtract vocals)
            instrumental = (y[0] - y[1]) / 2
            output = np.array([instrumental, instrumental])
        else:
            # For drums, bass, all - just return instrumental for now
            instrumental = (y[0] - y[1]) / 2
            output = np.array([instrumental, instrumental])
        
        # Save output
        output_path = os.path.join(tmpdir, f"{stem_type}.wav")
        sf.write(output_path, output.T, sr)
        
        print(f"[vocal_removal] Created {stem_type} stem: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Vocal removal error: {e}")
        raise HTTPException(status_code=500, detail=f"Vocal removal failed: {str(e)}")

def _format_duration(seconds: float) -> str:
    """Format duration in MM:SS format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)