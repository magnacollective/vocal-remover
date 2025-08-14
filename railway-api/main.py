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

            # Load audio with librosa (mono for analysis)
            y, sr = librosa.load(wav_path, sr=None, mono=True)
            duration_seconds = len(y) / sr

            # Separate harmonic/percussive for improved analysis
            y_harm, y_perc = librosa.effects.hpss(y)

            # Detect BPM from percussive/onset envelope; resolve half/double and expose candidates
            bpm, bpm_candidates = _detect_bpm_with_candidates(y_perc, sr)

            # Detect key from harmonic content with tuning + beat sync
            key, key_candidates = _detect_key_with_candidates(y_harm, sr)

            analysis_result = {
                "bpm": round(float(bpm), 1),
                "key": key,
                "duration": _format_duration(duration_seconds),
                "sample_rate": f"{sr} Hz",
                "bpm_candidates": [{"bpm": round(float(b), 1), "confidence": float(c)} for b, c in bpm_candidates[:5]],
                "key_candidates": [{"key": k, "confidence": float(c)} for k, c in key_candidates[:5]],
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

def _detect_bpm_with_candidates(y_perc: np.ndarray, sr: int) -> tuple[float, list[tuple[float, float]]]:
    """Detect BPM robustly using onset envelope, tempogram autocorrelation, and resolve half/double.
    Returns (best_bpm, candidates[(bpm, confidence)...])
    """
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length, aggregate=np.median)
    if onset_env.size == 0 or np.all(onset_env == 0):
        # Fallback to beat_track on raw signal
        tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
        tempo = float(tempo)
        if tempo < 90:
            tempo *= 2
        if tempo > 180:
            tempo /= 2
        return tempo, [(tempo, 0.5)]

    # Autocorrelation for candidate lags
    ac = librosa.autocorrelate(onset_env, max_size=onset_env.shape[0] // 2)
    ac[:2] = 0  # suppress 0/1 lag

    # Convert lags to BPM
    lags = np.arange(1, len(ac))
    bpms = 60.0 * sr / (hop_length * lags)

    # Keep plausible BPM range
    mask = (bpms >= 60) & (bpms <= 220)
    bpms = bpms[mask]
    strengths = ac[1:][mask]

    # Peak pick
    peaks, _ = signal.find_peaks(strengths, distance=2)
    if len(peaks) == 0:
        # fallback to librosa tempo candidates
        tempos = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=None)
        if tempos is None or len(tempos) == 0:
            tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
            return float(tempo), [(float(tempo), 0.5)]
        # Normalize confidences
        confs = np.linspace(1.0, 0.5, num=min(5, len(tempos)))
        chosen = list(zip(tempos[:5], confs))
        best = float(tempos[0])
        # half/double adjustment
        if best < 90:
            best *= 2
        if best > 180:
            best /= 2
        return float(best), [(float(b), float(c)) for b, c in chosen]

    cand_bpms = bpms[peaks]
    cand_strengths = strengths[peaks]

    # Normalize strengths to [0,1]
    if cand_strengths.max() > 0:
        cand_confs = (cand_strengths - cand_strengths.min()) / (cand_strengths.max() - cand_strengths.min() + 1e-9)
    else:
        cand_confs = np.zeros_like(cand_strengths)

    # Generate half/double variants and pick best scoring per equivalence class
    scored: list[tuple[float, float]] = []
    for bpm, conf in zip(cand_bpms, cand_confs):
        variants = {bpm}
        if bpm < 90:
            variants.add(bpm * 2)
        if bpm > 180:
            variants.add(bpm / 2)
        for v in variants:
            # score prefers mid-tempo and higher confidence
            score = conf - (abs(v - 120.0) / 200.0)
            scored.append((float(v), float(max(0.0, score))))

    # Deduplicate by rounding BPM
    agg: dict[int, float] = {}
    for b, s in scored:
        key = int(round(b))
        agg[key] = max(agg.get(key, 0.0), s)
    # Sort by score desc
    sorted_cands = sorted(((float(k), v) for k, v in agg.items()), key=lambda x: x[1], reverse=True)
    best_bpm = sorted_cands[0][0] if sorted_cands else float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr))
    return best_bpm, sorted_cands

def _detect_key_with_candidates(y_harm: np.ndarray, sr: int) -> tuple[str, list[tuple[str, float]]]:
    """Detect musical key and provide candidates with confidence using tuned, beat-synchronous chroma.
    Returns (best_key, [(key, confidence)...])
    """
    try:
        # Estimate tuning and compute chroma CQT on harmonic signal
        tuning = librosa.estimate_tuning(y=y_harm, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=tuning, bins_per_octave=12, n_chroma=12)

        # Beat-synchronous median aggregation
        tempo, beats = librosa.beat.beat_track(y=y_harm, sr=sr)
        if beats is not None and len(beats) > 1:
            chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.median)
        else:
            chroma_sync = chroma

        chroma_mean = np.mean(chroma_sync, axis=1)
        s = np.sum(chroma_mean)
        if s > 0:
            chroma_mean = chroma_mean / s

        # KS profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        candidates: list[tuple[str, float]] = []
        for i in range(12):
            cmaj = float(np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1])
            cmin = float(np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1])
            candidates.append((f"{keys[i]} Major", cmaj))
            candidates.append((f"{keys[i]} Minor", cmin))

        # Sort by correlation (confidence)
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_key, best_conf = candidates[0]

        # Normalize confidences to 0..1 by top score
        top = max(c for _, c in candidates) or 1.0
        norm_cands = [(k, c / top) for k, c in candidates]
        return best_key, norm_cands
    except Exception as e:
        print(f"Key detection error: {e}")
        return "C Major", [("C Major", 0.0)]

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