from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import shutil
import subprocess
import librosa
import numpy as np
from scipy import signal
import soundfile as sf

app = FastAPI(title="Vocal Remover & Audio Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

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
            y, sr = librosa.load(wav_path, sr=None)
            duration_seconds = len(y) / sr
            
            # Detect BPM
            bpm = _detect_bpm(y, sr)
            
            # Detect key
            key = _detect_key(y, sr)
            
            # Format results
            analysis_result = {
                "bpm": round(float(bpm), 1),
                "key": key,
                "duration": _format_duration(duration_seconds),
                "sample_rate": f"{sr} Hz"
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
    
    with tempfile.TemporaryDirectory() as tmpdir:
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
            
            # Return the separated audio file
            return FileResponse(
                output_path,
                media_type="audio/wav",
                filename=f"{stem_type}_separated.wav",
            )
            
        except Exception as e:
            print(f"[separate] Error: {e}")
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

def _detect_bpm(y: np.ndarray, sr: int) -> float:
    """Detect BPM using librosa's tempo detection."""
    try:
        # Use librosa's tempo detection
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)
    except:
        # Fallback: simple onset-based detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        if len(onset_frames) < 2:
            return 120.0  # Default BPM
        
        # Calculate average time between onsets
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        intervals = np.diff(onset_times)
        if len(intervals) == 0:
            return 120.0
            
        avg_interval = np.median(intervals)
        bpm = 60.0 / avg_interval if avg_interval > 0 else 120.0
        
        # Clamp to reasonable range
        return max(60.0, min(200.0, bpm))

def _detect_key(y: np.ndarray, sr: int) -> str:
    """Detect musical key using chroma features."""
    try:
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Normalize
        chroma_mean = chroma_mean / np.sum(chroma_mean)
        
        # Key profiles (Krumhansl-Schmuckler)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Normalize profiles
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)
        
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        
        best_correlation = -1
        best_key = "C Major"
        
        for i in range(12):
            # Test major
            major_shifted = np.roll(major_profile, i)
            correlation = np.corrcoef(chroma_mean, major_shifted)[0, 1]
            if correlation > best_correlation:
                best_correlation = correlation
                best_key = f"{keys[i]} Major"
            
            # Test minor
            minor_shifted = np.roll(minor_profile, i)
            correlation = np.corrcoef(chroma_mean, minor_shifted)[0, 1]
            if correlation > best_correlation:
                best_correlation = correlation
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