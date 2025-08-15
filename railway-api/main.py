from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from contextlib import asynccontextmanager
import tempfile
import os
import shutil
import subprocess
import librosa
import numpy as np
from scipy import signal
try:
    import madmom
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    _HAVE_MADMOM = True
except Exception as e:
    print(f"[WARNING] madmom not available: {e}")
    print("[INFO] Using librosa for all tempo detection (madmom disabled)")
    _HAVE_MADMOM = False

try:
    from demucs import pretrained
    from demucs.separate import load_track, apply_model
    from demucs.audio import convert_audio, save_audio
    import torch
    _HAVE_DEMUCS = True
except Exception:
    _HAVE_DEMUCS = False
    
import soundfile as sf

# Global model cache for performance
_CACHED_MODEL = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup for immediate availability."""
    global _CACHED_MODEL
    if _HAVE_DEMUCS:
        # Optimize torch for multi-CPU performance
        import torch
        num_threads = min(32, os.cpu_count() or 4)  # Use up to 32 threads
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        print(f"[startup] Configured torch with {num_threads} threads")
        
        print("[startup] Loading htdemucs model...")
        _CACHED_MODEL = pretrained.get_model('htdemucs')
        _CACHED_MODEL.eval()
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            _CACHED_MODEL = _CACHED_MODEL.cuda()
            print("[startup] Model loaded on GPU")
        else:
            print("[startup] Model loaded on CPU")
        print("[startup] Model ready - all endpoints operational")
    else:
        print("[startup] Demucs not available, using fallback methods")
    yield
    # Cleanup on shutdown
    _CACHED_MODEL = None

app = FastAPI(title="Vocal Remover & Audio Analysis API", lifespan=lifespan)

def get_cached_model():
    """Get cached htdemucs model (loaded at startup)."""
    return _CACHED_MODEL

# CORS configuration (env-driven with safe defaults)
ALLOWED_ORIGINS_STR = os.getenv("ALLOWED_ORIGINS", "https://www.studiobuddy.xyz,https://studiobuddy.xyz,https://studio-buddy-web.vercel.app,http://localhost:3000")
# Clean up origins: handle both comma and semicolon separators, remove quotes
if ALLOWED_ORIGINS_STR == "*":
    ALLOWED_ORIGINS = ["*"]
else:
    # Replace semicolons with commas, then split
    cleaned = ALLOWED_ORIGINS_STR.replace(';', ',')
    ALLOWED_ORIGINS = [
        origin.strip().strip('"').strip("'") 
        for origin in cleaned.split(",") 
        if origin.strip() and origin.strip() != ','
    ]
print(f"[CORS] Allowed origins: {ALLOWED_ORIGINS}")

# Use only FastAPI's CORSMiddleware to avoid duplicate headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "service": "vocal-remover-api", "port": os.getenv("PORT", "8000")}

@app.get("/version")
def version():
    """Expose build metadata to verify deployments."""
    sha = os.getenv("RAILWAY_GIT_COMMIT_SHA") or os.getenv("GIT_SHA") or "unknown"
    ts = os.getenv("BUILD_TIME") or "unknown"
    return {"commit": sha, "build_time": ts, "allowed_origins": ALLOWED_ORIGINS}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": _CACHED_MODEL is not None,
        "demucs_available": _HAVE_DEMUCS
    }

@app.get("/warmup")
def warmup():
    """Warmup endpoint to preload model if needed."""
    model = get_cached_model()
    return {
        "status": "warmed",
        "model_ready": model is not None,
        "demucs_available": _HAVE_DEMUCS
    }

@app.post("/analyze")
async def analyze_audio(
    audio: UploadFile = File(...),
    window_sec: int = Query(75, ge=15, le=180, description="Center window length to analyze in seconds"),
    prefer_min_bpm: int = Query(90, ge=40, le=200, description="Lower bound of preferred BPM range"),
    prefer_max_bpm: int = Query(180, ge=60, le=240, description="Upper bound of preferred BPM range"),
    genre: str | None = Query(None, description="Optional genre hint (edm, hiphop, pop, house, techno, trap)"),
    profile: str = Query("fast", regex="^(fast|accurate)$", description="Analysis profile"),
    backend: str = Query("librosa", regex="^(librosa|pro)$", description="Analysis backend"),
):
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

            # Adjust defaults by genre/profile
            if genre:
                g = genre.lower()
                if g in {"edm", "house", "techno", "trance"}:
                    prefer_min_bpm, prefer_max_bpm = 110, 200
                elif g in {"hiphop", "trap"}:
                    prefer_min_bpm, prefer_max_bpm = 60, 180
                elif g in {"pop", "rock"}:
                    prefer_min_bpm, prefer_max_bpm = 70, 190

            if profile == "accurate":
                window = min(60, max(30, window_sec))
            else:
                window = min(30, max(15, window_sec))

            # Convert to speedy analysis WAV (mono, 22.05kHz, middle window)
            wav_path = _prepare_analysis_wav(input_path, tmpdir, window_sec=window)
            print(f"[analyze] Processing: {wav_path}")

            # Load audio with librosa (mono for analysis)
            y, sr = librosa.load(wav_path, sr=22050, mono=True)
            duration_seconds = len(y) / sr

            if profile == "fast":
                # Fast but reasonably accurate
                bpm, bpm_candidates = _detect_bpm_fast(y, sr, prefer_min_bpm, prefer_max_bpm)
                key, key_candidates = _detect_key_fast(y, sr)
            else:
                # Accurate path: use better algorithms
                y_harm, y_perc = librosa.effects.hpss(y)
                # Use librosa's robust BPM detection
                bpm, bpm_candidates = _detect_bpm_with_candidates(y_perc, sr, prefer_min_bpm, prefer_max_bpm)
                # Use better key detection 
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

        # Use AI-based separation if available, fallback to simple method
        if _HAVE_DEMUCS:
            output_path = _ai_vocal_separation(wav_path, tmpdir, stem_type)
        else:
            print("[separate] Warning: Using fallback simple vocal removal (low quality)")
            output_path = _simple_vocal_removal(wav_path, tmpdir, stem_type)

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Stem separation failed: output not created")

        # Stream the file to avoid occasional FileResponse stat race on ephemeral filesystems
        file_handle = open(output_path, "rb")

        def _cleanup():
            try:
                try:
                    file_handle.close()
                except Exception:
                    pass
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

        return StreamingResponse(
            file_handle,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={stem_type}_separated.wav"
            },
            background=BackgroundTask(_cleanup),
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

def _prepare_analysis_wav(input_path: str, tmpdir: str, window_sec: int = 30) -> str:
    """Convert input to fast-to-analyze mono 22.05kHz WAV and trim to the most informative middle window.
    This speeds up analysis and reduces intro/outro bias.
    """
    probe = os.path.join(tmpdir, "probe.wav")
    # First convert to mono 22.05kHz
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "warning",
        "-i", input_path,
        "-vn", "-ac", "1", "-ar", "22050", "-c:a", "pcm_s16le",
        probe,
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {e}")

    # Load to get duration and trim middle segment (use soundfile for speed)
    data, sr = sf.read(probe, dtype='float32', always_2d=False)
    if data.ndim > 1:
        y = data.mean(axis=1)
    else:
        y = data
    total_sec = len(y) / sr
    win = max(15, min(window_sec, int(total_sec)))
    start = max(0, int((total_sec - win) / 2))
    end = start + win
    trimmed = y[start * sr : end * sr]
    out = os.path.join(tmpdir, "analysis.wav")
    sf.write(out, trimmed, sr)
    return out

def _detect_bpm_with_candidates(y_perc: np.ndarray, sr: int, prefer_min: int = 90, prefer_max: int = 180) -> tuple[float, list[tuple[float, float]]]:
    """Detect BPM robustly using onset envelope, tempogram autocorrelation, and resolve half/double.
    Returns (best_bpm, candidates[(bpm, confidence)...])
    """
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length, aggregate=np.median)
    if onset_env.size == 0 or np.all(onset_env == 0):
        # Fallback to beat_track on raw signal
        tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
        tempo = float(tempo)
        if tempo < prefer_min:
            tempo *= 2
        if tempo > prefer_max:
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
        if best < prefer_min:
            best *= 2
        if best > prefer_max:
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
        if bpm < prefer_min:
            variants.add(bpm * 2)
        if bpm > prefer_max:
            variants.add(bpm / 2)
        for v in variants:
            # score prefers mid-tempo and higher confidence
            mid = (prefer_min + prefer_max) / 2.0
            score = conf - (abs(v - mid) / (prefer_max - prefer_min))
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

# _madmom_bpm function removed - madmom disabled for Python 3.11+ compatibility

def _detect_bpm_fast(y: np.ndarray, sr: int, prefer_min: int, prefer_max: int) -> tuple[float, list[tuple[float, float]]]:
    """Fast but accurate BPM detection using librosa's tempo detection."""
    try:
        # Use librosa's built-in tempo detection with multiple candidates
        tempo_candidates = librosa.beat.tempo(y=y, sr=sr, aggregate=None, max_tempo=300)
        
        if tempo_candidates is None or len(tempo_candidates) == 0:
            # Fallback to beat_track
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            return float(tempo), [(float(tempo), 0.8)]
        
        # Score candidates based on preference range
        scored_candidates = []
        for tempo in tempo_candidates[:5]:  # Top 5 candidates
            # Prefer tempos in the specified range
            score = 1.0
            if tempo < prefer_min:
                score = 0.5
                tempo = tempo * 2  # Try doubling
            elif tempo > prefer_max:
                score = 0.5
                tempo = tempo / 2  # Try halving
            
            # Final range check after adjustment
            if prefer_min <= tempo <= prefer_max:
                score += 0.3
                
            scored_candidates.append((float(tempo), float(score)))
        
        # Sort by score and return best
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_tempo = scored_candidates[0][0]
        
        return best_tempo, scored_candidates[:3]
        
    except Exception as e:
        print(f"Fast BPM detection error: {e}")
        # Fallback
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo), [(float(tempo), 0.5)]

def _detect_key_fast(y: np.ndarray, sr: int) -> tuple[str, list[tuple[str, float]]]:
    """Simple and reliable key detection using basic chroma features."""
    try:
        # Use CQT chroma for better pitch resolution
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
        
        # Use weighted average - emphasize stronger frames
        frame_strength = np.sum(chroma, axis=0)
        weights = frame_strength / (frame_strength.sum() + 1e-8)
        chroma_mean = np.average(chroma, axis=1, weights=weights)
        
        # Normalize 
        if chroma_mean.sum() > 0:
            chroma_mean = chroma_mean / chroma_mean.sum()
        
        # Improved templates with chord emphasis (root, third, fifth)
        major_template = np.array([3.0, 0.5, 2.0, 0.5, 2.5, 2.0, 0.5, 3.0, 0.5, 2.0, 0.5, 1.5])  # Emphasize I, iii, V
        minor_template = np.array([3.0, 0.5, 1.5, 2.5, 0.5, 2.0, 0.5, 3.0, 2.0, 0.5, 1.5, 0.5])  # Emphasize i, III, v
        
        major_template = major_template / major_template.sum()
        minor_template = minor_template / minor_template.sum()
        
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        scores = []
        
        # Test all 24 keys (12 major + 12 minor)
        for i in range(12):
            # Rotate templates to different keys
            maj_rotated = np.roll(major_template, i)
            min_rotated = np.roll(minor_template, i)
            
            # Calculate similarity (dot product)
            maj_score = np.dot(chroma_mean, maj_rotated)
            min_score = np.dot(chroma_mean, min_rotated)
            
            scores.append((f"{keys[i]} Major", float(maj_score)))
            scores.append((f"{keys[i]} Minor", float(min_score)))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return best match
        best_key = scores[0][0]
        
        # Boost confidence based on separation from second-best
        top_score = scores[0][1]
        second_score = scores[1][1] if len(scores) > 1 else 0.0
        separation = top_score - second_score
        confidence_boost = min(1.0, separation * 2.0)  # Boost if clear winner
        
        # Normalize confidence scores with boost
        max_score = scores[0][1] if scores[0][1] > 0 else 1.0
        normalized_scores = []
        for i, (key, score) in enumerate(scores[:5]):
            conf = score / max_score
            if i == 0:  # Boost top result
                conf = min(1.0, conf + confidence_boost * 0.2)
            normalized_scores.append((key, conf))
        
        return best_key, normalized_scores
        
    except Exception as e:
        print(f"Key detection error: {e}")
        return "C Major", [("C Major", 1.0)]

def _detect_key_with_candidates(y_harm: np.ndarray, sr: int) -> tuple[str, list[tuple[str, float]]]:
    """Detect musical key with two robust chroma variants and pick the stronger one.
    Returns (best_key, [(key, confidence)...])
    """
    try:
        # KS profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        def corr_candidates(chroma_mat: np.ndarray) -> list[tuple[str, float]]:
            chroma_mean = np.mean(chroma_mat, axis=1)
            s = float(np.sum(chroma_mean))
            if s > 0:
                chroma_vec = chroma_mean / s
            else:
                chroma_vec = chroma_mean
            cands: list[tuple[str, float]] = []
            best = -1.0
            for i in range(12):
                cmaj = float(np.corrcoef(chroma_vec, np.roll(major_profile, i))[0, 1])
                cmin = float(np.corrcoef(chroma_vec, np.roll(minor_profile, i))[0, 1])
                cands.append((f"{keys[i]} Major", cmaj))
                cands.append((f"{keys[i]} Minor", cmin))
                best = max(best, cmaj, cmin)
            cands.sort(key=lambda x: x[1], reverse=True)
            # Normalize by top
            top = cands[0][1] if cands else 1.0
            return [(k, (c / top) if top else 0.0) for k, c in cands]

        # Variant A: tuned CQT chroma at 36 bins, fold to 12
        tuning = librosa.estimate_tuning(y=y_harm, sr=sr)
        chroma36 = librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=tuning, bins_per_octave=36, n_chroma=36)
        tempo, beats = librosa.beat.beat_track(y=y_harm, sr=sr)
        if beats is not None and len(beats) > 1:
            chroma36 = librosa.util.sync(chroma36, beats, aggregate=np.median)
        # fold 36 -> 12 by summing every 3 bins
        if chroma36.shape[0] == 36:
            chroma12_a = chroma36.reshape(12, 3, -1).sum(axis=1)
        else:
            chroma12_a = librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=tuning, bins_per_octave=12, n_chroma=12)

        cands_a = corr_candidates(chroma12_a)
        best_a = cands_a[0][1] if cands_a else 0.0

        # Variant B: chroma CENS (robust to dynamics)
        chroma_cens = librosa.feature.chroma_cens(y=y_harm, sr=sr)
        if beats is not None and len(beats) > 1:
            chroma_cens = librosa.util.sync(chroma_cens, beats, aggregate=np.median)
        cands_b = corr_candidates(chroma_cens)
        best_b = cands_b[0][1] if cands_b else 0.0

        # Choose variant with stronger top confidence, return merged top-10 for transparency
        chosen = cands_a if best_a >= best_b else cands_b
        best_key = chosen[0][0] if chosen else "C Major"
        return best_key, chosen[:10]
    except Exception as e:
        print(f"Key detection error: {e}")
        return "C Major", [("C Major", 0.0)]

def _ai_vocal_separation(wav_path: str, tmpdir: str, stem_type: str) -> str:
    """AI-based vocal separation using Demucs with chunking for large files."""
    try:
        print(f"[ai_separation] Starting AI separation for {stem_type}")
        
        # Use cached htdemucs model
        model = get_cached_model()
        if model is None:
            raise HTTPException(status_code=500, detail="Demucs model not available")
        
        # Load and convert audio
        wav = load_track(wav_path, model.audio_channels, model.samplerate)
        print(f"[ai_separation] Loaded audio: {wav.shape}")
        
        # Determine device (model may already be on GPU from startup)
        device = next(model.parameters()).device
        print(f"[ai_separation] Using device: {device}")
        
        # Check if file is large and needs chunking (>3 minutes = 180 seconds)
        duration_sec = wav.shape[-1] / model.samplerate
        chunk_size_sec = 60  # Process in 60-second chunks
        
        if duration_sec > 180:
            print(f"[ai_separation] Large file ({duration_sec:.1f}s), using chunking")
            sources = _process_in_chunks(model, wav, device, chunk_size_sec)
        else:
            print(f"[ai_separation] Small file ({duration_sec:.1f}s), processing whole")
            ref = wav.mean(0)  
            wav = (wav - ref.mean()) / ref.std()  # Normalize
            sources = apply_model(model, wav[None], device=device, progress=True)[0]
            sources = sources * ref.std() + ref.mean()  # Denormalize
        
        # Get source names (typically: drums, bass, other, vocals)
        source_names = model.sources
        print(f"[ai_separation] Available sources: {source_names}")
        
        # Map requested stem to model sources
        if stem_type == "instrumental":
            # Combine all non-vocal sources
            vocal_idx = source_names.index('vocals') if 'vocals' in source_names else -1
            if vocal_idx >= 0:
                # Sum all sources except vocals
                output_audio = torch.sum(sources[[i for i in range(len(sources)) if i != vocal_idx]], dim=0)
            else:
                output_audio = torch.sum(sources, dim=0)  # Fallback
        elif stem_type in source_names:
            # Get specific source
            source_idx = source_names.index(stem_type)
            output_audio = sources[source_idx]
        else:
            raise HTTPException(status_code=400, detail=f"Stem type '{stem_type}' not available in model sources: {source_names}")
        
        # Save output
        output_path = os.path.join(tmpdir, f"{stem_type}.wav")
        print(f"[ai_separation] Writing to: {output_path}")
        print(f"[ai_separation] Output shape: {output_audio.shape}, dtype: {output_audio.dtype}")
        
        # Convert to numpy and save
        output_numpy = output_audio.detach().cpu().numpy()
        # Demucs outputs (channels, samples), transpose for soundfile (samples, channels)
        if output_numpy.ndim == 2:
            output_numpy = output_numpy.T
        
        sf.write(output_path, output_numpy, model.samplerate)
        
        # Verify file was created
        if not os.path.exists(output_path):
            print(f"[ai_separation] ERROR: File was not created at {output_path}")
            raise HTTPException(status_code=500, detail=f"Failed to create output file: {output_path}")
        
        file_size = os.path.getsize(output_path)
        print(f"[ai_separation] Created {stem_type} stem: {output_path} (size: {file_size} bytes)")
        return output_path
        
    except Exception as e:
        print(f"[ai_separation] Error: {e}")
        print(f"[ai_separation] Falling back to simple vocal removal")
        # Fallback to simple method if AI separation fails
        return _simple_vocal_removal(wav_path, tmpdir, stem_type)

def _process_in_chunks(model, wav, device, chunk_size_sec=60):
    """Process large audio files in chunks to reduce memory usage and improve progress feedback."""
    import torch
    
    sr = model.samplerate
    chunk_samples = int(chunk_size_sec * sr)
    total_samples = wav.shape[-1]
    n_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    
    print(f"[chunking] Processing {n_chunks} chunks of {chunk_size_sec}s each")
    
    # Initialize output tensors
    all_sources = []
    
    for i in range(n_chunks):
        start_idx = i * chunk_samples
        end_idx = min((i + 1) * chunk_samples, total_samples)
        chunk = wav[..., start_idx:end_idx]
        
        print(f"[chunking] Processing chunk {i+1}/{n_chunks}")
        
        # Normalize chunk
        ref = chunk.mean(0)
        chunk_norm = (chunk - ref.mean()) / (ref.std() + 1e-8)
        
        # Process chunk
        with torch.no_grad():  # Save memory
            chunk_sources = apply_model(model, chunk_norm[None], device=device, progress=False)[0]
            chunk_sources = chunk_sources * ref.std() + ref.mean()
        
        all_sources.append(chunk_sources.cpu())  # Move to CPU to save GPU memory
        
    # Concatenate all chunks
    sources = torch.cat(all_sources, dim=-1)
    return sources

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
            output = np.stack([vocals, vocals])
        elif stem_type == "instrumental":
            # Remove center channel (subtract vocals)
            instrumental = (y[0] - y[1]) / 2
            output = np.stack([instrumental, instrumental])
        else:
            # For drums, bass, all - just return instrumental for now
            instrumental = (y[0] - y[1]) / 2
            output = np.stack([instrumental, instrumental])
        
        # Save output
        output_path = os.path.join(tmpdir, f"{stem_type}.wav")
        print(f"[vocal_removal] Writing to: {output_path}")
        print(f"[vocal_removal] Output shape: {output.shape}, dtype: {output.dtype}")
        
        # Write as (frames, channels) - transpose to get correct shape
        try:
            sf.write(output_path, output.T, sr)
            print(f"[vocal_removal] sf.write completed")
        except Exception as write_error:
            print(f"[vocal_removal] sf.write failed: {write_error}")
            raise
        
        # Verify file was actually created
        if not os.path.exists(output_path):
            print(f"[vocal_removal] ERROR: File was not created at {output_path}")
            raise HTTPException(status_code=500, detail=f"Failed to create output file: {output_path}")
        
        file_size = os.path.getsize(output_path)
        print(f"[vocal_removal] Created {stem_type} stem: {output_path} (size: {file_size} bytes)")
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
    # Railway needs PORT env variable
    port = int(os.getenv("PORT", 8000))
    print(f"[MAIN] Starting server on port {port}", flush=True)
    print(f"[MAIN] Environment PORT: {os.getenv('PORT', 'NOT SET')}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")