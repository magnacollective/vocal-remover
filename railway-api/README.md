# Vocal Remover & Audio Analysis Railway API

This is a FastAPI service that provides:
- **BPM Detection** - Analyze tempo of audio files
- **Key Detection** - Identify musical key using chroma features  
- **Vocal Removal** - Simple center-channel vocal extraction
- **Stem Separation** - Basic instrumental/vocal separation

## API Endpoints

### POST /analyze
Analyze audio file for BPM, key, duration, and sample rate.

**Request:** 
- `audio`: Audio file (MP3, WAV, etc.)

**Response:**
```json
{
    "bpm": 128.0,
    "key": "A Minor", 
    "duration": "3:24",
    "sample_rate": "44100 Hz"
}
```

### POST /separate
Separate audio into stems.

**Request:**
- `audio`: Audio file
- `stem_type`: "vocals", "instrumental", "drums", "bass", or "all"

**Response:** WAV file of separated stem

## Railway Deployment

1. Create new Railway project
2. Connect to GitHub repository
3. Set root directory to `railway-api/`
4. Railway will automatically detect Dockerfile and deploy

## Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## Dependencies

- FastAPI - Web framework
- librosa - Audio analysis  
- soundfile - Audio I/O
- FFmpeg - Audio conversion
- scipy/numpy - Signal processing