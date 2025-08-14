#!/usr/bin/env python3
"""Minimal test server to debug Railway deployment"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Minimal Test API")

# Simple CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "ok", 
        "message": "minimal server working",
        "port": os.getenv("PORT", "unknown")
    }

@app.get("/health")
def health():
    return {"status": "healthy", "minimal": True}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"[MINIMAL] Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)