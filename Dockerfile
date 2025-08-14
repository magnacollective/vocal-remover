FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Ensure latest packaging tools for building wheels
# Preinstall build/runtime deps needed by legacy setup.py packages (madmom needs Cython & numpy at egg_info time)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir Cython==0.29.36 numpy==1.24.3 scipy==1.11.3 && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchaudio==2.8.0+cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port (Railway uses PORT env var)
EXPOSE $PORT

# Run the application with Railway's PORT variable
CMD uvicorn main:app --host 0.0.0.0 --port $PORT