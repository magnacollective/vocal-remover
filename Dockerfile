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
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port (Railway uses PORT env var)
EXPOSE $PORT

# Run the application with Railway's PORT variable
CMD uvicorn main:app --host 0.0.0.0 --port $PORT