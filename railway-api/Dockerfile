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
# Preinstall build/runtime deps needed by legacy setup.py packages (madmom needs Cython & numpy at egg_info time)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir Cython==0.29.36 numpy==1.24.3 scipy==1.11.3 && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchaudio==2.8.0+cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Railway dynamically assigns ports
EXPOSE 8000

# Use Python directly to handle PORT env variable properly
CMD ["python", "main.py"]