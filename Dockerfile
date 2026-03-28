FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    gcc \
    g++ \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies (madmom installed separately due to build quirks)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir madmom==0.16.1

# Copy application code
COPY app.py .

# Railway provides PORT env variable
ENV PORT=8000
EXPOSE 8000

CMD ["python", "app.py"]
