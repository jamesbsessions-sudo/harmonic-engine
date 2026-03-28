FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    gcc \
    g++ \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install numpy first (required by vamp/chord-extractor setup.py)
RUN pip install --no-cache-dir numpy==1.26.4

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8000
EXPOSE 8000

CMD ["python", "app.py"]
