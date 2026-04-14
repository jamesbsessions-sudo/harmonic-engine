FROM python:3.10-slim
RUN apt-get update && apt-get install -y ffmpeg git gcc g++ libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir torch==2.1.2+cpu torchaudio==2.1.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('lj1995/VoiceConversionWebUI', 'rmvpe.pt', local_dir='models/')"
RUN python -c "from basic_pitch.inference import predict; print('Basic Pitch ONNX model OK')"
COPY app.py .
COPY rmvpe_model.py .
COPY rmvpe_pitch.py .
ENV PORT=8000
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV KLANGIO_API_KEY=0xkl-a93cae920b443f0888a48830ac27821a
EXPOSE 8000
CMD ["python", "app.py"]
