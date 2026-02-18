FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV & matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ── CPU-only ML wheels (saves ~3-4 GB vs GPU versions) ──────
# Install CPU-only PyTorch BEFORE sentence-transformers picks up the 2 GB CUDA build
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install CPU-only TensorFlow BEFORE deepface pulls the 2 GB GPU build
RUN pip install --no-cache-dir tensorflow-cpu

# Now install everything else (already-installed packages are skipped)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
