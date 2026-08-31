# onnxruntime-gpu 1.26 needs CUDA 12 + cuDNN 9; ubuntu24.04 gives python 3.12
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "import cv2, numpy, onnxruntime, fastapi, uvicorn, multipart"

COPY deoldify_onnx.py server.py ./

# mount the models dir at run time: -v ./models:/app/models
EXPOSE 8080
ENTRYPOINT ["python", "server.py"]
