FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FACEFUSION_ROOT=/opt/facefusion \
    FACEFUSION_PYTHON=/opt/facefusion/.venv/bin/python \
    MODEL_WEIGHTS_DIR=/app/weights \
    LD_LIBRARY_PATH=/opt/facefusion/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:\
/opt/facefusion/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:\
/opt/facefusion/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:\
/opt/facefusion/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:\
/opt/facefusion/.venv/lib/python3.12/site-packages/nvidia/cufft/lib:\
/opt/facefusion/.venv/lib/python3.12/site-packages/nvidia/curand/lib:\
/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    python3.12 \
    python3.12-venv \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/facefusion
RUN git init . \
  && git remote add origin https://github.com/facefusion/facefusion.git \
  && git fetch --depth 1 origin decc6c9d8f8e014878624452c6e8fe208726b4ec \
  && git checkout --detach FETCH_HEAD \
  && python3.12 -m venv .venv

ENV PATH=/opt/facefusion/.venv/bin:$PATH
RUN python install.py cuda@12 --skip-conda \
  && python -m pip install --no-cache-dir onnxruntime-gpu[cuda,cudnn]==1.24.4

WORKDIR /app
COPY face_swap_video_worker/requirements.txt requirements.txt
RUN python3.12 -m venv /opt/worker-venv \
  && /opt/worker-venv/bin/pip install --no-cache-dir -r requirements.txt

COPY services/__init__.py services/__init__.py
COPY services/video_job_manager.py services/video_job_manager.py
COPY utils/__init__.py utils/__init__.py
COPY utils/model_weights.py utils/model_weights.py
COPY face_swap_video_worker face_swap_video_worker

RUN mkdir -p /app/weights/face-swap/facefusion/.assets \
  && ln -s /app/weights/face-swap/facefusion/.assets /opt/facefusion/.assets \
  && addgroup --gid 1001 fastapigroup \
  && adduser --uid 1001 --disabled-password --gecos "" --ingroup fastapigroup fastapiuser \
  && chown -R fastapiuser:fastapigroup /app /opt/facefusion

USER fastapiuser
EXPOSE 8001
CMD ["/opt/worker-venv/bin/uvicorn", "face_swap_video_worker.main:app", "--host", "0.0.0.0", "--port", "8001"]
