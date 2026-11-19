FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS sd-server-build

RUN sed -i 's|http://security.ubuntu.com|https://security.ubuntu.com|g' \
    /etc/apt/sources.list.d/ubuntu.sources \
  && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /src/stable-diffusion.cpp
RUN git clone --depth 1 --recurse-submodules https://github.com/leejet/stable-diffusion.cpp.git . \
  && cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89-virtual \
    -DSD_CUDA=ON \
    -DSD_SERVER_BUILD_FRONTEND=OFF \
  && cmake --build build --config Release --target sd-server --parallel 2

FROM python:3.13-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FACE_SWAP_SD_CPP_SERVER_PATH=/usr/local/bin/sd-server \
    MODEL_WEIGHTS_DIR=/app/weights \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libstdc++6 \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

COPY --from=sd-server-build /src/stable-diffusion.cpp/build/bin/sd-server /usr/local/bin/sd-server
COPY --from=sd-server-build /usr/local/cuda-12.8/targets/x86_64-linux/lib/ /usr/local/cuda/lib64/
COPY --from=sd-server-build /usr/lib/x86_64-linux-gnu/libnccl.so.2* /usr/lib/x86_64-linux-gnu/

RUN addgroup --gid 1001 fastapigroup && adduser --uid 1001 \
    --disabled-password \
    --gecos "" fastapiuser \
    --ingroup fastapigroup \
  && mkdir -p /app/weights \
  && chown -R fastapiuser:fastapigroup /app

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
  && python -m pip install --no-cache-dir -r requirements.txt

COPY . .
USER fastapiuser
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]