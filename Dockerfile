FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN addgroup --gid 1001 fastapigroup && adduser --uid 1001  \
    --disabled-password  \
    --gecos "" fastapiuser  \
    --ingroup fastapigroup

RUN apt-get update && apt-get install -y --no-install-recommends  \
    libgl1 \
    libglib2.0-0 \
    libstdc++6 \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
  && python -m pip install --no-cache-dir -r requirements.txt

COPY . .
USER fastapiuser
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]