# Machine Learning FastAPI back-end

## Deploy Using Docker with NVIDIA GPU

### Prerequisites

Docker Desktop with NVIDIA GPU access enabled, an NVIDIA driver, and the NVIDIA
Container Toolkit (or the equivalent GPU support for your Docker host).

---

### Steps

1. Build the image from the repository root and push it to Docker Hub:

```shell
docker login
docker build -t lendoo73/ml-api:gpu -f Dockerfile .
docker push lendoo73/ml-api:gpu
```

2. On the NVIDIA-enabled desktop, clone the repository and open a terminal in
   its `deploy` folder to pull and start the published image:

```shell
cd path/to/ml-api/deploy
docker compose pull
docker compose up -d
```
3. Confirm that PyTorch can see the GPU:
```shell
docker compose exec api python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```
The final value should be `True`. Open the application in your browser:
👉 http://localhost:8002/

Downloaded model weights are stored under `/app/weights` and persisted in the
`model_weights` Docker volume across container recreations. `docker compose down -v`
also deletes this volume and its downloaded models.

The[web UI](https://github.com/lendoo73/ml-react) is served by FastAPI from the /build directory  

The API docs are available at:
👉 http://localhost:8002/api/docs  
👉 http://localhost:8002/api/redoc

## Sending email

`POST /api/mail/send` accepts `multipart/form-data` with `to` (recipient email),
`text` (plain text or HTML), and `image` (JPEG, PNG, GIF, or WebP attachment; up to
10 MB). Configure `MAIL_SMTP_SERVER` and `MAIL_FROM_ADDRESS`; optional SMTP settings
are listed in `deploy/sample.env`. The endpoint sends HTML as the message body and
the image as an attachment.

## Pose angles

Pose prediction endpoints accept an optional JSON-encoded `angles` parameter. Each triplet identifies three keypoints in order; the middle keypoint is the pivot. Use either three COCO keypoint names or three zero-based indices per triplet:

```json
[["left_shoulder", "left_elbow", "left_wrist"], [5, 7, 9]]
```

Send `angles` as a URL-encoded query parameter for all HTTP requests, and as a query parameter in the webcam WebSocket URL. For file uploads, only the media file (`source`) is sent in the multipart request body; `angles` is not a form field. Angle results require JSON output; combining `angles` with `output=image` or `output=video` returns a validation error. Angle results include clockwise image-plane degrees in `[0, 360)`, resolved names and indices, keypoint confidence values, and validity. Confidence values are informational; clients choose whether to use low-confidence measurements. An angle is unavailable only when its coordinates are unusable or its vectors are degenerate. Video and webcam angle requests automatically enable tracking and include each pose's `track_id`; image requests do not enable tracking.

For example, request angles for both elbows on a video upload:

```shell
curl -X POST --url-query 'angles=[["left_shoulder","left_elbow","left_wrist"],["right_shoulder","right_elbow","right_wrist"]]' \
  "http://localhost:8002/api/pose_estimation/predict/video" \
  -F "source=@bicep-curl.mp4"
```

Video JSON predictions run as background jobs. The response contains a `job_id`; use `/api/pose_estimation/predict/video/progress/{job_id}` to receive progress and the completed per-frame results.

## Face swapping

The face-swap endpoints use the BFS head-swap LoRA and Viggle turbo adapter with
Qwen-Image-2.1. The image endpoint accepts `source` (portrait) and `target`
(target image) multipart fields and returns a JPEG. The video endpoint accepts
`source` and `target_video` multipart fields, processes frames sequentially, and
returns an MP4 with the original audio when present. The webcam WebSocket expects
the portrait as its first binary message, then target frames as binary messages;
each prediction is returned as a JPEG binary message. All three endpoints accept
the `model` query parameter:

- `qwen-image-2.1-bf16-local` (default) runs the existing Diffusers backend.
- `qwen-image-2.1-gguf-q3-k-xl-local` runs the Q3_K_XL GGUF through
  `stable-diffusion.cpp`, with CPU offload enabled.
- `qwen-image-2.1-huggingface-space` uses the public [Qwen-Image-2.1 Hugging
  Face Space](https://hugging-apps-qwen-image-2-1.hf.space/) API. It uploads
  images to that Space for processing and may be slower or
  unavailable when the shared free service is busy. Set
  `FACE_SWAP_HF_SPACE_TIMEOUT` to change the generation timeout (default: 1800
  seconds). The hosted Space does not use this repository's local BFS/Viggle
  adapters, so its face-swap results may differ from the local backends.

Use `POST /api/face_swap/predict/image?model=qwen-image-2.1-bf16-local` or
`POST /api/face_swap/predict/video?model=qwen-image-2.1-gguf-q3-k-xl-local`.
Connect to `ws://<host>/api/face_swap/predict/webcam?model=qwen-image-2.1-gguf-q3-k-xl-local`
for webcam frames.

The BF16 backend requires a CUDA-enabled GPU and compatible CUDA-enabled PyTorch.
When running outside Docker, the GGUF backend requires a `sd-server` executable
from `stable-diffusion.cpp`; set `FACE_SWAP_SD_CPP_SERVER_PATH` to its path, or
put it on `PATH`. The Docker image includes its own CUDA-enabled `sd-server`. It
starts the server on first use, downloads missing model assets
from Hugging Face, and stores them under
`face_swap/weights/qwen-image-2.1-gguf-q3-k-xl-local/` (ignored by Git). Set
`FACE_SWAP_SD_CPP_URL` to use an already-running compatible server instead; that
server must have the matching model and both LoRAs available. Optional settings
are `FACE_SWAP_SD_CPP_PORT` (default `1235`),
`FACE_SWAP_SD_CPP_STARTUP_TIMEOUT` (default `900` seconds), and
`FACE_SWAP_SD_CPP_TIMEOUT` (default `1800` seconds per generation).

On the available 4 GB RTX 500 Ada Laptop GPU, one 512×512 test image took about
80 seconds with CPU offload. The GGUF backend limits generated images to about
512×512 pixels to fit this GPU; video frames are resized back to the source
resolution after generation. A webcam or video request processes frames
sequentially and is not real-time on that hardware. The Docker host must provide NVIDIA GPU support and a compatible NVIDIA driver.
To use a separately managed server instead, set `FACE_SWAP_SD_CPP_URL` to its
reachable URL. The base Qwen
model and adapters have separate license terms, including non-commercial
research/evaluation restrictions; check the model cards before redistribution
or deployment. Use only media for which you have permission.
