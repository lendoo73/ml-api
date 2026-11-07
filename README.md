# Machine Learning FastAPI back-end

## Deploy Using Docker

### Prerequisites

Docker Desktop installed on your machine  
👉 https://docs.docker.com/desktop/

---

### Steps

1. Copy the /deploy folder from this repository to your local machine.
2. Open a terminal and navigate into the copied folder:
```shell
cd path/to/deploy
```
3. Start the container:
```shell
docker compose up -d
```
4. Open the application in your browser:  
👉 http://localhost:8002/

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