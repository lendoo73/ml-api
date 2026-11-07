# Frontend model selection

The consolidated prediction endpoints select a model with the `model` query
parameter instead of using a model-specific URL. Model names are fixed
server-side selectors; do not send weight filenames or filesystem paths.

## Selectable models

| Domain | `model` values | Supported sources |
| --- | --- | --- |
| Object detection | `yolo26x`, `face-mask-detect`, `road-sign-detect`, `knife` | Image, video, webcam, image search |
| Classification | `yolo26x-cls`, `chest-xray`, `weather` | `chest-xray`: image upload, image URL, image search. `yolo26x-cls` and `weather`: image upload, image URL, image search, video, webcam |
| Oriented bounding box | `yolo26x-obb`, `car-parking-lot` | Image, video, webcam |
| Segmentation | `yolo26x-seg`, `brain-tumor`, `pothole` | `brain-tumor`: image and ZIP batch only. `yolo26x-seg` and `pothole`: image, video, webcam |

Use the model list for the specific source endpoint. For example, do not offer
`chest-xray` as a video or webcam choice, or `brain-tumor` for video or webcam.
The segmentation batch endpoint is available only for `brain-tumor`.

## Endpoint pattern

Image upload requests use `POST` with a multipart `source` file. Image URLs and
image-search requests use `GET`. Video uploads use `POST`; video URLs use `GET`.
The selected model is a query parameter for all these requests:

```text
/api/{domain}/predict?model={model-name}
/api/{domain}/predict/video?model={model-name}
/api/{domain}/predict/webcam?model={model-name}
```

Omitting `model` selects the domain's default model. Prefer sending it
explicitly so the selected model is clear in the request and frontend state.

## Examples

Upload an image to the knife detector:

```bash
curl -X POST "https://<api-host>/api/object_detection/predict?model=knife" \
  -F "source=@image.jpg"
```

Classify an image with the chest-X-ray model:

```bash
curl -X POST "https://<api-host>/api/classification/predict?model=chest-xray" \
  -F "source=@scan.png"
```

Classify an image URL with the weather model:

```text
GET https://<api-host>/api/classification/predict?model=weather&source=https%3A%2F%2Fexample.com%2Fimage.jpg
```

Run the car-parking OBB model instead of using a separate car-parking endpoint:

```bash
curl -X POST "https://<api-host>/api/oriented_bounding_box/predict?model=car-parking-lot" \
  -F "source=@parking.jpg"
```

Run pothole segmentation with its model-specific options. Pothole `classes`
must be zero-based numeric IDs:

```bash
curl -X POST "https://<api-host>/api/segmentation/predict?model=pothole&imgsz=640&max_det=300&retina_masks=true&classes=0" \
  -F "source=@road.jpg"
```

The brain-tumor model supports ZIP batch prediction:

```bash
curl -X POST "https://<api-host>/api/segmentation/predict/batch?model=brain-tumor" \
  -F "source=@scans.zip"
```

WebSocket examples use the same selector in the connection URL:

```text
/api/object_detection/predict/webcam?model=road-sign-detect
/api/classification/predict/webcam?model=weather
/api/oriented_bounding_box/predict/webcam?model=car-parking-lot
/api/segmentation/predict/webcam?model=pothole
```

Send each webcam frame as binary image bytes; each frame receives a JSON
prediction response. JSON video prediction requests return a `job_id`; poll the
domain's `/predict/video/progress/{job_id}` endpoint for progress and results.

## Previous model-specific URLs

Use these replacements when migrating callers:

| Previous URL | Replacement |
| --- | --- |
| `/api/object_detection/face_mask/predict` | `/api/object_detection/predict?model=face-mask-detect` |
| `/api/object_detection/road_signs/predict` | `/api/object_detection/predict?model=road-sign-detect` |
| `/api/classification/chest_xray/predict` | `/api/classification/predict?model=chest-xray` |
| `/api/classification/weather/predict` | `/api/classification/predict?model=weather` |
| `/api/car_parking_lot/predict` | `/api/oriented_bounding_box/predict?model=car-parking-lot` |
| `/api/segmentation/pothole/predict` | `/api/segmentation/predict?model=pothole` |
| `/api/segmentation/brain_tumor/predict` | `/api/segmentation/predict?model=brain-tumor` |

The separate `/api/multi_model/predict` API is unchanged. It accepts its own
model identifiers to run multiple models in one request:

```text
object-detection
road-sign-detection
face-mask-detection
pose-estimation
segmentation
pothole-segmentation
brain-tumor-segmentation
classification
chest-xray-classification
weather-classification
oriented-bounding-box
car-parking-lot
```

These are not substitutes for the per-domain `model` values listed above. In
particular, the standalone selector `knife` is not currently a multi-model ID.
