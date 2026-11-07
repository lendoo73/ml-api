API_DESCRIPTION = """
# ML REST API 🚀

## WebSocket webcam endpoints

WebSocket routes are not included in the generated OpenAPI schema, so they are
listed here for reference. Optional query parameters are listed below. After
connecting, send each video frame as a binary WebSocket message; each frame
receives a JSON array containing that frame's prediction results.

## Object detection and general segmentation class filtering

Object detection and general segmentation prediction endpoints accept an optional
`classes` query parameter. Omit it to predict all classes, or repeat it to select
multiple zero-based class IDs or class names (for example, `?classes=0&classes=person`).
IDs and names are specific to the model. For COCO-trained models, see the
[Ultralytics COCO class list](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).
For custom models, use the names and ordering that the model was trained with.
Classification uses `model=yolo26x-cls`, `model=chest-xray`, or `model=weather`
on image endpoints. Chest X-ray classification is image-only; video and webcam
endpoints allow `yolo26x-cls` or `weather`.

| Model | WebSocket endpoint | Optional query parameters |
| --- | --- | --- |
| Object detection | `/api/object_detection/predict/webcam` | `model` (`yolo26x`, `face-mask-detect`, `road-sign-detect`, or `knife`); `classes` (optional, repeatable IDs or names); `tracking` (`false`); `tracker` (`bytetrack` or `botsort`) |
| Classification | `/api/classification/predict/webcam` | `model` (`yolo26x-cls` or `weather`) |
| Oriented bounding box | `/api/oriented_bounding_box/predict/webcam` | `model` (`yolo26x-obb` or `car-parking-lot`); `tracking` (`false`); `tracker` (`bytetrack` or `botsort`) |
| Segmentation | `/api/segmentation/predict/webcam` | `model` (`yolo26x-seg` or `pothole`); `classes` (optional, repeatable IDs or names; IDs only for pothole); `tracking` (`false`); `tracker` (`bytetrack`) |
| Pose estimation | `/api/pose_estimation/predict/webcam` | `tracking` (`false`); `tracker` (`bytetrack`); `angles` (optional JSON-encoded list of three-keypoint lists; using it enables tracking) |

For pose estimation, angle triplets must use either all keypoint names or all
zero-based COCO keypoint indices, with the middle keypoint as the pivot. See the
[Ultralytics COCO pose dataset configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml)
for the keypoint ordering.

Segmentation image endpoints accept `yolo26x-seg`, `brain-tumor`, and `pothole`.
The brain-tumor model is image-only, with ZIP batch prediction at
`/api/segmentation/predict/batch`; it is not selectable on video or webcam
endpoints. Pothole segmentation also accepts `imgsz`, `max_det`, and
`retina_masks` on image endpoints.

"""
