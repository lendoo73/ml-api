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

| Model | WebSocket endpoint | Optional query parameters |
| --- | --- | --- |
| Object detection | `/api/object_detection/predict/webcam` | `classes` (optional, repeatable IDs or names); `tracking` (`false`); `tracker` (`bytetrack`; `bytetrack` or `botsort`) |
| Road sign detection | `/api/object_detection/road_signs/predict/webcam` | `classes` (optional, repeatable IDs or names); `tracking` (`false`); `tracker` (`bytetrack`; `bytetrack` or `botsort`) |
| Face mask detection | `/api/object_detection/face_mask/predict/webcam` | `classes` (optional, repeatable IDs or names); `tracking` (`false`); `tracker` (`bytetrack`; `bytetrack` or `botsort`). Tracking is rejected if unsupported by the model. |
| Classification | `/api/classification/predict/webcam` | None |
| Chest X-ray classification | `/api/classification/chest_xray/predict/webcam` | None |
| Weather classification | `/api/classification/weather/predict/webcam` | None |
| Oriented bounding box | `/api/oriented_bounding_box/predict/webcam` | `tracking` (`false`); `tracker` (`bytetrack`, or `botsort` when supported) |
| Car parking lot | `/api/car_parking_lot/predict/webcam` | `tracking` (`false`); `tracker` (`bytetrack`, or `botsort` when supported) |
| Segmentation | `/api/segmentation/predict/webcam` | `classes` (optional, repeatable IDs or names); `tracking` (`false`); `tracker` (`bytetrack`) |
| Pothole segmentation | `/api/segmentation/pothole/predict/webcam` | `tracking` (`false`); `tracker` (`bytetrack`) |
| Pose estimation | `/api/pose_estimation/predict/webcam` | `tracking` (`false`); `tracker` (`bytetrack`); `angles` (optional JSON-encoded list of three-keypoint lists; using it enables tracking) |

For pose estimation, angle triplets must use either all keypoint names or all
zero-based COCO keypoint indices, with the middle keypoint as the pivot. See the
[Ultralytics COCO pose dataset configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml)
for the keypoint ordering.

"""
