# Oriented Bounding Box Domain Implementation Plan

## Objective

Create a new `oriented_bounding_box` FastAPI domain for the official YOLO26x OBB model. The domain must support the same source types currently supported by the computer-vision endpoints:

- image uploads;
- image URLs;
- video uploads;
- video URLs;
- asynchronous video prediction and progress streaming;
- webcam frames over WebSocket.

The domain must be structured so that future custom OBB models, such as a car-parking-lot model, can reuse the same controller and inference contract without duplicating the media-processing infrastructure.

The official YOLO26x OBB weights have been downloaded to
`oriented_bounding_box/weights/yolo26x-obb.pt`. This plan still does not change application code.

## Naming and API boundary

Use `oriented_bounding_box` as the Python package/domain name because it is the complete, unambiguous computer-vision term. Use concise OBB terminology inside class and file names where appropriate.

Recommended API surface:

```text
/api/oriented_bounding_box/predict
/api/oriented_bounding_box/predict/video
/api/oriented_bounding_box/predict/video/progress/{job_id}
/api/oriented_bounding_box/predict/webcam
```

Use the OpenAPI tag `oriented bounding box`.

The controller accepts a `model` literal and the repository maps each allowed model name to server-configured weights. Current choices are `yolo26x-obb` and `car-parking-lot`.

## Planned package structure

```text
oriented_bounding_box/
├── __init__.py
├── controllers/
│   ├── __init__.py
│   ├── oriented_bounding_box_rest_controller.py
├── models/
│   ├── __init__.py
│   ├── dtos/
│   │   ├── __init__.py
│   │   └── oriented_bounding_box_dto.py
│   └── entities/
│       ├── __init__.py
│       ├── oriented_bounding_box.py
│       └── oriented_detection.py
├── repositories/
│   ├── __init__.py
│   └── yolo_oriented_bounding_box_model_repository.py
├── services/
│   ├── __init__.py
│   └── oriented_bounding_box_service.py
└── weights/
    └── yolo26x-obb.pt  # downloaded official YOLO26x OBB weights
```

The repository must use this exact file as its repository-relative default and must not substitute
the existing axis-aligned `object_detection/weights/yolo26x.pt` model.

## Architecture

### Controller and router

Implement a router factory based on the existing object-detection, pose, and segmentation endpoint patterns. Keep handlers responsible only for:

- FastAPI input validation;
- constructing task-specific plot/prediction options;
- calling the service;
- mapping JSON, image, video, and WebSocket responses.

Reuse existing helpers for:

- upload decoding;
- image and video URL downloads;
- temporary files;
- background cleanup;
- asynchronous video jobs;
- progress streaming;
- thread-pool offloading.

The controller must not load YOLO models or transform Ultralytics result tensors.

### Service

Create a concrete `OrientedBoundingBoxService`, even if most initial behavior is inherited from shared vision infrastructure. It should:

- expose the asynchronous image, URL, video, and webcam boundary;
- delegate inference and result transformation to the repository;
- preserve the existing non-blocking HTTP behavior;
- provide an extension point for future custom OBB services.

If the current `BaseVisionService` cannot express OBB result types cleanly, extract only the genuinely shared media orchestration into a narrowly scoped base abstraction. Do not make the generic base know about `boxes`, `masks`, `keypoints`, and `obb` simultaneously unless that remains type-safe and readable.

### Repository

Create a concrete repository with the existing lazy, thread-local model-loading pattern:

- default path resolved relative to `oriented_bounding_box/weights`;
- environment override such as `YOLO_ORIENTED_BOUNDING_BOX_MODEL`;
- one lazily initialized `YOLO` instance per worker thread;
- synchronous model inference methods called through the service's thread pool.

The repository should implement:

- single-frame JSON prediction;
- annotated single-frame output;
- streamed video JSON prediction with progress callbacks;
- annotated video generation;
- webcam-frame prediction;
- optional tracking only if the installed OBB model and repository behavior support it reliably.

Tracking must not be copied from axis-aligned detection blindly. Confirm the installed Ultralytics OBB tracking behavior before exposing `tracking`, `tracker`, `track_id`, or trajectories. If unsupported or unreliable, omit those parameters from the initial OBB API rather than accepting and silently ignoring them.

## OBB entity and DTO contract

Do not reuse the axis-aligned `BoundingBox` as the primary OBB result. Define an explicit oriented geometry contract.

The planned geometry should contain:

- center `x` and `y`;
- width and height;
- angle;
- four corner points, if available from the Ultralytics result;
- a documented angle unit and range.

Use a concrete detection entity/DTO containing:

- `class_id`;
- `class_name`;
- `confidence`;
- OBB geometry;
- optional tracking fields only if tracking is implemented.

The repository should derive the values from `Results.obb` rather than from `Results.boxes`. The implementation must document the chosen angle convention, because Ultralytics exposes OBB-specific tensors and `xywhr`/`xyxyxyxy` representations and clients need a stable API contract.

Video JSON should preserve the repository's existing frame envelope:

```json
{
  "frame_index": 0,
  "timestamp_ms": 0,
  "detections": []
}
```

Use a domain-appropriate response filename such as `oriented_bounding_box.mp4`.

## Official Ultralytics contract to verify during implementation

The installed dependency is `ultralytics==8.4.7`. The implementation must verify the local package behavior, not rely only on online examples.

Official documentation identifies OBB as its own YOLO task and uses OBB-specific model weights and result data. The relevant references are:

- [Ultralytics OBB task](https://docs.ultralytics.com/tasks/obb/)
- [Ultralytics prediction mode](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics OBB predictor API](https://docs.ultralytics.com/reference/models/yolo/obb/predict/)

Before defining query parameters, inspect the installed source/signatures and run a small result inspection with the official YOLO26x OBB weights. Verify:

- the downloaded `oriented_bounding_box/weights/yolo26x-obb.pt` asset and its task metadata;
- `Results.obb` availability;
- `obb.xywh`, `obb.xywhr`, `obb.xyxyxyxy`, `obb.cls`, and `obb.conf` shapes/types;
- angle units and normalization;
- `result.plot()` behavior for OBB annotations;
- whether `model.track()` supports the selected OBB model and which tracker options are valid;
- whether video streaming returns the same OBB result contract as image inference.

### Initial optional prediction parameters

Expose only options that the installed Ultralytics predictor accepts and that every supported source path can honor:

- `conf`, constrained to `0..1`;
- `iou`, constrained to `0..1`, if verified for OBB NMS;
- `imgsz`, with the repository's established scalar or tuple validation;
- `device`, only if the existing API convention supports exposing it;
- `classes`, if the repository has a suitable typed representation;
- `max_det`, if verified for the installed OBB predictor;
- `agnostic_nms`, if verified and useful for the OBB task;
- `verbose` should remain an internal value and default to `False`, not a user-facing option.

For rendered image/video output, expose only plotting options supported by `Results.plot()` for OBB:

- confidence labels;
- line width;
- font size/font;
- labels;
- boxes;
- color mode;
- text color, if the existing custom-color plotting path can correctly render OBB corners.

Do not expose segmentation-only parameters such as `masks` or `retina_masks`, pose-only keypoint options, or axis-aligned-only assumptions. Do not add a query parameter unless it is passed through to inference/rendering and covered by an endpoint test.

## Router registration and configuration

Update the application router registration to include the new controller. Add only the OBB-specific model configuration required by the repository's existing environment/configuration conventions.

Do not change the default axis-aligned object-detection model configuration. The OBB model must have its own environment variable and repository-relative default.

## Tests

Add integration coverage following `tests/test_api_integration.py`:

1. image upload returns a list with OBB detection fields;
2. image output returns an annotated JPEG;
3. image URL returns OBB predictions;
4. webcam accepts an image frame and returns OBB detections;
5. uploaded video returns frame-level OBB results;
6. uploaded video with `output=video` returns an annotated MP4;
7. video URL returns an asynchronous job or the repository's established response shape;
8. video progress route returns the existing SSE contract;
9. OpenAPI contains all OBB routes and the intended optional parameters;
10. invalid parameter ranges return validation errors;
11. missing video jobs return `404`;
12. model/configuration failures use the application's established error handling.

Use a deterministic fixture or the official model only in the same way the existing integration suite handles other model domains. Avoid mock-heavy tests. Add focused unit tests only for isolated OBB tensor-to-DTO conversion or angle/corner normalization that cannot be covered reliably through endpoints.

## Implementation sequence

1. Inspect the downloaded `oriented_bounding_box/weights/yolo26x-obb.pt` asset and confirm its task metadata.
2. Inspect the installed Ultralytics OBB result and plotting behavior.
3. Define the OBB entity and DTO contract, including angle convention.
4. Implement the repository with lazy model loading and OBB result conversion.
5. Implement the concrete service using shared asynchronous media helpers.
6. Implement the reusable router factory and the default OBB controller.
7. Register the router and model configuration.
8. Add integration tests for every exposed source and response mode.
9. Run the focused integration tests, inspect OpenAPI output, and run the relevant broader suite.
10. Review the diff for duplicated media logic, unsupported parameters, incorrect model paths, and accidental generated artifacts.

## Future custom-model extension

Future models must not require copying the generic OBB controller. A custom model should provide its own repository/service only when it changes weights, classes, result transformation, or business interpretation.

For example, a parking-lot model may later add:

```text
oriented_bounding_box/
├── repositories/
│   └── parking_lot_obb_model_repository.py
└── services/
    └── parking_lot_obb_service.py
```

The generic OBB route should remain available for raw oriented detections. A parking-lot route/service is justified only when it adds domain behavior such as occupancy, parking zones, availability, or aggregation; a different weights file alone should be represented through model configuration.
