---
name: custom-yolo-model-endpoint
description: Create reusable controllers, services, repositories, routes, tests, weights, and TensorBoard assets for a new YOLO model.
---

# Custom YOLO Model Endpoint Agent

You are responsible for implementing a complete domain for a new user-provided YOLO model in this FastAPI repository. The model type and domain are supplied by the user at runtime. Reuse the shared base classes and router factories; never copy-paste an existing domain's inference or media-handling implementation.

## Required intake

Before changing code:

1. Confirm the model/controller name. If it cannot be derived from the request, ask for it.
2. Require the user to provide the new YOLO weights file or an accessible repository path. If it is missing, ask for it. Never invent a weights path and never substitute an existing model.
3. Confirm the parent YOLO model variant used for training (`yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`, or the applicable variant). If it is missing, ask for it.
4. Use the exact user-provided weights path or tagged asset, but rename the copied weight file to `<parent-model>_<model-name>.pt` so the training variant is always visible. Do not assume a filename, task type, or domain.
5. Inspect the analogous domains, shared base classes, application router registration, dependency configuration, and existing integration tests before implementation.
6. Confirm the TensorBoard run directory or event-file path when it is not included in the request. If missing, ask for it before copying assets.

If either required input is missing, stop and ask one focused question at a time. Do not create placeholder model paths or silently substitute another model.

## Implementation scope

Create a complete domain named from the user's model/controller name, including:

- `controllers` / router
- concrete service
- concrete repository
- DTO and entity models
- router registration and model configuration required by the application
- the supplied weights copied into the model's corresponding weights directory and renamed as `<parent-model>_<model-name>.pt` (for example, `yolo26x_face_mask_detect.pt`);
- supplied TensorBoard event files copied into `data/tensorboard/<task>/<model-name>/`;
- integration tests for the new endpoints
- focused unit tests only for behavior that cannot be exercised reliably through integration tests

Unless the user explicitly excludes a modality, every new model must provide prediction endpoints for:

- images, including uploaded files and supported URL sources;
- videos, including asynchronous processing/progress behavior where established by the repository;
- webcam frames through the existing realtime transport.

If the user explicitly states that a modality is not required (for example, image prediction is not applicable to the model), omit only that modality and preserve the others. Record the requested exception in the implementation and tests rather than silently skipping an endpoint.

Keep the implementation asynchronous at the HTTP boundary and compatible with the repository's existing async execution patterns. Do not block the event loop with model inference or file/video processing; reuse established thread-pool, background-job, and streaming abstractions.

## Design requirements

- Use existing domains as examples, especially segmentation, pose estimation, and object detection, but do not copy-paste their code.
- Follow the repository's `controller -> service -> repository` boundary:
  - controllers validate input, depend on services, and map responses;
  - services contain orchestration and business behavior;
  - repositories own model loading, inference, and persistence/data-access concerns;
  - models contain DTO/entity contracts only.
- Prefer shared base classes and reusable helpers where they fit. Extend or parameterize an existing abstraction instead of duplicating common YOLO inference, media handling, plotting, or job-management logic.
- Always create concrete domain classes for this model, even when they inherit all behavior from a shared base.
- Keep model loading lazy and thread-safe according to existing repository conventions. Configure the supplied weights through the project's established configuration mechanism, with a safe repository-relative default when appropriate.
- Preserve existing API behavior and response conventions. Use a route prefix and tags derived from the user's domain name.
- Infer whether the supplied model performs detection, segmentation, pose estimation, classification, or another supported YOLO task from the request and weights metadata; if that cannot be determined reliably, ask the user before implementing task-specific DTOs and endpoints.

## Task-specific Ultralytics parameters

Ultralytics prediction arguments are task-dependent. Do not use one fixed parameter list for every new model. Before designing the DTOs, query parameters, repository call, or tests:

1. Determine the model task (`classify`, `detect`, `segment`, `pose`, `obb`, or another task supported by the installed Ultralytics version) from the user request, weights metadata, and the model's documented interface.
2. Consult the official Ultralytics documentation for the installed/API-compatible version, including the relevant task page and `model.predict()` argument reference. If the local package version differs from the online documentation, prefer the installed package's supported signature and behavior.
3. Build a task-specific parameter contract. For example, classification must not expose segmentation-only options such as masks or polygon rendering, while pose estimation must account for keypoint-specific output/rendering options.
4. Reuse only common parameters when they are actually supported by the selected task and inference path. Typical common parameters may include `conf`, `imgsz`, `device`, `classes`, `max_det`, or `verbose`, but verify each one instead of assuming it is available.
5. Add task-specific parameters only when the model and endpoint can honor them. Examples include detection NMS options, segmentation mask/retina options, pose keypoint rendering options, classification top-k options, and OBB-specific settings.

Expose only parameters supported by the selected Ultralytics task and endpoint type. Validate query values at the FastAPI boundary, document defaults, and pass every accepted parameter through to inference without silently discarding it. Keep image, URL, video, webcam, and progress endpoints consistent with the capabilities actually supported by that task. Update DTOs, entities, response mapping, and integration tests to match the task-specific prediction result rather than copying another domain's schema.

## Testing workflow

Write integration tests first, and ensure every new endpoint exposed by the model's controller is covered by at least one integration test. This includes image upload and URL endpoints, video upload and URL endpoints, video progress endpoints, and webcam endpoints when those modalities are supported. Cover:

- successful prediction and response shape for the new model;
- URL or media input supported by the domain;
- optional YOLO parameter propagation;
- validation failures for invalid parameter ranges and missing/invalid input;
- annotated output and asynchronous video-job behavior when those endpoints are exposed;
- missing model/configuration or inference errors using the repository's established error handling.

Use real application fixtures and the existing test client patterns. Avoid mock-heavy tests; use a small deterministic fixture or test model only when the repository already provides one. Add unit tests only for isolated transformation or validation logic that is impractical to cover through an endpoint.

Run the narrowest existing test command covering the changed domain, then the relevant broader suite if needed. Do not add testing tools or dependencies unless the repository requires them.

## Completion checklist

Before finishing:

- verify imports and router registration;
- verify the exact user-supplied weights path/configuration is used and no unrelated model is substituted;
- verify every exposed parameter reaches the inference layer;
- verify all request handlers are async and blocking work is offloaded;
- verify integration tests pass;
- review the diff for duplicated logic, unrelated changes, and accidental secrets or large generated artifacts.
