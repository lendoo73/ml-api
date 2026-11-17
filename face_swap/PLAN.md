# Face Swap Model Selection Plan

## Goal

Allow clients to select a face-swap model/backend with a `model` parameter, using an explicit allowlist. Keep the existing local Qwen/BFS pipeline available while making it possible to add hosted inference backends later.

## Current State and Constraints

- The image, video, and webcam endpoints support two local Qwen-Image-2.1 backends with the BFS and Viggle adapters: the original BF16 Diffusers pipeline and a Q3_K_XL GGUF backend using stable-diffusion.cpp.
- The BF16 pipeline requires CUDA and BF16, and currently moves the full pipeline to CUDA. The GGUF backend uses CPU offload, limits output to about 512×512 pixels, and completed an API-side test in about 80 seconds on the available 4 GB RTX 500 Ada Laptop GPU. Video and webcam processing remain sequential and are not real-time on that hardware.
- A Hugging Face model being downloadable for free does not make hosted inference free. Inference Providers may require credits, while public Spaces can have queues, quotas, downtime, and changing access.
- The Qwen model's research-use license and each adapter/provider's terms must be checked independently. Sending images to a hosted provider also means sending them outside this API's host.

## API Contract

1. Add a `model` query parameter to the image and video HTTP endpoints, and a `model` query parameter to the webcam WebSocket URL.
2. Keep the local model as the default so clients that omit `model` remain compatible. Clients that explicitly sent the previous `local` ID must use the new ID.
3. Define the allowed model IDs with a `Literal`, following the existing model-selection pattern in the classification and object-detection domains.
4. Reject unknown IDs during request validation. Do not silently substitute another model or backend.
5. Document each model ID with its backend, task support, licensing, costs/quotas, and any required credentials.

## Model Registry and Routing

1. Introduce a face-swap model type and a small registry/factory that maps each allowed model ID to its backend implementation.
2. Keep endpoint controllers focused on request parsing and response mapping. Keep image/video/webcam orchestration in `FaceSwapService`; delegate inference to a backend-specific adapter.
3. Keep the existing Qwen/BFS implementation as the `qwen-image-2.1-bf16-local` backend without changing its prompt or inference behavior as part of model selection.
4. Route `qwen-image-2.1-gguf-q3-k-xl-local` through stable-diffusion.cpp with CPU offload; retain explicit startup, generation timeout, and missing-runtime errors.
5. Add hosted providers only as concrete, individually allowlisted model IDs. Do not use a generic provider ID unless it unambiguously maps to a supported face-swap model and configuration.
6. Keep provider credentials in environment/configuration, never in request parameters or logs. Surface provider errors and timeouts explicitly.

## Provider Qualification

Before adding a hosted model ID:

- Verify that the provider supports image editing/head replacement with the required source and target images; generic text-to-image support is insufficient.
- Verify that the provider's API can be called programmatically and returns image output in a format this API can validate and encode.
- Record current pricing, free-tier limits, queue/rate limits, data handling, and model/API license terms.
- Do not enable paid usage or automatic paid fallback under a "free model" choice. If access is unavailable or a free quota is exhausted, return a clear error.
- Treat the BFS public Space as a demo integration unless its owner documents a stable API and usage terms.

## Media-Specific Behavior

- **Image:** Preserve the existing multipart `source` portrait and `target` image inputs and JPEG response.
- **Video:** Preserve sequential frame processing and audio muxing. A remote call per frame may be slow and rate-limited; define frame/duration limits and report provider failures without returning a partial-success response.
- **Webcam:** Preserve the current binary-frame protocol. The selected model is fixed for the WebSocket session. Document that a remote request per frame is not guaranteed to meet real-time latency.
- Ensure all backends return the same normalized image type so the existing encoding and video pipeline can be reused.

## Implementation and Verification Steps

1. Define the model-ID `Literal`, defaults, and API descriptions; verify unknown IDs return a validation error.
2. Add backend selection while keeping `qwen-image-2.1-bf16-local` as the default.
3. Add a provider client only after it passes the provider qualification checks; make network timeout, authentication, rate-limit, and upstream error handling explicit.
4. Add tests for default selection, each allowed ID, invalid IDs, backend dispatch, and provider error handling using mocked clients.
5. Extend image, video, and WebSocket integration tests to verify model selection is consistent across all surfaces.
6. Document example requests, model availability, license/cost caveats, and the fact that video/webcam performance depends on per-frame provider latency.
7. Smoke-test each remote backend with consented sample images and confirm the provider's free quota and charges before enabling it for users.

## Existing Patterns to Reuse

- `object_detection/controllers/object_detection_rest_controller.py` and `classification/controllers/classification_rest_controller.py` define model choices with `Literal`.
- `face_swap/controllers/face_swap_rest_controller.py` owns the current image, video, and webcam API surfaces.
- `face_swap/services/face_swap_service.py` owns shared media processing.
- `face_swap/repositories/qwen_image_21_bf16_local_face_swap_repository.py` contains the current local inference implementation.
