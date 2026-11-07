# Multi-model prediction endpoint plan

## Goal

Add a multi-model API that accepts one image, video, or webcam frame stream and a
list of supported model identifiers. The API runs the selected models against
the same source, then returns their results together. For video, results are
combined by frame. The front end remains responsible for combining prediction
results into application-specific behavior.

Existing model-specific endpoints remain available and unchanged.

## Supported models

- Define an explicit model literal/allowlist. Each value identifies one
  supported model implementation and maps to its service and options adapter.
  For example, `object-detection` maps to the general YOLO26 detector at
  `/api/object_detection`, while `road-sign-detection` maps to the custom
  detector at `/api/object_detection/road_signs`.
- The literal can include general or custom models from any supported task,
  including specialized models such as pothole segmentation. Being
  single-class does not prevent a model from being supported.
- Do not expose arbitrary model loading or caller-provided model paths.
- Validate the requested model list: it must be non-empty, contain only
  allowlisted identifiers, and not contain duplicates.
- Define model-specific option validation and adaptation for each listed
  model. Pass validated options to that model's service; do not call its router
  or make an HTTP request to this API.
- Adding multi-model support for a project model requires adding its identifier
  and service/options adapter to the literal mapping.

## Request parameters

Every request has three logical inputs:

1. **`source`**: an image or video upload, a public image/video URL, or a
   webcam frame sent over the WebSocket.
2. **`models`**: a non-empty list of model identifiers from the allowlist.
3. **`model_options`**: an object keyed by selected model identifier, with
   options validated against that model's schema.

Example model selection and options:

```json
{
  "models": ["object-detection", "pose-estimation", "segmentation"],
  "model_options": {
    "object-detection": {
      "classes": ["knife"],
      "tracking": true
    },
    "pose-estimation": {
      "tracking": true,
      "angles": [
        ["left_shoulder", "left_elbow", "left_wrist"],
        ["right_shoulder", "right_elbow", "right_wrist"]
      ]
    },
    "segmentation": {
      "classes": ["person", "knife"]
    }
  }
}
```

Transport-specific encodings may differ: image/video upload routes receive
`source` as multipart data with model selection and options as request fields;
URL routes receive the source URL and model selection/options as query
parameters. The webcam WebSocket receives model selection and options once for
the connection, then receives each frame as binary data. Options must correspond
only to selected models.

## API shape

Use one API feature with source-specific routes, since an uploaded image,
uploaded video, and continuous webcam frames have different request and
response lifecycles:

- **Image:** multipart upload or image URL plus model identifiers and their
  options; return a `predictions` object keyed by model identifier.
- **Video:** multipart upload or video URL plus model identifiers and their
  options; submit one job and return its job ID. Report progress through the
  existing video-job progress mechanism. Completed results are grouped by
  frame index, with each frame containing results keyed by model identifier.
- **Webcam:** Use a WebSocket route, matching the existing webcam API. The
  client supplies model identifiers and their options once for the connection,
  then sends each captured frame as binary image bytes. The server processes
  that frame and sends one JSON response containing results keyed by model
  identifier. Do not add a frame counter; the client decides whether to retain
  or discard each response.

Keep each model's result in its existing native shape rather than forcing
detection boxes, poses, classifications, and other outputs into one schema.
Include a stable frame index in video results so clients can align predictions
with source frames. Webcam responses correspond to individual WebSocket frame
messages and do not need a back-end frame index.

## Processing design

1. Validate model identifiers, ensure options are provided only for selected
   models, and validate each model's options before inference.
2. Decode each image/frame once.
3. Call each selected model's service directly with its validated
   model-specific options. Do not route internally through existing HTTP
   endpoints.
4. Return each service's native DTO-backed prediction shape, grouped under its
   model identifier; FastAPI's response models/OpenAPI should document the
   response contract.
5. For video, read/decode the source once and process frames in source order.
   For each frame, run selected model services in parallel, await all selected
   models, then combine their results under that frame's index. Advance the
   aggregate video-job progress only after all selected models finish the frame.
6. For webcam, process each received binary frame and return its combined
   result before receiving/processing the next frame, matching existing
   WebSocket behavior. Scope optional model tracking state to the connection
   and reset it when the WebSocket disconnects.

Do not process multiple frames concurrently through the same stateful tracker;
each selected model must see video frames in order when tracking is enabled.
Parallelize across selected models for the current frame, with bounded
concurrency. Do not use the presumed slowest model as a proxy for overall
progress or completion: the aggregate job waits for all selected models per
frame and completes only after all frames have been processed. Do not queue an
entire video in memory.

## Likely integration points

- Add a multi-model controller/router and register it from the API router.
- Add the model literal mapping, per-model option validation/adapters, and
  multi-model orchestration in the service layer.
- Reuse existing model services, pose/detection DTOs, upload decoding, video
  job manager, and progress endpoint where their contracts fit.
- Reuse class filtering already supported by general segmentation. Add each
  other model's options through its service adapter; for example, pothole
  segmentation can be allowlisted and receive its supported class-index
  options.
- Pose returns all keypoints. The optional `angles` setting adds requested
  angle measurements; clients can filter keypoints locally.
- Call specialized model services directly when their model identifier is
  selected; do not exclude them because they have one class or different
  prediction options.
- Keep orchestration out of model repositories and avoid changing the behavior
  of existing single-model routes.

## Tests and acceptance criteria

- An image request selects multiple allowlisted models, decodes the upload once,
  invokes each selected model, and returns all results under the correct model
  identifiers.
- Unknown, empty, or duplicate model selections produce a clear validation
  error and run no inference.
- Per-model options are validated by that model's schema; unsupported options,
  options for unselected models, and malformed option values are rejected.
- A video request uploads and decodes once, processes frames in source order,
  runs selected models in parallel per frame, and returns correctly aligned
  results with monotonically increasing frame indices. Progress advances only
  after all selected models complete the corresponding frame.
- Each webcam frame sent over the WebSocket receives one combined response;
  the client can retain or discard it without requiring a server-generated
  sequence number.
- A failure in any selected model fails that combined frame/request explicitly;
  it is never represented as a successful empty prediction or partial result.
- Existing model-specific endpoint tests continue to pass.
- Exercise representative model combinations and the selected video
  concurrency strategy to confirm CPU/GPU resources remain bounded and
  tracking stays correct.

## Implementation decisions and limitation

- Model inference runs in parallel across selected models for one frame, while
  frames remain sequential to preserve tracker state and alignment.
- A failure in one selected model fails the combined operation; partial
  success payloads are not returned.
- Video frames are decoded and processed incrementally, but the existing video
  job manager retains the completed result in memory. Result size and retention
  limits remain a limitation for long or high-resolution videos.
