import json
import os

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import AnyUrl

from exceptions import InvalidImageSourceError, InvalidVideoSourceError
from multi_model.models.multi_model_dto import (
    MultiModelImagePredictionDto,
    MultiModelJobDto,
)
from multi_model.services.multi_model_prediction_service import (
    ModelId,
    MultiModelPredictionService,
    parse_configuration,
)
from services.video_job_manager import video_job_manager
from utils.casting import decode_upload_file_to_ndarray
from utils.image import download_image_to_ndarray
from utils.video import download_video_to_temp, save_uploadfile_to_temp


router = APIRouter(prefix="/multi_model/predict", tags=["multi-model prediction"])
MULTI_MODEL_TAG_DESCRIPTION = """
The image and video upload routes accept repeated `models` form values and an
optional `model_options` form value containing a JSON object keyed by model
identifier. Image and video URL sources are also supported with GET requests;
send `source`, repeat `models` as query parameters, and pass `model_options` as
a JSON-encoded query parameter.

For example, select `object-detection` and `pose-estimation` and provide:

```json
{
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
  }
}
```

For image URL sources, an `image/*` content type is preferred. Some servers
return valid image bytes as `application/octet-stream`; these are accepted
when the content can be decoded as an image.

The same model options are used for each source type. Pose responses include
all keypoints; the optional `angles` setting adds angle measurements. Options
vary by model; unsupported options are rejected. Image responses are keyed by
model identifier. Video requests return a `job_id`; poll
`/api/multi_model/predict/video/progress/{job_id}` for aggregate progress and
the combined per-frame result. A video frame is complete only after all
selected models finish it.

For `/api/multi_model/predict/webcam`, send a JSON configuration as the first
WebSocket message, with `models` as a list and `model_options` as an optional
object. Then send each frame as binary image bytes. Each frame receives one
combined JSON response; no server-generated webcam frame index is included.

Supported model identifiers: `object-detection`, `road-sign-detection`,
`face-mask-detection`, `pose-estimation`, `segmentation`,
`pothole-segmentation`, `brain-tumor-segmentation`, `classification`,
`chest-xray-classification`, `weather-classification`,
`oriented-bounding-box`, and `car-parking-lot`.
"""
MODEL_OPTIONS_DESCRIPTION = (
    "Optional JSON object keyed by selected model identifier. Each value contains "
    "that model's options; unsupported options are rejected."
)


def _parse_options(raw_options: str | None) -> dict[str, dict]:
    if raw_options is None or raw_options == "":
        return {}
    try:
        options = json.loads(raw_options)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=422, detail="model_options must be a valid JSON object."
        ) from error
    if not isinstance(options, dict) or any(
        not isinstance(value, dict) for value in options.values()
    ):
        raise HTTPException(
            status_code=422,
            detail="model_options must be an object of model identifiers and option objects.",
        )
    return options


def _validated_configuration(
    models: list[ModelId], raw_options: str | None
) -> tuple[list[str], dict[str, dict]]:
    try:
        return parse_configuration(
            [model.value for model in models],
            _parse_options(raw_options),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("", response_model=MultiModelImagePredictionDto)
async def predict_image(
    source: UploadFile = File(..., description="Image to analyze with selected models."),
    models: list[ModelId] = Form(..., description="Repeat to select supported models."),
    model_options: str | None = Form(None, description=MODEL_OPTIONS_DESCRIPTION),
):
    model_ids, options = _validated_configuration(models, model_options)
    try:
        image = await decode_upload_file_to_ndarray(source)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid image source.") from error
    try:
        service = MultiModelPredictionService(model_ids, options)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    try:
        return {"predictions": await service.predict_frame(image)}
    finally:
        service.close()


@router.get("", response_model=MultiModelImagePredictionDto)
async def predict_image_from_url(
    source: AnyUrl,
    models: list[ModelId] = Query(..., description="Repeat to select supported models."),
    model_options: str | None = Query(None, description=MODEL_OPTIONS_DESCRIPTION),
):
    model_ids, options = _validated_configuration(models, model_options)
    try:
        image = await run_in_threadpool(
            download_image_to_ndarray,
            str(source),
            allow_octet_stream=True,
        )
    except InvalidImageSourceError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    try:
        service = MultiModelPredictionService(model_ids, options)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    try:
        return {"predictions": await service.predict_frame(image)}
    finally:
        service.close()


@router.post("/video", response_model=MultiModelJobDto)
async def predict_video(
    source: UploadFile = File(..., description="Video to analyze with selected models."),
    models: list[ModelId] = Form(..., description="Repeat to select supported models."),
    model_options: str | None = Form(None, description=MODEL_OPTIONS_DESCRIPTION),
):
    model_ids, options = _validated_configuration(models, model_options)
    video_path = await save_uploadfile_to_temp(source)
    try:
        service = MultiModelPredictionService(model_ids, options)
    except ValueError as error:
        os.remove(video_path)
        raise HTTPException(status_code=422, detail=str(error)) from error

    async def process(path, progress_callback):
        return await service.predict_video(path, progress_callback)
    return {
        "job_id": video_job_manager.submit_path(video_path, process)
    }


@router.get("/video", response_model=MultiModelJobDto)
async def predict_video_from_url(
    source: AnyUrl,
    models: list[ModelId] = Query(..., description="Repeat to select supported models."),
    model_options: str | None = Query(None, description=MODEL_OPTIONS_DESCRIPTION),
):
    model_ids, options = _validated_configuration(models, model_options)
    try:
        video_path = await run_in_threadpool(download_video_to_temp, str(source))
    except InvalidVideoSourceError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    try:
        service = MultiModelPredictionService(model_ids, options)
    except ValueError as error:
        os.remove(video_path)
        raise HTTPException(status_code=422, detail=str(error)) from error

    async def process(path, progress_callback):
        return await service.predict_video(path, progress_callback)

    return {
        "job_id": video_job_manager.submit_path(video_path, process)
    }


@router.get("/video/progress/{job_id}")
async def video_progress(job_id: str):
    if video_job_manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Video prediction job was not found.")
    return StreamingResponse(
        video_job_manager.stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.websocket("/webcam")
async def predict_webcam(websocket: WebSocket):
    await websocket.accept()
    try:
        configuration = await websocket.receive_json()
        if not isinstance(configuration, dict):
            await websocket.close(code=1008, reason="Initial message must be a model configuration object.")
            return
        raw_models = configuration.get("models")
        if not isinstance(raw_models, list) or not all(
            isinstance(model, str) for model in raw_models
        ):
            await websocket.close(code=1008, reason="models must be a list of model identifiers.")
            return
        models = []
        for model in raw_models:
            try:
                models.append(ModelId(model))
            except ValueError:
                await websocket.close(code=1008, reason=f"Unsupported model identifier: {model}.")
                return
        options = configuration.get("model_options", {})
        if not isinstance(options, dict) or any(
            not isinstance(value, dict) for value in options.values()
        ):
            await websocket.close(code=1008, reason="model_options must be an object of option objects.")
            return
        try:
            model_ids, validated_options = parse_configuration(
                [model.value for model in models], options
            )
        except ValueError as error:
            await websocket.close(code=1008, reason=str(error))
            return

        try:
            service = MultiModelPredictionService(model_ids, validated_options)
        except ValueError as error:
            await websocket.close(code=1008, reason=str(error))
            return
        try:
            while True:
                frame_bytes = await websocket.receive_bytes()
                try:
                    frame = await run_in_threadpool(_decode_webcam_frame, frame_bytes)
                except ValueError as error:
                    await websocket.close(code=1003, reason=str(error))
                    return
                predictions = await service.predict_frame(frame)
                await websocket.send_json({"predictions": predictions})
        finally:
            service.close()
    except WebSocketDisconnect:
        return


def _decode_webcam_frame(frame_bytes: bytes):
    import cv2
    import numpy as np

    frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image frame.")
    return frame
