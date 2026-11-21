# FaceFusion video analysis precheck

The video analysis step is a precheck before FaceFusion extracts and processes
the video frames. If the check flags the video, FaceFusion returns error code
`3` from the analysis task. This does not mean that no face was found; it means
the content analyzer rejected the video before face-swap processing.

## Function and source paths

The FaceFusion workflow wrapper is:

`weights\face-swap\facefusion\facefusion\workflows\to_video.py`

Function: `analyse_video() -> ErrorCode` (around line 20). It calls
`content_analyser.analyse_video(...)`; if that returns `True`, the wrapper
returns `3`, otherwise it returns `0`.

The wrapper is scheduled first by the video workflow in:

`weights\face-swap\facefusion\facefusion\workflows\image_to_video.py`

The underlying frame analysis is in:

`weights\face-swap\facefusion\facefusion\content_analyser.py`

Function: `analyse_video(video_path, trim_frame_start, trim_frame_end)`
(around line 168). It samples frames, and returns `True` when the flagged-frame
rate is greater than `10%` (around line 197). A frame is classified as NSFW
when at least two of the three NSFW detectors agree.

For local development, these files are in the FaceFusion checkout under
`weights\face-swap\facefusion`. In the deployed worker container, FaceFusion
source is installed separately from the mounted model weights:

- Workflow wrapper: `/opt/facefusion/facefusion/workflows/to_video.py`
- Video workflow: `/opt/facefusion/facefusion/workflows/image_to_video.py`
- Analyzer implementation: `/opt/facefusion/facefusion/content_analyser.py`

The host folder `model_weights\face-swap\facefusion` contains model assets
(including `.assets`), not the FaceFusion source files. The analyzer check is
separate from face detection and face swapping.

The current containerized deployment bind-mounts the workflow file from
`C:\Docker\Images\dockerhub\lendoo73\ml-rest-api\facefusion\workflows\to_video.py`
to `/opt/facefusion/facefusion/workflows/to_video.py`. Edit the host-side file
and restart the video worker for changes to take effect; no image rebuild is
needed for this file.
