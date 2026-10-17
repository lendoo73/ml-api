from collections import defaultdict
from pathlib import Path
import os

from object_detection.repositories.yolo_model_repository import YoloModelRepository
from oriented_bounding_box.models.entities.oriented_detection import (
    OrientedDetection, OrientedGeometry,
)
from oriented_bounding_box.models.entities.oriented_tracking_detection import (
    OrientedTrackingDetection,
)


class YoloOrientedBoundingBoxModelRepository(YoloModelRepository):
    def __init__(self, model_path: str | None = None):
        default = Path(__file__).resolve().parent.parent / "weights" / "yolo26x-obb.pt"
        super().__init__(model_path or os.getenv("YOLO_ORIENTED_BOUNDING_BOX_MODEL", str(default)))
        self._tracking_trajectories = defaultdict(list)

    @staticmethod
    def _detections_from_result(result) -> list[OrientedDetection]:
        obb = result.obb
        if obb is None:
            return []
        corners = obb.xyxyxyxy.cpu().tolist()
        xywhr = obb.xywhr.cpu().tolist()
        return [
            OrientedDetection(
                class_id=int(cls), class_name=result.names[int(cls)],
                confidence=float(conf),
                obb=OrientedGeometry(
                    x=float(geo[0]), y=float(geo[1]), width=float(geo[2]),
                    height=float(geo[3]), angle=float(geo[4]),
                    corners=[[float(v) for v in point] for point in corner],
                ),
            )
            for geo, corner, cls, conf in zip(xywhr, corners, obb.cls.cpu().tolist(), obb.conf.cpu().tolist())
        ]

    def predict(self, source, conf: float = 0.5):
        return self._detections_from_result(self._predict_result(source, conf))

    def track(self, source, conf=0.5, tracker="bytetrack", trajectory_length=30):
        result = self.model.track(
            source=source, conf=conf, tracker=f"{tracker}.yaml",
            persist=True, verbose=False,
        )[0]
        return self._tracking_detections_from_result(result, trajectory_length)

    def track_video(self, source, conf=0.5, tracker="bytetrack", progress_callback=None):
        total_frames = self._probe_frame_count(source)
        frames = []
        self.reset_tracking()
        try:
            for frame_index, result in enumerate(self.model.track(
                source=source, conf=conf, tracker=f"{tracker}.yaml",
                persist=True, stream=True, verbose=False,
            )):
                if progress_callback:
                    progress_callback(frame_index + 1, total_frames)
                frames.append({
                    "frame_index": frame_index,
                    "timestamp_ms": None,
                    "detections": [item.model_dump() for item in self._tracking_detections_from_result(result)],
                })
        finally:
            self.reset_tracking()
        return frames

    def track_annotated_video(self, source, plot_options=None, conf=0.5, tracker="bytetrack",
                              progress_callback=None):
        self.reset_tracking()
        try:
            return self._process_annotated_video(
                source, plot_options, conf, progress_callback,
                lambda frame: self.model.track(
                    source=frame, conf=conf, tracker=f"{tracker}.yaml",
                    persist=True, verbose=False,
                )[0],
            )
        finally:
            self.reset_tracking()

    def reset_tracking(self):
        self._tracking_trajectories.clear()
        model = getattr(self._thread_models, "model", None)
        if model is not None:
            model.predictor = None

    def _tracking_detections_from_result(self, result, trajectory_length=30):
        obb = result.obb
        if obb is None:
            return []
        ids = obb.id.int().cpu().tolist() if obb.id is not None else [None] * len(obb)
        detections = []
        for geo, corner, cls, conf, track_id in zip(
            obb.xywhr.cpu().tolist(), obb.xyxyxyxy.cpu().tolist(),
            obb.cls.cpu().tolist(), obb.conf.cpu().tolist(), ids,
        ):
            geometry = OrientedGeometry(
                x=float(geo[0]), y=float(geo[1]), width=float(geo[2]),
                height=float(geo[3]), angle=float(geo[4]),
                corners=[[float(v) for v in point] for point in corner],
            )
            trajectory = None
            if track_id is not None:
                center = (round(geometry.x), round(geometry.y))
                trajectory = self._tracking_trajectories[track_id]
                trajectory.append(center)
                del trajectory[:-trajectory_length]
            detections.append(OrientedTrackingDetection(
                class_id=int(cls), class_name=result.names[int(cls)],
                confidence=float(conf), obb=geometry, track_id=track_id,
                trajectory=trajectory,
            ))
        return detections
