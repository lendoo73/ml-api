from pose_estimation.services.pose_angle_service import (
    AngleDefinition,
    PoseAngleService,
)
from pose_estimation.repositories.yolo_pose_model_repository import YoloPoseModelRepository
from services.base_vision_service import BaseVisionService
from services.tracking_service import TrackingService


class PoseEstimationService(BaseVisionService):
    def __init__(self):
        super().__init__(YoloPoseModelRepository())
        self.tracking_service = TrackingService(self.model_repository)
        self.pose_angle_service = PoseAngleService()

    async def predict_image(
        self,
        source,
        output="json",
        plot_options=None,
        conf=0.5,
        angles: list[AngleDefinition] | None = None,
    ):
        poses = await super().predict_image(source, output, plot_options, conf)
        return self.pose_angle_service.add_angles(poses, angles)

    async def predict_url(
        self,
        source_url,
        output="json",
        plot_options=None,
        conf=0.5,
        angles: list[AngleDefinition] | None = None,
    ):
        poses = await super().predict_url(source_url, output, plot_options, conf)
        return self.pose_angle_service.add_angles(poses, angles)

    def predict_frame_bytes(
        self,
        frame_bytes,
        conf=0.5,
        tracking=False,
        tracker="bytetrack",
        angles: list[AngleDefinition] | None = None,
    ):
        poses = (
            self.tracking_service.track_frame_bytes(frame_bytes, conf, tracker)
            if tracking
            else super().predict_frame_bytes(frame_bytes, conf)
        )
        return self.pose_angle_service.add_angles(poses, angles)

    async def predict_video_path_with_angles(
        self,
        video_path,
        conf=0.5,
        progress_callback=None,
        tracking=False,
        tracker="bytetrack",
        angles: list[AngleDefinition] | None = None,
    ):
        frames = (
            await self.predict_tracked_video_path(
                video_path, conf, progress_callback, tracker
            )
            if tracking
            else await super().predict_video_path(video_path, conf, progress_callback)
        )
        return self.pose_angle_service.add_angles_to_video_frames(frames, angles)

    async def predict_video_url_with_angles(
        self,
        source_url,
        conf=0.5,
        progress_callback=None,
        tracking=False,
        tracker="bytetrack",
        angles: list[AngleDefinition] | None = None,
    ):
        frames = (
            await self.predict_tracked_video_url(
                source_url,
                output="json",
                conf=conf,
                progress_callback=progress_callback,
                tracker=tracker,
            )
            if tracking
            else await super().predict_video_url(
                source_url,
                output="json",
                conf=conf,
                progress_callback=progress_callback,
            )
        )
        return self.pose_angle_service.add_angles_to_video_frames(frames, angles)
