from __future__ import annotations

from typing import Any

from datasets.types import CameraIntrinsics, FrontendBatchOutput, RGBDFrame
from utils.instance_builder import PoseInstanceBuilder


class FrontendInferDataset(object):
    def __init__(
        self,
        frame: RGBDFrame,
        frontend_output: FrontendBatchOutput,
        builder: PoseInstanceBuilder,
        device: str = "cuda",
    ) -> None:
        self.frame = frame
        self.frontend_output = frontend_output
        self.builder = builder
        self.device = device
        self._cached_pose_batch = None
        self._cached_legacy_batch = None

    @classmethod
    def from_raw_frame(
        cls,
        data: dict[str, Any],
        frontend,
        builder: PoseInstanceBuilder,
        device: str = "cuda",
    ) -> "FrontendInferDataset":
        frame = RGBDFrame(
            rgb=data["color"],
            depth=data["depth"],
            intrinsics=CameraIntrinsics.from_any(data["meta"]),
            meta=data.get("meta"),
        )
        frontend_output = frontend.predict(frame, data=data)
        return cls(
            frame=frame, frontend_output=frontend_output, builder=builder, device=device
        )

    def get_pose_batch(self):
        if self._cached_pose_batch is None:
            self._cached_pose_batch = self.builder.build_batch(
                self.frame, self.frontend_output
            )
        return self._cached_pose_batch

    def get_objects(self):
        if self._cached_legacy_batch is None:
            self._cached_legacy_batch = self.builder.to_legacy_batch_dict(
                self.get_pose_batch()
            )
        return self._cached_legacy_batch

    @property
    def color(self):
        return self.frame.rgb

    @color.setter
    def color(self, color):
        self.frame.rgb = color

    @property
    def depth(self):
        return self.frame.depth

    @property
    def cam_intrinsics(self):
        if self.frame.meta is not None and not isinstance(self.frame.meta, dict):
            return self.frame.meta.camera.intrinsics
        return self.frame.intrinsics
