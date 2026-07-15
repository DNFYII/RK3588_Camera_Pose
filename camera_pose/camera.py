from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
import cv2


@dataclass(frozen=True)
class CameraSpec:
    source: str | int
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    backend: str = "auto"


def parse_source(device: str) -> str | int:
    if device.isdigit():
        return int(device)
    return device


def gst_v4l2_pipeline(device: str, width: int | None, height: int | None, fps: int | None) -> str:
    caps = ["format=NV12"]
    if width:
        caps.append(f"width={width}")
    if height:
        caps.append(f"height={height}")
    if fps:
        caps.append(f"framerate={fps}/1")
    return (
        f"v4l2src device={device} io-mode=mmap ! "
        f"video/x-raw,{','.join(caps)} ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def open_capture(spec: CameraSpec) -> cv2.VideoCapture:
    source = spec.source
    uses_gstreamer = False
    if isinstance(source, str) and source.startswith("gst:"):
        capture = cv2.VideoCapture(source[4:], cv2.CAP_GSTREAMER)
        uses_gstreamer = True
    elif (
        isinstance(source, str)
        and source.startswith("/dev/video")
        and spec.backend in {"auto", "gst"}
    ):
        capture = cv2.VideoCapture(
            gst_v4l2_pipeline(source, spec.width, spec.height, spec.fps),
            cv2.CAP_GSTREAMER,
        )
        uses_gstreamer = True
    elif isinstance(source, int):
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        capture = cv2.VideoCapture(str(source), cv2.CAP_V4L2)

    if uses_gstreamer:
        return capture

    if spec.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
    if spec.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
    if spec.fps:
        capture.set(cv2.CAP_PROP_FPS, spec.fps)

    return capture


def read_frame(capture: cv2.VideoCapture, warmup: int = 0):
    frame = None
    ok = False
    for _ in range(max(0, warmup)):
        capture.read()
    for _ in range(5):
        ok, frame = capture.read()
        if ok and frame is not None and frame.size:
            return True, frame
    return ok, frame


def list_video_nodes() -> list[Path]:
    return sorted(Path("/dev").glob("video[0-9]*"), key=lambda path: path.name)
