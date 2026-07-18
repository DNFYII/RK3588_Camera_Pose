from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)

from .camera import CameraSpec, list_video_nodes, open_capture, parse_source, read_frame
from .calibration import (
    CalibrationParameters,
    build_undistort_maps,
    chessboard_line_metrics,
    evaluate_models,
    make_new_camera_matrix,
    mapping_metrics,
    undistort_image,
    undistort_points,
)
from .chessboard import (
    draw_axes,
    find_corners,
    find_corners_precise,
    make_object_points,
    orient_corners_with_h,
    solve_planar_pose,
)
from .io import append_markdown_log, ensure_dir, read_yaml, write_image, write_yaml
from .residual_impact import run_residual_pose_impact_experiment

DATA_DIR = Path("data")
DEFAULT_IMAGE_DIR = DATA_DIR / "calibration_images"
DEFAULT_CALIBRATION = DATA_DIR / "calibration.yaml"
DEFAULT_LOG = DATA_DIR / "process_log.md"
DEFAULT_PATTERN = [10, 7]
DEFAULT_VIDEO = DATA_DIR / "calibration_source.avi"


class PreviewState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.running = True


def start_preview_server(state: PreviewState, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            if self.path not in {"/", "/stream.mjpg"}:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while state.running:
                with state.lock:
                    jpeg = state.jpeg
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                except BrokenPipeError:
                    break
                time.sleep(0.05)

    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def overlay_recording_preview(
    frame: np.ndarray,
    pattern: tuple[int, int],
    elapsed: float,
    duration: float,
    frame_count: int,
    min_sharpness: float,
    status: dict,
    recording: bool = True,
) -> np.ndarray:
    preview = frame.copy()
    found = bool(status.get("found", False))
    corners = status.get("corners")
    sharpness = float(status.get("sharpness", 0.0))
    if found and corners is not None:
        cv2.drawChessboardCorners(preview, pattern, corners, found)
    color = (0, 220, 0) if found and sharpness >= min_sharpness else (0, 0, 255)
    if recording:
        duration_text = f"{duration:.0f}s" if duration > 0 else "manual"
        lines = [
            f"REC {elapsed:5.1f}/{duration_text}  frames={frame_count}",
            f"chessboard={'YES' if found else 'NO'}  sharpness={sharpness:.1f}  threshold={min_sharpness:.1f}",
            "Move slowly; keep full board visible; include edges/corners",
        ]
    else:
        lines = [
            "READY - PREVIEW ONLY - NOT RECORDING",
            f"chessboard={'YES' if found else 'NO'}  sharpness={sharpness:.1f}  threshold={min_sharpness:.1f}",
            "Waiting for START; keep the full board visible",
        ]
    y = 34
    for line in lines:
        cv2.putText(preview, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
        y += 34
    return preview


def parse_pattern(values: list[str]) -> tuple[int, int]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("pattern requires two integers: cols rows")
    cols, rows = int(values[0]), int(values[1])
    if cols <= 1 or rows <= 1:
        raise argparse.ArgumentTypeError("pattern values must be greater than 1")
    return cols, rows


def add_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="/dev/video11", help="camera node, index, or gst:<pipeline>")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--backend", choices=["auto", "gst", "v4l2"], default="auto")


def add_log_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)


def camera_spec(args: argparse.Namespace) -> CameraSpec:
    return CameraSpec(parse_source(args.device), args.width, args.height, args.fps, args.backend)


def cmd_probe(args: argparse.Namespace) -> int:
    nodes = [Path(args.device)] if args.device else list_video_nodes()
    if not nodes:
        print("no /dev/video* nodes found")
        return 1

    for node in nodes:
        spec = CameraSpec(str(node), args.width, args.height, args.fps, args.backend)
        capture = open_capture(spec)
        ok, frame = read_frame(capture, warmup=args.warmup)
        capture.release()
        if ok and frame is not None:
            print(f"OK   {node}  shape={frame.shape}  mean={frame.mean():.1f}")
            if args.save:
                out = DATA_DIR / f"probe_{node.name}.jpg"
                write_image(out, frame)
                print(f"     saved {out}")
        else:
            print(f"FAIL {node}")
    return 0


def corner_shift(corners: np.ndarray | None, previous: np.ndarray | None) -> float:
    if corners is None or previous is None or corners.shape != previous.shape:
        return float("inf")
    return float(np.mean(np.linalg.norm(corners.reshape(-1, 2) - previous.reshape(-1, 2), axis=1)))


def corner_geometry_shift(corners: np.ndarray, other: np.ndarray) -> float:
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    reference = np.asarray(other, dtype=np.float32).reshape(-1, 2)
    direct = np.linalg.norm(points - reference, axis=1).mean()
    reversed_order = np.linalg.norm(points - reference[::-1], axis=1).mean()
    return float(min(direct, reversed_order))


def chessboard_sharpness(image: np.ndarray, corners: np.ndarray | None = None) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if corners is not None and corners.size:
        pts = corners.reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(pts.astype(np.float32))
        margin = int(max(w, h) * 0.08)
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(gray.shape[1], x + w + margin)
        y1 = min(gray.shape[0], y + h + margin)
        roi = gray[y0:y1, x0:x1]
        if roi.size:
            gray = roi
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def cmd_capture(args: argparse.Namespace) -> int:
    pattern = tuple(args.pattern)
    ensure_dir(args.output)
    capture = open_capture(camera_spec(args))
    if not capture.isOpened():
        print(f"failed to open camera: {args.device}", file=sys.stderr)
        return 1

    if args.no_window and not args.auto:
        print("--no-window needs --auto so frames can be saved without keyboard input", file=sys.stderr)
        return 1
    if args.no_window:
        print("headless capture: auto-saving frames when corners are detected")
    else:
        print("press SPACE to save a frame, q/ESC to quit")
    count = len(list(args.output.glob("*.jpg")))
    last_save = 0.0
    last_status = 0.0
    last_saved_corners: np.ndarray | None = None
    saved_paths: list[Path] = []
    saved_quality: list[dict] = []
    append_markdown_log(
        args.log,
        "开始采集标定图",
        [
            f"camera={args.device}, backend={args.backend}, size={args.width}x{args.height}, fps={args.fps}",
            f"pattern={pattern[0]}x{pattern[1]}, output={args.output}",
            f"auto={args.auto}, max_images={args.max_images}, min_corner_shift={args.min_corner_shift}px",
            f"min_sharpness={args.min_sharpness}",
        ],
    )
    while True:
        ok, frame = read_frame(capture)
        if not ok or frame is None:
            print("failed to read frame", file=sys.stderr)
            break

        found, corners, _ = find_corners(frame, pattern)
        sharpness = chessboard_sharpness(frame, corners if found else None)
        preview = frame.copy()
        if found and corners is not None:
            cv2.drawChessboardCorners(preview, pattern, corners, found)
        cv2.putText(
            preview,
            f"saved={count} corners={'yes' if found else 'no'} sharp={sharpness:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0) if found and sharpness >= args.min_sharpness else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        key = -1
        if not args.no_window:
            cv2.imshow("capture calibration images", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
        enough_motion = corner_shift(corners, last_saved_corners) >= args.min_corner_shift
        sharp_enough = sharpness >= args.min_sharpness
        should_save = key == ord(" ") or (
            args.auto
            and found
            and enough_motion
            and sharp_enough
            and time.time() - last_save > args.auto_interval
        )
        if should_save:
            path = args.output / f"calib_{count:03d}.jpg"
            write_image(path, frame)
            print(f"saved {path} sharpness={sharpness:.2f}")
            saved_paths.append(path)
            saved_quality.append({"file": str(path), "sharpness": sharpness})
            if corners is not None:
                last_saved_corners = corners.copy()
            count += 1
            last_save = time.time()
            if args.max_images and count >= args.max_images:
                break
        elif args.no_window and time.time() - last_status > args.status_interval:
            reasons = []
            if not found:
                reasons.append("未检测到完整棋盘")
            if found and not enough_motion:
                reasons.append("姿态变化不足")
            if found and not sharp_enough:
                reasons.append(f"清晰度不足({sharpness:.1f}<{args.min_sharpness:.1f})")
            print(f"waiting saved={count}/{args.max_images or '-'} sharpness={sharpness:.1f} {'; '.join(reasons)}")
            last_status = time.time()

    capture.release()
    if not args.no_window:
        cv2.destroyAllWindows()
    sharpness_summary = (
        ", ".join(f"{Path(item['file']).name}:{item['sharpness']:.1f}" for item in saved_quality)
        if saved_quality
        else "none"
    )
    append_markdown_log(
        args.log,
        "结束采集标定图",
        [
            f"saved_count={len(saved_paths)}",
            f"saved_files={', '.join(str(path) for path in saved_paths) if saved_paths else 'none'}",
            f"sharpness={sharpness_summary}",
        ],
    )
    return 0


def collect_calibration_points(
    image_dir: Path,
    pattern: tuple[int, int],
    square_size: float,
    precise: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int], list[Path]]:
    object_template = make_object_points(pattern, square_size)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used: list[Path] = []
    image_size: tuple[int, int] | None = None

    for path in sorted(image_dir.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        detector = find_corners_precise if precise else find_corners
        found, corners, gray = detector(image, pattern)
        if not found or corners is None:
            print(f"skip {path}: corners not found")
            continue
        object_points.append(object_template.copy())
        image_points.append(corners)
        used.append(path)
        image_size = (gray.shape[1], gray.shape[0])
        print(f"use  {path}")

    if image_size is None:
        raise RuntimeError("no usable chessboard images found")
    return object_points, image_points, image_size, used


def cmd_calibrate(args: argparse.Namespace) -> int:
    if not 0.0 <= args.undistort_alpha <= 1.0:
        print("--undistort-alpha must be between 0 and 1", file=sys.stderr)
        return 1
    pattern = tuple(args.pattern)
    object_points, image_points, image_size, used = collect_calibration_points(
        args.images,
        pattern,
        args.square_size,
    )
    if len(used) < args.min_images:
        print(
            f"need at least {args.min_images} usable images, found {len(used)}",
            file=sys.stderr,
        )
        return 1

    evaluation, selected_fit = evaluate_models(
        object_points,
        image_points,
        image_size,
        pattern,
        folds=args.cv_folds,
        alpha=args.undistort_alpha,
    )
    model = str(evaluation["selected_model"])
    camera_matrix = np.asarray(selected_fit["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(selected_fit["dist_coeffs"], dtype=np.float64).reshape(-1)
    rms = float(selected_fit["rms_reprojection_error"])
    new_camera_matrix, valid_roi = make_new_camera_matrix(
        model,
        camera_matrix,
        dist_coeffs,
        image_size,
        args.undistort_alpha,
    )
    calibration = CalibrationParameters(
        model=model,
        image_size=image_size,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs.reshape(-1, 1),
        new_camera_matrix=new_camera_matrix,
        undistort_alpha=args.undistort_alpha,
        valid_roi=valid_roi,
    )
    map_quality = mapping_metrics(calibration)
    model_constraints = {
        "pinhole_radial2": ["p1=0", "p2=0", "k3=0"],
        "pinhole_radial3": ["p1=0", "p2=0"],
        "pinhole_tangent3": [],
        "fisheye": ["skew=0"],
    }[model]
    data = {
        "pattern": list(pattern),
        "square_size": args.square_size,
        "image_size": list(image_size),
        "corner_detector": "findChessboardCorners+cornerSubPix",
        "model": model,
        "model_constraints": model_constraints,
        "rms_reprojection_error": float(rms),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs.reshape(-1),
        "new_camera_matrix": new_camera_matrix,
        "undistortion": {
            "scope": "global_full_frame",
            "alpha": args.undistort_alpha,
            "pixel_aspect_policy": "square_pixels_fx_equals_fy",
            "target_max_invalid_output_ratio": 0.0,
            "output_size": list(image_size),
            "valid_roi_xywh": list(valid_roi),
            **map_quality,
        },
        "used_images": [str(path) for path in used],
        "views": len(used),
    }
    write_yaml(args.output, data)
    write_yaml(args.evaluation_output, evaluation)
    append_markdown_log(
        args.log,
        "完成相机内参标定",
        [
            f"pattern={pattern[0]}x{pattern[1]}, square_size={args.square_size} mm",
            f"image_size={image_size[0]}x{image_size[1]}, views={len(used)}",
            f"selected_model={model}, corner_detector=findChessboardCorners+cornerSubPix",
            f"rms_reprojection_error={rms:.6f}",
            f"new_camera_matrix={new_camera_matrix.tolist()}",
            f"global_mapping={map_quality}",
            f"calibration_file={args.output}",
            f"model_evaluation_file={args.evaluation_output}",
        ],
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.evaluation_output}")
    print(f"selected model: {model}")
    print(f"rms reprojection error: {rms:.4f}")
    return 0


def cmd_detect_pattern(args: argparse.Namespace) -> int:
    image = cv2.imread(str(args.image))
    if image is None:
        print(f"failed to read image: {args.image}", file=sys.stderr)
        return 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    matches: list[tuple[int, int]] = []
    priority = [
        tuple(args.pattern),
        (10, 7),
        (9, 6),
        (10, 6),
        (9, 7),
        (8, 6),
        (8, 7),
        (11, 8),
    ]
    seen: set[tuple[int, int]] = set()

    def candidates():
        for item in priority:
            if item not in seen:
                seen.add(item)
                yield item
        if args.all:
            for cols in range(args.min_cols, args.max_cols + 1):
                for rows in range(args.min_rows, args.max_rows + 1):
                    item = (cols, rows)
                    if item not in seen:
                        seen.add(item)
                        yield item

    for cols, rows in candidates():
        if cols < args.min_cols or cols > args.max_cols or rows < args.min_rows or rows > args.max_rows:
            continue
        ok, _ = cv2.findChessboardCorners(
            gray,
            (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if ok:
            matches.append((cols, rows))
            if not args.all:
                break
    if not matches:
        print("no chessboard pattern found")
        return 1
    matches.sort(key=lambda item: item[0] * item[1], reverse=True)
    print("candidate inner-corner patterns:")
    for cols, rows in matches:
        print(f"  {cols} {rows}  corners={cols * rows}")
    print(f"best guess: --pattern {matches[0][0]} {matches[0][1]}")
    return 0


def quality_rows(
    image_dir: Path,
    pattern: tuple[int, int],
    square_size: float | None = None,
    calibration: Path | None = None,
) -> list[dict]:
    rows = []
    camera_matrix = None
    dist_coeffs = None
    object_points = None
    if calibration and calibration.exists() and square_size is not None:
        camera_matrix, dist_coeffs = load_calibration(calibration)
        object_points = make_object_points(pattern, square_size)

    for path in sorted(image_dir.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = cv2.imread(str(path))
        if image is None:
            rows.append({"file": str(path), "readable": False})
            continue
        found, corners, _ = find_corners_precise(image, pattern)
        row = {
            "file": str(path),
            "readable": True,
            "corners_found": bool(found),
            "sharpness": chessboard_sharpness(image, corners if found else None),
        }
        if found and corners is not None and camera_matrix is not None and object_points is not None:
            ok, rvec, tvec, _ = solve_planar_pose(
                object_points,
                corners,
                camera_matrix,
                dist_coeffs,
            )
            if ok:
                projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
                errors = np.linalg.norm(corners.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
                row["mean_reprojection_error_px"] = float(errors.mean())
                row["max_reprojection_error_px"] = float(errors.max())
        rows.append(row)
    return rows


def cmd_quality(args: argparse.Namespace) -> int:
    pattern = tuple(args.pattern)
    rows = quality_rows(args.images, pattern, args.square_size, args.calibration)
    if not rows:
        print(f"no images found in {args.images}", file=sys.stderr)
        return 1

    write_yaml(args.output, {"images": rows})
    lines = [
        f"images={args.images}",
        f"pattern={pattern[0]}x{pattern[1]}",
        f"quality_file={args.output}",
    ]
    print("file,corners,sharpness,mean_err_px,max_err_px")
    for row in rows:
        mean_err = row.get("mean_reprojection_error_px")
        max_err = row.get("max_reprojection_error_px")
        mean_text = f"{mean_err:.3f}" if mean_err is not None else ""
        max_text = f"{max_err:.3f}" if max_err is not None else ""
        print(
            f"{Path(row['file']).name},"
            f"{row.get('corners_found', False)},"
            f"{row.get('sharpness', 0.0):.2f},"
            f"{mean_text},"
            f"{max_text}"
        )
        if mean_err is not None:
            lines.append(
                f"{Path(row['file']).name}: sharpness={row['sharpness']:.2f}, "
                f"mean_err={mean_err:.3f}px, max_err={max_err:.3f}px"
            )
        else:
            lines.append(
                f"{Path(row['file']).name}: sharpness={row.get('sharpness', 0.0):.2f}, "
                f"corners_found={row.get('corners_found', False)}"
            )
    append_markdown_log(args.log, "完成标定图质量检查", lines)
    return 0


def cmd_record_video(args: argparse.Namespace) -> int:
    ensure_dir(args.output.parent)
    capture = open_capture(camera_spec(args))
    if not capture.isOpened():
        print(f"failed to open camera: {args.device}", file=sys.stderr)
        return 1

    pattern = tuple(args.pattern)
    preview_state: PreviewState | None = None
    preview_server: ThreadingHTTPServer | None = None
    if args.preview_port:
        preview_state = PreviewState()
        preview_server = start_preview_server(preview_state, args.preview_host, args.preview_port)
        print(f"live preview: http://{args.preview_host}:{args.preview_port}/")

    start_requested = threading.Event()
    if args.wait_for_start:
        def wait_for_start_signal() -> None:
            try:
                line = sys.stdin.readline()
            except (OSError, ValueError):
                return
            if line:
                start_requested.set()

        threading.Thread(target=wait_for_start_signal, daemon=True).start()
        print("preview ready; recording has NOT started")
        print("press Enter to start recording")
    else:
        start_requested.set()

    writer: cv2.VideoWriter | None = None
    started: float | None = None
    last_status = time.time()
    frames = 0
    captured_frames = 0
    status = {"found": False, "corners": None, "sharpness": 0.0}
    interrupted = False
    start_failed = False
    try:
        while True:
            if started is not None and args.duration > 0 and time.time() - started >= args.duration:
                break
            ok, frame = read_frame(capture)
            if not ok or frame is None:
                print("failed to read frame", file=sys.stderr)
                break
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

            captured_frames += 1
            if writer is None and start_requested.is_set():
                fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
                writer = cv2.VideoWriter(
                    str(args.output),
                    fourcc,
                    args.fps,
                    (args.width, args.height),
                )
                if not writer.isOpened():
                    print(f"failed to open video writer: {args.output}", file=sys.stderr)
                    writer = None
                    start_failed = True
                    break
                started = time.time()
                last_status = started
                append_markdown_log(
                    args.log,
                    "开始录制标定视频",
                    [
                        f"camera={args.device}, backend={args.backend}, size={args.width}x{args.height}, fps={args.fps}",
                        f"duration={'manual' if args.duration <= 0 else f'{args.duration}s'}, output={args.output}, fourcc={args.fourcc}",
                        f"preview_port={args.preview_port}, min_sharpness={args.min_sharpness}, pattern={pattern[0]}x{pattern[1]}",
                    ],
                )
                if args.duration > 0:
                    print(f"recording {args.duration:.1f}s to {args.output}")
                else:
                    print(f"recording until Ctrl+C to {args.output}")

            if writer is not None:
                writer.write(frame)
                frames += 1

            if captured_frames % max(1, args.preview_every) == 0:
                found, corners, _ = find_corners(frame, pattern)
                sharpness = chessboard_sharpness(frame, corners if found else None)
                status = {"found": found, "corners": corners, "sharpness": sharpness}
                recording = writer is not None and started is not None
                annotated = overlay_recording_preview(
                    frame,
                    pattern,
                    time.time() - started if started is not None else 0.0,
                    args.duration,
                    frames,
                    args.min_sharpness,
                    status,
                    recording=recording,
                )
                write_image(args.preview_image, annotated)
                if preview_state is not None:
                    ok_jpg, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                    if ok_jpg:
                        with preview_state.lock:
                            preview_state.jpeg = encoded.tobytes()
                if args.preview_window:
                    cv2.imshow("recording preview", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if writer is None and key in (10, 13, ord("s")):
                        start_requested.set()
                    if key in (27, ord("q")):
                        interrupted = True
                        break

            now = time.time()
            if now - last_status >= args.status_interval:
                if started is None:
                    print(
                        "preview ready; NOT RECORDING "
                        f"corners={'yes' if status.get('found') else 'no'} "
                        f"sharpness={float(status.get('sharpness', 0.0)):.1f}"
                    )
                else:
                    elapsed = now - started
                    print(
                        f"recording elapsed={elapsed:.1f}s frames={frames} "
                        f"corners={'yes' if status.get('found') else 'no'} "
                        f"sharpness={float(status.get('sharpness', 0.0)):.1f}"
                    )
                last_status = now
    except KeyboardInterrupt:
        interrupted = True
        print("recording interrupted" if started is not None else "preview interrupted before recording")
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        if args.preview_window:
            cv2.destroyAllWindows()
        if preview_state is not None:
            preview_state.running = False
        if preview_server is not None:
            preview_server.shutdown()
            preview_server.server_close()

    if started is None:
        if start_failed:
            return 1
        print("preview stopped; no video was recorded")
        return 0

    elapsed = time.time() - started
    append_markdown_log(
        args.log,
        "结束录制标定视频",
        [
            f"output={args.output}",
            f"elapsed={elapsed:.3f}s",
            f"frames={frames}",
            f"approx_fps={frames / elapsed:.3f}" if elapsed > 0 else "approx_fps=unknown",
            f"interrupted={interrupted}",
        ],
    )
    print(f"wrote {args.output} frames={frames} elapsed={elapsed:.1f}s")
    return 0 if frames > 0 else 1


def cmd_extract_frames(args: argparse.Namespace) -> int:
    pattern = tuple(args.pattern)
    ensure_dir(args.output)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        print(f"failed to open video: {args.video}", file=sys.stderr)
        return 1

    for old in args.output.glob("*.jpg"):
        old.unlink()

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * args.sample_interval))) if fps > 0 else max(1, args.frame_step)
    candidates: list[dict] = []
    scanned = 0
    frame_index = 0
    append_markdown_log(
        args.log,
        "开始从视频抽取标定帧",
        [
            f"video={args.video}",
            f"output={args.output}",
            f"target_frames={args.max_images}, pattern={pattern[0]}x{pattern[1]}",
            f"min_sharpness={args.min_sharpness}, min_corner_shift={args.min_corner_shift}px",
            f"fps={fps:.3f}, total_frames={total}, sample_step={step}",
        ],
    )

    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if frame_index % step != 0:
            frame_index += 1
            continue
        scanned += 1
        found, corners, _ = find_corners(frame, pattern)
        if found and corners is not None:
            sharpness = chessboard_sharpness(frame, corners)
            if sharpness >= args.min_sharpness:
                candidates.append(
                    {
                        "video_frame": frame_index,
                        "time_sec": frame_index / fps if fps > 0 else None,
                        "sharpness": sharpness,
                        "corners": corners.copy(),
                    }
                )
        if scanned % args.status_every == 0:
            print(f"extract scanned={scanned} candidates={len(candidates)} frame={frame_index}")
        frame_index += 1

    capture.release()
    if len(candidates) < args.max_images:
        print(
            f"only found {len(candidates)}/{args.max_images} clear candidate frames",
            file=sys.stderr,
        )
        return 1

    sharpest = max(range(len(candidates)), key=lambda index: candidates[index]["sharpness"])
    selected_indices = [sharpest]
    remaining = set(range(len(candidates))) - {sharpest}
    maximum_sharpness = max(float(candidate["sharpness"]) for candidate in candidates)
    while remaining and len(selected_indices) < args.max_images:
        best_index: int | None = None
        best_score = -1.0
        best_distance = 0.0
        for index in remaining:
            distance = min(
                corner_geometry_shift(candidates[index]["corners"], candidates[chosen]["corners"])
                for chosen in selected_indices
            )
            quality = float(candidates[index]["sharpness"]) / maximum_sharpness
            score = distance * (0.8 + 0.2 * np.sqrt(max(0.0, quality)))
            if score > best_score:
                best_index = index
                best_score = score
                best_distance = distance
        if best_index is None or best_distance < args.min_corner_shift:
            break
        candidates[best_index]["nearest_selected_shift"] = best_distance
        selected_indices.append(best_index)
        remaining.remove(best_index)

    if len(selected_indices) < args.max_images:
        print(
            f"only selected {len(selected_indices)}/{args.max_images} frames with "
            f"minimum corner shift {args.min_corner_shift:.1f}px",
            file=sys.stderr,
        )
        return 1

    candidates[sharpest]["nearest_selected_shift"] = None
    selected_candidates = sorted(
        (candidates[index] for index in selected_indices),
        key=lambda item: int(item["video_frame"]),
    )
    selected_by_frame = {
        int(candidate["video_frame"]): (output_index, candidate)
        for output_index, candidate in enumerate(selected_candidates)
    }
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        print(f"failed to reopen video: {args.video}", file=sys.stderr)
        return 1
    selected: list[dict] = []
    frame_index = 0
    while selected_by_frame:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        target = selected_by_frame.pop(frame_index, None)
        if target is not None:
            output_index, candidate = target
            path = args.output / f"calib_{output_index:03d}.jpg"
            write_image(path, frame)
            selected.append(
                {
                    "file": str(path),
                    "video_frame": frame_index,
                    "time_sec": candidate["time_sec"],
                    "sharpness": candidate["sharpness"],
                    "nearest_selected_shift": candidate["nearest_selected_shift"],
                }
            )
            print(f"saved {path} frame={frame_index} sharpness={candidate['sharpness']:.2f}")
        frame_index += 1
    capture.release()
    selected.sort(key=lambda item: int(item["video_frame"]))
    metadata = {
        "video": str(args.video),
        "candidate_count": len(candidates),
        "selection": (
            f"full-video diverse selection of {args.max_images} frames by checkerboard "
            "corner geometry, weighted by sharpness"
        ),
        "minimum_corner_shift_px": args.min_corner_shift,
        "frames": selected,
    }
    write_yaml(args.output.parent / "extracted_frames.yaml", metadata)
    append_markdown_log(
        args.log,
        "结束从视频抽取标定帧",
        [
            f"video={args.video}",
            f"candidate_count={len(candidates)}, saved_count={len(selected)}",
            f"selected_metadata={args.output.parent / 'extracted_frames.yaml'}",
            f"saved_files={', '.join(item['file'] for item in selected) if selected else 'none'}",
        ],
    )
    if len(selected) < args.max_images:
        print(f"only extracted {len(selected)}/{args.max_images} frames", file=sys.stderr)
        return 1
    return 0


def load_calibration_parameters(path: Path) -> CalibrationParameters:
    return CalibrationParameters.from_mapping(read_yaml(path))


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    calibration = load_calibration_parameters(path)
    return calibration.camera_matrix, calibration.dist_coeffs


def estimate_pose(
    frame: np.ndarray,
    pattern: tuple[int, int],
    square_size: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    point_transform=None,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None, dict | None]:
    found, corners, _ = find_corners(frame, pattern)
    if not found or corners is None:
        return False, None, None, corners, None
    corners, h_orientation = orient_corners_with_h(frame, corners, pattern)
    if point_transform is not None:
        corners = point_transform(corners)
    object_points = make_object_points(pattern, square_size)
    ok, rvec, tvec, _ = solve_planar_pose(
        object_points,
        corners,
        camera_matrix,
        dist_coeffs,
    )
    return bool(ok), rvec, tvec, corners, h_orientation


def pose_payload(
    rvec: np.ndarray,
    tvec: np.ndarray,
    pattern: tuple[int, int],
    square_size: float,
) -> dict:
    rotation, _ = cv2.Rodrigues(rvec)
    camera_position_board = -rotation.T @ tvec.reshape(3, 1)
    return {
        "timestamp": time.time(),
        "pattern": list(pattern),
        "square_size": square_size,
        "rvec_board_to_camera": rvec.reshape(3),
        "tvec_board_to_camera": tvec.reshape(3),
        "rotation_matrix_board_to_camera": rotation,
        "camera_position_board": camera_position_board.reshape(3),
    }


def cmd_pose(args: argparse.Namespace) -> int:
    pattern = tuple(args.pattern)
    calibration = load_calibration_parameters(args.calibration)
    if args.raw:
        camera_matrix = calibration.camera_matrix
        dist_coeffs = calibration.dist_coeffs
        maps = None
        image_space = "raw_distorted"
    else:
        camera_matrix = calibration.new_camera_matrix
        dist_coeffs = calibration.corrected_dist_coeffs
        maps = build_undistort_maps(calibration)
        image_space = "globally_undistorted"
    capture = open_capture(camera_spec(args))
    if not capture.isOpened():
        print(f"failed to open camera: {args.device}", file=sys.stderr)
        return 1

    axis_length = args.axis_length or args.square_size * min(pattern) * 0.5
    last_payload = None
    started = time.time()
    while True:
        if args.timeout and time.time() - started > args.timeout:
            print(f"timed out after {args.timeout:.1f}s waiting for chessboard", file=sys.stderr)
            break
        ok, frame = read_frame(capture)
        if not ok or frame is None:
            print("failed to read frame", file=sys.stderr)
            break

        pose_ok, rvec, tvec, corners, h_orientation = estimate_pose(
            frame,
            pattern,
            args.square_size,
            camera_matrix,
            dist_coeffs,
            point_transform=(
                (lambda points: undistort_points(points, calibration))
                if maps is not None
                else None
            ),
        )
        if maps is not None:
            frame, _ = undistort_image(frame, calibration, maps)
        preview = frame.copy()
        if corners is not None:
            cv2.drawChessboardCorners(preview, pattern, corners, pose_ok)
        if pose_ok and rvec is not None and tvec is not None:
            preview = draw_axes(preview, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
            last_payload = pose_payload(rvec, tvec, pattern, args.square_size)
            last_payload.update(
                {
                    "source_calibration": str(args.calibration),
                    "image_space": image_space,
                    "camera_matrix_used": camera_matrix,
                    "dist_coeffs_used": dist_coeffs.reshape(-1),
                    "h_orientation": h_orientation,
                }
            )
            write_yaml(args.output, last_payload)
            write_image(args.image_output, preview)
            if args.log_every_pose or args.once:
                cam = np.asarray(last_payload["camera_position_board"]).reshape(3)
                tv = tvec.reshape(3)
                append_markdown_log(
                    args.log,
                    "完成棋盘格位姿估计",
                    [
                        f"calibration_file={args.calibration}",
                        f"image_space={image_space}",
                        f"h_orientation={h_orientation}",
                        f"pose_file={args.output}, image_file={args.image_output}",
                        f"tvec_board_to_camera_mm=({tv[0]:.3f}, {tv[1]:.3f}, {tv[2]:.3f})",
                        f"camera_position_board_mm=({cam[0]:.3f}, {cam[1]:.3f}, {cam[2]:.3f})",
                    ],
                )
            t = tvec.reshape(3)
            cv2.putText(
                preview,
                f"t=({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if args.once:
                print(f"wrote {args.output}")
                break
        else:
            cv2.putText(
                preview,
                "chessboard not found",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if args.no_window:
            if args.once and last_payload is not None:
                break
            time.sleep(args.interval)
        else:
            cv2.imshow("pose", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    capture.release()
    if not args.no_window:
        cv2.destroyAllWindows()
    if last_payload is None:
        return 1
    return 0


def cmd_undistort_image(args: argparse.Namespace) -> int:
    image = cv2.imread(str(args.input))
    if image is None:
        print(f"failed to read image: {args.input}", file=sys.stderr)
        return 1
    calibration = load_calibration_parameters(args.calibration)
    try:
        corrected, valid_mask = undistort_image(image, calibration)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    write_image(args.output, corrected)
    pattern = tuple(args.pattern)
    raw_found, raw_corners, _ = find_corners_precise(image, pattern)
    corrected_found, corrected_corners, _ = find_corners_precise(corrected, pattern)
    metrics: dict[str, object] = {
        "source_image": str(args.input),
        "output_image": str(args.output),
        "calibration_file": str(args.calibration),
        "scope": "global_full_frame",
        "model": calibration.model,
        "image_size": list(calibration.image_size),
        "new_camera_matrix": calibration.new_camera_matrix,
        "remaining_dist_coeffs": calibration.corrected_dist_coeffs.reshape(-1),
        "invalid_output_ratio": float(np.mean(valid_mask == 0)),
        "raw_chessboard_found": bool(raw_found),
        "corrected_chessboard_found": bool(corrected_found),
    }
    if raw_found and raw_corners is not None:
        metrics["raw_chessboard_line_residual"] = chessboard_line_metrics(
            [raw_corners],
            pattern,
        )
    if corrected_found and corrected_corners is not None:
        metrics["corrected_chessboard_line_residual"] = chessboard_line_metrics(
            [corrected_corners],
            pattern,
        )
    write_yaml(args.metrics_output, metrics)
    append_markdown_log(
        args.log,
        "完成全图去畸变",
        [
            f"source={args.input}, output={args.output}",
            f"calibration={args.calibration}, model={calibration.model}",
            f"new_camera_matrix={calibration.new_camera_matrix.tolist()}",
            f"invalid_output_ratio={float(np.mean(valid_mask == 0)):.6f}",
            f"metrics={args.metrics_output}",
        ],
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.metrics_output}")
    return 0


def cmd_pose_image(args: argparse.Namespace) -> int:
    image = cv2.imread(str(args.input))
    if image is None:
        print(f"failed to read image: {args.input}", file=sys.stderr)
        return 1
    calibration = load_calibration_parameters(args.calibration)
    try:
        corrected, valid_mask = undistort_image(image, calibration)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    pattern = tuple(args.pattern)
    found, corners, _ = find_corners_precise(image, pattern)
    if not found or corners is None:
        print("chessboard not found in source image", file=sys.stderr)
        return 1
    oriented_corners, h_orientation = orient_corners_with_h(image, corners, pattern)
    corrected_corners = undistort_points(oriented_corners, calibration)
    object_points = make_object_points(pattern, args.square_size)
    ok, rvec, tvec, reprojection = solve_planar_pose(
        object_points,
        corrected_corners,
        calibration.new_camera_matrix,
        calibration.corrected_dist_coeffs,
    )
    if not ok or rvec is None or tvec is None:
        print("pose estimation failed", file=sys.stderr)
        return 1

    write_image(args.undistorted_output, corrected)
    preview = corrected.copy()
    cv2.drawChessboardCorners(preview, pattern, corrected_corners, True)
    axis_length = args.axis_length or args.square_size * min(pattern) * 0.5
    preview = draw_axes(
        preview,
        calibration.new_camera_matrix,
        calibration.corrected_dist_coeffs,
        rvec,
        tvec,
        axis_length,
    )
    translation = tvec.reshape(3)
    cv2.putText(
        preview,
        f"t=({translation[0]:.1f}, {translation[1]:.1f}, {translation[2]:.1f}) mm",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"t=({translation[0]:.1f}, {translation[1]:.1f}, {translation[2]:.1f}) mm",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    write_image(args.image_output, preview)

    payload = pose_payload(rvec, tvec, pattern, args.square_size)
    payload.update(
        {
            "source_image": str(args.input),
            "source_calibration": str(args.calibration),
            "image_space": "globally_undistorted",
            "undistorted_image": str(args.undistorted_output),
            "visualization_image": str(args.image_output),
            "camera_matrix_used": calibration.new_camera_matrix,
            "dist_coeffs_used": calibration.corrected_dist_coeffs.reshape(-1),
            "invalid_output_ratio": float(np.mean(valid_mask == 0)),
            "pose_solver": "IPPE 双解选择 + LM 精化",
            "reprojection": reprojection,
            "h_orientation": h_orientation,
            "h_usage": "H 外形用于消除棋盘格 180 度方向歧义；毫米位姿由 70 个棋盘格角点计算。",
        }
    )
    write_yaml(args.output, payload)
    camera_position = np.asarray(payload["camera_position_board"]).reshape(3)
    append_markdown_log(
        args.log,
        "完成全图去畸变后的 H 辅助棋盘格位姿估计",
        [
            f"source={args.input}, calibration={args.calibration}",
            f"undistorted_image={args.undistorted_output}",
            f"pose_file={args.output}, visualization={args.image_output}",
            f"h_orientation={h_orientation}",
            f"mean_reprojection_error_px={reprojection['mean_reprojection_error_px']:.6f}",
            f"tvec_board_to_camera_mm=({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f})",
            f"camera_position_board_mm=({camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f})",
        ],
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.image_output}")
    print(f"wrote {args.undistorted_output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RK3588 camera calibration and pose tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="probe /dev/video* capture nodes")
    probe.add_argument("--device", help="probe only this device")
    probe.add_argument("--width", type=int, default=1280)
    probe.add_argument("--height", type=int, default=720)
    probe.add_argument("--fps", type=int, default=30)
    probe.add_argument("--backend", choices=["auto", "gst", "v4l2"], default="auto")
    probe.add_argument("--warmup", type=int, default=5)
    probe.add_argument("--save", action="store_true")
    probe.set_defaults(func=cmd_probe)

    capture = subparsers.add_parser("capture", help="capture chessboard calibration images")
    add_camera_args(capture)
    add_log_arg(capture)
    capture.add_argument("--pattern", nargs=2, type=int, default=DEFAULT_PATTERN, metavar=("COLS", "ROWS"))
    capture.add_argument("--output", type=Path, default=DEFAULT_IMAGE_DIR)
    capture.add_argument("--auto", action="store_true", help="auto-save frames when corners are detected")
    capture.add_argument("--auto-interval", type=float, default=1.0)
    capture.add_argument("--max-images", type=int, default=30)
    capture.add_argument("--min-corner-shift", type=float, default=18.0)
    capture.add_argument("--min-sharpness", type=float, default=120.0)
    capture.add_argument("--status-interval", type=float, default=2.0)
    capture.add_argument("--no-window", action="store_true")
    capture.set_defaults(func=cmd_capture)

    calibrate = subparsers.add_parser("calibrate", help="calibrate camera intrinsics")
    add_log_arg(calibrate)
    calibrate.add_argument("--pattern", nargs=2, type=int, default=DEFAULT_PATTERN, metavar=("COLS", "ROWS"))
    calibrate.add_argument("--square-size", type=float, required=True)
    calibrate.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    calibrate.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION)
    calibrate.add_argument(
        "--evaluation-output",
        type=Path,
        default=DATA_DIR / "calibration_model_evaluation.yaml",
    )
    calibrate.add_argument("--min-images", type=int, default=8)
    calibrate.add_argument("--cv-folds", type=int, default=3)
    calibrate.add_argument("--undistort-alpha", type=float, default=0.0)
    calibrate.set_defaults(func=cmd_calibrate)

    pose = subparsers.add_parser("pose", help="estimate chessboard pose from live camera")
    add_camera_args(pose)
    add_log_arg(pose)
    pose.add_argument("--pattern", nargs=2, type=int, default=DEFAULT_PATTERN, metavar=("COLS", "ROWS"))
    pose.add_argument("--square-size", type=float, required=True)
    pose.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    pose.add_argument("--output", type=Path, default=DATA_DIR / "pose_latest.yaml")
    pose.add_argument("--image-output", type=Path, default=DATA_DIR / "pose_latest.jpg")
    pose.add_argument("--axis-length", type=float)
    pose.add_argument("--once", action="store_true")
    pose.add_argument("--no-window", action="store_true")
    pose.add_argument("--interval", type=float, default=0.05)
    pose.add_argument("--timeout", type=float, default=30.0)
    pose.add_argument("--log-every-pose", action="store_true")
    pose.add_argument("--raw", action="store_true", help="skip global undistortion")
    pose.set_defaults(func=cmd_pose)

    detect = subparsers.add_parser("detect-pattern", help="find likely chessboard inner-corner count")
    detect.add_argument("image", type=Path)
    detect.add_argument("--pattern", nargs=2, type=int, default=DEFAULT_PATTERN, metavar=("COLS", "ROWS"))
    detect.add_argument("--min-cols", type=int, default=4)
    detect.add_argument("--max-cols", type=int, default=14)
    detect.add_argument("--min-rows", type=int, default=4)
    detect.add_argument("--max-rows", type=int, default=12)
    detect.add_argument("--all", action="store_true", help="scan the full range instead of priority candidates")
    detect.set_defaults(func=cmd_detect_pattern)

    quality = subparsers.add_parser("quality", help="check calibration image sharpness and reprojection error")
    add_log_arg(quality)
    quality.add_argument("--pattern", nargs=2, type=int, default=DEFAULT_PATTERN, metavar=("COLS", "ROWS"))
    quality.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    quality.add_argument("--square-size", type=float, default=24.0)
    quality.add_argument("--calibration", type=Path)
    quality.add_argument("--output", type=Path, default=DATA_DIR / "quality_report.yaml")
    quality.set_defaults(func=cmd_quality)

    record = subparsers.add_parser("record-video", help="record a calibration/data-source video")
    add_camera_args(record)
    add_log_arg(record)
    record.add_argument("--output", type=Path, default=DEFAULT_VIDEO)
    record.add_argument("--duration", type=float, default=0.0, help="seconds to record; 0 records until Ctrl+C")
    record.add_argument("--fourcc", default="MJPG")
    record.add_argument("--status-interval", type=float, default=10.0)
    record.add_argument("--pattern", nargs=2, type=int, default=DEFAULT_PATTERN, metavar=("COLS", "ROWS"))
    record.add_argument("--min-sharpness", type=float, default=120.0)
    record.add_argument("--preview-host", default="0.0.0.0")
    record.add_argument("--preview-port", type=int, default=8080)
    record.add_argument("--preview-every", type=int, default=3)
    record.add_argument("--preview-image", type=Path, default=DATA_DIR / "live_preview.jpg")
    record.add_argument("--preview-window", action="store_true")
    record.add_argument(
        "--wait-for-start",
        action="store_true",
        help="show live preview first and wait for Enter before creating the video",
    )
    record.set_defaults(func=cmd_record_video)

    extract = subparsers.add_parser("extract-frames", help="extract calibration frames from a recorded video")
    add_log_arg(extract)
    extract.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    extract.add_argument("--pattern", nargs=2, type=int, default=DEFAULT_PATTERN, metavar=("COLS", "ROWS"))
    extract.add_argument("--output", type=Path, default=DEFAULT_IMAGE_DIR)
    extract.add_argument("--max-images", type=int, default=30)
    extract.add_argument("--min-sharpness", type=float, default=120.0)
    extract.add_argument("--min-corner-shift", type=float, default=24.0)
    extract.add_argument("--sample-interval", type=float, default=0.5)
    extract.add_argument("--frame-step", type=int, default=15)
    extract.add_argument("--status-every", type=int, default=25)
    extract.set_defaults(func=cmd_extract_frames)

    undistort = subparsers.add_parser("undistort-image", help="globally undistort one image")
    add_log_arg(undistort)
    undistort.add_argument("--input", type=Path, required=True)
    undistort.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    undistort.add_argument("--output", type=Path, default=DATA_DIR / "undistorted_full.jpg")
    undistort.add_argument(
        "--metrics-output",
        type=Path,
        default=DATA_DIR / "undistortion_quality.yaml",
    )
    undistort.add_argument(
        "--pattern",
        nargs=2,
        type=int,
        default=DEFAULT_PATTERN,
        metavar=("COLS", "ROWS"),
    )
    undistort.set_defaults(func=cmd_undistort_image)

    pose_image = subparsers.add_parser(
        "pose-image",
        help="globally undistort an image and estimate H-oriented chessboard pose",
    )
    add_log_arg(pose_image)
    pose_image.add_argument("--input", type=Path, required=True)
    pose_image.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    pose_image.add_argument(
        "--pattern",
        nargs=2,
        type=int,
        default=DEFAULT_PATTERN,
        metavar=("COLS", "ROWS"),
    )
    pose_image.add_argument("--square-size", type=float, required=True)
    pose_image.add_argument("--axis-length", type=float)
    pose_image.add_argument("--output", type=Path, default=DATA_DIR / "pose_latest.yaml")
    pose_image.add_argument("--image-output", type=Path, default=DATA_DIR / "pose_latest.jpg")
    pose_image.add_argument(
        "--undistorted-output",
        type=Path,
        default=DATA_DIR / "undistorted_full.jpg",
    )
    pose_image.set_defaults(func=cmd_pose_image)

    residual_impact = subparsers.add_parser(
        "residual-impact",
        help="verify how residual undistortion error affects chessboard pose",
    )
    residual_impact.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    residual_impact.add_argument(
        "--image-dir",
        action="append",
        type=Path,
        default=[DATA_DIR / "video_pose_experiment" / "frames"],
        help="extra validation image directory; can be passed multiple times",
    )
    residual_impact.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "residual_pose_impact_experiment",
    )
    residual_impact.add_argument(
        "--report",
        type=Path,
        default=Path("docs") / "residual_pose_impact_report.md",
    )
    residual_impact.add_argument(
        "--representative-quality",
        type=Path,
        default=DATA_DIR / "undistortion_quality.yaml",
    )

    def cmd_residual_impact(args: argparse.Namespace) -> int:
        summary = run_residual_pose_impact_experiment(
            args.calibration,
            args.image_dir,
            args.output_dir,
            args.report,
            args.representative_quality,
        )
        print(f"wrote {args.output_dir / 'residual_pose_impact.yaml'}")
        print(f"wrote {args.report}")
        print(
            "representative mean residual pose impact: "
            f"{summary['sensitivity']['representative_mean_outward']['translation_delta_norm_mm']['max']:.4f} mm max"
        )
        return 0

    residual_impact.set_defaults(func=cmd_residual_impact)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
