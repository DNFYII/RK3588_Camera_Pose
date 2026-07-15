from __future__ import annotations

import os

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)


def make_object_points(pattern: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern
    points = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] = grid * float(square_size)
    return points


def find_corners(
    image: np.ndarray,
    pattern: tuple[int, int],
) -> tuple[bool, np.ndarray | None, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    ok, corners = cv2.findChessboardCorners(
        gray,
        pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK,
    )
    if not ok:
        if hasattr(cv2, "findChessboardCornersSB"):
            ok, corners = cv2.findChessboardCornersSB(
                gray,
                pattern,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if ok:
                return True, corners.astype(np.float32), gray
        return False, None, gray

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined, gray


def find_corners_precise(
    image: np.ndarray,
    pattern: tuple[int, int],
) -> tuple[bool, np.ndarray | None, np.ndarray]:
    # The classic detector plus cornerSubPix is deterministic on this RK3588
    # build. The SB implementation showed run-to-run subpixel drift here.
    return find_corners(image, pattern)


def orient_corners_with_h(
    image: np.ndarray,
    corners: np.ndarray,
    pattern: tuple[int, int],
) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    cols, rows = pattern
    if pattern != (10, 7):
        return corners, {
            "detected": False,
            "reason": "H orientation template is defined for the 10x7 target only",
        }

    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    scale = 30.0
    min_y = 7.4
    max_y = 15.4
    output_size = (int(9.0 * scale), int((max_y - min_y) * scale))
    board_to_crop = np.array(
        [
            [scale, 0.0, 0.0],
            [0.0, scale, -min_y * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    h_polygon = np.array(
        [
            [3.40, 8.30],
            [7.60, 8.30],
            [7.60, 10.00],
            [5.60, 10.00],
            [5.60, 12.80],
            [7.60, 12.80],
            [7.60, 14.45],
            [1.45, 14.45],
            [1.45, 12.80],
            [3.40, 12.80],
        ],
        dtype=np.float64,
    )
    polygon_pixels = np.column_stack(
        [
            h_polygon[:, 0] * scale,
            (h_polygon[:, 1] - min_y) * scale,
        ]
    ).astype(np.int32)
    template = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    cv2.fillPoly(template, [polygon_pixels], 255)
    score_mask = np.zeros_like(template, dtype=bool)
    score_mask[
        int(0.5 * scale) : int(7.4 * scale),
        int(0.5 * scale) : int(8.5 * scale),
    ] = True

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    image_mask = np.full(gray.shape, 255, dtype=np.uint8)
    candidates = [corners.reshape(-1, 2), corners.reshape(-1, 2)[::-1]]
    scores: list[float] = []
    valid_ratios: list[float] = []
    for candidate in candidates:
        image_to_board, _ = cv2.findHomography(candidate, grid)
        if image_to_board is None:
            scores.append(0.0)
            valid_ratios.append(0.0)
            continue
        transform = board_to_crop @ image_to_board
        warped = cv2.warpPerspective(gray, transform, output_size)
        valid = cv2.warpPerspective(
            image_mask,
            transform,
            output_size,
            flags=cv2.INTER_NEAREST,
        ) > 0
        _, binary = cv2.threshold(
            warped,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        active = score_mask & valid
        intersection = np.sum((binary > 0) & (template > 0) & active)
        denominator = np.sum((binary > 0) & active) + np.sum((template > 0) & active)
        scores.append(float(2.0 * intersection / denominator) if denominator else 0.0)
        valid_ratios.append(float(valid.mean()))

    selected = int(scores[1] > scores[0])
    selected_corners = corners.reshape(-1, 2)
    if selected:
        selected_corners = selected_corners[::-1]
    confidence_margin = abs(scores[0] - scores[1])
    detected = max(scores) >= 0.75 and confidence_margin >= 0.15
    if not detected:
        selected_corners = corners.reshape(-1, 2)
        selected = 0
    return selected_corners.reshape(-1, 1, 2).astype(np.float32), {
        "detected": detected,
        "corners_reversed": bool(selected),
        "selected_score": float(scores[selected]),
        "alternative_score": float(scores[1 - selected]),
        "confidence_margin": float(confidence_margin),
        "selected_valid_ratio": float(valid_ratios[selected]),
    }


def solve_planar_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, dict[str, float]]:
    objects = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    images = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    coefficients = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)

    candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
    if hasattr(cv2, "SOLVEPNP_IPPE"):
        result = cv2.solvePnPGeneric(
            objects,
            images,
            camera_matrix,
            coefficients,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if result[0]:
            for rvec, tvec in zip(result[1], result[2]):
                projected, _ = cv2.projectPoints(
                    objects,
                    rvec,
                    tvec,
                    camera_matrix,
                    coefficients,
                )
                error = float(
                    np.linalg.norm(
                        projected.reshape(-1, 2) - images.reshape(-1, 2),
                        axis=1,
                    ).mean()
                )
                candidates.append((rvec, tvec, error))

    if not candidates:
        ok, rvec, tvec = cv2.solvePnP(
            objects,
            images,
            camera_matrix,
            coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return False, None, None, {}
        candidates.append((rvec, tvec, float("inf")))

    rvec, tvec, _ = min(candidates, key=lambda item: item[2])
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            objects,
            images,
            camera_matrix,
            coefficients,
            rvec,
            tvec,
        )
    projected, _ = cv2.projectPoints(
        objects,
        rvec,
        tvec,
        camera_matrix,
        coefficients,
    )
    errors = np.linalg.norm(
        projected.reshape(-1, 2) - images.reshape(-1, 2),
        axis=1,
    )
    metrics = {
        "mean_reprojection_error_px": float(errors.mean()),
        "p95_reprojection_error_px": float(np.percentile(errors, 95)),
        "max_reprojection_error_px": float(errors.max()),
        "ippe_candidate_count": len(candidates),
    }
    if len(candidates) > 1:
        ordered = sorted(candidate[2] for candidate in candidates)
        metrics["ippe_second_solution_mean_error_px"] = float(ordered[1])
    return True, rvec, tvec, metrics


def draw_axes(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_length: float,
) -> np.ndarray:
    axis = np.float32(
        [
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, -axis_length],
        ]
    )
    projected, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    pts = projected.reshape(-1, 2).astype(int)
    output = image.copy()
    origin = tuple(pts[0])
    cv2.line(output, origin, tuple(pts[1]), (0, 0, 255), 3)
    cv2.line(output, origin, tuple(pts[2]), (0, 255, 0), 3)
    cv2.line(output, origin, tuple(pts[3]), (255, 0, 0), 3)
    return output
