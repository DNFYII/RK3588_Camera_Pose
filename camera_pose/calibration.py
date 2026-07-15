from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np


PINHOLE_RADIAL2 = "pinhole_radial2"
PINHOLE_RADIAL3 = "pinhole_radial3"
PINHOLE_TANGENT3 = "pinhole_tangent3"
FISHEYE = "fisheye"


@dataclass(frozen=True)
class CalibrationParameters:
    model: str
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    new_camera_matrix: np.ndarray
    undistort_alpha: float
    valid_roi: tuple[int, int, int, int]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "CalibrationParameters":
        image_size = tuple(int(value) for value in data["image_size"])
        undistortion = data.get("undistortion", {})
        new_matrix = data.get("new_camera_matrix", data["camera_matrix"])
        valid_roi = undistortion.get(
            "valid_roi_xywh",
            [0, 0, image_size[0], image_size[1]],
        )
        return cls(
            model=str(data.get("model", PINHOLE_RADIAL3)),
            image_size=(image_size[0], image_size[1]),
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1),
            new_camera_matrix=np.asarray(new_matrix, dtype=np.float64),
            undistort_alpha=float(undistortion.get("alpha", 0.0)),
            valid_roi=tuple(int(value) for value in valid_roi),
        )

    @property
    def corrected_dist_coeffs(self) -> np.ndarray:
        return np.zeros((5, 1), dtype=np.float64)


def _as_calibration_points(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    objects = [np.asarray(points, dtype=np.float32).reshape(-1, 3) for points in object_points]
    images = [np.asarray(points, dtype=np.float32).reshape(-1, 1, 2) for points in image_points]
    return objects, images


def _as_fisheye_points(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    objects = [np.asarray(points, dtype=np.float64).reshape(-1, 1, 3) for points in object_points]
    images = [np.asarray(points, dtype=np.float64).reshape(-1, 1, 2) for points in image_points]
    return objects, images


def fit_fisheye(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    initial_camera_matrix: np.ndarray | None = None,
    initial_dist_coeffs: np.ndarray | None = None,
) -> dict[str, Any]:
    objects, images = _as_fisheye_points(object_points, image_points)
    width, height = image_size
    camera_matrix = (
        np.asarray(initial_camera_matrix, dtype=np.float64).copy()
        if initial_camera_matrix is not None
        else np.array(
            [
                [float(max(width, height)), 0.0, width / 2.0],
                [0.0, float(max(width, height)), height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    dist_coeffs = (
        np.asarray(initial_dist_coeffs, dtype=np.float64).reshape(4, 1).copy()
        if initial_dist_coeffs is not None
        else np.zeros((4, 1), dtype=np.float64)
    )
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_FIX_SKEW
        | cv2.fisheye.CALIB_CHECK_COND
    )
    if initial_camera_matrix is not None:
        flags |= cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
        objects,
        images,
        image_size,
        camera_matrix,
        dist_coeffs,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8),
    )
    return {
        "model": FISHEYE,
        "rms_reprojection_error": float(rms),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs.reshape(-1),
        "rvecs": rvecs,
        "tvecs": tvecs,
    }


def fit_pinhole(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    model: str,
    initial_camera_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    if model not in {PINHOLE_RADIAL2, PINHOLE_RADIAL3, PINHOLE_TANGENT3}:
        raise ValueError(f"unsupported pinhole model: {model}")

    objects, images = _as_calibration_points(object_points, image_points)
    if initial_camera_matrix is None:
        initial_camera_matrix = fit_fisheye(object_points, image_points, image_size)["camera_matrix"]
    camera_matrix = np.asarray(initial_camera_matrix, dtype=np.float64).copy()
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if model == PINHOLE_RADIAL2:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    elif model == PINHOLE_RADIAL3:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objects,
        images,
        image_size,
        camera_matrix,
        dist_coeffs,
        flags=flags,
    )
    return {
        "model": model,
        "rms_reprojection_error": float(rms),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs.reshape(-1),
        "rvecs": rvecs,
        "tvecs": tvecs,
    }


def make_new_camera_matrix(
    model: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    alpha: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    if model == FISHEYE:
        new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix,
            dist_coeffs,
            image_size,
            np.eye(3),
            balance=float(alpha),
            new_size=image_size,
        )
        return new_matrix, (0, 0, image_size[0], image_size[1])
    new_matrix, valid_roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        float(alpha),
        image_size,
    )
    width, height = image_size
    source_mask = np.ones((height, width), dtype=np.uint8)

    def invalid_ratio(focal_length: float) -> float:
        candidate = new_matrix.copy()
        candidate[0, 0] = focal_length
        candidate[1, 1] = focal_length
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            candidate,
            image_size,
            cv2.CV_32FC1,
        )
        valid = cv2.remap(
            source_mask,
            map1,
            map2,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return float(np.mean(valid == 0))

    lower = float(min(new_matrix[0, 0], new_matrix[1, 1]))
    upper = float(max(new_matrix[0, 0], new_matrix[1, 1]))
    target_invalid_ratio = 0.0
    while invalid_ratio(upper) > target_invalid_ratio:
        upper *= 1.05
    for _ in range(18):
        middle = (lower + upper) * 0.5
        if invalid_ratio(middle) <= target_invalid_ratio:
            upper = middle
        else:
            lower = middle
    new_matrix[0, 0] = upper
    new_matrix[1, 1] = upper
    return new_matrix, tuple(int(value) for value in valid_roi)


def build_undistort_maps(
    calibration: CalibrationParameters,
) -> tuple[np.ndarray, np.ndarray]:
    if calibration.model == FISHEYE:
        return cv2.fisheye.initUndistortRectifyMap(
            calibration.camera_matrix,
            calibration.dist_coeffs,
            np.eye(3),
            calibration.new_camera_matrix,
            calibration.image_size,
            cv2.CV_32FC1,
        )
    return cv2.initUndistortRectifyMap(
        calibration.camera_matrix,
        calibration.dist_coeffs,
        None,
        calibration.new_camera_matrix,
        calibration.image_size,
        cv2.CV_32FC1,
    )


def undistort_image(
    image: np.ndarray,
    calibration: CalibrationParameters,
    maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if (image.shape[1], image.shape[0]) != calibration.image_size:
        raise ValueError(
            f"image size {(image.shape[1], image.shape[0])} does not match "
            f"calibration size {calibration.image_size}"
        )
    map1, map2 = maps if maps is not None else build_undistort_maps(calibration)
    corrected = cv2.remap(
        image,
        map1,
        map2,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    source_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
    valid_mask = cv2.remap(
        source_mask,
        map1,
        map2,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return corrected, valid_mask


def undistort_points(
    points: np.ndarray,
    calibration: CalibrationParameters,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if calibration.model == FISHEYE:
        corrected = cv2.fisheye.undistortPoints(
            points,
            calibration.camera_matrix,
            calibration.dist_coeffs,
            P=calibration.new_camera_matrix,
        )
    else:
        corrected = cv2.undistortPoints(
            points,
            calibration.camera_matrix,
            calibration.dist_coeffs,
            P=calibration.new_camera_matrix,
        )
    return corrected.astype(np.float32)


def _raw_reprojection_errors(
    model: str,
    fit: dict[str, Any],
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
) -> np.ndarray:
    camera_matrix = np.asarray(fit["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(fit["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    errors: list[np.ndarray] = []
    for objects, images in zip(object_points, image_points):
        objects64 = np.asarray(objects, dtype=np.float64).reshape(-1, 3)
        images64 = np.asarray(images, dtype=np.float64).reshape(-1, 1, 2)
        if model == FISHEYE:
            normalized = cv2.fisheye.undistortPoints(images64, camera_matrix, dist_coeffs)
            ok, rvec, tvec = cv2.solvePnP(objects64, normalized, np.eye(3), None)
            if not ok:
                continue
            projected, _ = cv2.fisheye.projectPoints(
                objects64.reshape(-1, 1, 3),
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(
                objects64,
                images64,
                camera_matrix,
                dist_coeffs,
            )
            if not ok:
                continue
            projected, _ = cv2.projectPoints(
                objects64,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
        errors.append(
            np.linalg.norm(
                projected.reshape(-1, 2) - images64.reshape(-1, 2),
                axis=1,
            )
        )
    return np.concatenate(errors) if errors else np.array([], dtype=np.float64)


def _line_distances(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered)
    return np.abs(centered @ axes[1])


def chessboard_line_metrics(
    image_points: list[np.ndarray],
    pattern: tuple[int, int],
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float]:
    cols, rows = pattern
    distances: list[np.ndarray] = []
    normalized: list[np.ndarray] = []
    for points in image_points:
        corrected = transform(points) if transform is not None else np.asarray(points)
        corrected = np.asarray(corrected, dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(corrected).all():
            continue
        grid = corrected.reshape(rows, cols, 2)
        spacing = np.median(
            np.concatenate(
                [
                    np.linalg.norm(np.diff(grid, axis=1), axis=2).reshape(-1),
                    np.linalg.norm(np.diff(grid, axis=0), axis=2).reshape(-1),
                ]
            )
        )
        view_distances = np.concatenate(
            [_line_distances(grid[row]) for row in range(rows)]
            + [_line_distances(grid[:, col]) for col in range(cols)]
        )
        distances.append(view_distances)
        if np.isfinite(spacing) and spacing > 1e-9:
            normalized.append(view_distances / spacing)
        else:
            normalized.append(np.full(view_distances.shape, np.inf, dtype=np.float64))
    if not distances:
        return {
            "mean_px": float("inf"),
            "p95_px": float("inf"),
            "max_px": float("inf"),
            "mean_square_fraction": float("inf"),
            "p95_square_fraction": float("inf"),
        }
    all_distances = np.concatenate(distances)
    all_normalized = np.concatenate(normalized)
    finite_normalized = all_normalized[np.isfinite(all_normalized)]
    if finite_normalized.size == 0:
        finite_normalized = np.array([float("inf")], dtype=np.float64)
    return {
        "mean_px": float(all_distances.mean()),
        "p95_px": float(np.percentile(all_distances, 95)),
        "max_px": float(all_distances.max()),
        "mean_square_fraction": float(finite_normalized.mean()),
        "p95_square_fraction": float(np.percentile(finite_normalized, 95)),
    }


def mapping_metrics(calibration: CalibrationParameters) -> dict[str, float]:
    map1, map2 = build_undistort_maps(calibration)
    mask = np.ones((calibration.image_size[1], calibration.image_size[0]), dtype=np.uint8)
    valid = cv2.remap(mask, map1, map2, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    step = 3
    ys, xs = np.mgrid[
        0 : calibration.image_size[1] : step,
        0 : calibration.image_size[0] : step,
    ]
    source_points = np.column_stack([xs.reshape(-1), ys.reshape(-1)]).astype(np.float64)
    corrected = undistort_points(source_points, calibration).reshape(-1, 2)
    finite = np.isfinite(corrected).all(axis=1)
    inside = (
        finite
        & (corrected[:, 0] >= 0)
        & (corrected[:, 0] < calibration.image_size[0])
        & (corrected[:, 1] >= 0)
        & (corrected[:, 1] < calibration.image_size[1])
    )
    return {
        "invalid_output_ratio": float(np.mean(valid == 0)),
        "source_sample_coverage_ratio": float(np.mean(inside)),
        "finite_source_mapping_ratio": float(np.mean(finite)),
    }


def evaluate_models(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    pattern: tuple[int, int],
    folds: int = 3,
    alpha: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    models = [PINHOLE_RADIAL2, PINHOLE_RADIAL3, PINHOLE_TANGENT3, FISHEYE]
    full_fisheye = fit_fisheye(object_points, image_points, image_size)
    initial_camera_matrix = np.asarray(full_fisheye["camera_matrix"], dtype=np.float64)
    full_fits: dict[str, dict[str, Any]] = {FISHEYE: full_fisheye}
    for model in models[:-1]:
        full_fits[model] = fit_pinhole(
            object_points,
            image_points,
            image_size,
            model,
            initial_camera_matrix,
        )

    evaluation: dict[str, Any] = {
        "selection_policy": {
            "cross_validation_folds": folds,
            "maximum_cv_gap_from_best_px": 0.005,
            "minimum_source_coverage_ratio": 0.895,
            "maximum_invalid_output_ratio": 0.0,
            "tie_breaker": "在验证误差相当且全局映射合格时，选择参数更少的模型",
        },
        "models": {},
    }
    for model in models:
        fold_rows = []
        for fold in range(folds):
            train_indices = [index for index in range(len(object_points)) if index % folds != fold]
            test_indices = [index for index in range(len(object_points)) if index % folds == fold]
            train_objects = [object_points[index] for index in train_indices]
            train_images = [image_points[index] for index in train_indices]
            test_objects = [object_points[index] for index in test_indices]
            test_images = [image_points[index] for index in test_indices]
            if model == FISHEYE:
                fold_fit = fit_fisheye(
                    train_objects,
                    train_images,
                    image_size,
                    np.asarray(full_fits[model]["camera_matrix"]),
                    np.asarray(full_fits[model]["dist_coeffs"]),
                )
            else:
                fold_fit = fit_pinhole(
                    train_objects,
                    train_images,
                    image_size,
                    model,
                    np.asarray(full_fits[model]["camera_matrix"]),
                )
            errors = _raw_reprojection_errors(
                model,
                fold_fit,
                test_objects,
                test_images,
            )
            fold_rows.append(
                {
                    "fold": fold,
                    "train_rms_px": fold_fit["rms_reprojection_error"],
                    "test_mean_px": float(errors.mean()),
                    "test_p95_px": float(np.percentile(errors, 95)),
                    "test_max_px": float(errors.max()),
                }
            )

        fit = full_fits[model]
        new_matrix, valid_roi = make_new_camera_matrix(
            model,
            np.asarray(fit["camera_matrix"]),
            np.asarray(fit["dist_coeffs"]),
            image_size,
            alpha,
        )
        calibration = CalibrationParameters(
            model=model,
            image_size=image_size,
            camera_matrix=np.asarray(fit["camera_matrix"]),
            dist_coeffs=np.asarray(fit["dist_coeffs"]).reshape(-1, 1),
            new_camera_matrix=new_matrix,
            undistort_alpha=alpha,
            valid_roi=valid_roi,
        )
        mapping = mapping_metrics(calibration)
        line_metrics = chessboard_line_metrics(
            image_points,
            pattern,
            transform=lambda points, current=calibration: undistort_points(points, current),
        )
        evaluation["models"][model] = {
            "parameter_count": {
                PINHOLE_RADIAL2: 2,
                PINHOLE_RADIAL3: 3,
                PINHOLE_TANGENT3: 5,
                FISHEYE: 4,
            }[model],
            "full_fit_rms_px": fit["rms_reprojection_error"],
            "camera_matrix": fit["camera_matrix"],
            "dist_coeffs": fit["dist_coeffs"],
            "new_camera_matrix": new_matrix,
            "valid_roi_xywh": list(valid_roi),
            "cross_validation": {
                "folds": fold_rows,
                "mean_px": float(np.mean([row["test_mean_px"] for row in fold_rows])),
                "p95_px": float(np.mean([row["test_p95_px"] for row in fold_rows])),
                "max_px": float(np.mean([row["test_max_px"] for row in fold_rows])),
            },
            "mapping": mapping,
            "corrected_chessboard_line_residual": line_metrics,
        }

    best_cv = min(
        row["cross_validation"]["mean_px"] for row in evaluation["models"].values()
    )
    stable_models = []
    eligible = []
    for model in models:
        row = evaluation["models"][model]
        mapping = row["mapping"]
        mapping_is_stable = (
            mapping["source_sample_coverage_ratio"] >= 0.895
            and mapping["invalid_output_ratio"] <= 0.0
            and mapping["finite_source_mapping_ratio"] >= 0.999
        )
        if mapping_is_stable:
            stable_models.append(model)
        if (
            mapping_is_stable
            and row["cross_validation"]["mean_px"] <= best_cv + 0.005
        ):
            eligible.append(model)
    if not eligible:
        if stable_models:
            eligible = [
                min(
                    stable_models,
                    key=lambda name: evaluation["models"][name]["cross_validation"]["mean_px"],
                )
            ]
            selection_reason = "验证误差窗口内无全图合格模型，选择全图合格模型中验证误差最低者"
        else:
            eligible = [
                min(models, key=lambda name: evaluation["models"][name]["cross_validation"]["mean_px"])
            ]
            selection_reason = "没有模型通过全图门槛，保留验证误差最低者并标记风险"
    else:
        selection_reason = "验证误差窗口与全图门槛均通过，按参数数量择优"
    selected_model = min(
        eligible,
        key=lambda name: (
            evaluation["models"][name]["parameter_count"],
            evaluation["models"][name]["cross_validation"]["mean_px"],
        ),
    )
    evaluation["selected_model"] = selected_model
    evaluation["eligible_models"] = eligible
    evaluation["stable_models"] = stable_models
    evaluation["selection_reason"] = selection_reason
    evaluation["raw_chessboard_line_residual"] = chessboard_line_metrics(
        image_points,
        pattern,
    )
    return evaluation, full_fits[selected_model]
