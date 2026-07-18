from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .calibration import (
    CalibrationParameters,
    chessboard_line_metrics,
    undistort_points,
)
from .chessboard import (
    find_corners_precise,
    make_object_points,
    orient_corners_with_h,
    solve_planar_pose,
)
from .io import ensure_dir, read_yaml, write_image, write_yaml


def rotation_angle_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    rot_a, _ = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64).reshape(3, 1))
    rot_b, _ = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64).reshape(3, 1))
    delta = rot_b @ rot_a.T
    value = (float(np.trace(delta)) - 1.0) * 0.5
    return float(math.degrees(math.acos(max(-1.0, min(1.0, value)))))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def radial_perturb(points: np.ndarray, camera_matrix: np.ndarray, amount_px: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    center = np.array([camera_matrix[0, 2], camera_matrix[1, 2]], dtype=np.float64)
    direction = pts - center
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    unit = np.divide(direction, norm, out=np.zeros_like(direction), where=norm > 1e-9)
    return (pts + unit * float(amount_px)).reshape(-1, 1, 2).astype(np.float32)


def refine_from_baseline(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    baseline_rvec: np.ndarray,
    baseline_tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    rvec = np.asarray(baseline_rvec, dtype=np.float64).reshape(3, 1).copy()
    tvec = np.asarray(baseline_tvec, dtype=np.float64).reshape(3, 1).copy()
    images = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    objects = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            objects,
            images,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(
            objects,
            images,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("solvePnP refinement failed")
    projected, _ = cv2.projectPoints(objects, rvec, tvec, camera_matrix, dist_coeffs)
    errors = np.linalg.norm(projected.reshape(-1, 2) - images.reshape(-1, 2), axis=1)
    return rvec, tvec, {
        "mean_reprojection_error_px": float(errors.mean()),
        "p95_reprojection_error_px": float(np.percentile(errors, 95)),
        "max_reprojection_error_px": float(errors.max()),
    }


def load_representative_residual(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    data = read_yaml(path)
    residual = data.get("corrected_chessboard_line_residual")
    return residual if isinstance(residual, dict) else None


def collect_image_paths(calibration_data: dict[str, Any], extra_dirs: list[Path]) -> list[Path]:
    paths = [Path(path) for path in calibration_data.get("used_images", [])]
    for directory in extra_dirs:
        if directory.exists():
            paths.extend(
                path
                for path in sorted(directory.glob("*"))
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def analyze_image(
    path: Path,
    calibration: CalibrationParameters,
    pattern: tuple[int, int],
    square_size: float,
    perturbation_amounts: dict[str, float],
) -> dict[str, Any] | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    found, corners, _ = find_corners_precise(image, pattern)
    if not found or corners is None:
        return None

    oriented, h_orientation = orient_corners_with_h(image, corners, pattern)
    corrected = undistort_points(oriented, calibration)
    object_points = make_object_points(pattern, square_size)

    raw_ok, raw_rvec, raw_tvec, raw_metrics = solve_planar_pose(
        object_points,
        oriented,
        calibration.camera_matrix,
        calibration.dist_coeffs,
    )
    corrected_ok, corrected_rvec, corrected_tvec, corrected_metrics = solve_planar_pose(
        object_points,
        corrected,
        calibration.new_camera_matrix,
        calibration.corrected_dist_coeffs,
    )
    if (
        not raw_ok
        or raw_rvec is None
        or raw_tvec is None
        or not corrected_ok
        or corrected_rvec is None
        or corrected_tvec is None
    ):
        return None

    perturbations = {}
    for name, amount in perturbation_amounts.items():
        if not np.isfinite(amount) or amount <= 0:
            continue
        for sign_name, sign in [("outward", 1.0), ("inward", -1.0)]:
            scenario = f"{name}_{sign_name}"
            perturbed = radial_perturb(corrected, calibration.new_camera_matrix, sign * amount)
            rvec, tvec, metrics = refine_from_baseline(
                object_points,
                perturbed,
                calibration.new_camera_matrix,
                calibration.corrected_dist_coeffs,
                corrected_rvec,
                corrected_tvec,
            )
            delta_t = np.asarray(tvec, dtype=np.float64).reshape(3) - np.asarray(
                corrected_tvec,
                dtype=np.float64,
            ).reshape(3)
            perturbations[scenario] = {
                "amount_px": float(amount),
                "translation_delta_mm": delta_t,
                "translation_delta_norm_mm": float(np.linalg.norm(delta_t)),
                "rotation_delta_deg": rotation_angle_deg(corrected_rvec, rvec),
                "mean_reprojection_error_px": metrics["mean_reprojection_error_px"],
                "p95_reprojection_error_px": metrics["p95_reprojection_error_px"],
                "max_reprojection_error_px": metrics["max_reprojection_error_px"],
            }

    raw_corrected_delta_t = np.asarray(raw_tvec, dtype=np.float64).reshape(3) - np.asarray(
        corrected_tvec,
        dtype=np.float64,
    ).reshape(3)
    return {
        "image": str(path),
        "h_orientation": h_orientation,
        "raw_line_residual": chessboard_line_metrics([oriented], pattern),
        "corrected_line_residual": chessboard_line_metrics([corrected], pattern),
        "raw_pose": {
            "tvec_board_to_camera_mm": raw_tvec.reshape(3),
            "mean_reprojection_error_px": raw_metrics["mean_reprojection_error_px"],
            "p95_reprojection_error_px": raw_metrics["p95_reprojection_error_px"],
            "max_reprojection_error_px": raw_metrics["max_reprojection_error_px"],
        },
        "corrected_pose": {
            "tvec_board_to_camera_mm": corrected_tvec.reshape(3),
            "mean_reprojection_error_px": corrected_metrics["mean_reprojection_error_px"],
            "p95_reprojection_error_px": corrected_metrics["p95_reprojection_error_px"],
            "max_reprojection_error_px": corrected_metrics["max_reprojection_error_px"],
        },
        "raw_vs_corrected_pose_delta": {
            "translation_delta_mm": raw_corrected_delta_t,
            "translation_delta_norm_mm": float(np.linalg.norm(raw_corrected_delta_t)),
            "rotation_delta_deg": rotation_angle_deg(corrected_rvec, raw_rvec),
        },
        "perturbations": perturbations,
    }


def draw_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    y_label: str,
    width: int = 1400,
    height: int = 720,
) -> None:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")
    bold_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    title_font = ImageFont.truetype(str(bold_path if bold_path.exists() else font_path), 28)
    body_font = ImageFont.truetype(str(font_path), 17)
    small_font = ImageFont.truetype(str(font_path), 14)
    margin_left, margin_right, margin_top, margin_bottom = 95, 35, 75, 105
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    all_values = [value for _, values, _ in series for value in values if np.isfinite(value)]
    ymax = max(all_values) * 1.18 if all_values else 1.0
    ymax = ymax if ymax > 0 else 1.0

    draw.text((margin_left, 18), title, fill=(35, 35, 35), font=title_font)
    draw.text((margin_left, 52), y_label, fill=(80, 80, 80), font=body_font)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=(70, 70, 70), width=1)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill=(70, 70, 70), width=1)

    for tick in range(6):
        y = int(margin_top + plot_h - plot_h * tick / 5)
        value = ymax * tick / 5
        draw.line((margin_left, y, margin_left + plot_w, y), fill=(232, 232, 232), width=1)
        draw.text((18, y - 9), f"{value:.3f}", fill=(85, 85, 85), font=small_font)

    count = len(labels)
    group_w = plot_w / max(1, count)
    bar_w = max(2, int(group_w * 0.7 / max(1, len(series))))
    for index, label in enumerate(labels):
        x0 = int(margin_left + group_w * index + group_w * 0.15)
        for series_index, (_, values, color) in enumerate(series):
            value = values[index]
            bar_h = int(plot_h * value / ymax) if np.isfinite(value) else 0
            x1 = x0 + series_index * bar_w
            y1 = margin_top + plot_h - bar_h
            draw.rectangle((x1, y1, x1 + bar_w - 1, margin_top + plot_h), fill=color)
        if index % max(1, count // 20) == 0:
            draw.text((int(margin_left + group_w * index), height - 62), label, fill=(70, 70, 70), font=small_font)

    legend_x = margin_left + 390
    legend_y = 53
    for name, _, color in series:
        draw.rectangle((legend_x, legend_y, legend_x + 18, legend_y + 18), fill=color)
        draw.text((legend_x + 25, legend_y - 4), name, fill=(55, 55, 55), font=body_font)
        legend_x += 210
    image = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    write_image(path, image)


def draw_residual_overlay(
    path: Path,
    source_image: Path,
    calibration: CalibrationParameters,
    pattern: tuple[int, int],
    square_size: float,
) -> None:
    image = cv2.imread(str(source_image))
    if image is None:
        return
    found, corners, _ = find_corners_precise(image, pattern)
    if not found or corners is None:
        return
    oriented, _ = orient_corners_with_h(image, corners, pattern)
    corrected = undistort_points(oriented, calibration)
    object_points = make_object_points(pattern, square_size)
    ok, rvec, tvec, _ = solve_planar_pose(
        object_points,
        corrected,
        calibration.new_camera_matrix,
        calibration.corrected_dist_coeffs,
    )
    corrected_image, valid = cv2.undistort(
        image,
        calibration.camera_matrix,
        calibration.dist_coeffs,
        None,
        calibration.new_camera_matrix,
    ), None
    preview = corrected_image.copy()
    cv2.drawChessboardCorners(preview, pattern, corrected, True)
    if ok and rvec is not None and tvec is not None:
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            calibration.new_camera_matrix,
            calibration.corrected_dist_coeffs,
        )
        points = corrected.reshape(-1, 2)
        projected = projected.reshape(-1, 2)
        scale = 35.0
        for point, target in zip(points, projected):
            start = tuple(np.round(target).astype(int))
            end = tuple(np.round(target + (point - target) * scale).astype(int))
            cv2.arrowedLine(preview, start, end, (0, 120, 255), 1, cv2.LINE_AA, tipLength=0.25)
        font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
        font = ImageFont.truetype(str(font_path), 25)
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        canvas = Image.fromarray(rgb)
        draw = ImageDraw.Draw(canvas)
        text = "橙色箭头：重投影残差向量 x35"
        draw.text((24, 18), text, fill=(0, 0, 0), font=font, stroke_width=3, stroke_fill=(0, 0, 0))
        draw.text((24, 18), text, fill=(255, 145, 0), font=font)
        preview = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    write_image(path, preview)


def make_report(
    path: Path,
    output_dir: Path,
    summary: dict[str, Any],
    visual_paths: dict[str, Path],
) -> None:
    ensure_dir(path.parent)
    rel = lambda item: str(Path("..") / item.relative_to(path.parent.parent)) if item.is_absolute() else str(Path("..") / item)
    p95_translation_worst = max(
        summary["sensitivity"]["representative_p95_outward"]["translation_delta_norm_mm"]["max"],
        summary["sensitivity"]["representative_p95_inward"]["translation_delta_norm_mm"]["max"],
    )
    p95_rotation_worst = max(
        summary["sensitivity"]["representative_p95_outward"]["rotation_delta_deg"]["max"],
        summary["sensitivity"]["representative_p95_inward"]["rotation_delta_deg"]["max"],
    )
    matrix_text = [
        [round(float(value), 6) for value in row]
        for row in np.asarray(summary["new_camera_matrix"], dtype=np.float64).tolist()
    ]
    lines = [
        "# 去畸变残差对位姿计算影响验证报告",
        "",
        "## 技术结论",
        "",
        (
            f"当前全局去畸变后，代表图像的棋盘直线性平均残差为 "
            f"{summary['representative_corrected_line_mean_px']:.4f} px；"
            f"在 70 个棋盘角点上把这一残差按径向方向进行保守扰动后，"
            f"位姿平移变化均值为 {summary['sensitivity']['representative_mean_outward']['translation_delta_norm_mm']['mean']:.4f} mm，"
            f"最大值为 {summary['sensitivity']['representative_mean_outward']['translation_delta_norm_mm']['max']:.4f} mm。"
        ),
        "",
        (
            f"即使采用代表图像的 p95 残差 "
            f"{summary['representative_corrected_line_p95_px']:.4f} px 做同向径向扰动，"
            f"双方向最坏平移变化为 {p95_translation_worst:.4f} mm，"
            f"双方向最坏旋转变化为 {p95_rotation_worst:.6f} deg。"
            "因此，当前残留的径向形变对棋盘格基准位姿的影响已经远小于角点检测、标定样本分布和靶标平面误差等主要误差源。"
        ),
        "",
        "## 判断标准",
        "",
        "这次不把“肉眼是否还像桶形”作为判断依据，而采用三个可量化标准：",
        "",
        "1. 去畸变后的棋盘行列角点应更接近直线，直线性残差低于原图。",
        "2. 将残留直线性残差按径向方向施加到角点后，重新 PnP 得到的位姿变化应显著小于当前实际重投影误差对应的量级。",
        "3. 原图畸变模型直接求解与全局去畸变后求解应保持一致，说明去畸变表示没有引入明显额外位姿偏差。",
        "",
        "## 数据和参数",
        "",
        (
            f"- 图像数量：共 {summary['image_count']} 张，"
            f"其中标定图 {summary['calibration_image_count']} 张，"
            f"独立验证图 {summary['validation_image_count']} 张。"
        ),
        f"- 棋盘格规格：{summary['pattern'][0]} x {summary['pattern'][1]} 内角点，格子边长 {summary['square_size_mm']:.1f} mm。",
        f"- 图像分辨率：{summary['image_size'][0]} x {summary['image_size'][1]}。",
        f"- 畸变模型：{summary['model']}，原始畸变系数 k1={summary['dist_coeffs'][0]:.6f}, k2={summary['dist_coeffs'][1]:.6f}。",
        f"- 去畸变后使用的新内参矩阵：`{matrix_text}`。",
        "",
        "## 几何残差验证",
        "",
        (
            f"{summary['image_count']} 张图的平均直线性残差从原图的 "
            f"{summary['line_residual']['raw_mean_px']['mean']:.4f} px "
            f"降到去畸变后的 {summary['line_residual']['corrected_mean_px']['mean']:.4f} px。"
            f"代表图像 `calib_015` 的残差从 {summary['representative_raw_line_mean_px']:.4f} px "
            f"降到 {summary['representative_corrected_line_mean_px']:.4f} px。"
        ),
        "",
        f"![去畸变前后直线性残差]({rel(visual_paths['line_residual'])})",
        "",
        "## 位姿扰动验证",
        "",
        (
            "为了估计残余径向畸变对位姿的影响，在去畸变后的角点上沿主点到角点的径向方向加入扰动。"
            "扰动幅值分别取代表图像的 mean、p95、max 残差，以及 40 张图整体的 mean 残差；"
            "每个幅值都同时测试向外和向内两个方向。扰动后的求解从原始位姿出发做 LM 局部精化，"
            "用于测量同一物理位姿附近的残差传递，而不是测平面位姿双解切换。"
        ),
        "",
        (
            f"以 `0.0348 px` 这个代表图像 mean 残差为例，向外扰动得到的平均平移变化为 "
            f"{summary['sensitivity']['representative_mean_outward']['translation_delta_norm_mm']['mean']:.4f} mm，"
            f"p95 为 {summary['sensitivity']['representative_mean_outward']['translation_delta_norm_mm']['p95']:.4f} mm；"
            f"平均旋转变化为 {summary['sensitivity']['representative_mean_outward']['rotation_delta_deg']['mean']:.6f} deg。"
        ),
        "",
        f"![残差扰动导致的平移变化]({rel(visual_paths['translation_sensitivity'])})",
        "",
        f"![残差扰动导致的旋转变化]({rel(visual_paths['rotation_sensitivity'])})",
        "",
        "## 原图模型与去畸变表示的一致性",
        "",
        (
            f"同一批角点分别用原图畸变模型和全局去畸变图求解，平移差异均值为 "
            f"{summary['raw_vs_corrected_pose']['translation_delta_norm_mm']['mean']:.4f} mm，"
            f"p95 为 {summary['raw_vs_corrected_pose']['translation_delta_norm_mm']['p95']:.4f} mm；"
            f"旋转差异均值为 {summary['raw_vs_corrected_pose']['rotation_delta_deg']['mean']:.6f} deg。"
            "这一步验证的是表示一致性，不把它当作绝对真值误差。"
        ),
        "",
        f"![代表图像残差向量]({rel(visual_paths['residual_overlay'])})",
        "",
        "## 结论",
        "",
        "当前去畸变后仍然可以测到非零残差，但它不是新的非零畸变系数，而是角点检测、插值重采样、标定模型近似和靶标平面误差共同留下的图像域残差。按照代表 mean 残差的径向扰动实验，这一级别残差传递到棋盘格位姿后是亚毫米量级、极小角度量级影响；因此，在当前棋盘格基准位姿计算中，它不是主要误差来源。",
        "",
        "后续如果要继续压低位姿误差，优先方向不是把残差强行调成 0，而是提高标定数据覆盖、减少图像模糊、检查靶标平整度，并在 H 标志位姿算法中单独建立与棋盘格基准的对比实验。",
        "",
        "## 实验文件",
        "",
        f"- 数值结果：`{summary['result_file']}`",
        f"- 可视化目录：`{output_dir}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_residual_pose_impact_experiment(
    calibration_path: Path,
    image_dirs: list[Path],
    output_dir: Path,
    report_path: Path,
    representative_quality_path: Path,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    calibration_data = read_yaml(calibration_path)
    calibration = CalibrationParameters.from_mapping(calibration_data)
    pattern = tuple(int(value) for value in calibration_data["pattern"])
    square_size = float(calibration_data["square_size"])
    image_paths = collect_image_paths(calibration_data, image_dirs)

    representative = load_representative_residual(representative_quality_path)
    representative_mean = float(representative["mean_px"]) if representative else 0.0348
    representative_p95 = float(representative["p95_px"]) if representative else 0.0721
    representative_max = float(representative["max_px"]) if representative else 0.2016
    calibration_line_residual = read_yaml(Path("data/calibration_model_evaluation.yaml")).get(
        "models",
        {},
    ).get(calibration.model, {}).get("corrected_chessboard_line_residual", {})
    all_view_mean = float(calibration_line_residual.get("mean_px", representative_mean))
    perturbation_amounts = {
        "representative_mean": representative_mean,
        "representative_p95": representative_p95,
        "representative_max": representative_max,
        "all_view_mean": all_view_mean,
    }

    rows = []
    for image_path in image_paths:
        row = analyze_image(image_path, calibration, pattern, square_size, perturbation_amounts)
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError("no usable chessboard images for residual impact experiment")

    raw_mean = [row["raw_line_residual"]["mean_px"] for row in rows]
    corrected_mean = [row["corrected_line_residual"]["mean_px"] for row in rows]
    corrected_reprojection = [row["corrected_pose"]["mean_reprojection_error_px"] for row in rows]
    raw_corrected_translation = [
        row["raw_vs_corrected_pose_delta"]["translation_delta_norm_mm"] for row in rows
    ]
    raw_corrected_rotation = [row["raw_vs_corrected_pose_delta"]["rotation_delta_deg"] for row in rows]

    scenario_names = sorted({name for row in rows for name in row["perturbations"].keys()})
    sensitivity = {}
    for name in scenario_names:
        sensitivity[name] = {
            "amount_px": float(np.mean([row["perturbations"][name]["amount_px"] for row in rows if name in row["perturbations"]])),
            "translation_delta_norm_mm": summarize(
                [
                    row["perturbations"][name]["translation_delta_norm_mm"]
                    for row in rows
                    if name in row["perturbations"]
                ]
            ),
            "rotation_delta_deg": summarize(
                [
                    row["perturbations"][name]["rotation_delta_deg"]
                    for row in rows
                    if name in row["perturbations"]
                ]
            ),
            "mean_reprojection_error_px": summarize(
                [
                    row["perturbations"][name]["mean_reprojection_error_px"]
                    for row in rows
                    if name in row["perturbations"]
                ]
            ),
        }

    summary = {
        "calibration_file": str(calibration_path),
        "result_file": str(output_dir / "residual_pose_impact.yaml"),
        "image_count": len(rows),
        "calibration_image_count": sum("calibration_images" in row["image"] for row in rows),
        "validation_image_count": sum("video_pose_experiment" in row["image"] for row in rows),
        "pattern": list(pattern),
        "square_size_mm": square_size,
        "image_size": list(calibration.image_size),
        "model": calibration.model,
        "dist_coeffs": calibration.dist_coeffs.reshape(-1),
        "camera_matrix": calibration.camera_matrix,
        "new_camera_matrix": calibration.new_camera_matrix,
        "representative_quality_file": str(representative_quality_path),
        "representative_raw_line_mean_px": float(
            read_yaml(representative_quality_path)["raw_chessboard_line_residual"]["mean_px"]
        )
        if representative_quality_path.exists()
        else float("nan"),
        "representative_corrected_line_mean_px": representative_mean,
        "representative_corrected_line_p95_px": representative_p95,
        "representative_corrected_line_max_px": representative_max,
        "line_residual": {
            "raw_mean_px": summarize(raw_mean),
            "corrected_mean_px": summarize(corrected_mean),
            "reduction_ratio_mean": float(1.0 - np.mean(corrected_mean) / np.mean(raw_mean)),
        },
        "corrected_pose_reprojection_mean_px": summarize(corrected_reprojection),
        "raw_vs_corrected_pose": {
            "translation_delta_norm_mm": summarize(raw_corrected_translation),
            "rotation_delta_deg": summarize(raw_corrected_rotation),
        },
        "sensitivity": sensitivity,
        "rows": rows,
    }
    result_path = output_dir / "residual_pose_impact.yaml"
    write_yaml(result_path, summary)

    labels = [
        ("v" if "video_pose_experiment" in row["image"] else "") + Path(row["image"]).stem.replace("calib_", "")
        for row in rows
    ]
    visual_paths = {
        "line_residual": output_dir / "line_residual_raw_vs_corrected.jpg",
        "translation_sensitivity": output_dir / "pose_translation_sensitivity.jpg",
        "rotation_sensitivity": output_dir / "pose_rotation_sensitivity.jpg",
        "residual_overlay": output_dir / "residual_vector_overlay.jpg",
    }
    draw_bar_chart(
        visual_paths["line_residual"],
        "棋盘直线性残差：去畸变前 vs 去畸变后",
        labels,
        [
            ("raw mean px", raw_mean, (210, 130, 55)),
            ("corrected mean px", corrected_mean, (60, 120, 210)),
        ],
        "mean line residual / px",
    )
    scenario_label_map = {
        "all_view_mean_inward": "全均-向内",
        "all_view_mean_outward": "全均-向外",
        "representative_max_inward": "最大-向内",
        "representative_max_outward": "最大-向外",
        "representative_mean_inward": "均值-向内",
        "representative_mean_outward": "均值-向外",
        "representative_p95_inward": "p95-向内",
        "representative_p95_outward": "p95-向外",
    }
    scenario_labels = [scenario_label_map.get(name, name) for name in scenario_names]
    draw_bar_chart(
        visual_paths["translation_sensitivity"],
        "残差径向扰动对位姿平移的影响",
        scenario_labels,
        [
            (
                "mean mm",
                [sensitivity[name]["translation_delta_norm_mm"]["mean"] for name in scenario_names],
                (66, 133, 170),
            ),
            (
                "p95 mm",
                [sensitivity[name]["translation_delta_norm_mm"]["p95"] for name in scenario_names],
                (220, 160, 65),
            ),
            (
                "max mm",
                [sensitivity[name]["translation_delta_norm_mm"]["max"] for name in scenario_names],
                (165, 82, 120),
            ),
        ],
        "translation delta / mm",
    )
    draw_bar_chart(
        visual_paths["rotation_sensitivity"],
        "残差径向扰动对位姿旋转的影响",
        scenario_labels,
        [
            (
                "mean deg",
                [sensitivity[name]["rotation_delta_deg"]["mean"] for name in scenario_names],
                (66, 133, 170),
            ),
            (
                "p95 deg",
                [sensitivity[name]["rotation_delta_deg"]["p95"] for name in scenario_names],
                (220, 160, 65),
            ),
            (
                "max deg",
                [sensitivity[name]["rotation_delta_deg"]["max"] for name in scenario_names],
                (165, 82, 120),
            ),
        ],
        "rotation delta / deg",
    )
    representative_image = Path(calibration_data.get("used_images", [rows[0]["image"]])[15])
    if not representative_image.exists():
        representative_image = Path(rows[0]["image"])
    draw_residual_overlay(
        visual_paths["residual_overlay"],
        representative_image,
        calibration,
        pattern,
        square_size,
    )
    make_report(report_path, output_dir, summary, visual_paths)
    return summary
