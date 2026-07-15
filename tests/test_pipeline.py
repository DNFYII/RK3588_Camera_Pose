from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from camera_pose.calibration import (
    CalibrationParameters,
    undistort_image,
    undistort_points,
)
from camera_pose.chessboard import (
    find_corners_precise,
    make_object_points,
    orient_corners_with_h,
    solve_planar_pose,
)
from camera_pose.io import read_yaml


ROOT = Path(__file__).resolve().parents[1]


class GlobalUndistortionPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calibration = CalibrationParameters.from_mapping(
            read_yaml(ROOT / "data/calibration.yaml")
        )

    def read_image(self, name: str) -> np.ndarray:
        image = cv2.imread(str(ROOT / "data/calibration_images" / name))
        self.assertIsNotNone(image)
        return image

    def test_global_map_has_expected_shape_and_no_internal_holes(self) -> None:
        corrected, valid = undistort_image(
            self.read_image("calib_015.jpg"),
            self.calibration,
        )
        self.assertEqual(corrected.shape, (960, 1280, 3))
        self.assertEqual(float(np.mean(valid == 0)), 0.0)

    def test_h_shape_resolves_both_corner_orderings(self) -> None:
        image = self.read_image("calib_015.jpg")
        found, corners, _ = find_corners_precise(image, (10, 7))
        self.assertTrue(found)
        self.assertIsNotNone(corners)
        cases = [
            ("detector_order", corners, False),
            ("manual_reversed_order", corners.reshape(-1, 2)[::-1].reshape(-1, 1, 2), True),
        ]
        for name, input_corners, expected_reversed in cases:
            with self.subTest(order=name):
                _, status = orient_corners_with_h(image, input_corners, (10, 7))
                self.assertTrue(status["detected"])
                self.assertEqual(status["corners_reversed"], expected_reversed)
                self.assertGreater(status["confidence_margin"], 0.15)

    def test_pose_uses_new_intrinsics_with_subpixel_error(self) -> None:
        image = self.read_image("calib_015.jpg")
        found, corners, _ = find_corners_precise(image, (10, 7))
        self.assertTrue(found)
        self.assertIsNotNone(corners)
        oriented, status = orient_corners_with_h(image, corners, (10, 7))
        self.assertTrue(status["detected"])
        corrected = undistort_points(oriented, self.calibration)
        ok, _, _, metrics = solve_planar_pose(
            make_object_points((10, 7), 24.0),
            corrected,
            self.calibration.new_camera_matrix,
            self.calibration.corrected_dist_coeffs,
        )
        self.assertTrue(ok)
        self.assertLess(metrics["mean_reprojection_error_px"], 0.2)
        self.assertLess(metrics["max_reprojection_error_px"], 0.5)


if __name__ == "__main__":
    unittest.main()
