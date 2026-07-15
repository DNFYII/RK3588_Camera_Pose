
## 2026-07-14 15:16:39 - 开始从视频抽取标定帧

- video=data/calibration_source.avi
- output=data/video_pose_experiment_5/frames
- target_frames=5, pattern=10x7
- min_sharpness=500.0, min_corner_shift=80.0px
- fps=30.000, total_frames=838, sample_step=15

## 2026-07-14 15:16:58 - 结束从视频抽取标定帧

- video=data/calibration_source.avi
- candidate_count=41, saved_count=5
- selected_metadata=data/video_pose_experiment_5/extracted_frames.yaml
- saved_files=data/video_pose_experiment_5/frames/calib_000.jpg, data/video_pose_experiment_5/frames/calib_001.jpg, data/video_pose_experiment_5/frames/calib_002.jpg, data/video_pose_experiment_5/frames/calib_003.jpg, data/video_pose_experiment_5/frames/calib_004.jpg

## 2026-07-14 15:17:18 - 完成全图去畸变后的 H 辅助棋盘格位姿估计

- source=data/video_pose_experiment_5/frames/calib_000.jpg, calibration=data/calibration.yaml
- undistorted_image=data/video_pose_experiment_5/undistorted/calib_000_undistorted.jpg
- pose_file=data/video_pose_experiment_5/poses/calib_000.yaml, visualization=data/video_pose_experiment_5/visualizations/calib_000_pose.jpg
- h_orientation={'detected': True, 'corners_reversed': False, 'selected_score': 0.8988288003116706, 'alternative_score': 0.5205639118610201, 'confidence_margin': 0.3782648884506504, 'selected_valid_ratio': 1.0}
- mean_reprojection_error_px=0.219916
- tvec_board_to_camera_mm=(-14.491, -58.471, 1454.014)
- camera_position_board_mm=(90.806, -24.938, -1452.211)

## 2026-07-14 15:17:19 - 完成全图去畸变后的 H 辅助棋盘格位姿估计

- source=data/video_pose_experiment_5/frames/calib_001.jpg, calibration=data/calibration.yaml
- undistorted_image=data/video_pose_experiment_5/undistorted/calib_001_undistorted.jpg
- pose_file=data/video_pose_experiment_5/poses/calib_001.yaml, visualization=data/video_pose_experiment_5/visualizations/calib_001_pose.jpg
- h_orientation={'detected': True, 'corners_reversed': False, 'selected_score': 0.881161691603203, 'alternative_score': 0.5058982136838558, 'confidence_margin': 0.3752634779193472, 'selected_valid_ratio': 1.0}
- mean_reprojection_error_px=0.194980
- tvec_board_to_camera_mm=(62.555, -128.292, 1408.238)
- camera_position_board_mm=(182.490, 157.390, -1394.788)

## 2026-07-14 15:17:20 - 完成全图去畸变后的 H 辅助棋盘格位姿估计

- source=data/video_pose_experiment_5/frames/calib_002.jpg, calibration=data/calibration.yaml
- undistorted_image=data/video_pose_experiment_5/undistorted/calib_002_undistorted.jpg
- pose_file=data/video_pose_experiment_5/poses/calib_002.yaml, visualization=data/video_pose_experiment_5/visualizations/calib_002_pose.jpg
- h_orientation={'detected': True, 'corners_reversed': False, 'selected_score': 0.954595421306696, 'alternative_score': 0.504879706402066, 'confidence_margin': 0.4497157149046299, 'selected_valid_ratio': 1.0}
- mean_reprojection_error_px=0.212480
- tvec_board_to_camera_mm=(149.942, 176.997, 1417.900)
- camera_position_board_mm=(237.500, 163.937, -1407.469)

## 2026-07-14 15:17:21 - 完成全图去畸变后的 H 辅助棋盘格位姿估计

- source=data/video_pose_experiment_5/frames/calib_003.jpg, calibration=data/calibration.yaml
- undistorted_image=data/video_pose_experiment_5/undistorted/calib_003_undistorted.jpg
- pose_file=data/video_pose_experiment_5/poses/calib_003.yaml, visualization=data/video_pose_experiment_5/visualizations/calib_003_pose.jpg
- h_orientation={'detected': True, 'corners_reversed': False, 'selected_score': 0.9443156572974017, 'alternative_score': 0.5148374813001496, 'confidence_margin': 0.42947817599725213, 'selected_valid_ratio': 1.0}
- mean_reprojection_error_px=0.171689
- tvec_board_to_camera_mm=(210.689, 17.167, 1429.357)
- camera_position_board_mm=(146.349, 172.351, -1427.103)

## 2026-07-14 15:17:22 - 完成全图去畸变后的 H 辅助棋盘格位姿估计

- source=data/video_pose_experiment_5/frames/calib_004.jpg, calibration=data/calibration.yaml
- undistorted_image=data/video_pose_experiment_5/undistorted/calib_004_undistorted.jpg
- pose_file=data/video_pose_experiment_5/poses/calib_004.yaml, visualization=data/video_pose_experiment_5/visualizations/calib_004_pose.jpg
- h_orientation={'detected': True, 'corners_reversed': False, 'selected_score': 0.8034463435135892, 'alternative_score': 0.5172518243150214, 'confidence_margin': 0.28619451919856786, 'selected_valid_ratio': 1.0}
- mean_reprojection_error_px=0.267374
- tvec_board_to_camera_mm=(421.208, 73.861, 1317.711)
- camera_position_board_mm=(181.028, 256.822, -1349.261)
