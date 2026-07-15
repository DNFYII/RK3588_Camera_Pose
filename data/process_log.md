
## 2026-07-13 19:12:25 - 开始从视频抽取标定帧

- video=data/calibration_source.avi
- output=data/calibration_images
- target_frames=40, pattern=10x7
- min_sharpness=500.0, min_corner_shift=24.0px
- fps=30.000, total_frames=838, sample_step=8

## 2026-07-13 19:12:52 - 结束从视频抽取标定帧

- video=data/calibration_source.avi
- candidate_count=79, saved_count=40
- selected_metadata=data/extracted_frames.yaml
- saved_files=data/calibration_images/calib_000.jpg, data/calibration_images/calib_001.jpg, data/calibration_images/calib_002.jpg, data/calibration_images/calib_003.jpg, data/calibration_images/calib_004.jpg, data/calibration_images/calib_005.jpg, data/calibration_images/calib_006.jpg, data/calibration_images/calib_007.jpg, data/calibration_images/calib_008.jpg, data/calibration_images/calib_009.jpg, data/calibration_images/calib_010.jpg, data/calibration_images/calib_011.jpg, data/calibration_images/calib_012.jpg, data/calibration_images/calib_013.jpg, data/calibration_images/calib_014.jpg, data/calibration_images/calib_015.jpg, data/calibration_images/calib_016.jpg, data/calibration_images/calib_017.jpg, data/calibration_images/calib_018.jpg, data/calibration_images/calib_019.jpg, data/calibration_images/calib_020.jpg, data/calibration_images/calib_021.jpg, data/calibration_images/calib_022.jpg, data/calibration_images/calib_023.jpg, data/calibration_images/calib_024.jpg, data/calibration_images/calib_025.jpg, data/calibration_images/calib_026.jpg, data/calibration_images/calib_027.jpg, data/calibration_images/calib_028.jpg, data/calibration_images/calib_029.jpg, data/calibration_images/calib_030.jpg, data/calibration_images/calib_031.jpg, data/calibration_images/calib_032.jpg, data/calibration_images/calib_033.jpg, data/calibration_images/calib_034.jpg, data/calibration_images/calib_035.jpg, data/calibration_images/calib_036.jpg, data/calibration_images/calib_037.jpg, data/calibration_images/calib_038.jpg, data/calibration_images/calib_039.jpg

## 2026-07-13 19:13:11 - 完成标定图质量检查

- images=data/calibration_images
- pattern=10x7
- quality_file=data/quality_before_calibration.yaml
- calib_000.jpg: sharpness=1199.31, corners_found=True
- calib_001.jpg: sharpness=1231.85, corners_found=True
- calib_002.jpg: sharpness=741.24, corners_found=True
- calib_003.jpg: sharpness=587.23, corners_found=True
- calib_004.jpg: sharpness=566.00, corners_found=True
- calib_005.jpg: sharpness=1298.98, corners_found=True
- calib_006.jpg: sharpness=597.23, corners_found=True
- calib_007.jpg: sharpness=740.33, corners_found=True
- calib_008.jpg: sharpness=608.09, corners_found=True
- calib_009.jpg: sharpness=557.97, corners_found=True
- calib_010.jpg: sharpness=863.11, corners_found=True
- calib_011.jpg: sharpness=848.80, corners_found=True
- calib_012.jpg: sharpness=1576.50, corners_found=True
- calib_013.jpg: sharpness=1019.77, corners_found=True
- calib_014.jpg: sharpness=1069.76, corners_found=True
- calib_015.jpg: sharpness=1772.88, corners_found=True
- calib_016.jpg: sharpness=647.36, corners_found=True
- calib_017.jpg: sharpness=975.71, corners_found=True
- calib_018.jpg: sharpness=1465.74, corners_found=True
- calib_019.jpg: sharpness=1398.82, corners_found=True
- calib_020.jpg: sharpness=1272.99, corners_found=True
- calib_021.jpg: sharpness=804.93, corners_found=True
- calib_022.jpg: sharpness=1389.90, corners_found=True
- calib_023.jpg: sharpness=783.77, corners_found=True
- calib_024.jpg: sharpness=935.82, corners_found=True
- calib_025.jpg: sharpness=1018.70, corners_found=True
- calib_026.jpg: sharpness=1101.22, corners_found=True
- calib_027.jpg: sharpness=1383.74, corners_found=True
- calib_028.jpg: sharpness=1441.01, corners_found=True
- calib_029.jpg: sharpness=1435.34, corners_found=True
- calib_030.jpg: sharpness=1619.77, corners_found=True
- calib_031.jpg: sharpness=819.92, corners_found=True
- calib_032.jpg: sharpness=858.93, corners_found=True
- calib_033.jpg: sharpness=687.22, corners_found=True
- calib_034.jpg: sharpness=1665.69, corners_found=True
- calib_035.jpg: sharpness=1688.75, corners_found=True
- calib_036.jpg: sharpness=1579.00, corners_found=True
- calib_037.jpg: sharpness=916.51, corners_found=True
- calib_038.jpg: sharpness=631.63, corners_found=True
- calib_039.jpg: sharpness=1629.99, corners_found=True

## 2026-07-13 19:14:29 - 完成相机内参标定

- pattern=10x7, square_size=24.0 mm
- image_size=1280x960, views=40
- selected_model=pinhole_radial2, corner_detector=findChessboardCorners+cornerSubPix
- rms_reprojection_error=0.234098
- new_camera_matrix=[[1419.2691462533548, 0.0, 628.1336510914844], [0.0, 1419.2691462533548, 442.4134970922314], [0.0, 0.0, 1.0]]
- global_mapping={'invalid_output_ratio': 9.928385416666667e-05, 'source_sample_coverage_ratio': 0.8980825526932085, 'finite_source_mapping_ratio': 1.0}
- calibration_file=data/calibration.yaml
- model_evaluation_file=data/calibration_model_evaluation.yaml

## 2026-07-13 19:16:16 - 完成相机内参标定

- pattern=10x7, square_size=24.0 mm
- image_size=1280x960, views=40
- selected_model=pinhole_radial2, corner_detector=findChessboardCorners+cornerSubPix
- rms_reprojection_error=0.234098
- new_camera_matrix=[[1419.2691462533548, 0.0, 628.1336510914844], [0.0, 1419.2691462533548, 442.4134970922314], [0.0, 0.0, 1.0]]
- global_mapping={'invalid_output_ratio': 9.928385416666667e-05, 'source_sample_coverage_ratio': 0.8980825526932085, 'finite_source_mapping_ratio': 1.0}
- calibration_file=data/calibration.yaml
- model_evaluation_file=data/calibration_model_evaluation.yaml

## 2026-07-13 19:16:46 - 完成全图去畸变

- source=data/calibration_images/calib_015.jpg, output=data/undistorted_full.jpg
- calibration=data/calibration.yaml, model=pinhole_radial2
- new_camera_matrix=[[1419.2691462533548, 0.0, 628.1336510914844], [0.0, 1419.2691462533548, 442.4134970922314], [0.0, 0.0, 1.0]]
- invalid_output_ratio=0.000099
- metrics=data/undistortion_quality.yaml

## 2026-07-13 19:16:49 - 完成标定图质量检查

- images=data/calibration_images
- pattern=10x7
- quality_file=data/quality_report.yaml
- calib_000.jpg: sharpness=1199.31, mean_err=0.127px, max_err=0.476px
- calib_001.jpg: sharpness=1231.85, mean_err=0.108px, max_err=0.343px
- calib_002.jpg: sharpness=741.24, mean_err=0.148px, max_err=0.378px
- calib_003.jpg: sharpness=587.23, mean_err=0.293px, max_err=0.684px
- calib_004.jpg: sharpness=566.00, mean_err=0.251px, max_err=0.662px
- calib_005.jpg: sharpness=1298.98, mean_err=0.233px, max_err=0.667px
- calib_006.jpg: sharpness=597.23, mean_err=0.301px, max_err=0.766px
- calib_007.jpg: sharpness=740.33, mean_err=0.259px, max_err=0.699px
- calib_008.jpg: sharpness=608.09, mean_err=0.159px, max_err=0.504px
- calib_009.jpg: sharpness=557.97, mean_err=0.211px, max_err=0.578px
- calib_010.jpg: sharpness=863.11, mean_err=0.244px, max_err=0.974px
- calib_011.jpg: sharpness=848.80, mean_err=0.217px, max_err=0.509px
- calib_012.jpg: sharpness=1576.50, mean_err=0.149px, max_err=0.345px
- calib_013.jpg: sharpness=1019.77, mean_err=0.148px, max_err=0.366px
- calib_014.jpg: sharpness=1069.76, mean_err=0.150px, max_err=0.526px
- calib_015.jpg: sharpness=1772.88, mean_err=0.149px, max_err=0.420px
- calib_016.jpg: sharpness=647.36, mean_err=0.348px, max_err=1.248px
- calib_017.jpg: sharpness=975.71, mean_err=0.287px, max_err=0.781px
- calib_018.jpg: sharpness=1465.74, mean_err=0.204px, max_err=0.526px
- calib_019.jpg: sharpness=1398.82, mean_err=0.205px, max_err=0.474px
- calib_020.jpg: sharpness=1272.99, mean_err=0.132px, max_err=0.505px
- calib_021.jpg: sharpness=804.93, mean_err=0.163px, max_err=0.464px
- calib_022.jpg: sharpness=1389.90, mean_err=0.152px, max_err=0.430px
- calib_023.jpg: sharpness=783.77, mean_err=0.174px, max_err=0.410px
- calib_024.jpg: sharpness=935.82, mean_err=0.163px, max_err=0.465px
- calib_025.jpg: sharpness=1018.70, mean_err=0.117px, max_err=0.281px
- calib_026.jpg: sharpness=1101.22, mean_err=0.156px, max_err=0.543px
- calib_027.jpg: sharpness=1383.74, mean_err=0.152px, max_err=0.401px
- calib_028.jpg: sharpness=1441.01, mean_err=0.169px, max_err=0.445px
- calib_029.jpg: sharpness=1435.34, mean_err=0.137px, max_err=0.344px
- calib_030.jpg: sharpness=1619.77, mean_err=0.249px, max_err=0.636px
- calib_031.jpg: sharpness=819.92, mean_err=0.232px, max_err=0.601px
- calib_032.jpg: sharpness=858.93, mean_err=0.146px, max_err=0.452px
- calib_033.jpg: sharpness=687.22, mean_err=0.254px, max_err=0.856px
- calib_034.jpg: sharpness=1665.69, mean_err=0.216px, max_err=0.864px
- calib_035.jpg: sharpness=1688.75, mean_err=0.171px, max_err=0.506px
- calib_036.jpg: sharpness=1579.00, mean_err=0.134px, max_err=0.536px
- calib_037.jpg: sharpness=916.51, mean_err=0.194px, max_err=0.568px
- calib_038.jpg: sharpness=631.63, mean_err=0.192px, max_err=0.484px
- calib_039.jpg: sharpness=1629.99, mean_err=0.274px, max_err=0.909px

## 2026-07-13 19:18:50 - 完成相机内参标定

- pattern=10x7, square_size=24.0 mm
- image_size=1280x960, views=40
- selected_model=pinhole_radial2, corner_detector=findChessboardCorners+cornerSubPix
- rms_reprojection_error=0.234098
- new_camera_matrix=[[1421.5434457484635, 0.0, 628.1336510914844], [0.0, 1421.5434457484635, 442.4134970922314], [0.0, 0.0, 1.0]]
- global_mapping={'invalid_output_ratio': 0.0, 'source_sample_coverage_ratio': 0.895777224824356, 'finite_source_mapping_ratio': 1.0}
- calibration_file=data/calibration.yaml
- model_evaluation_file=data/calibration_model_evaluation.yaml

## 2026-07-13 19:19:05 - 完成全图去畸变

- source=data/calibration_images/calib_015.jpg, output=data/undistorted_full.jpg
- calibration=data/calibration.yaml, model=pinhole_radial2
- new_camera_matrix=[[1421.5434457484635, 0.0, 628.1336510914844], [0.0, 1421.5434457484635, 442.4134970922314], [0.0, 0.0, 1.0]]
- invalid_output_ratio=0.000000
- metrics=data/undistortion_quality.yaml

## 2026-07-13 19:19:09 - 完成标定图质量检查

- images=data/calibration_images
- pattern=10x7
- quality_file=data/quality_report.yaml
- calib_000.jpg: sharpness=1199.31, mean_err=0.127px, max_err=0.476px
- calib_001.jpg: sharpness=1231.85, mean_err=0.108px, max_err=0.343px
- calib_002.jpg: sharpness=741.24, mean_err=0.148px, max_err=0.378px
- calib_003.jpg: sharpness=587.23, mean_err=0.293px, max_err=0.684px
- calib_004.jpg: sharpness=566.00, mean_err=0.251px, max_err=0.662px
- calib_005.jpg: sharpness=1298.98, mean_err=0.233px, max_err=0.667px
- calib_006.jpg: sharpness=597.23, mean_err=0.301px, max_err=0.766px
- calib_007.jpg: sharpness=740.33, mean_err=0.259px, max_err=0.699px
- calib_008.jpg: sharpness=608.09, mean_err=0.159px, max_err=0.504px
- calib_009.jpg: sharpness=557.97, mean_err=0.211px, max_err=0.578px
- calib_010.jpg: sharpness=863.11, mean_err=0.244px, max_err=0.974px
- calib_011.jpg: sharpness=848.80, mean_err=0.217px, max_err=0.509px
- calib_012.jpg: sharpness=1576.50, mean_err=0.149px, max_err=0.345px
- calib_013.jpg: sharpness=1019.77, mean_err=0.148px, max_err=0.366px
- calib_014.jpg: sharpness=1069.76, mean_err=0.150px, max_err=0.526px
- calib_015.jpg: sharpness=1772.88, mean_err=0.149px, max_err=0.420px
- calib_016.jpg: sharpness=647.36, mean_err=0.348px, max_err=1.248px
- calib_017.jpg: sharpness=975.71, mean_err=0.287px, max_err=0.781px
- calib_018.jpg: sharpness=1465.74, mean_err=0.204px, max_err=0.526px
- calib_019.jpg: sharpness=1398.82, mean_err=0.205px, max_err=0.474px
- calib_020.jpg: sharpness=1272.99, mean_err=0.132px, max_err=0.505px
- calib_021.jpg: sharpness=804.93, mean_err=0.163px, max_err=0.464px
- calib_022.jpg: sharpness=1389.90, mean_err=0.152px, max_err=0.430px
- calib_023.jpg: sharpness=783.77, mean_err=0.174px, max_err=0.410px
- calib_024.jpg: sharpness=935.82, mean_err=0.163px, max_err=0.465px
- calib_025.jpg: sharpness=1018.70, mean_err=0.117px, max_err=0.281px
- calib_026.jpg: sharpness=1101.22, mean_err=0.156px, max_err=0.543px
- calib_027.jpg: sharpness=1383.74, mean_err=0.152px, max_err=0.401px
- calib_028.jpg: sharpness=1441.01, mean_err=0.169px, max_err=0.445px
- calib_029.jpg: sharpness=1435.34, mean_err=0.137px, max_err=0.344px
- calib_030.jpg: sharpness=1619.77, mean_err=0.249px, max_err=0.636px
- calib_031.jpg: sharpness=819.92, mean_err=0.232px, max_err=0.601px
- calib_032.jpg: sharpness=858.93, mean_err=0.146px, max_err=0.452px
- calib_033.jpg: sharpness=687.22, mean_err=0.254px, max_err=0.856px
- calib_034.jpg: sharpness=1665.69, mean_err=0.216px, max_err=0.864px
- calib_035.jpg: sharpness=1688.75, mean_err=0.171px, max_err=0.506px
- calib_036.jpg: sharpness=1579.00, mean_err=0.134px, max_err=0.536px
- calib_037.jpg: sharpness=916.51, mean_err=0.194px, max_err=0.568px
- calib_038.jpg: sharpness=631.63, mean_err=0.192px, max_err=0.484px
- calib_039.jpg: sharpness=1629.99, mean_err=0.274px, max_err=0.909px

## 2026-07-13 19:20:22 - 完成全图去畸变后的 H 辅助棋盘格位姿估计

- source=data/calibration_images/calib_015.jpg, calibration=data/calibration.yaml
- undistorted_image=data/undistorted_full.jpg
- pose_file=data/pose_latest.yaml, visualization=data/pose_latest.jpg
- h_orientation={'detected': True, 'corners_reversed': False, 'selected_score': 0.9387883785225785, 'alternative_score': 0.5102125196394608, 'confidence_margin': 0.42857585888311767, 'selected_valid_ratio': 1.0}
- mean_reprojection_error_px=0.136189
- tvec_board_to_camera_mm=(142.460, 3.604, 1457.791)
- camera_position_board_mm=(164.959, 219.616, -1438.756)

## 2026-07-13 19:21:50 - 完成全图去畸变

- source=data/calibration_images/calib_015.jpg, output=data/undistorted_full.jpg
- calibration=data/calibration.yaml, model=pinhole_radial2
- new_camera_matrix=[[1421.5434457484635, 0.0, 628.1336510914844], [0.0, 1421.5434457484635, 442.4134970922314], [0.0, 0.0, 1.0]]
- invalid_output_ratio=0.000000
- metrics=data/undistortion_quality.yaml

## 2026-07-13 19:21:50 - 完成全图去畸变后的 H 辅助棋盘格位姿估计

- source=data/calibration_images/calib_015.jpg, calibration=data/calibration.yaml
- undistorted_image=data/undistorted_full.jpg
- pose_file=data/pose_latest.yaml, visualization=data/pose_latest.jpg
- h_orientation={'detected': True, 'corners_reversed': False, 'selected_score': 0.9387883785225785, 'alternative_score': 0.5102125196394608, 'confidence_margin': 0.42857585888311767, 'selected_valid_ratio': 1.0}
- mean_reprojection_error_px=0.136189
- tvec_board_to_camera_mm=(142.460, 3.604, 1457.791)
- camera_position_board_mm=(164.959, 219.616, -1438.756)
