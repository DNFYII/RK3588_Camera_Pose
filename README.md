# RK3588 相机标定与棋盘格位姿计算

本项目在 RK3588 板卡上读取 `camera0` 相机，使用 `10 x 7` 个内角点、格长 `24 mm` 的棋盘格完成相机标定、全局去畸变和棋盘格位姿计算。当前正式实验使用 `/dev/video11`，图像分辨率为 `1280 x 960`。

当前成果只针对棋盘格标定与位姿流程。后续若做 H 标志独立检测，应单独新增代码和报告，不改动本次棋盘格成果，除非明确要求。

## 当前结果

最终模型为两参数径向针孔模型，固定 `p1=p2=k3=0`：

```text
K =
[1572.473515,    0.000000, 631.009383]
[   0.000000, 1569.155259, 449.135933]
[   0.000000,    0.000000,   1.000000]

D = [-0.958851, 1.240927, 0, 0, 0]

K_new =
[1421.543446,    0.000000, 628.133651]
[   0.000000, 1421.543446, 442.413497]
[   0.000000,    0.000000,   1.000000]
```

标定 RMS 重投影误差为 `0.234098 px`。全局去畸变输出尺寸仍为 `1280 x 960`，无效输出像素比例为 `0`，原始视场采样保留比例约为 `89.58%`。40 张标定图全部成功检测棋盘、完成 H 方向判定和位姿计算，回放平均重投影误差为 `0.180586 px`。

## 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

## 推荐完整流程

检查相机节点：

```bash
python3 -m camera_pose probe --device /dev/video11 --width 1280 --height 960 --save
```

启动实时预览，按回车后开始录制，结束时按 `Ctrl+C` 保存视频：

```bash
python3 -m camera_pose record-video \
  --device /dev/video11 \
  --width 1280 --height 960 --fps 30 \
  --wait-for-start \
  --preview-port 8080 \
  --preview-image data/live_preview.jpg \
  --output data/calibration_source.avi
```

录制时可在浏览器打开 `http://板卡IP:8080/` 查看实时画面、棋盘检测状态和清晰度。当前 AVI 文件保留相机实际采集帧，不强行补重复帧，因此播放器时长可能短于真实录制墙钟时间；这不影响逐帧抽帧和标定。

从视频抽取 40 张清晰且姿态分散的标定图：

```bash
python3 -m camera_pose extract-frames \
  --video data/calibration_source.avi \
  --pattern 10 7 \
  --max-images 40 \
  --min-sharpness 500 \
  --min-corner-shift 24 \
  --sample-interval 0.25 \
  --output data/calibration_images
```

标定前质量检查：

```bash
python3 -m camera_pose quality \
  --pattern 10 7 \
  --images data/calibration_images \
  --square-size 24 \
  --output data/quality_before_calibration.yaml
```

执行模型比较、相机标定和 `K_new` 计算：

```bash
python3 -m camera_pose calibrate \
  --pattern 10 7 \
  --square-size 24 \
  --images data/calibration_images \
  --output data/calibration.yaml \
  --evaluation-output data/calibration_model_evaluation.yaml \
  --min-images 40 \
  --cv-folds 3 \
  --undistort-alpha 0
```

检查最终模型在 40 张图上的重投影误差：

```bash
python3 -m camera_pose quality \
  --pattern 10 7 \
  --images data/calibration_images \
  --square-size 24 \
  --calibration data/calibration.yaml \
  --output data/quality_report.yaml
```

生成全局去畸变图和畸变评价：

```bash
python3 -m camera_pose undistort-image \
  --input data/calibration_images/calib_015.jpg \
  --calibration data/calibration.yaml \
  --pattern 10 7 \
  --output data/undistorted_full.jpg \
  --metrics-output data/undistortion_quality.yaml
```

计算单张图位姿并生成可视化：

```bash
python3 -m camera_pose pose-image \
  --input data/calibration_images/calib_015.jpg \
  --calibration data/calibration.yaml \
  --pattern 10 7 \
  --square-size 24 \
  --output data/pose_latest.yaml \
  --image-output data/pose_latest.jpg \
  --undistorted-output data/undistorted_full.jpg
```

实时计算位姿：

```bash
python3 -m camera_pose pose \
  --device /dev/video11 \
  --width 1280 --height 960 --fps 30 \
  --pattern 10 7 \
  --square-size 24 \
  --calibration data/calibration.yaml
```

## 输出文件

- `data/calibration_source.avi`：正式标定视频，也可作为后续目标检测数据源。
- `data/calibration_images/`：从视频筛选出的 40 张标定图。
- `data/calibration_contact_sheet.jpg`：40 张标定图总览，已人工检查。
- `data/extracted_frames.yaml`：候选帧和最终抽帧记录。
- `data/quality_before_calibration.yaml`：标定前清晰度与角点检查。
- `data/calibration.yaml`：最终内参、畸变系数、新内参和全局映射质量。
- `data/calibration_model_evaluation.yaml`：候选模型的误差、留出验证和全图映射对比。
- `data/global_undistortion_comparison.jpg`：原图及候选模型去畸变视觉对比。
- `data/quality_report.yaml`：最终模型逐图重投影误差。
- `data/undistorted_full.jpg`：最终全局去畸变图。
- `data/undistortion_quality.yaml`：去畸变前后棋盘直线残差。
- `data/pose_latest.yaml`：最终测试图的位姿数据。
- `data/pose_latest.jpg`：最终测试图的位姿可视化。
- `data/pose_validation.yaml`：40 张图的位姿回放验证。
- `data/process_log.md`：实验过程日志。
- `docs/calibration_pose_report.md`：完整中文实验报告。

## 坐标约定

`rvec_board_to_camera` 和 `tvec_board_to_camera` 将棋盘坐标系中的点变换到相机坐标系：

```text
X_camera = R * X_board + t
```

`camera_position_board = -R^T * t` 表示相机光心在棋盘坐标系中的位置。由于棋盘格边长为 `24 mm`，所有平移量均以毫米为单位。
