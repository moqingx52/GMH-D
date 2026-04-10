# GMH-D（已中文化说明）

GMH-D（Google MediaPipe Hands-Depth）是一种基于 **RGB-D 相机 + MediaPipe** 的 3D 手部关键点跟踪方法，用深度信息增强 MediaPipe Hands 的 3D 估计，适用于在线实时跟踪或离线处理录制数据。

本目录提供两套入口脚本：

- `GMHD_AzureKinect.py`：Azure Kinect（或兼容 Azure Kinect SDK 的 RGB-D 设备）
- `GMHD_RealSense.py`：Intel RealSense D4xx

同时提供导出与回放：

- `tracking_export.py`：将 GMH-D 内存中的 tracking 数据导出为 `xr_teleoperate`/HaMeR 可读取的 JSON
- `visualize_tracking_json.py`：3D 动画回放导出的 JSON（支持自动单位缩放、导出 mp4/gif）

---

## 论文引用

如你在研究工作中使用本代码，请考虑引用：

- G. Amprimo, C. Ferraris, G. Masi, G. Pettiti and L. Priano,
  “GMH-D: Combining Google MediaPipe and RGB-Depth Cameras for Hand Motor Skills Remote Assessment,”
  2022 IEEE International Conference on Digital Health (ICDH), Barcelona, Spain, 2022, pp. 132–141,
  doi: 10.1109/ICDH55609.2022.00029.

- G. Amprimo, G. Masi, G. Pettiti, G. Olmo, L. Priano, C. Ferraris,
  “Hand tracking for clinical applications: validation of the google mediapipe hand (GMH) and the depth-enhanced GMH-D frameworks.”
  Biomedical Signal Processing and Controls (2024) doi: 10.1016/j.bspc.2024.106508.

---

## 安装

### 1）Python 环境

建议使用 Python 3.9+（本项目依赖 `mediapipe` / OpenCV 等）。

```bash
python -m venv env
```

- Windows：

```bash
.\env\Scripts\activate
```

- Linux/macOS：

```bash
source env/bin/activate
```

### 2）安装依赖

```bash
pip install -r requirements.txt
```

> 注意：
>
>- 使用 **Azure Kinect** 需要你已正确安装 Azure Kinect SDK（以及 `pyk4a` 所需的运行时）。
>- 使用 **Intel RealSense** 需要你已正确安装 RealSense SDK（以及 `pyrealsense2` 的运行时）。

---

## 快速开始（最常用）

### A. RealSense 在线实时跟踪（边采集边跟踪）

```bash
python GMHD_RealSense.py --mode online --online-workflow stream --save yes --outputpath .\out --outputname demo --n_hands 2 --visualize yes --interval 10
```

### B. RealSense 在线先录制再处理（推荐：录制阶段会预览 RGB/深度）

该模式会：
1) 连接 RealSense 录制 bag（左侧 RGB、右侧深度预览，按 `ESC` 可提前停止录制）
2) 自动切换到离线模式处理刚录制的 bag 并导出 JSON

```bash
python GMHD_RealSense.py --mode online --online-workflow record_then_process --save yes --outputpath .\out --outputname demo --visualize yes --interval 10
```

如果想显式指定录制出来的 `.bag` 路径：

```bash
python GMHD_RealSense.py --mode online --online-workflow record_then_process --record-bag-path .\out\my_record.bag --save yes --outputpath .\out --outputname demo
```

### C. Azure Kinect 在线实时跟踪

```bash
python GMHD_AzureKinect.py --mode online --save yes --outputpath .\out --outputname demo --n_hands 2 --visualize yes --interval 10
```

---

## 命令行参数说明（两套脚本通用）

两套入口脚本使用 `click` 解析参数，参数名以 `--xxx` 形式传入。

### 通用参数

- `--mode`：`online` 或 `offline`
  - `GMHD_AzureKinect.py` 默认 `online`
  - `GMHD_RealSense.py` 默认 `offline`
- `--save`：是否导出追踪结果，`yes` / `no`（默认 `yes`）
- `--outputpath`：输出目录（默认 `.`）
- `--outputname`：输出文件名（不含后缀，默认 `tracking_data`；最终写出 `*.json`）
- `--n_hands`：跟踪手的数量（默认 `2`）
- `--handconf`：手检测置信度阈值（默认 `0.5`）
- `--rerun_pd`：Palm detector 重跑阈值（默认 `0.2`）
- `--jointconf`：关键点跟踪置信度阈值（默认 `0.5`）
- `--interval`：运行时长（秒，默认 `10`）
  - 代码逻辑是“超过该时长就结束”。想长时间运行可设大一些（例如 `3600`）。
- `--visualize`：是否在处理时显示窗口，`yes` / `no`（默认 `yes`）
- `--debug`：调试开关（布尔 flag）。开启后会输出统计信息（例如导出手的数量、过滤原因等）。

### `--cam2base-json`（可选，对接 HaMeR / xr_teleoperate）

`GMHD_AzureKinect.py` / `GMHD_RealSense.py` 都支持：

- `--cam2base-json <path>`：一个 JSON 文件，包含 4×4 外参矩阵 `T_cam2base`。

当提供该参数时，导出的 tracking JSON 会同时写入：

- `p_wrist_base` / `R_wrist_base`（base 坐标系下的手腕位姿）

不提供时只写：

- `p_wrist` / `R_wrist`（相机坐标系下的手腕位姿）

> 这份外参 JSON 的格式与 HaMeR `demo.py --cam2base_json`、以及 `xr_teleoperate` 的约定保持一致。

---

## 离线处理（offline）

### 1）RealSense 离线处理 `.bag`

```bash
python GMHD_RealSense.py --mode offline --bagfilepath I:\data --bagfilename demo.bag --save yes --outputpath .\out --outputname demo --visualize no
```

- `--bagfilepath`：`.bag` 所在目录
- `--bagfilename`：`.bag` 文件名

### 2）Azure Kinect 离线处理 `.mkv`

```bash
python GMHD_AzureKinect.py --mode offline --mkvfilepath I:\data --mkvfilename demo.mkv --save yes --outputpath .\out --outputname demo --visualize no
```

- `--mkvfilepath`：`.mkv` 所在目录
- `--mkvfilename`：`.mkv` 文件名

---

## 在线处理（online）

### 1）RealSense 在线模式额外参数

`GMHD_RealSense.py` 额外支持：

- `--online-workflow`：仅 `mode=online` 有效
  - `stream`：边采集边跟踪（默认）
  - `record_then_process`：先录制并预览，再转离线处理录制出的 bag
- `--record-bag-path`：仅 `online-workflow=record_then_process` 时使用
  - 如果不传，默认写到 `outputpath/outputname.bag`

示例：

```bash
python GMHD_RealSense.py --mode online --online-workflow stream --save no --visualize yes --interval 30
```

```bash
python GMHD_RealSense.py --mode online --online-workflow record_then_process --save yes --outputpath .\out --outputname demo --visualize yes --interval 10 --debug
```

### 2）Azure Kinect 在线模式

```bash
python GMHD_AzureKinect.py --mode online --save yes --outputpath .\out --outputname demo --visualize yes --interval 10
```

---

## 输出文件说明（tracking JSON）

当 `--save yes` 时，会在 `outputpath` 下写出：

- `outputname.json`

该 JSON 是为 `xr_teleoperate` / HaMeR 的读入逻辑准备的“平面记录列表”结构：

- 顶层对象：`{"format": ..., "frames": [ ... ]}`
- `frames` 内每条记录对应 “某一帧 + 某只手”
- 常见字段：
  - `frame_idx`
  - `hand_side`：`left` / `right`
  - `timestamp_sec`
  - `p_wrist`、`R_wrist`（相机系）
  - 可选 `p_wrist_base`、`R_wrist_base`（提供 `--cam2base-json` 时写入）
  - `keypoints_3d_local`：21×3，局部关键点（相对 wrist）

> 坐标单位：不同相机/SDK 下导出的数值单位可能是“毫米”或“米”。本仓库的回放脚本支持自动检测并缩放，也可以手动指定缩放。

---

## 结果回放与可视化（3D）

使用 `visualize_tracking_json.py`：

### 1）直接播放（自动检测 mm/m 并缩放）

```bash
python visualize_tracking_json.py --json .\out\demo.json --fps 30
```

### 2）强制缩放（例如 mm -> m）

```bash
python visualize_tracking_json.py --json .\out\demo.json --force-scale 0.001
```

### 3）导出视频

```bash
python visualize_tracking_json.py --json .\out\demo.json --output .\out\demo.mp4 --fps 30 --no-show
```

---

## 常见问题（FAQ）

### 1）第一次运行会下载 MediaPipe 模型

脚本会自动下载 `hand_landmarker.task` 到 `./MP_model/`。

### 2）`--visualize no` 仍然想跑完整流程

可以把 `--visualize` 设为 `no`，用于服务器/无显示环境下运行（但注意 OpenCV/后端可能仍需图形依赖）。

### 3）在线模式想一直跑

`--interval` 是“超过多少秒就结束”的逻辑。想长时间跑可设为较大值，例如 `--interval 3600`。

---

## License

MIT License（同原项目约定）。
