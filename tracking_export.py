"""
将 GMH-D 内存中的 Frame / GMHDHand 序列导出为 xr_teleoperate 可读的 JSON。

约定与 ``xr_teleoperate/teleop/input_source/hamer_input.py`` 中 ``HamerJsonReader`` 一致：
顶层为 ``{"frames": [ ... ]}``（或等价地仅列表），每条记录含 ``frame_idx``、``hand_side``、
``p_wrist`` + ``R_wrist``（相机系），可选 ``p_wrist_base`` + ``R_wrist_base``（当提供
与 HaMeR 相同的 ``T_cam2base`` JSON 时写入）。
"""
from __future__ import annotations

import json
import os
from typing import Any, List, Optional, Sequence

import numpy as np

# 与 mediapipe.solutions.hands.HandLandmark 枚举顺序一致（WRIST 为首）
_WRIST = 0
_THUMB_CMC = 1
_INDEX_FINGER_MCP = 5
_MIDDLE_FINGER_MCP = 9
_PINKY_MCP = 17


def load_T_cam2base_from_json(path: str) -> np.ndarray:
    """与 HaMeR ``demo.py --cam2base_json`` / xr_teleoperate 相同：含 4×4 ``T_cam2base``。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or "T_cam2base" not in raw:
        raise ValueError("cam2base JSON 须为对象且包含键 T_cam2base（4×4 矩阵）")
    t = np.asarray(raw["T_cam2base"], dtype=np.float64)
    if t.shape != (4, 4):
        raise ValueError(f"T_cam2base 形状应为 (4, 4)，当前为 {t.shape}")
    return t


def wrist_pose_cam_to_base(
    p_cam: np.ndarray, r_cam: np.ndarray, t_cam2base: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """与 ``hamer_input.wrist_pose_cam_to_base`` 一致。"""
    r_cb = t_cam2base[:3, :3]
    t_cb = t_cam2base[:3, 3]
    p_cam = np.asarray(p_cam, dtype=np.float64).reshape(3)
    r_cam = np.asarray(r_cam, dtype=np.float64).reshape(3, 3)
    p_base = r_cb @ p_cam + t_cb
    r_base = r_cb @ r_cam
    return p_base, r_base


def _norm_hand_side(category_name: str) -> Optional[str]:
    s = str(category_name).strip().lower()
    if s.startswith("l") or "left" == s:
        return "left"
    if s.startswith("r") or "right" == s:
        return "right"
    return None


def _landmarks_to_xyz_array(joints: Sequence[Any]) -> Optional[np.ndarray]:
    coords: List[List[float]] = []
    for j in joints:
        x, y, z = getattr(j, "x", None), getattr(j, "y", None), getattr(j, "z", None)
        if x is None or y is None or z is None:
            coords.append([float("nan")] * 3)
        else:
            coords.append([float(x), float(y), float(z)])
    arr = np.asarray(coords, dtype=np.float64)
    if not np.all(np.isfinite(arr[_WRIST])):
        return None
    return arr


def estimate_R_wrist_cam(joints_xyz: np.ndarray, side: str) -> np.ndarray:
    """
    用手腕与掌面关键点在相机坐标系下构造稳定的近似手腕旋转，供 IK 使用（非 MANO 真值）。

    目标：让左右手得到同语义的腕部坐标系，避免镜像手直接套同一套叉乘顺序后
    某一轴翻转，进而把双臂 IK 解成一高一低。

    约定：
    - z 轴：手腕 -> 中指 MCP（朝手指根方向）
    - x 轴：掌面横向；左手取 index -> pinky，右手取 pinky -> index，使左右手 x 轴同语义
    - y 轴：由 z × x 得到，保证正交右手系
    列向量为手坐标系 x/y/z 在相机系下的表达。
    """
    side = _norm_hand_side(side)
    if side is None:
        return np.eye(3, dtype=np.float64)

    p0 = joints_xyz[_WRIST]
    p5 = joints_xyz[_INDEX_FINGER_MCP]
    p9 = joints_xyz[_MIDDLE_FINGER_MCP]
    p17 = joints_xyz[_PINKY_MCP]
    if not np.all(np.isfinite(np.stack([p0, p5, p9, p17]))):
        return np.eye(3, dtype=np.float64)

    v_z = p9 - p0
    nz = float(np.linalg.norm(v_z))
    if nz < 1e-9:
        return np.eye(3, dtype=np.float64)
    v_z = v_z / nz

    if side == "left":
        v_x = p17 - p5
    else:
        v_x = p5 - p17
    nx = float(np.linalg.norm(v_x))
    if nx < 1e-9:
        return np.eye(3, dtype=np.float64)
    v_x = v_x / nx

    v_y = np.cross(v_z, v_x)
    ny = float(np.linalg.norm(v_y))
    if ny < 1e-9:
        return np.eye(3, dtype=np.float64)
    v_y = v_y / ny

    v_x = np.cross(v_y, v_z)
    nx = float(np.linalg.norm(v_x))
    if nx < 1e-9:
        return np.eye(3, dtype=np.float64)
    v_x = v_x / nx
    return np.stack([v_x, v_y, v_z], axis=1)


def _wrist_score(joints: Sequence[Any]) -> float:
    w = joints[_WRIST]
    vis = float(getattr(w, "visibility", 1.0) or 1.0)
    pres = float(getattr(w, "presence", 1.0) or 1.0)
    return float(np.clip(min(vis, pres), 0.0, 1.0))


def tracking_frames_to_records(
    tracking_data: Sequence[Any],
    *,
    cam2base_json: Optional[str] = None,
    timestamp_to_sec: float = 1e-3,
) -> List[dict]:
    """
    将 GMH-D 的 ``Frame`` 列表转为 ``HamerJsonReader`` 可解析的平面记录列表
    （每手每帧一条）。
    """
    t_cam2base: Optional[np.ndarray] = None
    if cam2base_json:
        p = os.path.abspath(os.path.expanduser(cam2base_json))
        if not os.path.isfile(p):
            raise FileNotFoundError(f"未找到 cam2base JSON: {p}")
        t_cam2base = load_T_cam2base_from_json(p)

    records: List[dict] = []
    for frame_idx, frame in enumerate(tracking_data):
        ts_raw = float(getattr(frame, "timestamp", 0.0))
        timestamp_sec = ts_raw * float(timestamp_to_sec)
        hands = getattr(frame, "hands", None) or []
        for hand in hands:
            joints = getattr(hand, "joints", None)
            if not joints:
                continue
            handedness = getattr(hand, "handedness", None)
            if not handedness:
                continue
            side = _norm_hand_side(handedness[0].category_name)
            if side is None:
                continue
            arr = _landmarks_to_xyz_array(joints)
            if arr is None:
                continue
            p_wrist = arr[_WRIST]
            r_cam = estimate_R_wrist_cam(arr, side)
            score = _wrist_score(joints)
            rec: dict[str, Any] = {
                "frame_idx": int(frame_idx),
                "hand_side": side,
                "score": score,
                "timestamp_sec": timestamp_sec,
                "p_wrist": p_wrist.tolist(),
                "R_wrist": r_cam.tolist(),
                "keypoints_3d_local": (arr - arr[_WRIST:_WRIST + 1]).tolist(),
            }
            if t_cam2base is not None:
                p_b, r_b = wrist_pose_cam_to_base(p_wrist, r_cam, t_cam2base)
                rec["p_wrist_base"] = p_b.tolist()
                rec["R_wrist_base"] = r_b.tolist()
            records.append(rec)
    return records


def save_xr_teleop_tracking_json(
    tracking_data: Sequence[Any],
    filepath: str,
    *,
    cam2base_json: Optional[str] = None,
    code_version: str = "GMHD_1.0",
    timestamp_to_sec: float = 1e-3,
) -> str:
    """
    写入 UTF-8 JSON；自动创建输出目录；返回最终文件路径。

    若仅含 ``p_wrist``/``R_wrist``，在 xr_teleoperate 侧需使用 ``--hamer-cam2base-json``
    指向同一外参文件；若本函数传入了 ``cam2base_json``，则会同时写入 ``p_wrist_base``/
    ``R_wrist_base``，可直接回放而无需在遥操作端再变换。
    """
    filepath = os.path.abspath(os.path.expanduser(str(filepath)))
    if not filepath.lower().endswith(".json"):
        filepath += ".json"
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    records = tracking_frames_to_records(
        tracking_data,
        cam2base_json=cam2base_json,
        timestamp_to_sec=timestamp_to_sec,
    )
    payload = {
        "format": "xr_teleoperate_hamer_compatible",
        "code_version": code_version,
        "frames": records,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return filepath
