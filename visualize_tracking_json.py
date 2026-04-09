#!/usr/bin/env python3
"""Visualize structured hand-tracking JSON in 3D.

Supports JSON shaped like:
- {"frames": [...records...]}
- [...records...]

Each record should contain at least:
- frame_idx
- hand_side: left/right
- wrist position: p_wrist_base or p_wrist
Optional:
- keypoints_3d_local: 21x3 local hand keypoints relative to wrist

If the source data is in millimeters, this script can auto-detect and convert to meters.
It can show an interactive animation and/or export MP4/GIF when ffmpeg is available.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install it with: pip install matplotlib"
    ) from exc


LEFT_COLOR = "#3da5ff"
RIGHT_COLOR = "#ff5b7f"
EDGE_COLOR = "#d6d6d6"
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


@dataclass
class HandFrame:
    side: str
    wrist: np.ndarray  # (3,)
    keypoints: Optional[np.ndarray]  # (21, 3) in world/base/cam frame


@dataclass
class FrameBundle:
    frame_idx: int
    timestamp_sec: float
    hands: Dict[str, HandFrame]


def _norm_side(value: str) -> Optional[str]:
    s = str(value).strip().lower()
    if s in ("l", "left"):
        return "left"
    if s in ("r", "right"):
        return "right"
    return None


def _load_records(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "frames" in raw:
        records = raw["frames"]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("JSON must be a list of records or an object with key 'frames'")
    return [r for r in records if isinstance(r, dict)]


def _as_vec3(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"Expected vec3, got shape {arr.shape}")
    return arr[:3].copy()


def _record_timestamp_sec(rec: dict) -> float:
    if "timestamp_sec" in rec:
        try:
            return float(rec["timestamp_sec"])
        except Exception:
            return 0.0
    return 0.0


def _detect_scale(records: Iterable[dict], force_scale: Optional[float]) -> Tuple[float, str]:
    if force_scale is not None:
        return float(force_scale), "user-specified"

    samples: List[float] = []
    bone_samples: List[float] = []
    for rec in records:
        wrist = rec.get("p_wrist_base", rec.get("p_wrist"))
        if wrist is not None:
            try:
                samples.append(float(np.linalg.norm(_as_vec3(wrist))))
            except Exception:
                pass
        kp = rec.get("keypoints_3d_local")
        if kp is not None:
            try:
                kp_arr = np.asarray(kp, dtype=np.float64).reshape(-1, 3)
                if kp_arr.shape[0] >= 10:
                    bone_samples.append(float(np.linalg.norm(kp_arr[9] - kp_arr[0])))
            except Exception:
                pass
        if len(samples) >= 50 and len(bone_samples) >= 50:
            break

    median_wrist = float(np.median(samples)) if samples else float("nan")
    median_bone = float(np.median(bone_samples)) if bone_samples else float("nan")

    # Heuristic: human wrist position in meters is usually < 5 from origin;
    # local wrist->middle_mcp in meters is usually ~0.03..0.12.
    likely_mm = False
    if math.isfinite(median_wrist) and median_wrist > 10.0:
        likely_mm = True
    if math.isfinite(median_bone) and median_bone > 1.0:
        likely_mm = True

    if likely_mm:
        return 0.001, f"auto-detected millimeters (median wrist norm={median_wrist:.3f}, median wrist->middle_mcp={median_bone:.3f})"
    return 1.0, f"auto-detected meters (median wrist norm={median_wrist:.3f}, median wrist->middle_mcp={median_bone:.3f})"


def load_frames(path: str, scale: float) -> List[FrameBundle]:
    records = _load_records(path)
    by_frame: Dict[int, FrameBundle] = {}

    for rec in records:
        side = _norm_side(rec.get("hand_side", ""))
        if side is None:
            continue
        wrist_key = "p_wrist_base" if "p_wrist_base" in rec else "p_wrist" if "p_wrist" in rec else None
        if wrist_key is None:
            continue

        try:
            wrist = _as_vec3(rec[wrist_key]) * scale
        except Exception:
            continue

        kp_world: Optional[np.ndarray] = None
        if "keypoints_3d_local" in rec:
            try:
                kp_local = np.asarray(rec["keypoints_3d_local"], dtype=np.float64).reshape(-1, 3)
                if kp_local.shape[0] >= 21:
                    kp_world = wrist[None, :] + kp_local[:21] * scale
            except Exception:
                kp_world = None

        frame_idx = int(rec.get("frame_idx", rec.get("frame", rec.get("idx", 0))))
        ts = _record_timestamp_sec(rec)
        if frame_idx not in by_frame:
            by_frame[frame_idx] = FrameBundle(frame_idx=frame_idx, timestamp_sec=ts, hands={})
        by_frame[frame_idx].hands[side] = HandFrame(side=side, wrist=wrist, keypoints=kp_world)
        by_frame[frame_idx].timestamp_sec = ts

    return [by_frame[k] for k in sorted(by_frame.keys())]


def _collect_points(frames: List[FrameBundle]) -> np.ndarray:
    pts: List[np.ndarray] = []
    for fr in frames:
        for hand in fr.hands.values():
            pts.append(hand.wrist.reshape(1, 3))
            if hand.keypoints is not None:
                pts.append(hand.keypoints)
    if not pts:
        raise ValueError("No plottable wrist/keypoint data found")
    return np.concatenate(pts, axis=0)


def _set_axes_equal_3d(ax, points: np.ndarray) -> None:
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_hand(ax, hand: HandFrame, draw_edges: bool) -> None:
    color = LEFT_COLOR if hand.side == "left" else RIGHT_COLOR
    ax.scatter(
        [hand.wrist[0]], [hand.wrist[1]], [hand.wrist[2]],
        s=220, c=color, edgecolors="white", linewidths=1.0, depthshade=True,
    )
    if hand.keypoints is None:
        return

    kp = hand.keypoints
    ax.scatter(kp[1:, 0], kp[1:, 1], kp[1:, 2], s=26, c=color, alpha=0.95, depthshade=True)
    ax.scatter([kp[0, 0]], [kp[0, 1]], [kp[0, 2]], s=1, c=color, alpha=0.0)

    if draw_edges:
        for i, j in HAND_EDGES:
            if i < kp.shape[0] and j < kp.shape[0]:
                ax.plot(
                    [kp[i, 0], kp[j, 0]],
                    [kp[i, 1], kp[j, 1]],
                    [kp[i, 2], kp[j, 2]],
                    color=EDGE_COLOR,
                    linewidth=1.1,
                    alpha=0.8,
                )


def animate_frames(
    frames: List[FrameBundle],
    fps: float,
    title: str,
    output: Optional[str],
    draw_edges: bool,
    show: bool,
) -> None:
    points = _collect_points(frames)
    fig = plt.figure(figsize=(10, 8), facecolor="#0f1116")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#171a21")
    fig.patch.set_facecolor("#0f1116")

    def _style_axes() -> None:
        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.set_zlabel("Z", color="white")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.zaxis.label.set_color("white")
        try:
            ax.xaxis.set_pane_color((0.12, 0.13, 0.16, 1.0))
            ax.yaxis.set_pane_color((0.12, 0.13, 0.16, 1.0))
            ax.zaxis.set_pane_color((0.12, 0.13, 0.16, 1.0))
        except Exception:
            pass
        ax.grid(True, alpha=0.25)
        _set_axes_equal_3d(ax, points)

    def update(i: int):
        ax.cla()
        _style_axes()
        fr = frames[i]
        for side in ("left", "right"):
            hand = fr.hands.get(side)
            if hand is not None:
                _draw_hand(ax, hand, draw_edges=draw_edges)
        ax.set_title(
            f"{title}\nframe={fr.frame_idx}  t={fr.timestamp_sec:.3f}s  shown={i + 1}/{len(frames)}",
            color="white",
            pad=18,
        )
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / max(fps, 1e-6), blit=False)

    if output:
        output = os.path.abspath(os.path.expanduser(output))
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        ext = os.path.splitext(output)[1].lower()
        if ext == ".gif":
            anim.save(output, writer="pillow", fps=fps)
        else:
            anim.save(output, writer="ffmpeg", fps=fps, dpi=140)
        print(f"Saved animation to: {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D visualize hand-tracking JSON")
    parser.add_argument("--json", required=True, help="Path to structured hand JSON")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback/export FPS")
    parser.add_argument("--output", type=str, default=None, help="Optional output .mp4 or .gif path")
    parser.add_argument("--title", type=str, default="Hand Tracking 3D Replay", help="Plot title")
    parser.add_argument("--force-scale", type=float, default=None, help="Multiply all coordinates by this scale, e.g. 0.001 for mm->m")
    parser.add_argument("--no-edges", action="store_true", help="Do not draw hand skeleton edges")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = os.path.abspath(os.path.expanduser(args.json))
    records = _load_records(json_path)
    scale, reason = _detect_scale(records, args.force_scale)
    frames = load_frames(json_path, scale=scale)
    if not frames:
        raise SystemExit("No frames loaded from JSON")

    print(f"Loaded {len(records)} records from: {json_path}")
    print(f"Grouped into {len(frames)} frame(s)")
    print(f"Coordinate scale: {scale} ({reason})")
    print("Using wrist as big sphere, finger keypoints as small spheres")

    animate_frames(
        frames=frames,
        fps=float(args.fps),
        title=args.title,
        output=args.output,
        draw_edges=not args.no_edges,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
