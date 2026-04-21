from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


RED_TRAJECTORY = "#E53935"
BLUE_BORDER = "#1976D2"
NAVY_POINT = "#1E3A8A"


def ensure_dir(path: os.PathLike | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_videos(base_dir: os.PathLike | str, keyword: str) -> list[Path]:
    base_path = Path(base_dir)
    return sorted(
        p for p in base_path.rglob("*.mp4")
        if keyword in p.name
    )


def safe_fps(cap: cv2.VideoCapture, default: float = 30.0) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        return default
    return float(fps)


def extract_background(
    video_path: os.PathLike | str,
    frame_count: int = 50,
    skip_seconds: float = 5.0,
    grayscale: bool = True,
) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = safe_fps(cap)
    start_frame = int(skip_seconds * fps)
    if total_frames <= 0:
        cap.release()
        return None

    if start_frame >= total_frames:
        start_frame = 0

    frame_ids = np.linspace(start_frame, total_frames - 1, num=max(1, frame_count), dtype=int)
    frames: list[np.ndarray] = []

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, frame = cap.read()
        if not ok:
            continue
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    cap.release()
    if not frames:
        return None
    return np.median(frames, axis=0).astype(np.uint8)


def load_blue_mask(
    annotated_img_path: os.PathLike | str,
    target_shape: tuple[int, int],
    close_kernel: int = 5,
) -> Optional[np.ndarray]:
    img = cv2.imread(str(annotated_img_path))
    if img is None:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([140, 255, 255]))
    if cv2.countNonZero(blue_mask) < 100:
        return None

    h, w = target_shape[:2]
    blue_mask = cv2.resize(blue_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    return blue_mask


def largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found in mask.")
    return max(contours, key=cv2.contourArea)


def contour_centroid(mask: np.ndarray) -> tuple[int, int]:
    m = cv2.moments(mask)
    if m["m00"] == 0:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Empty mask.")
        return int(xs.mean()), int(ys.mean())
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def group_to_day(label: str) -> str:
    text = str(label)
    if "前" in text or "光栓" in text or "Day 0" in text or "day0" in text.lower():
        return "Day 0"
    if "14" in text:
        return "Day 14"
    if "11" in text:
        return "Day 11"
    if "7" in text:
        return "Day 7"
    if "3" in text:
        return "Day 3"
    if "1" in text:
        return "Day 1"
    return text


def extract_subgroup(group_name: str, video_name: str) -> str:
    day = group_to_day(group_name)
    match = re.search(r"[-_](\d{1,2})(?!\d)", str(video_name))
    if match:
        return f"{day}-{int(match.group(1)):02d}"
    return day


def detect_mouse_centroid(
    bg_gray: np.ndarray,
    frame_bgr: np.ndarray,
    roi_mask: np.ndarray,
    threshold: int = 35,
    min_area: int = 50,
    max_area: int = 8000,
) -> Optional[tuple[int, int]]:
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.subtract(bg_gray, frame_gray)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    if not valid:
        return None
    m = cv2.moments(max(valid, key=cv2.contourArea))
    if m["m00"] == 0:
        return None
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def pairwise_distance(points: Iterable[tuple[int, int]], pixel_to_unit: float = 1.0) -> float:
    pts = list(points)
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for p0, p1 in zip(pts[:-1], pts[1:]):
        total += float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    return total * pixel_to_unit


def infer_unit(pixel_to_cm: Optional[float]) -> str:
    return "cm" if pixel_to_cm is not None else "px"


def plot_mask_outline(ax: plt.Axes, mask: np.ndarray, color: str = BLUE_BORDER, linewidth: float = 3.0) -> None:
    contour = largest_contour(mask)
    epsilon = 0.002 * cv2.arcLength(contour, True)
    smooth = cv2.approxPolyDP(contour, epsilon, True)
    xs = np.append(smooth[:, 0, 0], smooth[0, 0, 0])
    ys = np.append(smooth[:, 0, 1], smooth[0, 0, 1])
    ax.plot(xs, ys, color=color, linewidth=linewidth, zorder=1)


def save_red_trajectory(
    output_path: os.PathLike | str,
    x_coords: list[int],
    y_coords: list[int],
    mask: np.ndarray,
    bg_shape: tuple[int, int],
    title: Optional[str] = None,
) -> None:
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = plt.gca()
    ax.set_aspect("equal")
    plot_mask_outline(ax, mask)
    ax.plot(x_coords, y_coords, color=RED_TRAJECTORY, linewidth=1.4, alpha=0.95, zorder=2)
    ax.scatter(x_coords[0], y_coords[0], facecolors="white", edgecolors=NAVY_POINT, s=90, linewidths=2.2, zorder=3)
    ax.scatter(x_coords[-1], y_coords[-1], color=NAVY_POINT, s=85, zorder=3)
    ax.text(
        x_coords[0], y_coords[0], "S", ha="center", va="center", color="white", fontsize=12,
        fontweight="bold", bbox=dict(facecolor=NAVY_POINT, edgecolor="none", boxstyle="square,pad=0.2")
    )
    ax.text(
        x_coords[-1], y_coords[-1], "E", ha="center", va="center", color="white", fontsize=12,
        fontweight="bold", bbox=dict(facecolor=NAVY_POINT, edgecolor="none", boxstyle="square,pad=0.2")
    )
    if title:
        ax.set_title(title)
    ax.set_ylim(bg_shape[0], 0)
    ax.set_xlim(0, bg_shape[1])
    ax.axis("off")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_heatmap(
    output_path: os.PathLike | str,
    x_coords: list[int],
    y_coords: list[int],
    bg_gray: np.ndarray,
    mask: np.ndarray,
    sigma: float = 3.0,
) -> None:
    h, w = bg_gray.shape[:2]
    masked_bg = bg_gray.copy()
    masked_bg[mask == 0] = 255
    heatmap, _, _ = np.histogram2d(
        y_coords,
        x_coords,
        bins=(max(10, h // 5), max(10, w // 5)),
        range=[[0, h], [0, w]],
    )
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(masked_bg, cmap="gray")
    plt.imshow(cv2.resize(heatmap, (w, h)), cmap="jet", alpha=0.5)
    plt.axis("off")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_dataframe(df: pd.DataFrame, csv_path: os.PathLike | str) -> None:
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


@dataclass
class TrackingConfig:
    skip_seconds: float = 5.0
    analyze_seconds: Optional[float] = 300.0
    background_frames: int = 50
    threshold: int = 35
    min_area: int = 50
    max_area: int = 8000
    pixel_to_cm: Optional[float] = None

    @property
    def pixel_to_unit(self) -> float:
        return float(self.pixel_to_cm) if self.pixel_to_cm is not None else 1.0

    @property
    def unit(self) -> str:
        return infer_unit(self.pixel_to_cm)
