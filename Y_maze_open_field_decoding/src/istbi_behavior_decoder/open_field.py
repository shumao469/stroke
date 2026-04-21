from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .common import (
    TrackingConfig,
    contour_centroid,
    detect_mouse_centroid,
    ensure_dir,
    extract_background,
    extract_subgroup,
    find_videos,
    group_to_day,
    largest_contour,
    load_blue_mask,
    pairwise_distance,
    save_dataframe,
    save_heatmap,
    save_red_trajectory,
    safe_fps,
)


CENTER_COLOR = "#FFD54F"
PERIPHERY_COLOR = "#90CAF9"


def build_center_mask(roi_mask: np.ndarray, center_ratio: float = 0.5) -> np.ndarray:
    contour = largest_contour(roi_mask)
    x, y, w, h = cv2.boundingRect(contour)
    cx, cy = x + w / 2.0, y + h / 2.0
    half_w = (w * center_ratio) / 2.0
    half_h = (h * center_ratio) / 2.0

    rect = np.zeros_like(roi_mask)
    x0 = int(round(cx - half_w))
    y0 = int(round(cy - half_h))
    x1 = int(round(cx + half_w))
    y1 = int(round(cy + half_h))
    cv2.rectangle(rect, (x0, y0), (x1, y1), 255, thickness=-1)
    return cv2.bitwise_and(rect, roi_mask)


def save_open_field_zone_debug(output_path: Path, roi_mask: np.ndarray, center_mask: np.ndarray) -> None:
    vis = np.zeros((roi_mask.shape[0], roi_mask.shape[1], 3), dtype=np.uint8)
    vis[roi_mask > 0] = [235, 245, 255]
    vis[(roi_mask > 0) & (center_mask == 0)] = [255, 200, 120]
    vis[center_mask > 0] = [80, 210, 255]
    cv2.putText(vis, "Center", contour_centroid(center_mask), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(str(output_path), vis)


def process_open_field_video(
    video_path: Path,
    roi_mask: np.ndarray,
    center_mask: np.ndarray,
    output_dir: Path,
    config: TrackingConfig,
) -> Optional[dict]:
    bg_gray = extract_background(
        video_path,
        frame_count=config.background_frames,
        skip_seconds=config.skip_seconds,
        grayscale=True,
    )
    if bg_gray is None:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = safe_fps(cap)
    max_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(config.skip_seconds * fps)
    if config.analyze_seconds is None:
        max_frames = max(0, max_total_frames - start_frame)
    else:
        max_frames = min(int(config.analyze_seconds * fps), max(0, max_total_frames - start_frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    points: list[tuple[int, int]] = []
    time_series: list[float] = []
    speed_series: list[float] = []
    zone_series: list[str] = []
    center_entries = 0
    current_zone: Optional[str] = None
    last_point: Optional[tuple[int, int]] = None
    frames_processed = 0

    while frames_processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        point = detect_mouse_centroid(
            bg_gray=bg_gray,
            frame_bgr=frame,
            roi_mask=roi_mask,
            threshold=config.threshold,
            min_area=config.min_area,
            max_area=config.max_area,
        )
        if point is None and last_point is not None:
            point = last_point
        if point is None:
            frames_processed += 1
            continue

        points.append(point)
        dist = 0.0
        if last_point is not None:
            dist = float(np.hypot(point[0] - last_point[0], point[1] - last_point[1])) * config.pixel_to_unit
        last_point = point

        x, y = point
        zone = "Center" if center_mask[y, x] > 0 else "Periphery"
        if current_zone is not None and zone != current_zone and zone == "Center":
            center_entries += 1
        current_zone = zone

        time_series.append(frames_processed / fps)
        speed_series.append(dist * fps)
        zone_series.append(zone)
        frames_processed += 1

    cap.release()
    if not points:
        return None

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    base_name = video_path.stem
    unit = config.unit

    save_red_trajectory(output_dir / f"{base_name}_trajectory.jpg", x_coords, y_coords, roi_mask, bg_gray.shape)
    save_heatmap(output_dir / f"{base_name}_heatmap.jpg", x_coords, y_coords, bg_gray, roi_mask)
    save_open_field_session_summary(
        output_path=output_dir / f"{base_name}_session_summary.png",
        time_series=time_series,
        speed_series=speed_series,
        zone_series=zone_series,
        unit=unit,
    )

    total_distance = pairwise_distance(points, pixel_to_unit=config.pixel_to_unit)
    duration_sec = len(points) / fps
    mean_speed = total_distance / duration_sec if duration_sec > 0 else 0.0
    center_time = sum(1 for z in zone_series if z == "Center") / fps
    periphery_time = sum(1 for z in zone_series if z == "Periphery") / fps

    return {
        "Group": video_path.parent.name,
        "Subgroup": extract_subgroup(video_path.parent.name, video_path.stem),
        "Video": base_name,
        "Duration_sec": round(duration_sec, 4),
        f"Total_Distance_{unit}": round(total_distance, 4),
        f"Mean_Speed_{unit}_per_s": round(mean_speed, 4),
        "Center_Time_sec": round(center_time, 4),
        "Periphery_Time_sec": round(periphery_time, 4),
        "Center_Ratio": round(center_time / duration_sec if duration_sec > 0 else 0.0, 4),
        "Center_Entries": int(center_entries),
    }


def save_open_field_session_summary(
    output_path: Path,
    time_series: list[float],
    speed_series: list[float],
    zone_series: list[str],
    unit: str,
) -> None:
    if not time_series:
        return
    sns.set_theme(style="ticks", font_scale=1.0)
    smoothed_speed = pd.Series(speed_series).rolling(window=30, center=True, min_periods=1).mean()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(time_series, smoothed_speed, color="#212121", linewidth=1.5)
    axes[0].fill_between(time_series, smoothed_speed, color="#BDBDBD", alpha=0.5)
    axes[0].set_ylabel(f"Speed\n({unit}/s)", fontweight="bold")
    axes[0].set_title("Open Field: Speed and Zone Timeline", fontweight="bold")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)

    zone_numeric = [1 if z == "Center" else 0 for z in zone_series]
    axes[1].fill_between(time_series, zone_numeric, step="pre", alpha=0.65, color="#FFD54F")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Periphery", "Center"])
    axes[1].set_xlabel("Time (s)", fontweight="bold")
    axes[1].set_ylabel("Zone", fontweight="bold")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_open_field_cohort(df: pd.DataFrame, output_dir: Path, unit: str) -> None:
    if df.empty:
        return
    df = df.copy()
    df["DayGroup"] = df["Group"].map(group_to_day)
    day_order = [d for d in ["Day 0", "Day 1", "Day 3", "Day 7", "Day 11", "Day 14"] if d in set(df["DayGroup"])]
    if not day_order:
        day_order = sorted(df["DayGroup"].unique())

    sns.set_theme(style="ticks", font_scale=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.barplot(data=df, x="DayGroup", y=f"Total_Distance_{unit}", order=day_order, errorbar=("se", 1), ax=axes[0], color="#90CAF9")
    sns.stripplot(data=df, x="DayGroup", y=f"Total_Distance_{unit}", order=day_order, ax=axes[0], color="#D32F2F", size=6, jitter=True)
    axes[0].set_title("Total Distance", fontweight="bold")
    axes[0].set_ylabel(unit)

    sns.barplot(data=df, x="DayGroup", y=f"Mean_Speed_{unit}_per_s", order=day_order, errorbar=("se", 1), ax=axes[1], color="#A5D6A7")
    sns.stripplot(data=df, x="DayGroup", y=f"Mean_Speed_{unit}_per_s", order=day_order, ax=axes[1], color="#1E88E5", size=6, jitter=True)
    axes[1].set_title("Mean Speed", fontweight="bold")
    axes[1].set_ylabel(f"{unit}/s")

    sns.barplot(data=df, x="DayGroup", y="Center_Ratio", order=day_order, errorbar=("se", 1), ax=axes[2], color="#FFE082")
    sns.stripplot(data=df, x="DayGroup", y="Center_Ratio", order=day_order, ax=axes[2], color="#6A1B9A", size=6, jitter=True)
    axes[2].set_title("Center Occupancy", fontweight="bold")
    axes[2].set_ylabel("ratio")

    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(output_dir / "OFT_Cohort_Summary.png", dpi=300)
    plt.close(fig)


def run_open_field(
    base_dir: str,
    skip_seconds: float = 5.0,
    analyze_seconds: Optional[float] = 300.0,
    pixel_to_cm: Optional[float] = None,
    center_ratio: float = 0.5,
) -> pd.DataFrame:
    base_path = Path(base_dir)
    output_dir = ensure_dir(base_path)
    config = TrackingConfig(skip_seconds=skip_seconds, analyze_seconds=analyze_seconds, pixel_to_cm=pixel_to_cm)
    videos = find_videos(base_path, "旷场")
    if not videos:
        raise FileNotFoundError(f"No open-field video containing '旷场' was found under: {base_dir}")

    results: list[dict] = []
    for video_path in videos:
        roi_candidates = [video_path.with_name(f"{video_path.stem}_ROI_debug.jpg")]
        roi_candidates.extend(sorted(video_path.parent.glob("*_ROI_debug.jpg")))

        bg = extract_background(video_path, frame_count=5, skip_seconds=skip_seconds, grayscale=True)
        if bg is None:
            continue

        roi_mask = None
        roi_path = None
        for candidate in roi_candidates:
            roi_mask = load_blue_mask(candidate, bg.shape, close_kernel=5)
            if roi_mask is not None:
                roi_path = candidate
                break
        if roi_mask is None:
            continue

        center_mask = build_center_mask(roi_mask, center_ratio=center_ratio)
        save_open_field_zone_debug(video_path.parent / f"{video_path.stem}_Zone_Debug.jpg", roi_mask, center_mask)
        data = process_open_field_video(video_path, roi_mask, center_mask, video_path.parent, config)
        if data is not None:
            data["ROI_File"] = roi_path.name if roi_path else ""
            results.append(data)

    if not results:
        raise RuntimeError("No open-field video was successfully decoded.")

    df = pd.DataFrame(results)
    csv_path = output_dir / "OFT_Results.csv"
    save_dataframe(df, csv_path)
    plot_open_field_cohort(df, output_dir, config.unit)
    return df
