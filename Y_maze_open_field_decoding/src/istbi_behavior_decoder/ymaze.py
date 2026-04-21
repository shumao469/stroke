from __future__ import annotations

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
    load_blue_mask,
    pairwise_distance,
    save_dataframe,
    save_heatmap,
    save_red_trajectory,
    safe_fps,
)


def auto_zone_ymaze(blue_mask: np.ndarray) -> dict[str, np.ndarray]:
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No Y-maze contour found.")
    main_contour = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(main_contour, returnPoints=False)
    defects = cv2.convexityDefects(main_contour, hull) if hull is not None and len(hull) >= 3 else None
    dist_transform = cv2.distanceTransform(blue_mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)

    center_mask = np.zeros_like(blue_mask)
    defect_pts: list[tuple[float, tuple[int, int]]] = []
    if defects is not None:
        for i in range(defects.shape[0]):
            _, _, f, d = defects[i, 0]
            defect_pts.append((float(d), tuple(main_contour[f][0])))
    defect_pts.sort(key=lambda x: x[0], reverse=True)

    if len(defect_pts) >= 3:
        triangle_pts = np.array([defect_pts[0][1], defect_pts[1][1], defect_pts[2][1]], dtype=np.int32)
        cv2.fillPoly(center_mask, [triangle_pts], 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        center_mask = cv2.dilate(center_mask, kernel, iterations=1)
        center_mask = cv2.bitwise_and(center_mask, blue_mask)
    else:
        cv2.circle(center_mask, max_loc, int(max_val * 1.2), 255, -1)
        center_mask = cv2.bitwise_and(center_mask, blue_mask)

    zone_masks: dict[str, np.ndarray] = {"Center": center_mask}
    arms_mask = cv2.subtract(blue_mask, center_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(arms_mask, connectivity=8)
    arm_candidates = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 500]
    valid_arms = sorted(arm_candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:3]

    arms_info: list[tuple[float, np.ndarray]] = []
    for label in valid_arms:
        arm = np.zeros_like(blue_mask)
        arm[labels == label] = 255
        cx, cy = contour_centroid(arm)
        angle = np.arctan2(cy - max_loc[1], cx - max_loc[0])
        arms_info.append((angle, arm))

    arms_info.sort(key=lambda x: x[0])
    for idx, (_, arm_mask) in enumerate(arms_info, start=1):
        zone_masks[f"Arm {idx}"] = arm_mask
    return zone_masks


def save_ymaze_zone_debug(output_path: Path, zone_masks: dict[str, np.ndarray]) -> None:
    sample_mask = next(iter(zone_masks.values()))
    vis = np.zeros((sample_mask.shape[0], sample_mask.shape[1], 3), dtype=np.uint8)
    color_map = {
        "Center": (0, 255, 255),
        "Arm 1": (0, 255, 0),
        "Arm 2": (255, 0, 0),
        "Arm 3": (255, 0, 255),
    }
    for name, mask in zone_masks.items():
        vis[mask > 0] = color_map.get(name, (200, 200, 200))
        cx, cy = contour_centroid(mask)
        cv2.putText(vis, name, (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(output_path), vis)


def calc_sap(arm_sequence: list[str]) -> float:
    if len(arm_sequence) < 3:
        return 0.0
    alternations = sum(1 for i in range(len(arm_sequence) - 2) if len(set(arm_sequence[i : i + 3])) == 3)
    return (alternations / (len(arm_sequence) - 2)) * 100.0


def calc_transition_matrix(arm_sequence: list[str]) -> np.ndarray:
    arm_to_idx = {"Arm 1": 0, "Arm 2": 1, "Arm 3": 2}
    matrix = np.zeros((3, 3), dtype=float)
    valid_seq = [arm_to_idx[a] for a in arm_sequence if a in arm_to_idx]
    for curr, nxt in zip(valid_seq[:-1], valid_seq[1:]):
        matrix[curr, nxt] += 1.0
    row_sums = matrix.sum(axis=1)
    for i in range(3):
        if row_sums[i] > 0:
            matrix[i] /= row_sums[i]
    return matrix


def process_ymaze_video(
    video_path: Path,
    blue_mask: np.ndarray,
    zone_masks: dict[str, np.ndarray],
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
    start_frame = int(config.skip_seconds * fps)
    max_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if config.analyze_seconds is None:
        max_frames = max(0, max_total_frames - start_frame)
    else:
        max_frames = min(int(config.analyze_seconds * fps), max(0, max_total_frames - start_frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    zone_names = ["Center", "Arm 1", "Arm 2", "Arm 3"]
    stats = {z: {"distance": 0.0, "duration": 0.0, "entries": 0} for z in zone_names}

    points: list[tuple[int, int]] = []
    arm_sequence: list[str] = []
    time_series: list[float] = []
    speed_series: list[float] = []
    zone_series: list[str] = []
    last_point: Optional[tuple[int, int]] = None
    current_zone = "Center"
    frames_processed = 0

    while frames_processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        point = detect_mouse_centroid(
            bg_gray=bg_gray,
            frame_bgr=frame,
            roi_mask=blue_mask,
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
        point_zone = current_zone
        for name in zone_names:
            mask = zone_masks.get(name)
            if mask is not None and mask[y, x] > 0:
                point_zone = name
                break

        stats[point_zone]["duration"] += 1.0 / fps
        stats[point_zone]["distance"] += dist
        if point_zone != current_zone:
            stats[point_zone]["entries"] += 1
            if point_zone.startswith("Arm"):
                if not arm_sequence or arm_sequence[-1] != point_zone:
                    arm_sequence.append(point_zone)
            current_zone = point_zone

        time_series.append(frames_processed / fps)
        speed_series.append(dist * fps)
        zone_series.append(current_zone)
        frames_processed += 1

    cap.release()
    if not points:
        return None

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    base_name = video_path.stem
    unit = config.unit

    save_red_trajectory(output_dir / f"{base_name}_trajectory.jpg", x_coords, y_coords, blue_mask, bg_gray.shape)
    save_heatmap(output_dir / f"{base_name}_heatmap.jpg", x_coords, y_coords, bg_gray, blue_mask)
    save_ymaze_session_summary(output_dir / f"{base_name}_session_summary.png", time_series, speed_series, zone_series, unit)

    total_distance = pairwise_distance(points, pixel_to_unit=config.pixel_to_unit)
    duration_sec = len(points) / fps
    mean_speed = total_distance / duration_sec if duration_sec > 0 else 0.0
    sap = calc_sap(arm_sequence)
    transition_matrix = calc_transition_matrix(arm_sequence)

    result = {
        "Group": video_path.parent.name,
        "Subgroup": extract_subgroup(video_path.parent.name, video_path.stem),
        "Video": base_name,
        "Duration_sec": round(duration_sec, 4),
        f"Total_Distance_{unit}": round(total_distance, 4),
        f"Mean_Speed_{unit}_per_s": round(mean_speed, 4),
        "Center_Time_sec": round(stats["Center"]["duration"], 4),
        "Arm1_Time_sec": round(stats["Arm 1"]["duration"], 4),
        "Arm2_Time_sec": round(stats["Arm 2"]["duration"], 4),
        "Arm3_Time_sec": round(stats["Arm 3"]["duration"], 4),
        "Center_Entries": int(stats["Center"]["entries"]),
        "Arm1_Entries": int(stats["Arm 1"]["entries"]),
        "Arm2_Entries": int(stats["Arm 2"]["entries"]),
        "Arm3_Entries": int(stats["Arm 3"]["entries"]),
        "SAP_percent": round(sap, 4),
        "Arm_Sequence": " > ".join(arm_sequence),
        "Transition_Matrix": transition_matrix.tolist(),
    }
    return result


def save_ymaze_session_summary(
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

    zone_to_num = {"Center": 0, "Arm 1": 1, "Arm 2": 2, "Arm 3": 3}
    numeric_zone = [zone_to_num.get(z, 0) for z in zone_series]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(time_series, smoothed_speed, color="#212121", linewidth=1.5)
    axes[0].fill_between(time_series, smoothed_speed, color="#BDBDBD", alpha=0.5)
    axes[0].set_ylabel(f"Speed\n({unit}/s)", fontweight="bold")
    axes[0].set_title("Y-Maze: Speed and Zone Timeline", fontweight="bold")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[1].step(time_series, numeric_zone, where="post", color="#1E88E5", linewidth=1.5)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(["Center", "Arm 1", "Arm 2", "Arm 3"])
    axes[1].set_xlabel("Time (s)", fontweight="bold")
    axes[1].set_ylabel("Zone", fontweight="bold")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_ymaze_cohort(df: pd.DataFrame, output_dir: Path, unit: str) -> None:
    if df.empty:
        return
    df = df.copy()
    df["DayGroup"] = df["Group"].map(group_to_day)
    day_order = [d for d in ["Day 0", "Day 1", "Day 3", "Day 7", "Day 11", "Day 14"] if d in set(df["DayGroup"])]
    if not day_order:
        day_order = sorted(df["DayGroup"].unique())

    sns.set_theme(style="ticks", font_scale=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.barplot(data=df, x="DayGroup", y="SAP_percent", order=day_order, errorbar=("se", 1), ax=axes[0], color="#EF9A9A")
    sns.stripplot(data=df, x="DayGroup", y="SAP_percent", order=day_order, ax=axes[0], color="#D32F2F", size=6, jitter=True)
    axes[0].set_title("Spontaneous Alternation", fontweight="bold")
    axes[0].set_ylabel("SAP (%)")

    sns.barplot(data=df, x="DayGroup", y=f"Total_Distance_{unit}", order=day_order, errorbar=("se", 1), ax=axes[1], color="#90CAF9")
    sns.stripplot(data=df, x="DayGroup", y=f"Total_Distance_{unit}", order=day_order, ax=axes[1], color="#1E88E5", size=6, jitter=True)
    axes[1].set_title("Total Distance", fontweight="bold")
    axes[1].set_ylabel(unit)

    sns.barplot(data=df, x="DayGroup", y=f"Mean_Speed_{unit}_per_s", order=day_order, errorbar=("se", 1), ax=axes[2], color="#A5D6A7")
    sns.stripplot(data=df, x="DayGroup", y=f"Mean_Speed_{unit}_per_s", order=day_order, ax=axes[2], color="#6A1B9A", size=6, jitter=True)
    axes[2].set_title("Mean Speed", fontweight="bold")
    axes[2].set_ylabel(f"{unit}/s")

    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(output_dir / "YMaze_Cohort_Summary.png", dpi=300)
    plt.close(fig)


def run_ymaze(
    base_dir: str,
    skip_seconds: float = 5.0,
    analyze_seconds: Optional[float] = 300.0,
    pixel_to_cm: Optional[float] = None,
) -> pd.DataFrame:
    base_path = Path(base_dir)
    output_dir = ensure_dir(base_path)
    config = TrackingConfig(skip_seconds=skip_seconds, analyze_seconds=analyze_seconds, pixel_to_cm=pixel_to_cm)
    videos = find_videos(base_path, "Y迷宫")
    if not videos:
        raise FileNotFoundError(f"No Y-maze video containing 'Y迷宫' was found under: {base_dir}")

    results: list[dict] = []
    for video_path in videos:
        bg = extract_background(video_path, frame_count=5, skip_seconds=skip_seconds, grayscale=True)
        if bg is None:
            continue

        roi_candidates = [video_path.with_name(f"{video_path.stem}_YMaze_ROI_debug.jpg")]
        roi_candidates.extend(sorted(video_path.parent.glob("*_YMaze_ROI_debug.jpg")))
        blue_mask = None
        roi_path = None
        for candidate in roi_candidates:
            blue_mask = load_blue_mask(candidate, bg.shape, close_kernel=5)
            if blue_mask is not None:
                roi_path = candidate
                break
        if blue_mask is None:
            continue

        zone_masks = auto_zone_ymaze(blue_mask)
        save_ymaze_zone_debug(video_path.parent / f"{video_path.stem}_Zone_Debug.jpg", zone_masks)
        data = process_ymaze_video(video_path, blue_mask, zone_masks, video_path.parent, config)
        if data is not None:
            data["ROI_File"] = roi_path.name if roi_path else ""
            results.append(data)

    if not results:
        raise RuntimeError("No Y-maze video was successfully decoded.")

    df = pd.DataFrame(results)
    csv_path = output_dir / "YMaze_Results.csv"
    save_dataframe(df.drop(columns=["Transition_Matrix"]), csv_path)
    plot_ymaze_cohort(df, output_dir, config.unit)
    return df
