import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import tempfile
import zipfile
import re
from math import pi
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 页面配置
st.set_page_config(page_title="Y-Maze Tracker AI", layout="wide")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

# ==========================================
# 核心算法模块 (从 y_maze_tracker.py 移植)
# ==========================================

def extract_background(video_path, frame_count=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    start_frame = int(5 * fps)
    
    frame_ids = np.linspace(start_frame, total_frames - 1, frame_count, dtype=int)
    frames = []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if ret: frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.median(frames, axis=0).astype(dtype=np.uint8) if frames else None

def extract_raw_blue_mask(img_path, bg_shape):
    img = cv2.imread(img_path)
    if img is None: return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([140, 255, 255]))
    if cv2.countNonZero(blue_mask) < 100: return None
    blue_mask = cv2.resize(blue_mask, (bg_shape[1], bg_shape[0]), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    return blue_mask

def auto_zone_ymaze(blue_mask, output_dir, base_name):
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(main_contour, returnPoints=False)
    defects = cv2.convexityDefects(main_contour, hull)
    dist_transform = cv2.distanceTransform(blue_mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)

    center_mask = np.zeros_like(blue_mask)
    zone_masks = {}
    defect_pts = []
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            defect_pts.append((d, tuple(main_contour[f][0])))
            
    defect_pts.sort(key=lambda x: x[0], reverse=True)
    if len(defect_pts) >= 3:
        p1, p2, p3 = defect_pts[0][1], defect_pts[1][1], defect_pts[2][1]
        triangle_pts = np.array([p1, p2, p3], dtype=np.int32)
        cv2.fillPoly(center_mask, [triangle_pts], 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        center_mask = cv2.dilate(center_mask, kernel, iterations=1)
        center_mask = cv2.bitwise_and(center_mask, blue_mask)
    else:
        cv2.circle(center_mask, max_loc, int(max_val * 1.2), 255, -1)
        center_mask = cv2.bitwise_and(center_mask, blue_mask)

    zone_masks['Center'] = center_mask

    arms_mask = cv2.subtract(blue_mask, center_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(arms_mask, connectivity=8)
    arm_candidates = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 500]
    valid_arms = sorted(arm_candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:3]
    
    arms_info = []
    for label in valid_arms:
        arm = np.zeros_like(blue_mask)
        arm[labels == label] = 255
        M = cv2.moments(arm)
        if M['m00'] > 0:
            cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
            angle = np.arctan2(cy - max_loc[1], cx - max_loc[0])
            arms_info.append((angle, arm))
            
    arms_info.sort(key=lambda x: x[0])
    for idx, (angle, a_mask) in enumerate(arms_info):
        zone_masks[f'Arm {idx+1}'] = a_mask

    debug_img = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
    debug_img[center_mask == 255] = [0, 255, 255]
    colors = [[0, 255, 0], [255, 0, 0], [255, 0, 255]] 
    for idx in range(1, 4):
        arm_name = f'Arm {idx}'
        if arm_name in zone_masks:
            mask = zone_masks[arm_name]
            debug_img[mask == 255] = colors[(idx-1) % 3]
            M = cv2.moments(mask)
            if M['m00'] > 0:
                cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                cv2.putText(debug_img, arm_name, (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
    cv2.putText(debug_img, 'Center', (max_loc[0]-35, max_loc[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_Zone_Debug.jpg"), debug_img)
    return zone_masks

def draw_vector_trajectory(x_coords, y_coords, blue_mask, bg_shape, base_name, output_dir):
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(main_contour, True)
    smooth_contour = cv2.approxPolyDP(main_contour, epsilon, True)
    cnt_x = np.append(smooth_contour[:, 0, 0], smooth_contour[0, 0, 0])
    cnt_y = np.append(smooth_contour[:, 0, 1], smooth_contour[0, 0, 1])

    plt.figure(figsize=(8, 8), facecolor='white')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.plot(cnt_x, cnt_y, color='#1976D2', linewidth=3.5, zorder=1)
    plt.plot(x_coords, y_coords, color='#E53935', linewidth=1.2, alpha=0.9, zorder=2)
    plt.scatter(x_coords[0], y_coords[0], facecolors='white', edgecolors='#1E3A8A', s=80, linewidths=2.5, zorder=3)
    plt.scatter(x_coords[-1], y_coords[-1], color='#1E3A8A', s=80, zorder=3)
             
    plt.ylim(bg_shape[0], 0); plt.xlim(0, bg_shape[1]); plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{base_name}_trajectory.jpg"), dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()

def get_bouts(t_series, val_series):
    bouts = {}
    if not len(t_series): return bouts
    current_val, start_t = val_series[0], t_series[0]
    for i in range(1, len(t_series)):
        if val_series[i] != current_val:
            if current_val not in bouts: bouts[current_val] = []
            bouts[current_val].append((start_t, t_series[i] - start_t))
            current_val, start_t = val_series[i], t_series[i]
    if current_val not in bouts: bouts[current_val] = []
    bouts[current_val].append((start_t, t_series[-1] - start_t))
    return bouts

def process_ymaze_video(video_path, blue_mask, zone_masks, output_dir, analyze_seconds=None):
    bg_gray = extract_background(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30 
    
    skip_frames = int(5 * fps)
    max_frames = int(analyze_seconds * fps) if analyze_seconds else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - skip_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)

    x_coords, y_coords = [], []
    last_center = None
    frames_processed = 0
    
    zones = ['Center', 'Arm 1', 'Arm 2', 'Arm 3']
    stats = {zone: {'distance': 0.0, 'duration': 0.0, 'entries': 0} for zone in zones}
    current_zone = 'Center'
    time_series, speed_series, zone_series = [], [], []
    arm_sequence = [] 

    while frames_processed < max_frames:
        ret, frame = cap.read()
        if not ret: break
            
        diff = cv2.subtract(bg_gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(thresh, thresh, mask=blue_mask)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cX, cY = None, None
        if contours:
            valid_contours = [c for c in contours if 50 < cv2.contourArea(c) < 8000]
            if valid_contours:
                M = cv2.moments(max(valid_contours, key=cv2.contourArea))
                if M["m00"] != 0: cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        dist = 0
        if cX is not None:
            if last_center: dist = np.sqrt((cX - last_center[0])**2 + (cY - last_center[1])**2)
            last_center = (cX, cY)
        elif last_center:
            cX, cY = last_center[0], last_center[1]
        
        if cX is not None:
            x_coords.append(cX); y_coords.append(cY)
            point_zone = current_zone 
            for z_name in zones:
                if z_name in zone_masks and zone_masks[z_name][cY, cX] == 255:
                    point_zone = z_name
                    break
                    
            if point_zone in stats:
                stats[point_zone]['duration'] += 1.0 / fps
                stats[point_zone]['distance'] += dist
                if point_zone != current_zone:
                    stats[point_zone]['entries'] += 1
                    if 'Arm' in point_zone:
                        if not arm_sequence or arm_sequence[-1] != point_zone:
                            arm_sequence.append(point_zone)
                    current_zone = point_zone
            
            time_series.append(frames_processed / fps)
            speed_series.append(dist * fps)
            zone_series.append(current_zone)

        frames_processed += 1
    cap.release()

    if not x_coords: return None
    base_name = os.path.basename(video_path).replace('.mp4', '')
    
    draw_vector_trajectory(x_coords, y_coords, blue_mask, bg_gray.shape, base_name, output_dir)

    masked_bg = bg_gray.copy()
    masked_bg[blue_mask == 0] = 255 
    heatmap_data, _, _ = np.histogram2d(y_coords, x_coords, bins=(bg_gray.shape[0]//5, bg_gray.shape[1]//5), range=[[0, bg_gray.shape[0]], [0, bg_gray.shape[1]]])
    plt.figure(figsize=(8, 8))
    plt.imshow(masked_bg, cmap='gray')
    plt.imshow(cv2.resize(gaussian_filter(heatmap_data, sigma=3), (bg_gray.shape[1], bg_gray.shape[0])), cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{base_name}_heatmap.jpg"), dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    
    plot_ymaze_session_charts(stats, base_name, output_dir)
    state_bouts = plot_spike_visualization(time_series, speed_series, zone_series, base_name, output_dir, fps)
    
    return stats, arm_sequence, state_bouts

def plot_ymaze_session_charts(stats, base_name, output_dir):
    zones = ['Center', 'Arm 1', 'Arm 2', 'Arm 3']
    distances = [stats[z]['distance'] for z in zones]
    durations = [stats[z]['duration'] for z in zones]
    entries = [stats[z]['entries'] for z in zones]
    colors = ['#FFD54F', '#81C784', '#64B5F6', '#BA68C8'] 
    
    sns.set_theme(style="ticks", font_scale=1.1, font="sans-serif")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.barplot(x=zones, y=distances, ax=axes[0], color='#90CAF9', edgecolor='#1565C0', linewidth=1.5)
    axes[0].set_title('Distance', fontweight='bold')
    sns.barplot(x=zones, y=durations, ax=axes[1], color='#A5D6A7', edgecolor='#2E7D32', linewidth=1.5)
    axes[1].set_title('Duration', fontweight='bold')
    sns.barplot(x=zones, y=entries, ax=axes[2], color='#FFAB91', edgecolor='#D84315', linewidth=1.5)
    axes[2].set_title('Entries', fontweight='bold')
    for ax in axes: ax.set_xlabel(''); sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_DataCharts.png"), dpi=300)
    plt.close()

def plot_spike_visualization(time_series, speed_series, zone_series, base_name, output_dir, fps):
    window_size = int(fps) if fps else 30
    smoothed_speed = pd.Series(speed_series).rolling(window=window_size, min_periods=1, center=True).mean().values
    
    state_series = []
    for s in smoothed_speed:
        if s < 2.0: state_series.append('Freezing')
        elif s < 15.0: state_series.append('Immobility')
        else: state_series.append('Mobility')

    zone_bouts = get_bouts(time_series, zone_series)
    state_bouts = get_bouts(time_series, state_series)
            
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1.5, 1.5]})
    sns.set_theme(style="white", font_scale=1.1, font="sans-serif")
    
    axes[0].plot(time_series, smoothed_speed, color='#212121', linewidth=1.5)
    axes[0].fill_between(time_series, smoothed_speed, color='#BDBDBD', alpha=0.5)
    axes[0].set_ylabel('Speed', fontweight='bold')
    axes[0].set_title('Behavioral Events Visualization', fontweight='bold', pad=15)
    
    zone_order, zone_colors = ['Center', 'Arm 1', 'Arm 2', 'Arm 3'], {'Center': '#FFB300', 'Arm 1': '#43A047', 'Arm 2': '#1E88E5', 'Arm 3': '#8E24AA'}
    for i, z in enumerate(zone_order):
        if z in zone_bouts: axes[1].broken_barh(zone_bouts[z], (i - 0.3, 0.6), facecolors=zone_colors[z], edgecolor='none')
    axes[1].set_yticks(range(len(zone_order))); axes[1].set_yticklabels(zone_order); axes[1].set_ylabel('Zone', fontweight='bold')
    
    state_order, state_colors = ['Freezing', 'Immobility', 'Mobility'], {'Freezing': '#1E88E5', 'Immobility': '#FDD835', 'Mobility': '#E53935'}
    for i, s in enumerate(state_order):
        if s in state_bouts: axes[2].broken_barh(state_bouts[s], (i - 0.3, 0.6), facecolors=state_colors[s], edgecolor='none')
    axes[2].set_yticks(range(len(state_order))); axes[2].set_yticklabels(state_order); axes[2].set_ylabel('State', fontweight='bold')
    
    for ax in axes: sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_Visualization.png"), dpi=300)
    plt.close()
    return state_bouts

def smart_group_mapping(video_name):
    v_name = str(video_name)
    day = 'Day 0'
    if '前' in v_name or '光栓' in v_name or 'Day 0' in v_name: day = 'Day 0'
    elif '14' in v_name: day = 'Day 14'
    elif '11' in v_name: day = 'Day 11'
    elif '7' in v_name: day = 'Day 7'
    elif '3' in v_name: day = 'Day 3'
    elif '1' in v_name: day = 'Day 1'
    
    match = re.search(r'[-_](\d{1,2})', v_name)
    if match: return f"{day}-{match.group(1).zfill(2)}"
    return day

def calc_sap(arm_sequence):
    if len(arm_sequence) < 3: return 0.0
    alternations = sum(1 for i in range(len(arm_sequence)-2) if len(set(arm_sequence[i:i+3])) == 3)
    return (alternations / (len(arm_sequence) - 2)) * 100

def calc_transition_matrix(arm_sequence):
    matrix = np.zeros((3, 3))
    arm_to_idx = {'Arm 1': 0, 'Arm 2': 1, 'Arm 3': 2}
    valid_seq = [arm_to_idx[arm] for arm in arm_sequence if arm in arm_to_idx]
    if len(valid_seq) > 1:
        for i in range(len(valid_seq) - 1): matrix[valid_seq[i], valid_seq[i+1]] += 1
    row_sums = matrix.sum(axis=1)
    for i in range(3):
        if row_sums[i] > 0: matrix[i] /= row_sums[i]
    return matrix

def plot_advanced_cohort_analysis(results, output_dir):
    df = pd.DataFrame(results)
    if df.empty: return
    existing_order = sorted(df['Group'].unique())
    sns.set_theme(style="ticks", font_scale=1.1, font="sans-serif")

    # Radar Chart
    categories = ['SAP (%)', 'Speed', 'Center Time', 'Total Distance']
    scaler = MinMaxScaler()
    radar_data = df[['SAP', 'Mean_Speed', 'Center_Time', 'Total_Distance']].fillna(0)
    if len(radar_data) > 0:
        scaled_data = scaler.fit_transform(radar_data)
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]
        ax.set_theta_offset(pi / 2); ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories, size=12, fontweight='bold')
        ax.set_rlabel_position(0); plt.yticks([0.25, 0.5, 0.75, 1.0], ["25", "50", "75", "100"], color="grey", size=10)
        plt.ylim(0, 1)
        colors = sns.color_palette("Set1", len(existing_order))
        for i, group in enumerate(existing_order):
            group_idx = df[df['Group'] == group].index
            if len(group_idx) > 0:
                values = scaled_data[group_idx].mean(axis=0).tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=group, color=colors[i])
                ax.fill(angles, values, color=colors[i], alpha=0.1)
        plt.title('Behavioral Fingerprint (Radar Chart)', size=15, fontweight='bold', pad=30)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Cohort_Radar_Clustering.png"), dpi=300)
        plt.close()

    # PCA Clustering
    pca_data = []
    for r in results:
        features = {'Group': r['Group'], 'DayGroup': r['DayGroup'], 'Total_Distance': r['Total_Distance'], 'SAP': r['SAP']}
        total_time = 0
        state_times = {'Freezing': 0, 'Immobility': 0, 'Mobility': 0}
        for state, bouts in r['State_Bouts'].items():
            state_times[state] = sum(duration for start, duration in bouts)
            total_time += state_times[state]
        for state in state_times: features[f'{state}_Ratio'] = state_times[state] / total_time if total_time > 0 else 0
        pca_data.append(features)
        
    df_pca = pd.DataFrame(pca_data)
    if len(df_pca) > 2:
        feature_cols = ['Total_Distance', 'SAP', 'Freezing_Ratio', 'Mobility_Ratio']
        X = df_pca[feature_cols].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_scaled)
        df_pca['PC1'], df_pca['PC2'] = pcs[:, 0], pcs[:, 1]
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='PC1', y='PC2', hue='DayGroup', style='DayGroup', data=df_pca, s=150, palette='Set1', edgecolor='black')
        for i in range(len(df_pca)): plt.text(df_pca['PC1'][i] + 0.1, df_pca['PC2'][i], df_pca['Group'][i], fontsize=9)
        plt.title('PCA of Behavioral Metrics', fontweight='bold')
        plt.xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        sns.despine(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Cohort_PCA_Clustering.png"), dpi=300)
        plt.close()

# ==========================================
# Streamlit Web UI 前端逻辑
# ==========================================

st.title("?? Y-Maze Behavioral Tracker & Cohort Analysis")
st.markdown("自动提取 Y 迷宫运动轨迹、热图、自发交替率 (SAP)、Spikes 行为光谱，并执行多维度 PCA 与雷达聚类。")

st.sidebar.header("?? Data Upload (上传数据)")
st.sidebar.info("请上传原视频 (.mp4) 与你手涂的蓝色蒙版图 (_ROI_debug.jpg)。系统会通过名字自动将视频与遮罩匹配。")

video_files = st.sidebar.file_uploader("上传视频文件", type=['mp4', 'avi'], accept_multiple_files=True)
mask_files = st.sidebar.file_uploader("上传蓝色掩码图", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

analyze_secs = st.sidebar.number_input("截取分析时长 (秒, 0代表分析全长)", min_value=0, value=300, step=10)
analyze_secs = analyze_secs if analyze_secs > 0 else None

if st.sidebar.button("?? 运行高级解码与聚类", type="primary"):
    if not video_files or not mask_files:
        st.warning("请同时上传至少一个视频和对应的蓝色掩码图片！")
    else:
        # 创建临时工作目录
        temp_dir = tempfile.mkdtemp()
        all_results = []
        
        progress_bar = st.progress(0)
        
        for i, video_file in enumerate(video_files):
            base_name = video_file.name.rsplit('.', 1)[0]
            st.subheader(f"正在处理: {base_name}")
            
            # 寻找匹配的 Mask
            matched_mask = next((m for m in mask_files if base_name in m.name), None)
            if not matched_mask:
                st.error(f"未找到与 {base_name} 匹配的掩码图，跳过...")
                continue
                
            # 保存到临时目录
            v_path = os.path.join(temp_dir, video_file.name)
            with open(v_path, "wb") as f: f.write(video_file.getbuffer())
            
            m_path = os.path.join(temp_dir, matched_mask.name)
            with open(m_path, "wb") as f: f.write(matched_mask.getbuffer())
            
            # 运行引擎
            with st.spinner('正在进行抗锯齿分割、轨迹提取与事件生成...'):
                bg_img = extract_background(v_path, frame_count=5)
                blue_mask = extract_raw_blue_mask(m_path, bg_img.shape)
                
                if blue_mask is not None:
                    zone_masks = auto_zone_ymaze(blue_mask, temp_dir, base_name)
                    stats, arm_sequence, state_bouts = process_ymaze_video(v_path, blue_mask, zone_masks, temp_dir, analyze_secs)
                    
                    total_dist = sum([s['distance'] for s in stats.values()])
                    total_time = sum([s['duration'] for s in stats.values()])
                    sap_score = calc_sap(arm_sequence)
                    trans_matrix = calc_transition_matrix(arm_sequence)
                    subgroup_name = smart_group_mapping(base_name)
                    
                    all_results.append({
                        'Group': subgroup_name,
                        'DayGroup': subgroup_name.split('-')[0] if '-' in subgroup_name else subgroup_name,
                        'Video': base_name,
                        'Total_Distance': total_dist,
                        'Mean_Speed': total_dist / total_time if total_time > 0 else 0,
                        'Center_Time': stats['Center']['duration'],
                        'SAP': sap_score,
                        'Zone_Stats': {z: stats[z]['duration'] for z in ['Center', 'Arm 1', 'Arm 2', 'Arm 3']},
                        'State_Bouts': state_bouts,
                        'Transition_Matrix': trans_matrix
                    })
                    
                    # 渲染当前处理的结果图片
                    col1, col2, col3 = st.columns(3)
                    with col1: st.image(os.path.join(temp_dir, f"{base_name}_trajectory_Shapes.jpg"), caption="Vector Trajectory")
                    with col2: st.image(os.path.join(temp_dir, f"{base_name}_heatmap.jpg"), caption="Heatmap")
                    with col3: st.image(os.path.join(temp_dir, f"{base_name}_Zone_Debug.jpg"), caption="Zone Mask")
                    
                    st.image(os.path.join(temp_dir, f"{base_name}_Visualization.png"), caption="Spike Event Visualization", use_container_width=True)
                else:
                    st.error("蓝色掩码图读取失败。")
            
            progress_bar.progress((i + 1) / len(video_files))
            
        # 所有视频处理完毕，进行高阶 Cohort 分析
        if all_results:
            st.success("全部视频解码完成！开始生成高阶特征聚类图表...")
            plot_advanced_cohort_analysis(all_results, temp_dir)
            
            # 展示高级图表
            st.markdown("### ?? Advanced Cohort & Subgroup Analysis")
            cohort_cols = st.columns(2)
            if os.path.exists(os.path.join(temp_dir, "Cohort_Radar_Clustering.png")):
                with cohort_cols[0]: st.image(os.path.join(temp_dir, "Cohort_Radar_Clustering.png"), caption="Behavioral Radar Fingerprint")
            if os.path.exists(os.path.join(temp_dir, "Cohort_PCA_Clustering.png")):
                with cohort_cols[1]: st.image(os.path.join(temp_dir, "Cohort_PCA_Clustering.png"), caption="PCA Trajectory Clustering")
            
            # 导出 CSV
            csv_path = os.path.join(temp_dir, "YMaze_Results.csv")
            df_export = pd.DataFrame(all_results).drop(columns=['Zone_Stats', 'Transition_Matrix', 'State_Bouts'])
            df_export.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            # 打包所有文件为 ZIP
            zip_path = os.path.join(temp_dir, "YMaze_Analysis_Outputs.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith(('.jpg', '.png', '.csv')) and file != "YMaze_Analysis_Outputs.zip":
                            zipf.write(os.path.join(root, file), file)
                            
            # 提供下载按钮
            with open(zip_path, "rb") as fp:
                st.download_button(
                    label="?? 一键下载所有数据和图表 (ZIP)",
                    data=fp,
                    file_name="YMaze_Analysis_Outputs.zip",
                    mime="application/zip",
                    type="primary"
                )