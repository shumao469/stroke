import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import io

# ==========================================
# 1. 核心算法 (移植自 XFastsort2)
# ==========================================

def bandpass(x, fs, low, high, order=3):
    nyq = fs * 0.5
    low = max(low, 0.01)
    high = min(high, nyq * 0.999)
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def detect_spikes_mad(x_spk, fs, thr_k=4.5, refractory_ms=1.5):
    """MAD-based 阈值检测"""
    med = np.median(x_spk)
    mad = np.median(np.abs(x_spk - med)) + 1e-12
    thr = med - thr_k * mad  # 负向阈值
    
    # 找到低于阈值的点
    idx = np.where(x_spk < thr)[0]
    if len(idx) == 0:
        return np.array([], dtype=np.int64), thr
    
    # 不应期处理 (Refractory period)
    refractory = int(refractory_ms * 1e-3 * fs)
    clean = []
    if len(idx) > 0:
        clean = [idx[0]]
        for i in idx[1:]:
            if i - clean[-1] > refractory:
                clean.append(i)
                
    return np.array(clean, dtype=np.int64), thr

def extract_waveforms(x_spk, spike_idx, fs, win_ms=1.0):
    """提取波形窗口"""
    win = int(win_ms * 1e-3 * fs)  # ±win
    waves = []
    valid = []
    T = len(x_spk)
    
    for s in spike_idx:
        if s - win >= 0 and s + win < T:
            waves.append(x_spk[s-win:s+win])
            valid.append(s)
            
    if len(waves) == 0:
        return np.zeros((0, 2*win), dtype=np.float32), np.array([], dtype=np.int64)
    return np.asarray(waves, dtype=np.float32), np.asarray(valid, dtype=np.int64)

def cluster_waveforms(waves, n_clusters=3, random_state=42):
    """PCA + KMeans 聚类"""
    if waves.shape[0] < n_clusters * 5:
        return np.zeros(waves.shape[0], dtype=int)
    
    # 1. PCA 降维
    pca = PCA(n_components=3, random_state=random_state)
    X_pca = pca.fit_transform(waves)
    
    # 2. KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X_pca)
    return labels

def calculate_qc(waves, spike_times, fs, min_isi_ms=1.5):
    """计算 SNR 和 ISI Violation"""
    if len(waves) == 0:
        return 0.0, 0.0
    
    # SNR
    mean_wf = waves.mean(0)
    p2p = mean_wf.max() - mean_wf.min()
    # 噪声估计: 减去平均波形后的残差的标准差
    noise = np.median(np.std(waves - mean_wf, axis=0)) + 1e-12
    snr = p2p / (2 * noise) # 近似公式
    
    # ISI
    isi = np.diff(np.sort(spike_times))
    isi_violation = 0.0
    if len(isi) > 0:
        violation_count = (isi < (min_isi_ms * 1e-3)).sum()
        isi_violation = (violation_count / len(isi)) * 100.0
        
    return snr, isi_violation

# ==========================================
# 2. Streamlit 界面逻辑
# ==========================================

st.set_page_config(page_title="XFastsort2 Dashboard", layout="wide", page_icon="⚡")

st.title("⚡ XFastsort2 Spike Sorting Dashboard")
st.markdown("基于 `CPU baseline pipeline` 的轻量级神经电生理信号分选工具。")

# --- 侧边栏：参数配置 ---
st.sidebar.header("1. Data & Config")

# 文件上传
uploaded_file = st.sidebar.file_uploader("Upload Trace (.csv/.txt)", type=["csv", "txt"])

# 参数设置
st.sidebar.subheader("Parameters")
fs = st.sidebar.number_input("Sampling Rate (Hz)", value=20000.0, step=1000.0)
bp_low = st.sidebar.number_input("Bandpass Low (Hz)", value=300.0)
bp_high = st.sidebar.number_input("Bandpass High (Hz)", value=6000.0)
threshold_k = st.sidebar.slider("Threshold (MAD k)", 3.0, 10.0, 4.5, 0.1)
n_clusters = st.sidebar.slider("Num Clusters (KMeans)", 1, 10, 3)

# 运行按钮
run_btn = st.sidebar.button("Run Sorting", type="primary")

# --- 主界面逻辑 ---

if uploaded_file is not None:
    # 缓存数据加载以提高性能
    @st.cache_data
    def load_data(file):
        # 尝试自动解析
        try:
            df = pd.read_csv(file, header=None, engine='python')
            # 简单的两列检测 logic
            if df.shape[1] < 2:
                st.error("Data must have at least 2 columns (Time, Value)")
                return None, None
            
            # 如果第一行是字符，去掉
            if isinstance(df.iloc[0,0], str):
                try:
                    float(df.iloc[0,0])
                except:
                    df = df.iloc[1:]
            
            t = df.iloc[:,0].astype(float).values
            x = df.iloc[:,1].astype(float).values
            return t, x
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None, None

    t, x_raw = load_data(uploaded_file)

    if t is not None:
        st.info(f"Loaded data: {len(t)} samples, {t[-1]-t[0]:.2f} seconds duration.")
        
        # 预览原始数据
        with st.expander("Data Preview (Raw Trace)", expanded=True):
            fig_raw, ax_raw = plt.subplots(figsize=(12, 2))
            # 只画前 1 秒或 20000 个点用于预览
            limit = int(fs) if len(t) > fs else len(t)
            ax_raw.plot(t[:limit], x_raw[:limit], lw=0.5, color='gray')
            ax_raw.set_title("Raw Signal (First 1s)")
            ax_raw.set_xlabel("Time (s)")
            ax_raw.set_ylabel("Amplitude")
            st.pyplot(fig_raw)

        # 点击运行后执行
        if run_btn:
            with st.spinner("Processing... Filtering > Detection > Clustering"):
                # 1. 滤波
                x_filt = bandpass(x_raw, fs, bp_low, bp_high)
                
                # 2. 检测
                spike_idx, thr = detect_spikes_mad(x_filt, fs, thr_k=threshold_k)
                st.write(f"Detected **{len(spike_idx)}** spikes. Threshold: {thr:.2f}")
                
                if len(spike_idx) > 10:
                    # 3. 提取波形
                    waves, valid_idx = extract_waveforms(x_filt, spike_idx, fs)
                    spike_times = t[valid_idx]
                    
                    # 4. 聚类
                    labels = cluster_waveforms(waves, n_clusters=n_clusters)
                    
                    # 5. 整理结果 & QC
                    results = []
                    unique_labels = np.sort(np.unique(labels))
                    
                    # 创建布局列
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("Mean Waveforms")
                        fig_wave, ax_wave = plt.subplots(figsize=(8, 5))
                        
                    with col2:
                        st.subheader("Cluster Quality")
                    
                    qc_data = []
                    
                    for cid in unique_labels:
                        mask = labels == cid
                        c_waves = waves[mask]
                        c_times = spike_times[mask]
                        
                        # 计算 Mean & Std
                        mean_w = c_waves.mean(0)
                        std_w = c_waves.std(0)
                        time_axis = np.arange(len(mean_w)) / fs * 1000  # ms
                        
                        # 绘制波形
                        color = plt.cm.tab10(cid % 10)
                        ax_wave.plot(time_axis, mean_w, color=color, lw=2, label=f"C{cid} (n={len(c_waves)})")
                        ax_wave.fill_between(time_axis, mean_w-std_w, mean_w+std_w, color=color, alpha=0.2)
                        
                        # 计算 QC
                        snr, isi_viol = calculate_qc(c_waves, c_times, fs)
                        qc_data.append({
                            "Cluster": f"C{cid}",
                            "Count": len(c_waves),
                            "SNR": f"{snr:.2f}",
                            "ISI Violf (%)": f"{isi_viol:.2f}"
                        })

                    ax_wave.legend()
                    ax_wave.set_xlabel("Time (ms)")
                    ax_wave.set_ylabel("Amplitude (uV)")
                    ax_wave.set_title(f"Mean Waveforms (k={n_clusters})")
                    col1.pyplot(fig_wave)
                    
                    # 显示 QC 表格
                    with col2:
                        st.dataframe(pd.DataFrame(qc_data), use_container_width=True)
                        
                    # --- Raster Plot ---
                    st.subheader("Raster Plot")
                    fig_raster, ax_raster = plt.subplots(figsize=(12, 3))
                    ax_raster.scatter(spike_times, labels, c=labels, cmap='tab10', s=10, alpha=0.6)
                    ax_raster.set_yticks(unique_labels)
                    ax_raster.set_ylabel("Cluster ID")
                    ax_raster.set_xlabel("Time (s)")
                    ax_raster.set_title("Spike Raster")
                    st.pyplot(fig_raster)
                    
                    # --- 下载数据 ---
                    st.divider()
                    st.subheader("Downloads")
                    
                    # 准备 CSV
                    df_res = pd.DataFrame({
                        "time_s": spike_times,
                        "cluster": labels
                    })
                    csv = df_res.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="Download Sorted Spikes (CSV)",
                        data=csv,
                        file_name="spikes_sorted.csv",
                        mime="text/csv",
                    )
                    
                else:
                    st.warning("Too few spikes detected. Try lowering the threshold.")

    else:
        st.write("Waiting for data upload...")

# 页脚
st.markdown("---")
st.caption("XFastsort2 Implementation | Python + Streamlit")