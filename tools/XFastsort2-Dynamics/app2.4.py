import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, welch, spectrogram, hilbert
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import io

# ==========================================
# 0. 配置与工具函数
# ==========================================

st.set_page_config(
    page_title="XFastsort2 Ultimate",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 美化
st.markdown("""
<style>
    .stApp {background-color: #0e1117;}
    .block-container {padding-top: 1rem;}
    h1, h2, h3 {color: #e0e0e0;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
    div[data-testid="stExpander"] div[role="button"] p {font-size: 1.1rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# 尝试导入可选库 (HMM / UMAP)
try:
    from hmmlearn import hmm
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ==========================================
# 1. 信号处理核心算法
# ==========================================

def generate_demo_data(duration_sec=30, fs=20000):
    """生成包含LFP、Spike和刺激伪迹的合成数据"""
    t = np.linspace(0, duration_sec, int(fs*duration_sec))
    # 1. 背景 LFP (Theta + Gamma) - 模拟状态切换
    lfp = 0.5 * np.sin(2 * np.pi * 6 * t)  # Theta
    # 前半段 Gamma 低，后半段 Gamma 高
    gamma_amp = np.linspace(0.1, 0.8, len(t))
    lfp += gamma_amp * np.sin(2 * np.pi * 40 * t)
    
    # 2. 噪声
    noise = np.random.normal(0, 0.2, len(t))
    
    # 3. Spikes (随机发放)
    spike_train = np.zeros_like(t)
    n_spikes = int(duration_sec * 15) # 15Hz firing rate
    spike_locs = np.random.choice(len(t)-100, n_spikes, replace=False)
    
    # 两种不同的 Spike 波形
    wf1 = np.array([0, -0.5, -2, -4, -1, 1, 0.5, 0]) * 3.0
    wf2 = np.array([0, 0.5, 1.5, -3, -5, -2, 1, 0]) * 2.5
    
    for i, loc in enumerate(spike_locs):
        wf = wf1 if i % 2 == 0 else wf2
        if loc + len(wf) < len(t):
            spike_train[loc:loc+len(wf)] += wf
            
    # 4. 刺激伪迹 (在中间几秒)
    stim = np.zeros_like(t)
    stim_start = int(len(t) * 0.4)
    stim_end = int(len(t) * 0.6)
    # 50Hz 干扰
    stim[stim_start:stim_end] = 30.0 * np.sin(2 * np.pi * 50 * t[stim_start:stim_end])
    
    x = lfp + noise + spike_train + stim
    return t, x

def bandpass(x, fs, low, high, order=3):
    nyq = fs * 0.5
    low = max(low, 0.01)
    high = min(high, nyq * 0.999)
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def detect_events_amplitude(x, fs, threshold_std=10.0):
    """自动检测高幅度事件（如刺激开始/结束）"""
    threshold = np.mean(x) + threshold_std * np.std(x)
    env = np.abs(x)
    is_event = env > threshold
    
    events = []
    if np.any(is_event):
        diff = np.diff(is_event.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) > 0 and len(ends) > 0:
            if starts[0] > ends[0]: ends = ends[1:]
            min_len = min(len(starts), len(ends))
            for i in range(min_len):
                dur = (ends[i] - starts[i]) / fs
                if dur > 0.05: # 忽略极短干扰
                    events.append((starts[i]/fs, ends[i]/fs))
    return events

def detect_spikes_mad(x_spk, fs, thr_k=4.5, refractory_ms=1.5):
    med = np.median(x_spk)
    mad = np.median(np.abs(x_spk - med)) + 1e-12
    thr = med - thr_k * mad
    idx = np.where(x_spk < thr)[0]
    if len(idx) == 0: return np.array([], dtype=np.int64), thr
    
    refractory = int(refractory_ms * 1e-3 * fs)
    clean = [idx[0]]
    for i in idx[1:]:
        if i - clean[-1] > refractory:
            clean.append(i)
    return np.array(clean, dtype=np.int64), thr

def extract_waveforms(x_spk, spike_idx, fs, win_ms=1.0):
    win = int(win_ms * 1e-3 * fs)
    waves, valid = [], []
    T = len(x_spk)
    for s in spike_idx:
        if s - win >= 0 and s + win < T:
            waves.append(x_spk[s-win:s+win])
            valid.append(s)
    if not waves: return np.zeros((0, 2*win)), np.array([])
    return np.asarray(waves), np.asarray(valid)

def compute_pac(x, fs, f_phase=(4, 12), f_amp=(30, 80), n_bins=18):
    """计算相位-幅度耦合 (MI)"""
    x_p = bandpass(x, fs, f_phase[0], f_phase[1])
    phase = np.angle(hilbert(x_p))
    
    x_a = bandpass(x, fs, f_amp[0], f_amp[1])
    amp = np.abs(hilbert(x_a))
    
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
        if np.any(mask):
            mean_amp[i] = np.mean(amp[mask])
    
    mean_amp /= np.sum(mean_amp)
    uniform_dist = np.ones(n_bins) / n_bins
    kl_div = np.sum(mean_amp * np.log((mean_amp + 1e-12) / (uniform_dist + 1e-12)))
    mi = kl_div / np.log(n_bins)
    return phase_bins, mean_amp, mi

# ==========================================
# 2. 状态管理
# ==========================================

if 'data' not in st.session_state:
    st.session_state['data'] = {'t': None, 'x': None, 'fs': 20000.0}
if 'events' not in st.session_state:
    st.session_state['events'] = []

# ==========================================
# 3. 侧边栏与数据加载
# ==========================================

with st.sidebar:
    st.title("🧠 XFastsort2")
    st.caption("Ultimate Electrophysiology Suite")
    
    module = st.radio(
        "Workflow",
        ["1. Data & Preprocess", 
         "2. Spike Sorting", 
         "3. PAC Analysis", 
         "4. Dynamics (t-SNE/HMM)"]
    )
    
    st.divider()
    
    input_source = st.radio("Data Source", ["📂 Upload File", "🎲 Load Demo Data"])
    
    if input_source == "🎲 Load Demo Data":
        if st.button("Generate Demo Data"):
            t, x = generate_demo_data(duration_sec=30, fs=20000)
            st.session_state['data'] = {'t': t, 'x': x, 'fs': 20000.0}
            st.session_state['events'] = []
            st.success("Demo Data Generated!")
            st.rerun()
            
    else:
        uploaded_file = st.file_uploader("Upload Trace (.csv/.txt)", type=["csv", "txt"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, header=None, engine='python')
                if isinstance(df.iloc[0,0], str):
                    df = pd.read_csv(uploaded_file)
                    t = df.iloc[:,0].values
                    x = df.iloc[:,1].values
                else:
                    t = df.iloc[:,0].astype(float).values
                    x = df.iloc[:,1].astype(float).values
                
                if 't' not in st.session_state['data'] or len(t) != len(st.session_state['data']['t']):
                    fs_est = 1 / np.median(np.diff(t))
                    st.session_state['data'] = {'t': t, 'x': x, 'fs': fs_est}
                    st.success(f"Loaded {len(t)} samples.")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state['data']['x'] is not None:
        fs_display = st.session_state['data']['fs']
        dur_display = st.session_state['data']['t'][-1]
        st.info(f"Signal Info:\nFS: {fs_display:.0f} Hz\nDur: {dur_display:.1f} s")

# ==========================================
# 4. 主界面模块
# ==========================================

if st.session_state['data']['x'] is None:
    st.warning("👈 Please upload data or load demo to begin.")
    st.stop()

t = st.session_state['data']['t']
x = st.session_state['data']['x']
fs = st.session_state['data']['fs']

# --- MODULE 1: Data & Preprocess ---
if module == "1. Data & Preprocess":
    st.header("🔍 Data Inspection & Event Detection")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Raw Signal Trace")
        fig, ax = plt.subplots(figsize=(10, 3))
        ds = int(fs / 2000) if fs > 2000 else 1
        ax.plot(t[::ds], x[::ds], 'k', lw=0.5, alpha=0.9, label='Raw LFP')
        
        for (start, end) in st.session_state['events']:
            ax.axvspan(start, end, color='red', alpha=0.2, label='Stimulus')
        
        # Unique legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label: ax.legend(by_label.values(), by_label.keys())
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.margins(x=0)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Auto-Detect")
        st.caption("Detect high-amplitude artifacts (e.g. Stim).")
        thr_std = st.number_input("Std Dev Threshold", 3.0, 50.0, 8.0)
        
        if st.button("Run Detection"):
            events = detect_events_amplitude(x, fs, threshold_std=thr_std)
            st.session_state['events'] = events
            st.success(f"Found {len(events)} events.")
            st.rerun()
        
        if st.button("Clear Events"):
            st.session_state['events'] = []
            st.rerun()
            
    if st.session_state['events']:
        st.markdown("#### Detected Events")
        st.dataframe(pd.DataFrame(st.session_state['events'], columns=["Start (s)", "End (s)"]), use_container_width=True)

# --- MODULE 2: Spike Sorting ---
elif module == "2. Spike Sorting":
    st.header("⚡ Spike Sorting Pipeline")
    
    col_param, col_viz = st.columns([1, 2])
    
    with col_param:
        with st.expander("Configuration", expanded=True):
            bp_low = st.number_input("BP Low (Hz)", 300, 1000, 300)
            bp_high = st.number_input("BP High (Hz)", 3000, 10000, 6000)
            thr_k = st.slider("Threshold (MAD)", 3.0, 10.0, 4.5)
            n_clusters = st.slider("Clusters (K)", 1, 6, 2)
            
            run_sort = st.button("🚀 Run Sorting", type="primary")

    if run_sort:
        with st.spinner("Filtering > Detecting > Clustering..."):
            # 1. Filter
            x_filt = bandpass(x, fs, bp_low, bp_high)
            
            # 2. Detect
            s_idx, thr = detect_spikes_mad(x_filt, fs, thr_k)
            
            if len(s_idx) > 20:
                # 3. Extract
                waves, valid_idx = extract_waveforms(x_filt, s_idx, fs)
                spike_times = t[valid_idx]
                
                # 4. Feature & Cluster
                pca = PCA(n_components=3)
                w_pca = pca.fit_transform(waves)
                
                # --- FIX: n_init=10 (Fixed Integer) to avoid string comparison error ---
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                labels = kmeans.fit_predict(w_pca)
                
                st.session_state['sort_res'] = {
                    'waves': waves,
                    'labels': labels,
                    'times': spike_times,
                    'pca': w_pca
                }
                st.toast(f"Success! Sorted {len(s_idx)} spikes.", icon="✅")
            else:
                st.error("Too few spikes found. Try lowering threshold.")

    # Visualization
    with col_viz:
        if 'sort_res' in st.session_state:
            res = st.session_state['sort_res']
            labels = res['labels']
            waves = res['waves']
            pca_data = res['pca']
            
            tab1, tab2, tab3 = st.tabs(["Waveforms", "PCA Space", "Raster Plot"])
            
            with tab1:
                fig_w, ax_w = plt.subplots(figsize=(8, 4))
                t_ms = np.arange(waves.shape[1]) / fs * 1000
                colors = plt.cm.tab10(np.linspace(0, 1, 10))
                
                for k in np.unique(labels):
                    mean_w = waves[labels==k].mean(0)
                    std_w = waves[labels==k].std(0)
                    c = colors[k % 10]
                    ax_w.plot(t_ms, mean_w, lw=2, color=c, label=f"Cluster {k} (n={sum(labels==k)})")
                    ax_w.fill_between(t_ms, mean_w-std_w, mean_w+std_w, color=c, alpha=0.15)
                    
                ax_w.set_xlabel("Time (ms)")
                ax_w.set_ylabel("Amplitude (uV)")
                ax_w.legend(loc='upper right')
                ax_w.set_title("Mean Waveforms")
                sns.despine()
                st.pyplot(fig_w)
                
            with tab2:
                fig_p, ax_p = plt.subplots(figsize=(8, 5))
                sc = ax_p.scatter(pca_data[:,0], pca_data[:,1], c=labels, cmap='tab10', alpha=0.7, s=20)
                ax_p.set_xlabel("PC 1")
                ax_p.set_ylabel("PC 2")
                ax_p.set_title("PCA Feature Space")
                # Add legend manually or via colorbar if many clusters
                sns.despine()
                st.pyplot(fig_p)
                
            with tab3:
                fig_r, ax_r = plt.subplots(figsize=(10, 3))
                ax_r.scatter(res['times'], labels, c=labels, cmap='tab10', s=15, marker='|')
                ax_r.set_yticks(np.unique(labels))
                ax_r.set_xlabel("Time (s)")
                ax_r.set_title("Raster Plot")
                
                for (start, end) in st.session_state['events']:
                    ax_r.axvspan(start, end, color='red', alpha=0.1)
                    
                sns.despine()
                st.pyplot(fig_r)

# --- MODULE 3: PAC Analysis ---
elif module == "3. PAC Analysis":
    st.header("📡 Phase-Amplitude Coupling (PAC)")
    
    # Pre-calc LFP
    target_fs = 1000.0
    q = int(fs / target_fs) if fs > target_fs else 1
    x_lfp = x[::q]
    fs_lfp = fs / q
    
    col_in, col_out = st.columns([1, 2])
    
    with col_in:
        with st.container(border=True):
            st.subheader("Bands Setup")
            phase_low = st.number_input("Phase Low (Hz)", 1, 10, 4)
            phase_high = st.number_input("Phase High (Hz)", 5, 20, 12)
            st.caption("Target Phase (e.g., Theta)")
            
            amp_low = st.number_input("Amp Low (Hz)", 20, 100, 30)
            amp_high = st.number_input("Amp High (Hz)", 50, 200, 80)
            st.caption("Target Amp (e.g., Gamma)")
            
            n_bins = st.slider("Phase Bins", 12, 36, 18)
            
            if st.button("Calculate MI", type="primary"):
                st.session_state['pac_res'] = compute_pac(x_lfp, fs_lfp, (phase_low, phase_high), (amp_low, amp_high), n_bins)

    with col_out:
        if 'pac_res' in st.session_state:
            bins, mean_amp, mi = st.session_state['pac_res']
            
            st.metric("Modulation Index (MI)", f"{mi:.5f}", delta="Higher is stronger coupling")
            
            fig_pac = plt.figure(figsize=(8, 5))
            ax1 = plt.subplot(111)
            
            width = (2*np.pi) / n_bins
            center = bins[:-1] + width/2
            
            # Double cycle for visualization
            center_double = np.concatenate([center, center + 2*np.pi])
            amp_double = np.concatenate([mean_amp, mean_amp])
            
            ax1.bar(center_double, amp_double, width=width, color='#4c72b0', edgecolor='white', alpha=0.9)
            ax1.set_xlim(-np.pi, 3*np.pi)
            ax1.set_xticks([0, np.pi, 2*np.pi])
            ax1.set_xticklabels(['0', '$\pi$', '2$\pi$'])
            
            # Fit sine
            def sine_func(x, a, b, c): return a * np.sin(x + b) + c
            try:
                popt, _ = curve_fit(sine_func, center, mean_amp)
                x_fit = np.linspace(-np.pi, 3*np.pi, 100)
                ax1.plot(x_fit, sine_func(x_fit, *popt), 'r--', lw=2, label='Sine Fit')
                ax1.legend()
            except: pass
            
            ax1.set_title(f"Phase-Amplitude Histogram")
            st.pyplot(fig_pac)

# --- MODULE 4: Dynamics (t-SNE/HMM) ---
elif module == "4. Dynamics (t-SNE/HMM)":
    st.header("🌀 Neural Population Dynamics")
    st.markdown("Feature Space Analysis: Sliding Window PSD -> t-SNE / PCA / HMM")
    
    # Params
    with st.expander("Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        win_sec = c1.slider("Window Size (s)", 0.5, 5.0, 1.0)
        tsne_perp = c2.slider("t-SNE Perplexity", 5, 50, 30)
        n_states = c3.slider("States (Cluster)", 2, 6, 3)
    
    if st.button("Run Dynamics Analysis", type="primary"):
        with st.spinner("Feature Extraction & Manifold Learning..."):
            # 1. Feature Extraction (Sliding Window PSD)
            target_fs = 500
            q = int(fs/target_fs) if fs > target_fs else 1
            x_lfp = x[::q]
            fs_lfp = fs/q
            
            bands = [(1,4), (4,8), (8,13), (13,30), (30,80)] # Delta, Theta, Alpha, Beta, Gamma
            nperseg = int(win_sec * fs_lfp)
            step = int(nperseg / 2) # 50% overlap
            
            feats = []
            times = []
            
            # Simple Welch Loop
            for i in range(0, len(x_lfp)-nperseg, step):
                seg = x_lfp[i:i+nperseg]
                f, p = welch(seg, fs=fs_lfp, nperseg=nperseg//2)
                row = []
                for (l, h) in bands:
                    mask = (f>=l) & (f<h)
                    pw = np.sum(p[mask]) if np.any(mask) else 1e-10
                    row.append(np.log10(pw))
                feats.append(row)
                times.append(i/fs_lfp * q) # Correct time
            
            if len(feats) < tsne_perp:
                st.error("Data too short for t-SNE perplexity setting.")
            else:
                X = StandardScaler().fit_transform(np.array(feats))
                
                # 2. Clustering (HMM or GMM)
                if HAS_HMM:
                    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
                    model.fit(X)
                    states = model.predict(X)
                    algo = "HMM"
                else:
                    model = GaussianMixture(n_components=n_states, random_state=42)
                    states = model.fit_predict(X)
                    algo = "GMM"
                
                # 3. Manifold Learning
                # PCA
                pca_res = PCA(n_components=3).fit_transform(X)
                
                # t-SNE
                tsne = TSNE(n_components=2, perplexity=tsne_perp, random_state=42, init='pca', learning_rate='auto')
                tsne_res = tsne.fit_transform(X)
                
                # UMAP
                umap_res = None
                if HAS_UMAP:
                    reducer = umap.UMAP(n_components=2, random_state=42)
                    umap_res = reducer.fit_transform(X)
                
                # --- VISUALIZATION ---
                
                st.subheader("State Trajectory")
                fig_t, ax_t = plt.subplots(figsize=(10, 2))
                ax_t.plot(times, states, drawstyle='steps-mid', color='k', lw=1)
                # Overlay Stim events
                for (s, e) in st.session_state['events']:
                    ax_t.axvspan(s, e, color='red', alpha=0.2)
                ax_t.set_yticks(range(n_states))
                ax_t.set_xlabel("Time (s)")
                ax_t.set_ylabel("State ID")
                st.pyplot(fig_t)
                
                tab_tsne, tab_pca, tab_umap = st.tabs(["t-SNE", "PCA (3D)", "UMAP"])
                
                with tab_tsne:
                    fig_ts, ax_ts = plt.subplots(figsize=(7, 5))
                    sc = ax_ts.scatter(tsne_res[:,0], tsne_res[:,1], c=states, cmap='viridis', s=30, alpha=0.8)
                    ax_ts.set_title(f"t-SNE Manifold ({algo} States)")
                    plt.colorbar(sc, label="State")
                    st.pyplot(fig_ts)
                    
                with tab_pca:
                    fig_3d = plt.figure(figsize=(7, 6))
                    ax_3d = fig_3d.add_subplot(111, projection='3d')
                    sc_3d = ax_3d.scatter(pca_res[:,0], pca_res[:,1], pca_res[:,2], c=states, cmap='viridis', s=30)
                    ax_3d.set_xlabel("PC1")
                    ax_3d.set_ylabel("PC2")
                    ax_3d.set_zlabel("PC3")
                    st.pyplot(fig_3d)
                    
                with tab_umap:
                    if umap_res is not None:
                        fig_u, ax_u = plt.subplots(figsize=(7, 5))
                        sc_u = ax_u.scatter(umap_res[:,0], umap_res[:,1], c=states, cmap='viridis', s=30)
                        ax_u.set_title("UMAP Projection")
                        plt.colorbar(sc_u)
                        st.pyplot(fig_u)
                    else:
                        st.info("Install `umap-learn` to see UMAP plots.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 XFastsort XSM Lab")