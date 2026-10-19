from pathlib import Path
import numpy as np

from umap_state.preprocess import zscore
from umap_state.segmentation import SegmentSpec, segment_by_time
from umap_state.embed import UMAPConfig, fit_umap
from umap_state.state import ClusterConfig, cluster_states

# Example: load PAC features (replace with your own feature matrix)
X = np.load(Path(__file__).parent / "data" / "UMAP_state_space_simple.npy")
if X.ndim == 1:
    X = X.reshape(-1, 1)

# IMPORTANT: fs here is feature sampling rate (bins per second), NOT raw LFP fs.
fs_feat = 1.0
t = np.arange(X.shape[0]) / fs_feat

spec = SegmentSpec(stim_onset=0.0, pre_dur=60.0, stim_dur=180.0, post_dur=600.0)
seg = segment_by_time(t, X, spec)

t_all = np.concatenate([seg["pre"][0], seg["stim"][0], seg["post"][0]])
X_all = np.concatenate([seg["pre"][1], seg["stim"][1], seg["post"][1]])

Xz = zscore(X_all, axis=0)
umcfg = UMAPConfig(n_components=2, n_neighbors=30, min_dist=0.1, random_state=0, use_gpu=False)
_, Z, backend = fit_umap(Xz, umcfg)

labels = cluster_states(Z, ClusterConfig(method="kmeans", n_clusters=5))

out = Path(__file__).parent / "outputs_demo"
out.mkdir(parents=True, exist_ok=True)
np.save(out/"Z.npy", Z)
np.save(out/"labels.npy", labels)
print("Saved", out, "backend:", backend)
