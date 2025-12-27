from pathlib import Path
import numpy as np

from xfastsort2_dynamics.preprocess import zscore
from xfastsort2_dynamics.segmentation import SegmentSpec, segment_by_time
from xfastsort2_dynamics.hmm import fit_hmm_gaussian, decode_hmm
from xfastsort2_dynamics.manifold import run_pca, run_tsne

# Load example feature matrix (PAC or other features)
X = np.load(Path(__file__).parent / "data" / "TI_10Hz_3min_rec10min_pre_cleaned_PAC.npy")
if X.ndim == 1:
    X = X.reshape(-1, 1)

# IMPORTANT: fs here is the feature sampling rate (bins per second).
# Adjust to match how PAC was computed. If unknown, treat each row as 1 second (fs=1) as a starting point.
fs_feat = 1.0
t = np.arange(X.shape[0]) / fs_feat

# Segment spec: stim onset = 0 and durations in seconds
spec = SegmentSpec(stim_onset=0.0, pre_dur=60.0, stim_dur=180.0, post_dur=600.0)
seg = segment_by_time(t, X, spec)

t_all = np.concatenate([seg["pre"][0], seg["stim"][0], seg["post"][0]])
X_all = np.concatenate([seg["pre"][1], seg["stim"][1], seg["post"][1]])
X_all = zscore(X_all, axis=0)

model = fit_hmm_gaussian(X_all, n_states=3)
res = decode_hmm(model, X_all)

pca = run_pca(X_all, n_components=3)
tsne = run_tsne(X_all, perplexity=30)

out = Path(__file__).parent / "outputs_10hz"
out.mkdir(parents=True, exist_ok=True)
np.save(out/"state_seq.npy", res.state_seq)
np.save(out/"pca.npy", pca.Z)
np.save(out/"tsne.npy", tsne)
print("Saved outputs to", out)
