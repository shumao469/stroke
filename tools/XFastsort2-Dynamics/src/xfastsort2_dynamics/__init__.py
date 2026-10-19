"""XFastsort2-Dynamics — HMM/PCA/tSNE pipelines for LFP & spike-derived features."""

from ._version import __version__

from .segmentation import SegmentSpec, segment_by_time
from .preprocess import zscore, smooth_1d
from .hmm import fit_hmm_gaussian, decode_hmm
from .manifold import run_pca, run_tsne
