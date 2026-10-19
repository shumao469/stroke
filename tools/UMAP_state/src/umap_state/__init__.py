"""UMAP State — UMAP-based state-space analysis for TI spike/LFP features."""

from ._version import __version__

from .segmentation import SegmentSpec, segment_by_time
from .preprocess import zscore, smooth_1d
from .embed import fit_umap, transform_umap
from .state import cluster_states, transition_matrix, state_occupancy
