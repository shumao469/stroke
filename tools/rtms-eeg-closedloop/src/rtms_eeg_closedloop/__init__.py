"""
rTMS-EEG task-state closed-loop analysis utilities (Python port of MATLAB scripts).
"""
from ._version import __version__

from .io import read_txt_to_numpy, zero_out_trigger_codes, save_mat, load_mat
from .preprocess import preprocess_task_eeg, epoch_from_trigger_channel
from .timefreq import compute_ersp_morlet, save_ersp_outputs
from .erd_ers import compute_erd_ers_ratio, compute_subject_metrics
from .stats_viz import compute_significance_matrices, plot_threshold_heatmaps
