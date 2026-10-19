from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any

def run_kilosort4(
    bin_path: str | Path,
    probe_path: str | Path,
    outdir: str | Path,
    *,
    sample_rate: float,
    n_channels: int,
    dtype: str = "int16",
    device: str = "cuda",
    extra_params: Optional[Dict[str, Any]] = None,
):
    """Run Kilosort4 (if installed) from Python.

    Notes
    -----
    Kilosort4 is developed by MouseLand (Pachitariu et al., Nature Methods 2024).
    This helper is a thin wrapper that calls the Kilosort python API if available.

    If you prefer the GUI:
      python -m kilosort

    Parameters
    ----------
    bin_path:
        Interleaved binary file.
    probe_path:
        Probe configuration (MAT recommended by Kilosort; PRB possible).
    outdir:
        Output folder (Phy-compatible).
    sample_rate, n_channels, dtype:
        Recording parameters needed for configuration.
    extra_params:
        Optional dict merged into Kilosort settings.

    Returns
    -------
    result_path: Path
        Directory containing Kilosort outputs (params.py, spike_times.npy, etc.).
    """
    bin_path = Path(bin_path)
    probe_path = Path(probe_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        import kilosort
    except Exception as e:
        raise ImportError(
            "Kilosort is not installed. Install with: python -m pip install kilosort\n"
            "(GPU strongly recommended; see docs/KILOSORT_GUIDE.md)\n"
            f"Original import error: {e}"
        )

    # Kilosort4 API may evolve; we keep this wrapper conservative.
    # Most users run via GUI or their own scripts. This function is a starting point.
    settings = {
        "data_path": str(bin_path),
        "probe_path": str(probe_path),
        "results_dir": str(outdir),
        "sample_rate": float(sample_rate),
        "n_channels": int(n_channels),
        "dtype": str(dtype),
        "device": str(device),
    }
    if extra_params:
        settings.update(extra_params)

    # Try common entry points
    if hasattr(kilosort, "run_kilosort"):
        kilosort.run_kilosort(**settings)
    elif hasattr(kilosort, "run"):
        kilosort.run(**settings)
    else:
        raise RuntimeError(
            "Could not find a supported Kilosort run function in the installed version.\n"
            "Please run via GUI: python -m kilosort\n"
            "or consult Kilosort documentation for the current API."
        )

    return outdir
