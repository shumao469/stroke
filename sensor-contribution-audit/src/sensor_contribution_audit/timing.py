"""Closed-form Gaussian timing-error calculations, with explicit time units.

For a sinusoid shifted by epsilon ~ N(mu, sigma_delta**2), independent of
the underlying phasor:
    E[exp(i*2*pi*f*epsilon)] = exp(i*2*pi*f*mu) * exp(-(2*pi*f*sigma_delta)**2/2)
sigma_delta and mu enter this equation in seconds. Public helpers take ms.
"""
import cmath
import math
from numbers import Real


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def coherent_amplitude(frequency_hz, relative_jitter_sd_ms):
    """Magnitude of the expected complex phasor relative to perfect alignment.

    This is |E[Z]|, rather than E[|Z|] or single-observation power.
    """
    f = _finite(frequency_hz, "frequency_hz")
    sigma_ms = _finite(relative_jitter_sd_ms, "relative_jitter_sd_ms")
    if f < 0 or sigma_ms < 0:
        raise ValueError("frequency and standard deviation must be non-negative")
    phase_sd = 2 * math.pi * f * sigma_ms / 1000
    return math.exp(-0.5 * phase_sd * phase_sd)


def expected_mean_phasor(frequency_hz, relative_jitter_sd_ms, mean_offset_ms=0):
    """Expected unit phasor; a fixed offset changes phase and preserves modulus."""
    f = _finite(frequency_hz, "frequency_hz")
    offset = _finite(mean_offset_ms, "mean_offset_ms")
    return coherent_amplitude(f, relative_jitter_sd_ms) * cmath.exp(
        1j * 2 * math.pi * f * offset / 1000
    )


def jitter_budget_ms(frequency_hz, minimum_retention=0.9):
    """Largest Gaussian relative timing SD satisfying H >= minimum_retention.

    The criterion is a user-selected analytical operating requirement, not a
    commercial-device acceptance threshold or an observed physiological limit.
    """
    f = _finite(frequency_hz, "frequency_hz")
    eta = _finite(minimum_retention, "minimum_retention")
    if f <= 0 or not 0 < eta <= 1:
        raise ValueError("frequency must be positive and retention must lie in (0, 1]")
    return math.sqrt(-2 * math.log(eta)) / (2 * math.pi * f) * 1000


def legacy_heatmap_grid():
    """The exact prior illustration grid: 40 frequencies x 21 jitter values."""
    return [
        {
            "frequency_hz": f,
            "relative_jitter_sd_ms": sigma,
            "coherent_amplitude_retained": coherent_amplitude(f, sigma),
        }
        for sigma in range(21)
        for f in range(1, 41)
    ]


def retention_contour(minimum_retention=0.9, samples=200):
    """Closed-form contour on the displayed 1-40 Hz, 0-20 ms domain.

    This samples the analytic curve directly; it does not estimate a contour
    by interpolating the 840 heatmap values.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ValueError("samples must be an integer of at least two")
    first = max(1.0, jitter_budget_ms(1, minimum_retention) / 20)
    if first > 40:
        return []
    return [
        {
            "frequency_hz": first + (40 - first) * i / (samples - 1),
            "relative_jitter_sd_ms": jitter_budget_ms(
                first + (40 - first) * i / (samples - 1), minimum_retention
            ),
            "coherent_amplitude_retained": float(minimum_retention),
        }
        for i in range(samples)
    ]
