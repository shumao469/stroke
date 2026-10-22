"""Validation against the retained equation-derived reference table."""
import csv
import math
from importlib.resources import files
from .timing import coherent_amplitude, jitter_budget_ms


def bundled_reference():
    return files("sensor_contribution_audit").joinpath("data/figure1a_reference.csv")


def validate_reference(reference=None, tolerance=1e-14):
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    source = bundled_reference() if reference is None else reference
    with source.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"frequency_hz", "relative_jitter_sd_ms", "coherent_amplitude_retained"}
    if len(rows) != 840 or set(rows[0]) != required:
        raise ValueError("reference must contain the 840-cell analytical grid and declared columns")
    expected = {(f, s) for s in range(21) for f in range(1, 41)}
    observed = set()
    maximum = 0.0
    for row in rows:
        f = float(row["frequency_hz"])
        sigma = float(row["relative_jitter_sd_ms"])
        observed.add((f, sigma))
        ref = float(row["coherent_amplitude_retained"])
        if not math.isfinite(ref) or not 0 <= ref <= 1:
            raise ValueError("reference amplitudes must be finite and lie in [0, 1]")
        maximum = max(maximum, abs(ref - coherent_amplitude(f, sigma)))
    if observed != expected:
        raise ValueError("reference coordinates must cover each declared grid point once")
    budgets = {str(f): jitter_budget_ms(f, 0.9) for f in (20, 40)}
    return {
        "reference_cells": len(rows),
        "frequency_grid_hz": {"start": 1, "stop": 40, "step": 1},
        "jitter_grid_ms": {"start": 0, "stop": 20, "step": 1},
        "maximum_absolute_difference": maximum,
        "absolute_tolerance": tolerance,
        "reference_grid_pass": maximum <= tolerance,
        "retention_criterion": 0.9,
        "analytical_budget_ms_by_frequency_hz": budgets,
        "budget_substitution_pass": all(abs(coherent_amplitude(f, budgets[str(f)]) - 0.9) < 1e-14 for f in (20, 40)),
        "evidence_scope": "Reproduction of an analytical illustration; no participant data or empirical model test",
    }
