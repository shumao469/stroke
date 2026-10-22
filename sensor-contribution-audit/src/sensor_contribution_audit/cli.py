"""Portable command-line interface, producing deterministic relative-path outputs."""
import argparse
import csv
import json
from pathlib import Path
from . import __version__
from .dimensions import dimension_audit
from .timing import legacy_heatmap_grid, retention_contour, jitter_budget_ms
from .validation import validate_reference
from .plotting import write_svg, write_pdf, write_png
from .reported import reported_values


def _json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analytical sensor timing-budget illustration; no participant analysis")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Write the 840-cell CSV, contour, JSON and editable SVG")
    generate.add_argument("--out", type=Path, default=Path("outputs"))
    generate.add_argument("--with-plots", action="store_true", help="Also write vector PDF and 300-dpi PNG using optional dependencies")
    validate = sub.add_parser("validate", help="Check the retained 840-cell analytical reference")
    validate.add_argument("--reference", type=Path)
    validate.add_argument("--out", type=Path)
    dimensions = sub.add_parser("dimensions", help="Audit declared feature and optional latent/readout dimensions")
    dimensions.add_argument("--d-x", type=int, required=True)
    dimensions.add_argument("--d-z", type=int, required=True)
    dimensions.add_argument("--h-x", type=int)
    dimensions.add_argument("--h-z", type=int)
    dimensions.add_argument("--quality", type=int, default=0)
    dimensions.add_argument("--availability", type=int, default=0)
    dimensions.add_argument("--out", type=Path)
    reported = sub.add_parser("reported-values", help="Export attributed aggregate article values; no new statistics")
    reported.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            reference_check = validate_reference()
            if not reference_check["reference_grid_pass"] or not reference_check["budget_substitution_pass"]:
                raise ValueError("the retained analytical reference must pass validation before generation")
            args.out.mkdir(parents=True, exist_ok=True)
            _csv(args.out / "gaussian_relative_jitter_heatmap.csv", legacy_heatmap_grid())
            _csv(args.out / "retention_0p9_contour.csv", retention_contour())
            summary = {
                "software_version": __version__,
                "equation": "H = abs(E[exp(i*2*pi*f*epsilon_delta)]) = exp(-(2*pi*f*sigma_delta)^2/2)",
                "units": {"frequency": "Hz", "displayed_standard_deviation": "ms", "equation_standard_deviation": "s"},
                "assumptions": ["zero-mean Gaussian residual relative timing error", "timing shifts independent of the underlying phasor", "coherent averaging of repeated complex observations"],
                "retention_criterion": 0.9,
                "budget_ms_at_20_hz": jitter_budget_ms(20),
                "budget_ms_at_40_hz": jitter_budget_ms(40),
                "validation": reference_check,
                "scope": "Analytical illustration; no participant data, physiological estimate or empirical model performance",
            }
            _json(args.out / "analytical_summary.json", summary)
            write_svg(args.out / "timing_budget.svg")
            if args.with_plots:
                write_pdf(args.out / "timing_budget.pdf")
                write_png(args.out / "timing_budget.pdf", args.out / "timing_budget_300dpi.png")
            result = {"status": "PASS", "files": sorted(p.name for p in args.out.iterdir() if p.is_file()), "reference_cells": 840}
        elif args.command == "validate":
            result = validate_reference(args.reference)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                _json(args.out, result)
            if not result["reference_grid_pass"] or not result["budget_substitution_pass"]:
                print(json.dumps(result, indent=2, sort_keys=True))
                return 1
        elif args.command == "dimensions":
            result = dimension_audit(args.d_x, args.d_z, args.h_x, args.h_z, args.quality, args.availability)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                _json(args.out, result)
        else:
            result = reported_values()
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                _json(args.out, result)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
