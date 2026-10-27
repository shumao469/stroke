"""Fixed generated engineering checks. Not a controller-superiority benchmark."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from strokeweave.numerics import prepare_problem, solve_exact, verify_candidate


def fixture(p):
    # Identity rows ensure rank without data-dependent sample removal.
    X = [[F(int(i == j)) for j in range(p)] for i in range(p)]
    X += [[F(((i + 3) * (j + 5) + j * j) % 19 - 9, 7)
           for j in range(p)] for i in range(3 * p + 3)]
    beta = tuple(F((-1) ** j * (j + 1), j + 2) for j in range(p))
    y = [sum((x * b for x, b in zip(row, beta)), F(0)) for row in X]
    return X, y, beta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "evidence" / "engineering_benchmark.json")
    args = parser.parse_args(argv)
    protocol = ROOT / "protocols" / "engineering_v1.json"
    specification = json.loads(protocol.read_text())
    rows = []
    started = time.perf_counter()
    for p in specification["solver_cost_check"]["parameter_sizes"]:
        X, y, beta = fixture(p)
        for repeat in range(specification["solver_cost_check"]["repetitions"]):
            methods = ("reference", "bareiss") if repeat % 2 == 0 else ("bareiss", "reference")
            for method in methods:
                begin = time.perf_counter()
                problem = prepare_problem(X, y, source_id=f"new-analytic-v1-p{p}")
                prepared = time.perf_counter()
                candidate = solve_exact(problem, method=method)
                solved = time.perf_counter()
                receipt = verify_candidate(problem, candidate)
                verified = time.perf_counter()
                correct = candidate.coefficients == beta and receipt["passed"] and candidate.rss == 0
                rows.append({"columns": p, "rows": len(X), "repetition": repeat,
                             "method": method, "statistics_seconds": prepared - begin,
                             "solve_seconds": solved - prepared,
                             "exact_residual_check_seconds": verified - solved,
                             "total_warm_seconds": verified - begin,
                             "rank": p, "exact_solution_equal": correct})
                if not correct:
                    raise RuntimeError("Algebraic mismatch; locked stop rule triggered")
    summaries = []
    for p in specification["solver_cost_check"]["parameter_sizes"]:
        data = [r for r in rows if r["columns"] == p]
        medians = {m: statistics.median(r["total_warm_seconds"] for r in data if r["method"] == m)
                   for m in ("reference", "bareiss")}
        summaries.append({"columns": p, "complete_warm_median_seconds": medians,
                          "bareiss_to_reference_ratio": medians["bareiss"] / medians["reference"]})
    report = {"protocol_id": specification["id"],
              "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
              "scope": "New generated common-kernel engineering check, not old NIST or patient evaluation",
              "cpu_exclusive_window": False, "cold_start_included": False,
              "all_baselines_share_improvements": True,
              "controller_superiority_established": False,
              "old_TACTIC_to_direct_target_met": False,
              "rows": rows, "summaries": summaries,
              "all_exact": all(r["exact_solution_equal"] for r in rows),
              "elapsed_seconds": time.perf_counter() - started}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError("Keep prior results; choose a new explicitly versioned output")
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
