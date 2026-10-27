"""Bounded generated-input release checks and explicit public-file manifest.

This is implementation QA, not cohort admission or institutional review.
No protected archive, old evaluation task, model or patient path is opened.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TOP = {".gitignore", "CITATION.cff", "LICENSE_PENDING.md", "pyproject.toml",
       "README.md", "SECURITY.md", "THIRD_PARTY_NOTICES.md", "run_strokeweave.py", "run_tests.py"}
DIRECTORIES = {"src", "tests", "tools", "docs", "protocols", "evidence"}
SUFFIXES = {".py", ".html", ".css", ".js", ".md", ".json", ".txt"}
GENERATED = {"evidence/FILE_MANIFEST.json", "evidence/RELEASE_SHA256.txt"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def public_paths():
    selected = []
    for file in sorted(ROOT.rglob("*")):
        if not file.is_file() or "__pycache__" in file.parts:
            continue
        relative = file.relative_to(ROOT)
        if len(relative.parts) == 1:
            include = relative.name in TOP
        else:
            include = relative.parts[0] in DIRECTORIES and file.suffix in SUFFIXES
        if include:
            if file.is_symlink() or file.stat().st_size > 256_000:
                raise ValueError(f"Disallowed publication member: {relative.as_posix()}")
            selected.append(file)
    return selected


def inspect_files(files):
    problems = []
    # High-specificity credential and private-path patterns, plus an explicit
    # path/type allowlist. This is not a universal secret-detection guarantee.
    expressions = {
        "AWS access key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        "GitHub token": r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
        "PEM secret": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "Signed download URL": r"X-Amz-(?:Signature|Credential)=",
        "Bearer credential": r"Bearer\s+[A-Za-z0-9_.-]{24,}",
        "Private absolute path": r"(?:[A-Z]:[/\\](?:Users|AgentAi)[/\\]|/home/[A-Za-z][A-Za-z0-9_-]{1,32}/)",
    }
    for file in files:
        text = file.read_text(encoding="utf-8")
        for kind, pattern in expressions.items():
            if re.search(pattern, text):
                problems.append({"file": file.relative_to(ROOT).as_posix(), "kind": kind})
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check the existing manifest without changing files")
    args = parser.parse_args(argv)
    manifest_path = ROOT / "evidence" / "FILE_MANIFEST.json"
    if args.check_only:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        failures = [row["path"] for row in manifest["files"]
                    if not (ROOT / row["path"]).is_file()
                    or sha((ROOT / row["path"]).read_bytes()) != row["sha256"]
                    or (ROOT / row["path"]).stat().st_size != row["bytes"]]
        sentinel = (ROOT / "evidence" / "RELEASE_SHA256.txt").read_text().strip().split()[0]
        observed = {p.relative_to(ROOT).as_posix() for p in public_paths() if p.relative_to(ROOT).as_posix() not in GENERATED}
        if observed != {row["path"] for row in manifest["files"]}:
            failures.append("allowlist_membership_changed")
        if sha(raw) != sentinel:
            failures.append("manifest_digest_mismatch")
        if failures:
            print(json.dumps({"status": "FAIL", "failures": failures}))
            return 1
        print(json.dumps({"status": "PASS", "files": len(manifest["files"]), "manifest_sha256": sentinel}))
        return 0

    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    if manifest_path.exists():
        raise FileExistsError("Existing release frozen; use --check-only or create a new version directory")
    started = time.perf_counter()
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    test = subprocess.run([sys.executable, "-B", str(ROOT / "run_tests.py")],
                          cwd=ROOT, capture_output=True, text=True, timeout=120, env=environment)
    transcript = test.stdout + test.stderr
    (evidence / "UNIT_TESTS.txt").write_text(transcript, encoding="utf-8")
    if test.returncode:
        print(transcript)
        return test.returncode
    count_match = re.search(r"Ran (\d+) tests? in", transcript)
    benchmark = json.loads((evidence / "engineering_benchmark.json").read_text())
    protocol = ROOT / "protocols" / "engineering_v1.json"
    if not benchmark["all_exact"] or benchmark["protocol_sha256"] != sha(protocol.read_bytes()):
        raise ValueError("Bounded benchmark or prospective protocol binding failed")
    initial_files = public_paths()
    issues = inspect_files(initial_files)
    if issues:
        raise ValueError(json.dumps(issues))
    validation = {
        "name": "StrokeWeave", "version": "0.1.0", "status": "FUNCTIONAL_RESEARCH_SNAPSHOT_PASS",
        "date": "2026-10-27", "python": platform.python_version(), "platform": platform.system(),
        "unit_tests": int(count_match[1]) if count_match else None, "tests_passed": test.returncode == 0,
        "automatic_independent_math_review": "Independent determinant/Cramer implementation tests, not institutional review",
        "bounded_generated_benchmark_rows": len(benchmark["rows"]), "benchmark_all_exact": benchmark["all_exact"],
        "browser_visual_acceptance": "NOT_COMPLETED_CONNECTION_UNAVAILABLE",
        "native_windows_tested": False, "real_model_plugins_enabled": False,
        "patient_execution": False, "training_enabled": False, "CAL_HOLD_test_labels_accessed": False,
        "old_TACTIC_to_direct_ratio": 0.983445494, "old_le_0_8_target_met": False,
        "controller_superiority_established": False, "NC_readiness_established": False,
        "code_licence": "Owner selection pending; not represented as MIT or Apache licensed",
        "secret_scan": "No findings in explicit publish allowlist; not a universal guarantee",
        "runtime_and_datasets_excluded": True, "github_destination": "shumao469/stroke new research branch only",
        "known_fixed_implementation_issues": ["source-checkout test import path", "retain owner on blocked shutdown",
            "reject database access after close", "single JSON/HTML export snapshot", "include artifact invalidation event"],
        "historical_campaign_unchanged": True, "elapsed_verification_seconds": time.perf_counter() - started,
    }
    (evidence / "RELEASE_VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    files = [p for p in public_paths() if p.relative_to(ROOT).as_posix() not in GENERATED]
    manifest = {"schema": "strokeweave-public-files-v1", "files": [
        {"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
        for p in files], "scope": "Explicit source/docs/generated-evidence allowlist; no runtime or restricted data"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    sentinel = sha(manifest_path.read_bytes())
    (evidence / "RELEASE_SHA256.txt").write_text(sentinel + "  FILE_MANIFEST.json\n", encoding="ascii")
    print(json.dumps({"status": validation["status"], "tests": validation["unit_tests"],
                      "public_files": len(files), "manifest_sha256": sentinel,
                      "original_performance_target_met": False, "visual_acceptance_completed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
