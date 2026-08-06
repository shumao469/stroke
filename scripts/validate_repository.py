from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_REGULAR_FILE = 100 * 1024 * 1024
SECRET_PATTERNS = {
    "GitHub token": re.compile(rb"(?:\bgh[pousr]_[A-Za-z0-9_]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b)"),
    "Private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI API key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__"}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        relative = path.relative_to(root)
        size = path.stat().st_size
        if size >= MAX_REGULAR_FILE:
            errors.append(f"File exceeds 100 MiB regular Git limit: {relative} ({size} bytes)")
        if size <= 2 * 1024 * 1024:
            try:
                data = path.read_bytes()
            except OSError:
                continue
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(data):
                    errors.append(f"Potential {label} in {relative}")
    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Repository validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
