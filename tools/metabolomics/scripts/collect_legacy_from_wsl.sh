#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/collect_legacy_from_wsl.sh /mnt/h/Data/Yuchun-yanshi
#
# Copies existing *.py analysis scripts into scripts/legacy/.
# DOES NOT copy data matrices.

SRC_DIR="${1:-/mnt/h/Data/Yuchun-yanshi}"
DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/legacy"

mkdir -p "${DEST_DIR}"

echo "Copying *.py from: ${SRC_DIR}"
find "${SRC_DIR}" -maxdepth 1 -type f -name "*.py" -print0 | while IFS= read -r -d '' f; do
  bn="$(basename "$f")"
  cp -v "$f" "${DEST_DIR}/${bn}"
done

echo "Done. Legacy scripts copied to: ${DEST_DIR}"
