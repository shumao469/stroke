#!/usr/bin/env bash
set -euo pipefail
EXCEL="${1:-HS_related_all_tables.xlsx}"
OUT="${2:-outputs}"

ms-taxonomy --excel "$EXCEL" --out "$OUT/taxonomy"
ms-network --excel "$EXCEL" --comparison HSvsNC --out "$OUT/network"
ms-network --excel "$EXCEL" --comparison ZSvsHS --out "$OUT/network"
ms-predict-demo --excel "$EXCEL" --out "$OUT/predict_demo"

echo "[DONE] All demo figures are in: $OUT"
