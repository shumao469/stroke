param(
  [string]$Excel="HS_related_all_tables.xlsx",
  [string]$Out="outputs"
)
ms-taxonomy --excel $Excel --out "$Out\taxonomy"
ms-network --excel $Excel --comparison HSvsNC --out "$Out\network"
ms-network --excel $Excel --comparison ZSvsHS --out "$Out\network"
ms-predict-demo --excel $Excel --out "$Out\predict_demo"
Write-Host "[DONE] All demo figures are in: $Out"
