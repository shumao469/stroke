# Research Upload Package: research_upload_s7b5n0tk.zip

This package was prepared for reproducible inspection and downstream analysis.

## Summary

- Archive size: **942.15 KiB**
- Files: **1,276**
- Uncompressed size: **10.37 MiB**
- Models detected: **0**
- Candidate entrypoints: **0**
- Potential secret findings: **0**
- Potential sensitive-data hints: **0**

## Research Upload Studio structure detected

- Manifest or inventory file detected
- Validation output detected
- Schema metadata detected
- SQL file detected

## Integrity and safety

See `research_uploads/research_upload_s7b5n0tk/scan_report.json`, `SHA256SUMS.txt`, and `MODEL_USAGE.md`.
Do not publish credentials, direct identifiers, raw clinical DICOM, or non-de-identified patient data to a public repository.

## DuckDB/Parquet examples

```python
import duckdb

con = duckdb.connect('path/to/database.duckdb', read_only=True)
print(con.sql('SHOW TABLES').fetchall())
```

```sql
SELECT * FROM read_parquet('path/to/parquet/**/*.parquet', union_by_name=true) LIMIT 20;
```
