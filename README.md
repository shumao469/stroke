# stroke

Research repository for stroke-related feature extraction, curated analysis packages, and deployable research utilities.

## Repository structure

- `tools/research_zip_publisher/`: Streamlit and local tools for scanning and publishing Research Upload Studio ZIP packages.
- `research_uploads/` or a user-selected folder: package metadata, integrity reports, model usage guides, and links to large release assets.
- `scripts/`: repository validation utilities.
- `.github/workflows/`: automated structural and secret-safety checks.

## Research ZIP Publisher

Run locally:

```bash
cd tools/research_zip_publisher
bash repair_and_run_wsl.sh
```

Windows:

```bat
cd toolsesearch_zip_publisher
repair_and_run_windows.cmd
```

For large packages, use the local CLI so the file is streamed and automatically split into release assets when required.

## Data governance

This repository is public. Do not upload protected health information, direct identifiers, credentials, or non-de-identified clinical source data. Publish models and data only when licensing, consent, ethics approval, and institutional policy permit it.

## Citation

See `CITATION.cff`.

Repository owner: [shumao469](https://github.com/shumao469)
