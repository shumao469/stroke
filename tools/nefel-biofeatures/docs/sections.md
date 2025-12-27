# Section-by-section guide (Section 1–7)

This document explains **inputs**, **outputs**, and **recommended settings** for each analysis section.

---

## Common conventions

### Image format
- All functions in `nefel.sections` and `nefel.markers.*` assume **RGB uint8** arrays:
  - shape: `(H, W, 3)`
  - dtype: `uint8`
- If you use OpenCV:
  - `cv2.imread()` returns **BGR**.
  - convert to RGB with:
    `rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)`

### Tissue mask
Most metrics are normalized by a tissue mask estimated from the grayscale image.
Default rule (in `nefel.core.tissue_mask`):
- `mask = gray < 245`, then morphological closing.

If your background is non-white (e.g., dark glass), adjust the threshold.

---

## Section 1: iNOS / Arg analysis

**Module:** `nefel.markers.inos_arg`

### Input
- Dual-channel IF image where:
  - iNOS is mainly **Red**
  - Arg/Arg1 is mainly **Green**

### Output
- `red_area_ratio`, `red_mean_intensity`, `red_integrated_intensity`
- `green_area_ratio`, `green_mean_intensity`, `green_integrated_intensity`
- `RG_ratio_area`, `RG_ratio_intensity`

### Notes
- Default uses HSV thresholding (robust across lighting shifts).
- If you see misclassification (e.g., autofluorescence), tighten HSV ranges.

---

## Section 2: Iba1 analysis (Day 1)

**Module:** `nefel.markers.iba1_day1`

### Output
- `iba1_area_ratio`
- `cell_count`
- `skeleton_total_length`
- `branching_index`
- `mean_intensity`

### QC output
Set `out_dir="qc_masks"` to write:
- `{prefix}_skeleton_overlay.jpg`
- `{prefix}_iba1_mask.jpg`

---

## Section 3: Claudin-5 analysis

**Module:** `nefel.markers.claudin5`

### Default method
`quantify_claudin5(rgb, q_thr=85)`:
1) Extract DAB OD (`rgb2hed`)
2) Use percentile threshold within tissue (`q_thr`)
3) Skeletonize and compute segment statistics

### Output
- `dab_pos_area_ratio`
- `dab_meanOD_pos`, `dab_meanOD_pos_x1000`
- `skeleton_total_length`, `segment_count`, `mean_segment_length`
- `p99_dab_tissue`

### QC output
Writes overlay + mask when `out_dir` is set:
- `{prefix}_overlay.jpg`
- `{prefix}_pos_mask.jpg`

---

## Section 4: CD31 analysis

**Module:** `nefel.markers.cd31`

### Output
- `cd31_area_ratio`
- `skeleton_total_length`
- `mean_segment_length`, `p90_segment_length`
- `segment_count`
- `mean_intensity`

### QC output
- `{prefix}_skeleton_overlay.jpg`
- `{prefix}_vessel_mask.jpg`

---

## Section 5: Synapse analysis

**Module:** `nefel.markers.synapse`

### Input
- An RGB image where two channels encode pre/post markers.

### Key parameters
- `thr_percentile` (default 99): puncta threshold
- `min_area`, `max_area`: puncta size filtering
- `max_dist` (default 3 px): colocalization radius

### Output
- `pre_count`, `post_count`
- `coloc_count`, `coloc_ratio`
- mean puncta area/intensity and densities

---

## Section 6: GAP43 analysis

**Module:** `nefel.markers.gap43`

### Output
- `gap43_area_ratio`
- `skeleton_total_length`
- `mean_fiber_length`, `p90_fiber_length`
- `segment_count`
- `mean_intensity`

### QC output
- `{prefix}_skeleton_overlay.jpg`
- `{prefix}_fiber_mask.jpg`

---

## Section 7: TUNEL analysis

**Module:** `nefel.markers.tunel`

### Input
- Uses DAPI-like nuclei detection (blue channel percentile)
- Uses TUNEL-like signal detection (red/green heuristic inside `detect_tunel_signal`)

### Output
- `total_cells`
- `tunel_pos_cells`
- `tunel_ratio`

### QC output
- `{prefix}_tunel_mask.jpg` (overlay)
