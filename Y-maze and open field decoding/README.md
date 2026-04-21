# Y-maze and open field decoding

A Python project designed for decoding video data from Mouse Open Field Tests (OFT) and Y-maze experiments. This repository was organized and refactored from previously uploaded notebook code, adopting a directory structure better suited for GitHub hosting, collaborative maintenance, and future expansion.

## Copyright Notice

Copyright (c) ISTBI, Fudan University, Xu Lab. All rights reserved.
Contact: shumaoxu@fudan.edu.cn

The code, documentation, visualizations, and derived outputs within this repository are, by default, the intellectual property of ISTBI, Fudan University, Xu Lab. For inquiries regarding reproduction, reuse, redistribution, or commercial collaboration, please contact the email address provided above.

## 1. Project Objectives

This project targets the following two types of behavioral videos:

1. Open Field Test (OFT)
- Objective: To track the movement trajectory of a mouse within an open arena. 
- Typical Outputs: Trajectory plots, heatmaps, velocity/zone timelines, total distance traveled, average velocity, time spent in the center zone, and number of entries into the center zone.

2. Y-maze Experiment
- Objective: To track the mouse's exploratory behavior across the three arms and the central zone. 
- Typical Outputs: Trajectory plots, heatmaps, velocity/zone timelines, time spent in each arm, number of entries into each arm, Spontaneous Alternation Rate (SAP), and arm entry sequence.


## 2. Core Methodology

This repository retains the most critical and robust technical pipeline derived from your original notebooks:

- Median Background Modeling: Uniformly samples frames from the video to generate a background image devoid of the mouse.
- Manual Blue ROI Annotation: Avoids heavy reliance on brightness thresholds by allowing the user to directly mark the valid experimental area using blue annotations.
- Background Subtraction + Morphological Cleaning: Extracts the mouse's silhouette from the foreground.
- Centroid Tracking: Uses the centroid of the silhouette to determine the mouse's current position.
- Trajectory Reconstruction: Plots the movement path as a red trajectory line, marking the start point with `S` and the end point with `E`. - Behavioral Quantification: Generates metrics such as distance traveled, velocity, dwell time within specific zones, and entry counts into those zones.

This protocol is particularly well-suited for the scenario you described earlier:

- The lower section of the video frame features a clearly delineated enclosure or maze structure.
- It is not feasible to rely solely on variations in lighting intensity to identify the floor surface.
- It places a strong emphasis on first precisely defining the experimental area, and then tracking the mouse exclusively within that designated region. ---

## 3. Repository Structure

```text
ISTBI_Behavior_Decoder/
├─ README.md
├─ COPYRIGHT.txt
├─ LICENSE.txt
├─ requirements.txt
├─ pyproject.toml
├─ .gitignore
├─ docs/
│  └─ OPERATION_GUIDE_CN.md
├─ notebooks/
│  └─ Mouse_Open_Field_and_Y-Maze_Video_Decoding.ipynb
├─ scripts/
│  ├─ run_open_field.py
│  └─ run_ymaze.py
└─ src/
└─ istbi_behavior_decoder/
├─ __init__.py
├─ common.py
├─ open_field.py
└─ ymaze.py
```

## 4. Environment Setup (WSL Recommended)

### 4.1 Create Virtual Environment

```bash
cd /path/to/ISTBI_Behavior_Decoder
python3 -m venv .venv
source .venv/bin/activate
```
### 4.2 Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Data Organization Suggestions

The suggested directory structure is as follows:

```text
/mnt/h/2024/VPL Electrophysiology/processed/Submit/Mouse 1/
├─ Pre-TI Stimulation/
│  ├─ Mouse A_Open Field.mp4
│  ├─ Mouse A_ROI_debug.jpg
│  ├─ Mouse A_Y-Maze.mp4
│  └─ Mouse A_YMaze_ROI_debug.jpg
├─ 1 Day Post-TI Stimulation/
│  ├─ Mouse B_Open Field.mp4
│  ├─ Mouse B_ROI_debug.jpg
│  ├─ Mouse B_Y-Maze.mp4
│  └─ Mouse B_YMaze_ROI_debug.jpg
└─ ...
```
### Key Naming Requirements

#### Open Field
- The video filename must contain: `Open Field`
- ROI It is recommended to name the image: `VideoName_ROI_debug.jpg`

#### Y-Maze
- The video filename should contain: `YMaze`
- It is recommended to name the ROI image: `VideoName_YMaze_ROI_debug.jpg`

## 6. Manual ROI Annotation Rules

### 6.1 Why Manual Blue Annotation is Recommended

For your specific data, directly and automatically identifying the floor or maze boundaries may be affected by the following factors:

- Reflections
- Shadows
- Uneven illumination at the edges
- Brief intrusion of the experimenter's hands into the frame
- Variations in the color of the apparatus floor

Therefore, the current solution employs manual blue ROI annotation. Its advantages are:

- High degree of controllability
- Greater robustness in complex lighting conditions
- Facilitates rapid manual correction
- Aligns more closely with the philosophy of "first ensuring the valid analysis area, then performing tracking"

### 6.2 Annotation Method

On the background image or any static frame:

- Use a drawing tool to paint the valid experimental area a distinct blue color.
- Open Field: Paint the entire bottom activity area.
- Y-Maze: Paint the entire Y-shaped activity area.
- You only need to ensure that the blue area covers the valid region; there is no need to strive for artistically precise edges.

The program will automatically extract the blue region (based on HSV values) to serve as a mask.

## 7. Open Field Experiment Decoding Logic

### 7.1 Inputs

- `*.mp4` video file
- Corresponding blue ROI image

### 7.2 Processing Workflow

1. Background Estimation
Skip the first few seconds of the video, uniformly sample a number of frames, and calculate the median to generate a background image.

2. ROI Extraction
Extract the blue region from the `_ROI_debug.jpg` file to obtain the mask for the Open Field floor. 3. Definition of the Center Zone
Within the bounding box of the ROI, a central rectangle is designated as the `Center` zone. The default setting is `center_ratio=0.5`, meaning the width and height of the center zone are each half the size of the overall ROI.

4. Mouse Detection
For each frame, the following operations are performed:
- Background subtraction
- Thresholding
- Morphological opening
- Contour area filtering
- Centroid calculation

5. Trajectory Reconstruction
Centroids are connected frame-by-frame to generate the trajectory line.

6. Behavioral Quantification
Outputs:
- Total distance traveled
- Average velocity
- Time spent in the center zone
- Time spent in the periphery zone
- Number of entries into the center zone
- Percentage of time spent in the center zone

### 7.3 Output Files

Generated within the directory containing each video:

- `*_trajectory.jpg`: Red trajectory plot, marked with `S` (Start) and `E` (End)
- `*_heatmap.jpg`: Heatmap
- `*_session_summary.png`: Velocity and zone occupancy timeline
- `*_Zone_Debug.jpg`: Visualization for verifying the ROI and center zone

Generated in the root directory:

- `OFT_Results.csv`
- `OFT_Cohort_Summary.png`

---

## 8. Y-Maze Decoding Logic

### 8.1 Inputs

- `*.mp4` video files
- Corresponding blue Y-maze ROI images

### 8.2 Processing Workflow

1. Background Estimation
Identical to the Open Field Test; a median background image is utilized.

2. Y-Maze ROI Extraction
The blue-colored region is extracted from the `_YMaze_ROI_debug.jpg` image. 3. Automatic Partitioning
Applied to the entire Y-maze mask:
- Uses convexity defects to identify the central intersection region.
- If insufficient convexity defects are detected, it falls back to estimating the center using a distance transform.
- Divides the remaining area into three distinct arms based on connected components.
- Sorts the arms into `Arm 1`, `Arm 2`, and `Arm 3` based on their relative angles to the center.

4. Mouse Detection and Tracking
Utilizes the following techniques:
- Background subtraction
- Thresholding
- Contour filtering within the ROI
- Centroid tracking

5. Region Determination
Based on the location of the centroid, determines whether the subject is currently situated in the `Center` or within one of the `Arms`. 6. Sequences and SAP Calculation
Record the sequence of arm entries; for example:

```text
Arm 1 > Arm 2 > Arm 3 > Arm 1 > Arm 3
```

Define a "valid alternation" as a sequence of three consecutive, non-repeating arm entries, and calculate the Spontaneous Alternation Percentage (SAP) as follows:

```
SAP = Number of Valid Alternations / (Total Number of Arm Entries - 2) × 100%
```

### 8.3 Output Files

Generated within each video directory:

- `*_trajectory.jpg`: Red trajectory plot, marked with `S` (Start) and `E` (End).
- `*_heatmap.jpg`: Heatmap.
- `*_session_summary.png`: Velocity + Zone Timeline.
- `*_Zone_Debug.jpg`: Debug image verifying the automatic partitioning of the Center and the three arms.

Generated in the root directory:

- `YMaze_Results.csv`
- `YMaze_Cohort_Summary.png`


## 9. Command-Line Execution

### 9.1 Running Open Field Analysis

```bash
python scripts/run_open_field.py \
--base-dir " \
--skip-seconds 5 \
--analyze-seconds 300
```

If the pixel-to-centimeter conversion ratio is already known, you may include it:

```bash
python scripts/run_open_field.py \
--base-dir " \
--skip-seconds 5 \
--analyze-seconds 300 \
--pixel-to-cm 0.045
```

### 9.2 Running Y-Maze Analysis

```bash
python scripts/run_ymaze.py \
--base-dir " \
--skip-seconds 5 \
--analyze-seconds 300
```

Similarly, you may include:

```bash
python scripts/run_ymaze.py \
--base-dir " \
--skip-seconds 5 \
--analyze-seconds 300 \
--pixel-to-cm 0.045
```

## 10. Suggestions for Interpreting Results

### Open Field

Common Interpretive Angles:

- Total Distance: General activity level
- Mean Speed: Level of locomotion speed
- Center Time / Center Ratio: One of the rough indicators of anxiety-like behavior
- Center Entries: Willingness to explore the central zone

### Y-Maze

Common Interpretive Angles:

- SAP_percent: Spatial working memory or Spontaneous Alternation Performance
- Arm entries: Exploratory drive
- Center vs Arm time: Balance between lingering in the center and exploring the arms
- Arm sequence: The original arm entry sequence, provided to facilitate manual verification.

---

## 11. Limitations and Important Considerations for the Current Version

1. Trajectory units default to pixels.
If the `--pixel-to-cm` argument is not provided, distance and velocity units default to px and px/s, respectively.

2. Relies on manually defined blue ROIs.
This constitutes the most critical source of robustness in the current version.

3. Uses the position from the previous frame to bridge gaps during severe occlusion.
This mechanism is designed to prevent trajectory fragmentation caused by transient detection failures.

4. Automatic zone partitioning for the Y-maze requires manual verification.
It is strongly recommended to always review the `*_Zone_Debug.jpg` file to ensure that the Center, Arm 1, Arm 2, and Arm 3 zones have been partitioned logically.

5. The definition of the Center Zone is an engineering-based approximation.
In the Open Field test, the Center Zone is currently defined based on a proportional ratio relative to the bounding box of the ROI. If you require strict adherence to your laboratory's established protocols, you may further adjust the `center_ratio` parameter or switch to defining the zone using fixed dimensions in centimeters.


## 12. Potential Directions for Future Expansion

Future enhancements could include:

- Batch video QC (Quality Control) reports
- More robust occlusion recovery capabilities
- Detection of body orientation (head/tail recognition)
- A more formalized statistical analysis pipeline
- Automatic generation of publication-quality figures
- Transitioning ROI annotation to an interactive GUI
- Outputting tables of behavioral events (e.g., freezing episodes, mobility bouts)


## 13. Citation and Attribution Guidelines

If this repository is utilized for internal group sharing, supplementary materials for publications, methodological documentation, or project archiving, it is recommended that the following attribution be retained:

> ISTBI, Fudan University, Xu Lab.
> Contact: shumaoxu@fudan.edu.cn
