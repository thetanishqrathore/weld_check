# Weld Check

Weld Check is a comprehensive computer vision pipeline designed for the automated analysis and classification of surface defects in Friction Stir Welding (FSW) optical images. It utilizes classical image processing, spatial-geometric heuristics, and morphological analysis to extract highly accurate defect measurements without the need for large training datasets.

## Pipeline Overview

The analysis is broken down into 5 sequential phases:

1. **Phase 1: Illumination Flattening** (`phase1_illumination_flattening.py`)
   - Corrects harsh lighting gradients using Homomorphic Filtering.
   - Suppresses and mathematically inpaints extreme specular highlights (glare).
   - Enhances local contrast of the metallic surface safely using CLAHE.

2. **Phase 2: Dynamic ROI Extraction** (`phase2_weld_roi_boundaries.py`)
   - Aggressively filters micro-textures to isolate macro plate boundaries.
   - Automatically crops the image to the strict vertical weld zone using Sobel-Y gradients and vertical projection profiling.

3. **Phase 3: Defect Segmentation** (`phase3_keyhole_module.py`, `phase3_1_traverse_edges.py`, `phase3_2_defect_masks.py`)
   - Identifies and isolates the terminal exit "keyhole".
   - Extracts and fuses massive structural anomalies (using Difference of Gaussians) and fine micro-fractures (using Sobel hysteresis growing).

4. **Phase 4: Spatial-Geometric Classification** (`phase4_classify_defects.py`)
   - Classifies contiguous defect blobs into formal defect categories:
     - **Crack/Tear**: Highly directional, eccentric micro-anomalies.
     - **Groove**: Elongated longitudinal trenches.
     - **Excess Expulsion (Flash)**: Anomalies residing on the extreme vertical shoulders of the weld.
     - **Void/Pit**: Generic volumetric porosities securely contained within the center weld zone.

5. **Phase 5: Metrics Extraction & Plots** (`phase5_metrics_and_plots.py`)
   - Computes precise geometric measurements (Area, True Aspect Ratio, Circularity, etc.) via PCA.
   - Generates visual bounding boxes and severity profiles across the weld length.
   - Exports all raw blob data to a comprehensive `features.csv`.

For an in-depth explanation of the algorithms powering each phase, please refer to [Walkthrough.md](./Walkthrough.md).

## Requirements

This project uses [`uv`](https://github.com/astral-sh/uv) as its package manager to handle dependencies and virtual environments swiftly. It explicitly manages known library conflicts (e.g. enforcing `numpy<2` to preserve compatibility).

- **Python 3.12+**
- **uv** (Install via `curl -LsSf https://astral.sh/uv/install.sh | sh` or `pip install uv`)

## Setup & Usage

1. **Prepare your Data**
   Place your raw FSW optical images (e.g., `.png`, `.jpg`) into the designated input directory. The directory must be created if it doesn't exist:
   ```bash
   mkdir -p images/raw_
   # Copy your images into images/raw_
   ```

2. **Run the Pipeline**
   The entire sequence of scripts is orchestrated by `main.py`. Because `uv` is configured via `pyproject.toml`, you can execute the pipeline seamlessly with one command. `uv` will automatically provision the environment and install necessary modules:
   ```bash
   uv run main.py
   ```

## Outputs

The pipeline generates progressive outputs organized by phase within the `images/` directory for every input image.

- `images/phase_1_illumination_flattening/`: Glare-corrected images.
- `images/phase_2_weld_roi_boundaries/`: The tight ROI crops.
- `images/phase_3_keyhole/`: Keyhole detections and masks.
- `images/phase_3_1_edges/`: Raw edge detection maps.
- `images/phase_3_2_defect_masks/`: Solid boolean defect segmentations.
- `images/phase_4_classification/`: Defect blobs separated by class.
- `images/phase_5_metrics/`: The final results. Look here for:
  - `features.csv`: All calculated structural and spatial metrics.
  - `overlay/annotated_blobs.png`: The original weld image overlaid with identified defect classes and localized bounding boxes.
  - `plots/`: Density profile graphs showing the severity of defects continuously along the weld plane.
