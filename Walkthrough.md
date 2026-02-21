# Weld Check Pipeline Walkthrough

This document outlines the entire end-to-end Friction Stir Welding (FSW) defect analysis pipeline implemented in `weld_check`. The pipeline processes raw optical images of welds through five distinct phases to extract, emphasize, classify, and measure surface anomalies.

The entire process is orchestrated by `main.py`, which sequences the scripts sequentially.

---

## Phase 1: Illumination Flattening (`phase1_illumination_flattening.py`)
**Goal:** Remove unequal lighting (vignetting/gradients) and suppress stark specular highlights without degrading the actual structural details of the weld bead.

**Steps:**
1. **Grayscale Conversion**: The colored input image is collapsed to grayscale.
2. **Homomorphic Filtering**: 
   - Converts the image to the logarithmic domain to separate illumination (multiplicative) from reflectance.
   - Applies a heavy Gaussian Blur (`sigma=45.0`) to estimate the low-frequency illumination map.
   - Subtracts this illumination map from the log image and exponents it back, leaving the flattened reflectance.
3. **Highlight Masking**: Isolates extreme specular highlights using a dynamic 99.5th percentile threshold. Small speckles are cleaned using morphological opening, and the rims of the glare are dilated.
4. **Telea Inpainting**: Reconstructs the burned-out glare pixels by mathematically interpolating the surrounding valid metal textures using the Fast Marching Method.
5. **Limited CLAHE**: Applies Contrast Limited Adaptive Histogram Equalization with a conservative clip limit (`2.0`) to improve local contrast separation without heavily inflating background noise.

---

## Phase 2: Dynamic ROI Extraction (`phase2_weld_roi_boundaries.py`)
**Goal:** Mathematically detect and crop the image to the exact vertical boundaries of the weld plate, discarding irrelevant background context.

**Steps:**
1. **Aggressive Low-Pass Filter**: Applies a massive 51x51 Gaussian Blur to blur out the internal "onion-ring" micro-textures of the weld, leaving only the macro-structural plate boundaries.
2. **Sobel-Y Edge Detection**: Computes absolute vertical gradients to intensely highlight the horizontal top and bottom crevices of the weld region.
3. **Otsu Binarization**: Converts the gradient magnitude into a stark black-and-white mask.
4. **Vertical Projection Profile**: Sums the white pixels across every row (y-axis) to create a 1D histogram of edge strength.
5. **Adaptive Thresholding & Padding**: Finds the strict `y_min` and `y_max` bounds by analyzing when the projection profile drops below a dynamic percentile floor. It aggressively expands these bounds outward until it safely hits the background noise floor, plus a hard 10px safety pad.
6. **ROI Crop**: The image is aggressively cropped to this exact band.

---

## Phase 3: Defect Segmentation

Phase 3 introduces a multi-stream approach to accurately segment extreme, macro-structural anomalies versus fine, micro-structural anomalies.

### Phase 3 Keyhole Detection (`phase3_keyhole_module.py`)
**Goal:** Identify and mask the massive, terminal exit hole ("keyhole") left by the welding tool, isolating it from the rest of the defect analysis pipeline.

**Steps:**
1. **Search Window Isolation**: Crops the search area to only the leftmost 20% of the ROI where the keyhole terminates.
2. **Edge Extraction**: Applies a light Gaussian blur, Canny edge detection, and morphological closing to extract contiguous circular borders.
3. **Primary Detection (Hough Circles)**: Runs the classic Hough Gradient method looking for circles mapping to ~12% to 48% of the ROI height. It scores candidates based on their leftmost bias and valid Canny edge support.
4. **Fallback Detection (Ellipse Fit)**: If Hough fails, it extracts contours and fits a bounding ellipse to the largest defect.
5. **Masking & Blackout**: Generates a solid mask of the keyhole and blacks it out on the working image, preventing it from polluting the subsequent traverse edge detection.

### Phase 3.1: Traverse Edge Detection (`phase3_1_traverse_edges.py`)
**Goal:** Enhance and extract the raw edge responses of defects, split into Macro and Micro streams.

**Steps:**
- **Texture Energy Weld Masking**: Uses local variance to identify the highly-textured weld zone against the smooth surrounding plate, preventing the pipeline from hallucinating defects in the safe background.
- **Stream A (Macro Edges)**: Uses the Difference of Gaussians (DoG) bandpass filter combined with Canny edge detection targeted at large, low-frequency structural anomalies (like massive expulsion or deep valleys).
- **Stream B (Micro Edges)**: Uses morphological opening-by-reconstruction to suppress the background, followed by Sobel magnitude. Extreme edge pixels act as seeds that hysteresis-grow into connected, sharp structural cracks using an upper threshold (99.2nd pctl) and a loose containment threshold (97.0th pctl).

### Phase 3.2: Defect Masks Morphological Fusion (`phase3_2_defect_masks.py`)
**Goal:** Thicken and fuse the shattered, pixel-level edge skeletons into solid, geometric component masks.

**Steps:**
- **Macro Mask**: Takes the top 8% easiest-to-see DoG pixels, applies a heavy morphological closing (9x5 block) to fuse them into broad, solid objects.
- **Micro Mask**: Takes the Sobel magnitude map and performs hysteresis growing. Cleans the binary mask by enforcing strict minimum area constraints and minimum bounding-box spans (rejecting tiny isolated speckles).
- **Fusion**: Performs a bitwise OR, merging the macro-trenches with the highly-resoluted micro-cracks into a single boolean master mask.

---

## Phase 4: Spatial-Geometric Classification (`phase4_classify_defects.py`)
**Goal:** Distinguish the solid, anonymous defect blobs into distinct physical categories based on their spatial orientation and mathematical geometry.

**Steps:**
1. **Keyhole Isolation**: Keyhole pixels are completely removed from all other masks to prevent contamination.
2. **Crack/Tear Extractor (Micro)**: Scans the micro-components. If a component is highly eccentric (Aspect Ratio $\ge$ 4.0) and spans at least 28 pixels, it is classified as a highly-directional, structural tear or crack.
3. **Groove Extractor (Macro + Micro)**: Scans combined segments that have been heavily fused horizontally. If an object is highly elongated (Aspect Ratio $\ge$ 2.0) but orientates within $\pm 35^\circ$ horizontally, it is classified as a longitudinal weld groove.
4. **Spatial Partitioning (Macro)**: Scans all remaining massive blobs. 
   - **Excess Expulsion (Flash)**: If a massive blob's centroid ($Y_{mid}$) resides within 22 pixels of the absolute top or bottom edge of the image (the boundary shoulders), it is mathematically classified as ejected flash material.
   - **Void/Pit (Catch-All)**: Any remaining macro-blob residing internally in the weld zone that failed the eccentricity checks is classified as a cavernous void or surface pit.

---

## Phase 5: Metrics Extraction & Plotting (`phase5_metrics_and_plots.py`)
**Goal:** Generate final human-readable overlays, plot localized severity distributions, and extract analytical metadata to a CSV.

**Steps:**
1. **Component Analysis**: Iterates over every distinct blob in every distinct class mask.
2. **PCA Geometry Extraction**: Uses Principal Component Analysis (Eigenvalue decomposition) on the blob's $(X, Y)$ coordinates to dynamically extract its true rotation angle, Major axis, Minor axis, and true Aspect Ratio.
3. **Contour Metrics**: Calculates perimeter and circularity ($4\pi \cdot \text{Area} / \text{Perimeter}^2$).
4. **Defect Profiling**: Stacks the binary maps and compresses them vertically onto the X-axis, generating a 1D line chart representing the density (severity) of defects across the entire weld length.
5. **Output Generation**:
   - Compiles all metadata explicitly into `features.csv`.
   - Draws bounding boxes and Class ID overlays directly over the ROI images.
   - Saves defect severity line-plots per class.
   - Saves a summary `.json` report containing the aggregate pixel ratios.
