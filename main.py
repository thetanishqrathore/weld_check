import os
import sys

# Ensure scripts directory is in path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from extract_roi import process_fsw_images
from mitigate_glare import process_glare
from gabor_filter import run_ab_test
from extract_features import extract_features

def main():
    base_dir = os.path.join(os.path.dirname(__file__), 'images')
    
    # Define directories
    raw_dir = os.path.join(base_dir, 'raw_')
    roi_dir = os.path.join(base_dir, 'phase1_roi')
    glare_dir = os.path.join(base_dir, 'phase2_glare')
    gabor_dir = os.path.join(base_dir, 'phase3_gabor')
    extraction_dir = os.path.join(base_dir, 'phase4_extraction')
    
    print("==================================================")
    print("   FRICTION STIR WELDING DEFECT PIPELINE v2.0     ")
    print("==================================================\\n")

    print("[1/4] Phase 1: Dynamic ROI Extraction")
    process_fsw_images(raw_dir, roi_dir)
    print("--------------------------------------------------")

    print("[2/4] Phase 2: Illumination Normalization (Glare Mitigation)")
    process_glare(roi_dir, glare_dir)
    print("--------------------------------------------------")
    
    print("[3/4] Phase 3: Gabor Filter Bank (Frequency Analysis)")
    # run_ab_test uses the phase1_roi images internally, so we pass roi_dir as input
    run_ab_test(roi_dir, gabor_dir)
    print("--------------------------------------------------")

    print("[4/4] Phase 4 & 5: Morphological Fusion & Spatial-Geometric Classification")
    # extract_features uses the phase1_roi images internally to re-create the pipeline before extracting
    extract_features(roi_dir, extraction_dir)
    print("==================================================")
    print("PIPELINE EXECUTION COMPLETE. Review images/phase4_extraction for Defect Analytics.")

if __name__ == "__main__":
    main()
