import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def build_filters():
    filters = []
    ksize = 31
    # We want to catch different widths of defects (cracks vs wide grooves)
    wavelengths = [5, 10, 15] 
    # Orientations: 0 (horizontal), 45, 90 (vertical), 135
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for theta in orientations:
        for lambd in wavelengths:
            # sigma defines the standard deviation of the Gaussian envelope
            sigma = 0.5 * lambd 
            # gamma is spatial aspect ratio (1.0 = circular footprint)
            gamma = 1.0 
            # psi is phase offset
            psi = 0 
            
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
            filters.append(kern)
    return filters

def process_gabor(img, filters):
    # Convolve image with all filters
    accum = np.zeros_like(img, dtype=np.float32)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_32F, kern)
        np.maximum(accum, np.absolute(fimg), accum)
    return accum

def run_ab_test(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # We load the output from Phase 2
    # The Telea inpainted (Column 3) is saved as norm_X, but wait, my previous script overwrote it.
    # Let's re-run the relevant part of Phase 2 here to dynamically generate the A vs B arrays.
    
    roi_paths = sorted(glob.glob(os.path.join('/Users/tanishqrathore/Developer/weld_check_v2/images/phase1_roi', 'roi_*.png')))
    filters = build_filters()
    
    for img_path in roi_paths:
        filename = os.path.basename(img_path)
        print(f"Running Gabor A/B Test on {filename}...")
        
        img = cv2.imread(img_path)
        if img is None: continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- PHASE 2 RE-CREATION FOR A/B ---
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # COLUMN 3: Inpainted Only (The "A" Test)
        inpainted = cv2.inpaint(img, dilated_mask, 3, cv2.INPAINT_TELEA)
        inpainted_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
        
        # COLUMN 4: Inpainted + CLAHE (The "B" Test)
        lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        clahe_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        clahe_gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
        
        # --- PHASE 3 GABOR EXECUTION ---
        # Generate Texture Energy Maps
        energy_map_A = process_gabor(inpainted_gray, filters)
        energy_map_B = process_gabor(clahe_gray, filters)
        
        # Calculate strict statistical threshold (Mean + 2 Standard Deviations)
        mean_A, std_A = cv2.meanStdDev(energy_map_A)
        thresh_val_A = mean_A[0][0] + (2.0 * std_A[0][0])
        _, binary_A = cv2.threshold(energy_map_A, thresh_val_A, 255, cv2.THRESH_BINARY)
        
        mean_B, std_B = cv2.meanStdDev(energy_map_B)
        thresh_val_B = mean_B[0][0] + (2.0 * std_B[0][0])
        _, binary_B = cv2.threshold(energy_map_B, thresh_val_B, 255, cv2.THRESH_BINARY)
        
        # Convert back to uint8 for visualization
        binary_A = np.uint8(binary_A)
        binary_B = np.uint8(binary_B)
        
        # Normalize energy maps for display purposes (0-255)
        vis_energy_A = cv2.normalize(energy_map_A, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        vis_energy_B = cv2.normalize(energy_map_B, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # --- VISUALIZATION ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Row 1: The "A" Test (Inpainted Only - Clean Signal)
        axes[0, 0].imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Input: Telea Inpaint Only')
        axes[0, 1].imshow(vis_energy_A, cmap='nipy_spectral')
        axes[0, 1].set_title('Texture Energy Map (Heatmap)')
        axes[0, 2].imshow(binary_A, cmap='gray')
        axes[0, 2].set_title('Defect Segmentation (Mean + 2σ)')
        
        # Row 2: The "B" Test (Inpainted + CLAHE - Corrupted Signal)
        axes[1, 0].imshow(cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Input: Inpaint + CLAHE')
        axes[1, 1].imshow(vis_energy_B, cmap='nipy_spectral')
        axes[1, 1].set_title('Texture Energy Map (CLAHE Corrupted)')
        axes[1, 2].imshow(binary_B, cmap='gray')
        axes[1, 2].set_title('Defect Segmentation (Failed Output)')
        
        for ax in axes.flatten(): ax.axis('off')
        
        plt.tight_layout()
        idx = filename.replace("roi_", "")
        out_path = os.path.join(output_dir, f'gabor_ab_{idx}')
        plt.savefig(out_path, dpi=150)
        plt.close()

if __name__ == "__main__":
    input_directory = '/Users/tanishqrathore/Developer/weld_check_v2/images/phase2_glare'
    output_directory = '/Users/tanishqrathore/Developer/weld_check_v2/images/phase3_gabor'
    run_ab_test(input_directory, output_directory)
