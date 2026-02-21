import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def process_glare(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(input_dir, 'roi_*.png')))
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        img = cv2.imread(img_path)
        if img is None: continue
            
        # 1. Glare Masking (Thresholding high intensity pixels)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 240 represents pure specular clipping on standard 8-bit sensors
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # Dilate mask to ensure we cover the optical fringes of the glare
        kernel = np.ones((5,5), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 2. Inpainting (Telea method)
        inpainted = cv2.inpaint(img, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # 3. Illumination Normalization (CLAHE via LAB space)
        lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L (Lightness) channel only
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        merged = cv2.merge((cl, a, b))
        clahe_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        # --- Visualization Generation ---
        idx = filename.replace("roi_", "").replace(".png", "")
        
        cv2.imwrite(os.path.join(output_dir, f'{idx}_01_glare_mask.png'), dilated_mask)
        cv2.imwrite(os.path.join(output_dir, f'{idx}_02_telea_inpainted.png'), inpainted)
        
        # We will still save CLAHE since the pipeline theoretically keeps it for documentation,
        # even if Phase 3 skips it.
        clahe_path = os.path.join(output_dir, f'{idx}_03_clahe_norm.png')
        cv2.imwrite(clahe_path, clahe_bgr)
        
    print(f"\nPhase 2 processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    input_directory = '/Users/tanishqrathore/Developer/weld_check_v2/images/phase1_roi'
    output_directory = '/Users/tanishqrathore/Developer/weld_check_v2/images/phase2_glare'
    process_glare(input_directory, output_directory)
