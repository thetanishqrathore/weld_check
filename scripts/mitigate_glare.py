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
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original ROI')
        axes[0].axis('off')
        
        axes[1].imshow(dilated_mask, cmap='gray')
        axes[1].set_title('Dilated Glare Mask (>240)')
        axes[1].axis('off')
        
        axes[2].imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Inpainted ROI (Telea)')
        axes[2].axis('off')
        
        axes[3].imshow(cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB))
        axes[3].set_title('Inpainted + CLAHE')
        axes[3].axis('off')
        
        plt.tight_layout()
        idx = filename.replace("roi_", "")
        summary_path = os.path.join(output_dir, f'summary_glare_{idx}')
        plt.savefig(summary_path, dpi=150)
        plt.close()
        
        out_path = os.path.join(output_dir, f'norm_{idx}')
        cv2.imwrite(out_path, clahe_bgr)
        
    print(f"\\nPhase 2 processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    input_directory = '/Users/tanishqrathore/Developer/weld_check_v2/images/phase1_roi'
    output_directory = '/Users/tanishqrathore/Developer/weld_check_v2/images/phase2_glare'
    process_glare(input_directory, output_directory)
