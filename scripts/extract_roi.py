import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def process_fsw_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. Aggressive Low-Pass Filter
        # Use a large kernel relative to image size. 51x51 works well for high-res.
        kernel_size = 51
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # 2. Directional Gradient Extraction (Sobel Y)
        # cv2.CV_64F prevents clipping of negative gradient values
        sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_y = np.absolute(sobel_y)
        # Normalize back to 8-bit for Otsu
        sobel_8u = np.uint8(255 * abs_sobel_y / np.max(abs_sobel_y))
        
        # 3. Binarization (Otsu's Method)
        _, binary = cv2.threshold(sobel_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. Vertical Projection Profile
        # Sum white pixels across every row
        projection = np.sum(binary == 255, axis=1)
        
        # 5. Boundary Extraction & Cropping
        max_peak = np.max(projection)
        # Dynamic noise floor: e.g., 20% of the maximum peak
        threshold_val = max_peak * 0.2
        
        y_min = 0
        for i in range(len(projection)):
            if projection[i] > threshold_val:
                y_min = i
                break
                
        y_max = len(projection) - 1
        for i in range(len(projection)-1, -1, -1):
            if projection[i] > threshold_val:
                y_max = i
                break
                
        # Optional: Add small padding
        padding = 10
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        roi = img[y_min:y_max, :]
        
        # --- Visualization Generation ---
        fig, axes = plt.subplots(1, 5, figsize=(24, 6))
        
        # Original with lines
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axhline(y=y_min, color='r', linestyle='-', linewidth=2)
        axes[0].axhline(y=y_max, color='r', linestyle='-', linewidth=2)
        axes[0].axis('off')
        
        # Blurred
        axes[1].imshow(blur, cmap='gray')
        axes[1].set_title(f'Gaussian Blur ({kernel_size}x{kernel_size})')
        axes[1].axis('off')
        
        # Binary Sobel
        axes[2].imshow(binary, cmap='gray')
        axes[2].set_title("Otsu's Binarization (Sobel Y)")
        axes[2].axis('off')
        
        # Projection Profile
        axes[3].plot(projection, range(len(projection)))
        axes[3].invert_yaxis()
        axes[3].set_title('Vertical Projection Profile')
        axes[3].axhline(y=y_min, color='r', linestyle='--', linewidth=1.5, label='y_min')
        axes[3].axhline(y=y_max, color='r', linestyle='--', linewidth=1.5, label='y_max')
        axes[3].set_xlabel('Sum of White Pixels')
        axes[3].set_ylabel('Y-coordinate (Rows)')
        axes[3].legend()
        
        # Cropped ROI
        axes[4].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        axes[4].set_title('Extracted ROI')
        axes[4].axis('off')
        
        plt.tight_layout()
        summary_path = os.path.join(output_dir, f'summary_{filename}')
        plt.savefig(summary_path, dpi=150)
        plt.close()
        
        roi_path = os.path.join(output_dir, f'roi_{filename}')
        cv2.imwrite(roi_path, roi)
        
    print(f"\\nProcessing complete. All results saved to {output_dir}")

if __name__ == "__main__":
    input_directory = '/Users/tanishqrathore/Developer/weld_check_v2/images/raw_'
    output_directory = '/Users/tanishqrathore/Developer/weld_check_v2/images/phase1_roi'
    process_fsw_images(input_directory, output_directory)
