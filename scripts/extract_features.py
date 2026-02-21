import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

def extract_features(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Reload original ROIs for overlay visualization
    roi_dir = '/Users/tanishqrathore/Developer/weld_check_v2/images/phase1_roi'
    
    # We load the Gabor filter script's raw binary mask from the A test (Column 3)
    # However, since gabor_filter.py didn't save the raw binary masks to disk,
    # we need to recreate the Gabor pipeline on the Telea inpainted ROIs internally here.
    
    # Import logic from gabor_filter.py
    from gabor_filter import build_filters, process_gabor
    filters = build_filters()
    
    image_paths = sorted(glob.glob(os.path.join(roi_dir, 'roi_*.png')))
    
    all_metrics = []
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img_id = filename.replace("roi_", "").replace(".png", "")
        print(f"Extracting features from Image {img_id}...")
        
        orig_img = cv2.imread(img_path)
        if orig_img is None: continue
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Telea Inpainting Background
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        inpainted = cv2.inpaint(orig_img, dilated_mask, 3, cv2.INPAINT_TELEA)
        inpainted_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
        
        # 2. Gabor Execution (Phase 3 Row 1)
        energy_map = process_gabor(inpainted_gray, filters)
        mean_val, std_val = cv2.meanStdDev(energy_map)
        thresh_val = mean_val[0][0] + (2.0 * std_val[0][0])
        _, binary = cv2.threshold(energy_map, thresh_val, 255, cv2.THRESH_BINARY)
        gabor_binary = np.uint8(binary)
        
        # 3. Morphological Fusion (The Bridge)
        # Using a heavy disk-shaped structuring element
        disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fused_binary = cv2.morphologyEx(gabor_binary, cv2.MORPH_CLOSE, disk_kernel)
        
        # Optional: remove tiny specks (noise) smaller than a certain area (e.g. 100px)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fused_binary, connectivity=8)
        clean_binary = np.zeros_like(fused_binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 50:
                clean_binary[labels == i] = 255
        
        # 4. Connected Components & Geometric Extraction
        contours, _ = cv2.findContours(clean_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        overlay = orig_img.copy()
        
        # Dimensions for spatial bounds
        h, w = orig_img.shape[:2]
        
        blob_data = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 50: continue # Skip noise
            
            # Spatial Metrics
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 10, 10
                
            # Geometric Metrics
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                center, axes, angle = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
            else:
                rect = cv2.minAreaRect(contour)
                center, dims, angle = rect
                major_axis = max(dims)
                minor_axis = min(dims)
            
            major_axis = max(1.0, major_axis)
            minor_axis = max(1.0, minor_axis)
            aspect_ratio = major_axis / minor_axis
            
            defect_class = "General Porosity/Void" # Tier 3: Exhaustive Catch-all
            
            # Tier 1: Spatial Filters
            if cY < (0.15 * h) or cY > (0.85 * h):
                defect_class = "Flash/Edge Defect"
            elif cX < (0.20 * w) and aspect_ratio < 2.5:
                defect_class = "Keyhole"
            else:
                # Tier 2: Geometric Filters (Scale-invariant)
                if aspect_ratio >= 3.0:
                    defect_class = "Crack/Groove"
            
            blob_data.append({
                "contour": contour,
                "Blob_ID": i,
                "Class": defect_class,
                "Centroid_X": cX,
                "Centroid_Y": cY,
                "Area_px": round(area, 2),
                "Aspect_Ratio": round(aspect_ratio, 2)
            })

        # Tier 1 Conflict Resolution: Master Keyhole Aggregation
        # A single weld run can only produce one terminal keyhole.
        keyholes = [b for b in blob_data if b["Class"] == "Keyhole"]
        if len(keyholes) > 1:
            # The true keyhole is the one closest to the physical extraction point (lowest X)
            keyholes.sort(key=lambda x: x["Centroid_X"])
            # Keep the first one, downgrade all others to Tier 3 Porosity/Void
            for false_keyhole in keyholes[1:]:
                false_keyhole["Class"] = "General Porosity/Void"

        # Pass 3: Draw and save metrics
        for blob in blob_data:
            defect_class = blob["Class"]
            cX = blob["Centroid_X"]
            cY = blob["Centroid_Y"]
            
            color = (200, 200, 200) # Gray
            if defect_class == "Flash/Edge Defect":
                color = (255, 0, 255) # Magenta
            elif defect_class == "Keyhole":
                color = (0, 0, 255) # Red
            elif defect_class == "Crack/Groove":
                color = (0, 255, 255) # Yellow
            
            cv2.drawContours(overlay, [blob["contour"]], -1, color, 2)
            cv2.putText(overlay, f"{defect_class}", (cX - 20, max(10, cY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            all_metrics.append({
                "Image": img_id,
                "Blob_ID": blob["Blob_ID"],
                "Class": defect_class,
                "Centroid_X": cX,
                "Centroid_Y": cY,
                "Area_px": blob["Area_px"],
                "Aspect_Ratio": blob["Aspect_Ratio"]
            })

            
        # Visualization output
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        axes[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original ROI')
        
        axes[1].imshow(gabor_binary, cmap='gray')
        axes[1].set_title('Phase 3: Shattered Gabor Mask')
        
        axes[2].imshow(clean_binary, cmap='gray')
        axes[2].set_title('Phase 3.5: Morphological Fusion (Disk 15x15)')
        
        axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[3].set_title('Phase 4: Geometric Class Overlay')
        
        for ax in axes: ax.axis('off')
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'extraction_{img_id}.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        
    df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(output_dir, 'defect_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\\nExtraction complete. Metrics saved to {csv_path}")
    print("\\nSample Metrics:")
    print(df.head(15))

if __name__ == "__main__":
    output_directory = '/Users/tanishqrathore/Developer/weld_check_v2/images/phase4_extraction'
    extract_features(None, output_directory)
