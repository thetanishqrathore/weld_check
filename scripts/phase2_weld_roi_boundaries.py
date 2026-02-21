import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
from pathlib import Path


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def widen_band_adaptive(projection, y_min, y_max, h,
                        bg_quantile=0.90,
                        stop_factor=1.20,
                        consec=6,
                        hard_pad=10,
                        max_expand=120):
    proj = projection.astype(np.float32)

    # background level from outer 10% rows (top+bottom)
    k = max(1, h // 10)
    bg = np.concatenate([proj[:k], proj[-k:]])
    bg_level = float(np.quantile(bg, bg_quantile))

    # expand upward
    y0 = y_min
    below = 0
    steps = 0
    while y0 > 0 and steps < max_expand:
        y0 -= 1
        steps += 1
        if proj[y0] <= (bg_level * stop_factor):
            below += 1
            if below >= consec:
                break
        else:
            below = 0

    # expand downward
    y1 = y_max
    below = 0
    steps = 0
    while y1 < h - 1 and steps < max_expand:
        y1 += 1
        steps += 1
        if proj[y1] <= (bg_level * stop_factor):
            below += 1
            if below >= consec:
                break
        else:
            below = 0

    # final pad
    y0 = max(0, y0 - hard_pad)
    y1 = min(h - 1, y1 + hard_pad)
    return int(y0), int(y1), bg_level, bg_level * stop_factor



def process_fsw_images(input_dir, output_dir,
                       kernel_size=51,
                       proj_frac=0.20,
                       proj_percentile=85,
                       padding=10):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    image_paths = sorted(
        [Path(p) for p in glob.glob(str(input_dir / "*.png"))]
    )

    if not image_paths:
        raise SystemExit(f"No .png images found in {input_dir}")

    for img_path in image_paths:
        stem = img_path.stem
        od = output_dir / stem
        ensure_dir(od)

        print(f"Processing {img_path.name}...")

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to load {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1) Aggressive low-pass to remove onion-ring / micro texture
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # 2) Sobel-Y on blurred image (horizontal edges)
        sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_y = np.absolute(sobel_y)

        # Normalize to 8-bit safely
        mx = np.max(abs_sobel_y)
        if mx < 1e-6:
            sobel_8u = np.zeros_like(gray, dtype=np.uint8)
        else:
            sobel_8u = np.uint8(255 * abs_sobel_y / mx)

        # 3) Otsu binarization on gradient magnitude
        _, binary = cv2.threshold(sobel_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4) Vertical projection: count edge pixels per row
        projection = np.sum(binary == 255, axis=1).astype(np.float32)

        max_peak = float(np.max(projection)) if projection.size else 0.0
        pctl = float(np.percentile(projection, proj_percentile)) if projection.size else 0.0

        # 5) Dynamic threshold floor (more robust than pure % of max)
        thr = max(max_peak * proj_frac, pctl)

        # Find y_min / y_max
        y_min = 0
        for i in range(len(projection)):
            if projection[i] > thr:
                y_min = i
                break

        y_max = len(projection) - 1
        for i in range(len(projection) - 1, -1, -1):
            if projection[i] > thr:
                y_max = i
                break

        # Padding
        y_min, y_max, bg_level, stop_thr = widen_band_adaptive(
    projection, y_min, y_max, h,
            bg_quantile=0.90,
            stop_factor=1.20,
            consec=6,
            hard_pad=10,
            max_expand=120
        )

        # ROI crop (full width, band height)
        roi = img[y_min:y_max + 1, :]

        # --- Save debug artifacts ---
        cv2.imwrite(str(od / "01_blur.png"), blur)
        cv2.imwrite(str(od / "02_sobel_binary.png"), binary)
        cv2.imwrite(str(od / "03_roi.png"), roi)

        # Original with lines
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.axhline(y=y_min, color='r', linewidth=2)
        ax1.axhline(y=y_max, color='r', linewidth=2)
        ax1.axis('off')
        plt.tight_layout()
        plt.savefig(str(od / "04_original_bounds.png"), dpi=150)
        plt.close(fig1)

        # Projection profile plot
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(projection, np.arange(len(projection)))
        ax2.invert_yaxis()
        ax2.axhline(y=y_min, color='r', linestyle='--', linewidth=1.5, label='y_min')
        ax2.axhline(y=y_max, color='r', linestyle='--', linewidth=1.5, label='y_max')
        ax2.axvline(x=thr, color='k', linestyle=':', linewidth=1.0, label='threshold')
        ax2.set_xlabel('Edge-pixel count per row')
        ax2.set_ylabel('Row (y)')
        ax2.axvline(x=stop_thr, color='g', linestyle=':', linewidth=1.0, label='stop_thr')
        ax2.legend()
        plt.tight_layout()
        plt.savefig(str(od / "05_projection_profile.png"), dpi=150)
        plt.close(fig2)

        # Meta
        meta = {
            "input": str(img_path),
            "shape": [int(h), int(w)],
            "params": {
                "kernel_size": int(kernel_size),
                "proj_frac": float(proj_frac),
                "proj_percentile": int(proj_percentile),
                "padding": int(padding),
                "thr": float(thr),
                "max_peak": float(max_peak),
                "proj_percentile_value": float(pctl),
            },
            "roi": {"y_min": int(y_min), "y_max": int(y_max)}
        }
        (od / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone. Phase 2 outputs saved to: {output_dir}")


if __name__ == "__main__":
    process_fsw_images(
        input_dir="images/raw_",
        output_dir="images/phase_2_weld_roi_boundaries",
        kernel_size=51,
        proj_frac=0.20,
        proj_percentile=85,
        padding=10
    )
