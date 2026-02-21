import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


CLASSES = ["keyhole", "crack_tear", "groove", "void_pit", "excess_expulsion", "all_defects"]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def imread_gray(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img


def save_png(p: Path, img: np.ndarray):
    ensure_dir(p.parent)
    cv2.imwrite(str(p), img)


def component_pca_metrics(mask_u8: np.ndarray):
    """Return angle_deg, major, minor, aspect using PCA on foreground pixels."""
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 10:
        return 0.0, 0.0, 0.0, 1.0
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    pts_c = pts - mean
    cov = np.cov(pts_c.T)
    vals, vecs = np.linalg.eigh(cov)
    major_vec = vecs[:, 1]
    angle = np.degrees(np.arctan2(major_vec[1], major_vec[0]))
    major = float(np.sqrt(max(vals[1], 0.0)) * 6.0)
    minor = float(np.sqrt(max(vals[0], 0.0)) * 6.0)
    aspect = major / (minor + 1e-6)
    return float(angle), major, minor, float(aspect)


def contour_metrics(component_mask_u8: np.ndarray):
    """Perimeter, circularity via contour."""
    cnts, _ = cv2.findContours(component_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    per = float(cv2.arcLength(c, True))
    circ = 0.0
    if per > 1e-6:
        circ = float(4.0 * np.pi * area / (per * per))
    return per, circ


def profiles_from_mask(mask_u8: np.ndarray):
    """Return (count_per_x, frac_per_x) for a binary mask."""
    m = (mask_u8 > 0).astype(np.uint8)
    H, W = m.shape
    count = m.sum(axis=0).astype(np.float32)               # white pixels per x
    frac = count / max(1.0, float(H))                      # normalized by ROI height
    return count, frac


def plot_profile(x_vals, y_vals, out_path: Path, title: str, ylabel: str):
    plt.figure()
    plt.plot(x_vals, y_vals)
    plt.title(title)
    plt.xlabel("x (column index along weld length)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=160)
    plt.close()


def annotate_blobs(gray: np.ndarray, blobs: list, out_path: Path):
    """
    blobs: list of dict with keys: cls, bbox(x,y,w,h), id
    Draw boxes + labels on ROI.
    """
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for b in blobs:
        x, y, w, h = b["bbox"]
        cls = b["cls"]
        bid = b["id"]
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            bgr,
            f"{cls}:{bid}",
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    save_png(out_path, bgr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase4_root", type=str, default="images/phase_4_classification")
    ap.add_argument("--roi_root", type=str, default="images/phase_3_2_defect_masks")
    ap.add_argument("--out_root", type=str, default="images/phase_5_metrics")
    ap.add_argument("--min_area", type=int, default=25)  # ignore tiny specks
    args = ap.parse_args()

    phase4_root = Path(args.phase4_root)
    roi_root = Path(args.roi_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    stems = sorted([p.name for p in phase4_root.iterdir() if p.is_dir()])
    if not stems:
        raise SystemExit(f"No stems found under {phase4_root}")

    for stem in stems:
        masks_dir = phase4_root / stem / "masks"
        if not masks_dir.exists():
            print(f"Skipping {stem} (no masks dir)")
            continue

        # ROI for overlay context
        roi_path = roi_root / stem / "01_input.png"
        if not roi_path.exists():
            # fallback (in case filename differs)
            cand = list((roi_root / stem).glob("*.png"))
            roi_path = cand[0] if cand else None
        if roi_path is None or not Path(roi_path).exists():
            print(f"Skipping {stem} (missing ROI image)")
            continue

        gray = imread_gray(Path(roi_path))
        H, W = gray.shape

        out_dir = out_root / stem
        plots_dir = out_dir / "plots"
        overlay_dir = out_dir / "overlay"
        ensure_dir(plots_dir)
        ensure_dir(overlay_dir)

        # Load class masks
        masks = {}
        for cls in CLASSES:
            p = masks_dir / f"{cls}.png"
            if p.exists():
                masks[cls] = (imread_gray(p) > 0).astype(np.uint8) * 255
            else:
                masks[cls] = np.zeros((H, W), np.uint8)

        # Features extraction
        rows = []
        blob_annots = []
        summary = {"shape": [int(H), int(W)], "classes": {}}

        for cls in ["keyhole", "crack_tear", "groove", "void_pit", "excess_expulsion"]:
            m = masks[cls]
            num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)

            kept = 0
            total_area = 0
            for i in range(1, num):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < args.min_area:
                    continue

                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])
                cx, cy = centroids[i]
                comp = (labels == i).astype(np.uint8) * 255

                angle, major, minor, aspect = component_pca_metrics(comp)
                per, circ = contour_metrics(comp)

                kept += 1
                total_area += area

                rows.append({
                    "stem": stem,
                    "class": cls,
                    "blob_id": kept,
                    "area_px": area,
                    "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
                    "centroid_x": float(cx), "centroid_y": float(cy),
                    "angle_deg": float(angle),
                    "major_est": float(major),
                    "minor_est": float(minor),
                    "aspect_est": float(aspect),
                    "perimeter": float(per),
                    "circularity": float(circ),
                })

                blob_annots.append({"cls": cls, "id": kept, "bbox": (x, y, w, h)})

            summary["classes"][cls] = {
                "blob_count": int(kept),
                "total_area_px": int(total_area),
                "area_ratio_of_roi": float(total_area / max(1, H * W)),
            }

        # Save CSV
        csv_path = out_dir / "features.csv"
        ensure_dir(csv_path.parent)
        with open(csv_path, "w", newline="") as f:
            fieldnames = list(rows[0].keys()) if rows else [
                "stem","class","blob_id","area_px","bbox_x","bbox_y","bbox_w","bbox_h",
                "centroid_x","centroid_y","angle_deg","major_est","minor_est","aspect_est",
                "perimeter","circularity"
            ]
            wri = csv.DictWriter(f, fieldnames=fieldnames)
            wri.writeheader()
            for r in rows:
                wri.writerow(r)

        # Profiles (per class + all)
        x = np.arange(W, dtype=np.int32)
        for cls in ["keyhole", "crack_tear", "groove", "void_pit", "excess_expulsion"]:
            count, frac = profiles_from_mask(masks[cls])
            plot_profile(
                x, count,
                plots_dir / f"profile_{cls}.png",
                title=f"{stem} | {cls} | defect pixels per x",
                ylabel="white pixels"
            )

        # All defects (from Phase 4 mask if present; else fuse)
        all_mask = masks["all_defects"]
        if np.sum(all_mask) == 0:
            all_mask = np.zeros((H, W), np.uint8)
            for cls in ["keyhole", "crack_tear", "groove", "void_pit", "excess_expulsion"]:
                all_mask = cv2.bitwise_or(all_mask, masks[cls])

        all_count, all_frac = profiles_from_mask(all_mask)
        plot_profile(
            x, all_count,
            plots_dir / "profile_all_defects.png",
            title=f"{stem} | all_defects | defect pixels per x",
            ylabel="white pixels"
        )
        plot_profile(
            x, all_frac,
            plots_dir / "area_fraction_all_defects.png",
            title=f"{stem} | all_defects | area fraction per x",
            ylabel="fraction of ROI height"
        )

        # Overlay annotated blobs
        annotate_blobs(gray, blob_annots, overlay_dir / "annotated_blobs.png")

        # Save summary JSON
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Done. Wrote Phase 5 metrics+plots to: {out_root}")


if __name__ == "__main__":
    main()