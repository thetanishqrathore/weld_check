# scripts/phase4_classify_defects.py
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


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


def cc_iter(mask_u8: np.ndarray):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        yield i, labels, area, x, y, w, h


def component_orientation_and_axes(component_mask_u8: np.ndarray):
    """
    Compute orientation and ellipse-like axes via PCA on foreground pixels.
    Returns: angle_deg, major, minor, aspect
    """
    ys, xs = np.where(component_mask_u8 > 0)
    if len(xs) < 20:
        return 0.0, 0.0, 0.0, 1.0

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    pts_c = pts - mean

    cov = np.cov(pts_c.T)
    vals, vecs = np.linalg.eigh(cov)  # ascending eigenvalues
    major_vec = vecs[:, 1]
    angle = np.degrees(np.arctan2(major_vec[1], major_vec[0]))

    # approximate axis lengths (std dev scaled)
    major = float(np.sqrt(max(vals[1], 0.0)) * 6.0)
    minor = float(np.sqrt(max(vals[0], 0.0)) * 6.0)
    aspect = major / (minor + 1e-6)
    return float(angle), major, minor, float(aspect)


def overlay_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    m = mask > 0
    bgr[m] = (0, 0, 255)
    return bgr


def overlay_classes(gray: np.ndarray, masks: dict):
    """
    Make a BGR overlay where each class paints a distinct color.
    """
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def paint(mask, color):
        m = mask > 0
        bgr[m] = color

    paint(masks["keyhole"], (255, 0, 255))         # magenta
    paint(masks["crack_tear"], (0, 255, 255))      # yellow
    paint(masks["groove"], (0, 255, 0))            # green
    paint(masks["void_pit"], (255, 0, 0))          # blue
    paint(masks["excess_expulsion"], (0, 0, 255))  # red
    return bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyhole_root", type=str, default="images/phase_3_keyhole")
    ap.add_argument("--masks_root", type=str, default="images/phase_3_2_defect_masks")
    ap.add_argument("--out_root", type=str, default="images/phase_4_classification")

    # Crack/Tear (micro) thresholds
    ap.add_argument("--crack_min_span", type=int, default=28)
    ap.add_argument("--crack_min_aspect", type=float, default=4.0)

    # Groove thresholds (now from macro ∪ micro)
    ap.add_argument("--groove_min_aspect", type=float, default=2.0)
    ap.add_argument("--groove_angle_deg", type=float, default=35.0)
    ap.add_argument("--groove_min_minor", type=float, default=4.0)
    ap.add_argument("--groove_close_w", type=int, default=21)  # connect fragments horizontally
    ap.add_argument("--groove_close_h", type=int, default=3)

    # Excess expulsion band (top/bottom)
    ap.add_argument("--edge_band_px", type=int, default=22)
    ap.add_argument("--expulsion_min_area", type=int, default=120)

    args = ap.parse_args()

    keyhole_root = Path(args.keyhole_root)
    masks_root = Path(args.masks_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    # FIXED: stems must be the parent folder above "macro"
    macro_paths = sorted(masks_root.glob("*/macro/03_macro_mask_clean.png"))
    if not macro_paths:
        raise SystemExit(f"No macro masks found under {masks_root}/*/macro/03_macro_mask_clean.png")

    stems = sorted({p.parents[1].name for p in macro_paths})

    for stem in stems:
        macro_p = masks_root / stem / "macro" / "03_macro_mask_clean.png"
        micro_p = masks_root / stem / "micro" / "04_micro_mask_clean.png"
        gray_p = masks_root / stem / "01_input.png"
        keyhole_p = keyhole_root / stem / "keyhole" / "04_keyhole_mask.png"

        if not (macro_p.exists() and micro_p.exists() and gray_p.exists()):
            print(f"Skipping {stem} (missing masks)")
            continue

        gray = imread_gray(gray_p)
        H, W = gray.shape

        macro = imread_gray(macro_p)
        micro = imread_gray(micro_p)

        keyhole = np.zeros((H, W), np.uint8)
        if keyhole_p.exists():
            keyhole = (imread_gray(keyhole_p) > 0).astype(np.uint8) * 255

        # Remove keyhole pixels from macro/micro so it can't contaminate classes
        inv_keyhole = (keyhole == 0).astype(np.uint8) * 255
        macro = cv2.bitwise_and(macro, inv_keyhole)
        micro = cv2.bitwise_and(micro, inv_keyhole)

        out_dir = out_root / stem
        masks_dir = out_dir / "masks"
        overlay_dir = out_dir / "overlay"
        ensure_dir(masks_dir)
        ensure_dir(overlay_dir)

        crack_mask = np.zeros((H, W), np.uint8)
        groove_mask = np.zeros((H, W), np.uint8)
        void_mask = np.zeros((H, W), np.uint8)
        expulsion_mask = np.zeros((H, W), np.uint8)

        # -------------------------
        # 1) Crack/Tear from MICRO components
        # -------------------------
        for i, labels, area, x, y, w, h in cc_iter(micro):
            comp = (labels == i).astype(np.uint8) * 255
            span = max(w, h)
            angle, major, minor, aspect = component_orientation_and_axes(comp)

            if span >= args.crack_min_span and aspect >= args.crack_min_aspect:
                crack_mask[labels == i] = 255

        # -------------------------
        # 2) Groove from (MACRO ∪ MICRO), connect fragments horizontally
        # -------------------------
        groove_cand = cv2.bitwise_or(macro, micro)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (args.groove_close_w, args.groove_close_h))
        groove_cand = cv2.morphologyEx(groove_cand, cv2.MORPH_CLOSE, k, iterations=1)

        for i, labels, area, x, y, w, h in cc_iter(groove_cand):
            comp = (labels == i).astype(np.uint8) * 255
            angle, major, minor, aspect = component_orientation_and_axes(comp)

            # normalize angle to [-90, 90]
            angle_n = ((angle + 90) % 180) - 90

            if (
                aspect >= args.groove_min_aspect
                and abs(angle_n) <= args.groove_angle_deg
                and minor >= args.groove_min_minor
            ):
                groove_mask[labels == i] = 255

        # -------------------------
        # 3) Macro components: expulsion vs void/pit
        #    (skip anything already marked as groove)
        # -------------------------
        top_band = args.edge_band_px
        bot_band = H - args.edge_band_px

        for i, labels, area, x, y, w, h in cc_iter(macro):
            # skip macro components overlapping groove
            if np.any((labels == i) & (groove_mask > 0)):
                continue

            y_mid = y + h / 2.0
            near_edge = (y_mid <= top_band) or (y_mid >= bot_band)

            if near_edge and area >= args.expulsion_min_area:
                expulsion_mask[labels == i] = 255
            else:
                void_mask[labels == i] = 255

        # Keyhole
        keyhole_mask = keyhole.copy()

        # Fuse
        all_defects = cv2.bitwise_or(keyhole_mask, crack_mask)
        all_defects = cv2.bitwise_or(all_defects, groove_mask)
        all_defects = cv2.bitwise_or(all_defects, void_mask)
        all_defects = cv2.bitwise_or(all_defects, expulsion_mask)

        # Save masks
        save_png(masks_dir / "keyhole.png", keyhole_mask)
        save_png(masks_dir / "crack_tear.png", crack_mask)
        save_png(masks_dir / "groove.png", groove_mask)
        save_png(masks_dir / "void_pit.png", void_mask)
        save_png(masks_dir / "excess_expulsion.png", expulsion_mask)
        save_png(masks_dir / "all_defects.png", all_defects)

        # Overlays
        save_png(overlay_dir / "overlay_all.png", overlay_mask(gray, all_defects))
        save_png(overlay_dir / "overlay_by_class.png", overlay_classes(gray, {
            "keyhole": keyhole_mask,
            "crack_tear": crack_mask,
            "groove": groove_mask,
            "void_pit": void_mask,
            "excess_expulsion": expulsion_mask
        }))

        meta = {
            "shape": [int(H), int(W)],
            "params": {
                "crack_min_span": args.crack_min_span,
                "crack_min_aspect": args.crack_min_aspect,
                "groove_min_aspect": args.groove_min_aspect,
                "groove_angle_deg": args.groove_angle_deg,
                "groove_min_minor": args.groove_min_minor,
                "groove_close": [args.groove_close_w, args.groove_close_h],
                "edge_band_px": args.edge_band_px,
                "expulsion_min_area": args.expulsion_min_area,
            },
            "counts": {
                "keyhole_px": int(np.sum(keyhole_mask > 0)),
                "crack_px": int(np.sum(crack_mask > 0)),
                "groove_px": int(np.sum(groove_mask > 0)),
                "void_px": int(np.sum(void_mask > 0)),
                "expulsion_px": int(np.sum(expulsion_mask > 0)),
                "all_px": int(np.sum(all_defects > 0)),
            }
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Done. Wrote Phase 4 classification to: {out_root}")


if __name__ == "__main__":
    main()