import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def keep_by_span(mask_u8: np.ndarray, min_span: int = 25, min_area: int = 15) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        span = max(w, h)
        if area >= min_area and span >= min_span:
            out[labels == i] = 255
    return out

def imread_gray(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img


def save_png(p: Path, img: np.ndarray):
    ensure_dir(p.parent)
    cv2.imwrite(str(p), img)


def to_u8_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.percentile(x, 1)
    x = x / (np.percentile(x, 99) + 1e-6)
    x = np.clip(x, 0, 1)
    return (255 * x).astype(np.uint8)


def dog_mag(gray: np.ndarray, s1: float, s2: float) -> np.ndarray:
    g1 = cv2.GaussianBlur(gray, (0, 0), s1)
    g2 = cv2.GaussianBlur(gray, (0, 0), s2)
    d = (g1.astype(np.float32) - g2.astype(np.float32))
    return np.abs(d)


def sobel_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def hysteresis_grow(mag_u8: np.ndarray, lo_thr: float, hi_thr: float, max_iters: int = 200) -> np.ndarray:
    allowed = (mag_u8 >= lo_thr).astype(np.uint8) * 255
    seed = (mag_u8 >= hi_thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cur = seed.copy()
    for _ in range(max_iters):
        prev = cur
        dil = cv2.dilate(cur, k, iterations=1)
        cur = cv2.bitwise_and(dil, allowed)
        if np.array_equal(cur, prev):
            break
    return seed, allowed, cur


def remove_small_cc(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == i] = 255
    return out


def overlay_mask_on_gray(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Red overlay where mask is on
    bgr[mask > 0] = (0, 0, 255)
    return bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, default="images/phase_3_keyhole")
    ap.add_argument("--pattern", type=str, default="*/keyhole/05_roi_keyhole_blacked.png")
    ap.add_argument("--out_root", type=str, default="images/phase_3_2_defect_masks")

    # Macro (DoG) parameters
    ap.add_argument("--dog_s1", type=float, default=2.0)
    ap.add_argument("--dog_s2", type=float, default=10.0)
    ap.add_argument("--macro_pctl", type=float, default=92.0)   # keep top X% DoG magnitude
    ap.add_argument("--macro_close_w", type=int, default=9)
    ap.add_argument("--macro_close_h", type=int, default=5)
    ap.add_argument("--macro_min_area", type=int, default=120)

    # Micro (Sobel) parameters (hysteresis)
    ap.add_argument("--micro_hi_pctl", type=float, default=99.1)
    ap.add_argument("--micro_lo_pctl", type=float, default=96.5)
    ap.add_argument("--micro_grow_iters", type=int, default=200)
    ap.add_argument("--micro_open", type=int, default=3)
    ap.add_argument("--micro_min_area", type=int, default=60)

    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    inputs = sorted(in_root.glob(args.pattern))
    if not inputs:
        raise SystemExit(f"No inputs found: {in_root}/{args.pattern}")

    for p in inputs:
        stem = p.parents[1].name
        od = out_root / stem
        ensure_dir(od)

        gray = imread_gray(p)
        H, W = gray.shape
        save_png(od / "01_input.png", gray)

        # -------------------------
        # Macro mask from DoG mag
        # -------------------------
        macro_dir = od / "macro"
        ensure_dir(macro_dir)

        dmag = dog_mag(gray, args.dog_s1, args.dog_s2)
        d_u8 = to_u8_norm(dmag)
        save_png(macro_dir / "01_dog_mag.png", d_u8)

        thr_macro = np.percentile(d_u8, args.macro_pctl)
        macro_raw = (d_u8 >= thr_macro).astype(np.uint8) * 255
        save_png(macro_dir / "02_macro_mask_raw.png", macro_raw)

        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (args.macro_close_w, args.macro_close_h))
        macro_clean = cv2.morphologyEx(macro_raw, cv2.MORPH_CLOSE, k_close, iterations=1)
        macro_clean = remove_small_cc(macro_clean, args.macro_min_area)
        save_png(macro_dir / "03_macro_mask_clean.png", macro_clean)

        # -------------------------
        # Micro mask from Sobel mag + hysteresis
        # -------------------------
        micro_dir = od / "micro"
        ensure_dir(micro_dir)

        smag = sobel_mag(gray)
        sm_u8 = to_u8_norm(smag)
        save_png(micro_dir / "01_sobel_mag.png", sm_u8)

        hi_thr = np.percentile(sm_u8, args.micro_hi_pctl)
        lo_thr = np.percentile(sm_u8, args.micro_lo_pctl)

        seed, allowed, micro_mask = hysteresis_grow(sm_u8, lo_thr, hi_thr, max_iters=args.micro_grow_iters)
        save_png(micro_dir / "02_micro_seed.png", seed)
        save_png(micro_dir / "03_micro_allowed.png", allowed)
        save_png(micro_dir / "03x_micro_mask_raw.png", micro_mask)

        # Clean micro mask
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.micro_open, args.micro_open))
        micro_mask = cv2.morphologyEx(micro_mask, cv2.MORPH_OPEN, k_open, iterations=1)
        micro_clean = keep_by_span(micro_mask, min_span=25, min_area=15)
        save_png(micro_dir / "04_micro_mask_clean.png", micro_clean)

        # -------------------------
        # Fuse
        # -------------------------
        fused_dir = od / "fused"
        ensure_dir(fused_dir)

        all_mask = cv2.bitwise_or(macro_clean, micro_clean)
        save_png(fused_dir / "01_all_defects_mask.png", all_mask)
        save_png(fused_dir / "02_overlay.png", overlay_mask_on_gray(gray, all_mask))

        meta = {
            "input": str(p),
            "shape": [int(H), int(W)],
            "macro": {
                "dog_s1": args.dog_s1,
                "dog_s2": args.dog_s2,
                "macro_pctl": args.macro_pctl,
                "thr_macro": float(thr_macro),
                "close": [args.macro_close_w, args.macro_close_h],
                "min_area": args.macro_min_area,
            },
            "micro": {
                "hi_pctl": args.micro_hi_pctl,
                "lo_pctl": args.micro_lo_pctl,
                "hi_thr": float(hi_thr),
                "lo_thr": float(lo_thr),
                "grow_iters": args.micro_grow_iters,
                "open": args.micro_open,
                "min_area": args.micro_min_area,
            },
        }
        (od / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Done. Wrote Phase 3.2 defect masks to: {out_root}")


if __name__ == "__main__":
    main()