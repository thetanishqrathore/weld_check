# phase1_illumination_flattening.py
# Input:  images/raw_/*
# Output: images/phase_1_illumination_flattening/<stem>/*
#
# Purpose:
# - Flatten illumination (vignetting / gradients) without "flattening the bead"
# - Suppress specular highlights by explicit detection + inpainting
# - Produce a consistent grayscale image for the paper pipeline (Phase 2+)
#
# Dependencies:
#   pip install opencv-python numpy
#
# Run:
#   python phase1_illumination_flattening.py \
#       --in_dir images/raw_ \
#       --out_dir images/phase_1_illumination_flattening

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def imread_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return img


def save_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def homomorphic_illumination_flatten(gray_u8: np.ndarray,
                                    sigma: float = 45.0,
                                    strength: float = 1.0) -> np.ndarray:
    """
    Homomorphic filtering:
      gray -> log -> subtract low-frequency illumination -> exp
    sigma controls how low-frequency the illumination estimate is.
    strength scales the correction (1.0 = full).
    """
    g = gray_u8.astype(np.float32) / 255.0
    g = np.clip(g, 1e-6, 1.0)
    logg = np.log(g)

    # Low-frequency illumination estimate by heavy Gaussian blur
    illum = cv2.GaussianBlur(logg, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Remove illumination, keep reflectance
    refl = logg - strength * illum

    # Normalize to [0,1] robustly
    r = np.exp(refl)
    r = r - np.percentile(r, 1)
    r = r / (np.percentile(r, 99) + 1e-6)
    r = np.clip(r, 0.0, 1.0)
    return (r * 255.0).astype(np.uint8)


def build_highlight_mask(gray_u8: np.ndarray,
                         p_hi: float = 99.5,
                         min_area: int = 80,
                         dilate_px: int = 3) -> np.ndarray:
    """
    Detect specular highlights (very bright pixels) and return a binary mask.
    We use a percentile threshold so it's adaptive across images.
    """
    thr = np.percentile(gray_u8, p_hi)
    thr = max(thr, 230)  # ensure we really focus on near-saturated glare
    mask = (gray_u8 >= thr).astype(np.uint8) * 255

    # Clean small speckles; keep meaningful blobs
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # Remove tiny connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == i] = 255
    mask = cleaned

    # Slightly dilate so we also cover the highlight rims
    if dilate_px > 0:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, kd, iterations=1)

    return mask


def inpaint_highlights(gray_u8: np.ndarray,
                       mask_u8: np.ndarray,
                       radius: int = 3,
                       method: str = "telea") -> np.ndarray:
    if method.lower() == "ns":
        m = cv2.INPAINT_NS
    else:
        m = cv2.INPAINT_TELEA
    return cv2.inpaint(gray_u8, mask_u8, inpaintRadius=radius, flags=m)


def clahe_limited(gray_u8: np.ndarray,
                  clip_limit: float = 2.0,
                  tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(gray_u8)


def summarize(gray_u8: np.ndarray, highlight_mask_u8: np.ndarray) -> dict:
    sat = int(np.sum(gray_u8 >= 250))
    total = int(gray_u8.size)
    hi = int(np.sum(highlight_mask_u8 > 0))
    return {
        "shape": [int(gray_u8.shape[0]), int(gray_u8.shape[1])],
        "min": int(gray_u8.min()),
        "max": int(gray_u8.max()),
        "mean": float(gray_u8.mean()),
        "p1": float(np.percentile(gray_u8, 1)),
        "p50": float(np.percentile(gray_u8, 50)),
        "p99": float(np.percentile(gray_u8, 99)),
        "saturated_px": sat,
        "saturated_ratio": sat / max(total, 1),
        "highlight_mask_px": hi,
        "highlight_mask_ratio": hi / max(total, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="images/raw_")
    ap.add_argument("--out_dir", type=str, default="images/phase_1_illumination_flattening")
    ap.add_argument("--ext", type=str, default="png,jpg,jpeg,webp")
    # homomorphic params
    ap.add_argument("--homo_sigma", type=float, default=45.0)
    ap.add_argument("--homo_strength", type=float, default=1.0)
    # highlight params
    ap.add_argument("--hi_percentile", type=float, default=99.5)
    ap.add_argument("--hi_min_area", type=int, default=80)
    ap.add_argument("--hi_dilate", type=int, default=3)
    ap.add_argument("--inpaint_radius", type=int, default=3)
    ap.add_argument("--inpaint_method", type=str, default="telea", choices=["telea", "ns"])
    # clahe params
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_tiles", type=int, default=8)

    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    exts = {e.strip().lower() for e in args.ext.split(",") if e.strip()}
    paths = []
    for p in in_dir.rglob("*"):
        if p.is_file() and p.suffix.lower().lstrip(".") in exts:
            paths.append(p)
    paths.sort()

    if not paths:
        raise SystemExit(f"No images found in {in_dir} with extensions {sorted(exts)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for p in paths:
        stem = p.stem
        od = out_dir / stem
        od.mkdir(parents=True, exist_ok=True)

        bgr = imread_bgr(p)
        gray0 = to_gray(bgr)

        # 01: raw gray
        save_png(od / "01_gray.png", gray0)

        # 02: illumination flattening (homomorphic)
        gray2 = homomorphic_illumination_flatten(gray0, sigma=args.homo_sigma, strength=args.homo_strength)
        save_png(od / "02_illum_corrected.png", gray2)

        # 03: highlight mask on illum-corrected (more stable than raw)
        hi_mask = build_highlight_mask(
            gray2,
            p_hi=args.hi_percentile,
            min_area=args.hi_min_area,
            dilate_px=args.hi_dilate,
        )
        save_png(od / "03_highlight_mask.png", hi_mask)

        # 04: inpaint highlights (remove specular islands/rims)
        gray4 = inpaint_highlights(gray2, hi_mask, radius=args.inpaint_radius, method=args.inpaint_method)
        save_png(od / "04_deglare_inpaint.png", gray4)

        # 06: constrained CLAHE to improve local separability (not too aggressive)
        # we are avoiding clahe as it casue high contrast which is not need
        gray6 = clahe_limited(gray4, clip_limit=args.clahe_clip, tile_grid_size=args.clahe_tiles)
        save_png(od / "06_phase1_clahe.png", gray6)
        # 05: final (no CLAHE)
        gray5 = gray4
        save_png(od / "05_phase1_final.png", gray5)

        # metadata``
        meta = {
            "input": str(p),
            "params": {
                "homo_sigma": args.homo_sigma,
                "homo_strength": args.homo_strength,
                "hi_percentile": args.hi_percentile,
                "hi_min_area": args.hi_min_area,
                "hi_dilate": args.hi_dilate,
                "inpaint_radius": args.inpaint_radius,
                "inpaint_method": args.inpaint_method,
                "clahe_clip": args.clahe_clip,
                "clahe_tiles": args.clahe_tiles,
            },
            "stats_before": summarize(gray0, np.zeros_like(gray0)),
            "stats_after": summarize(gray5, hi_mask),
        }
        (od / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Done. Wrote Phase 1 outputs to: {out_dir}")


if __name__ == "__main__":
    main()