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

def hysteresis_grow(mag_u8: np.ndarray, lo_thr: float, hi_thr: float, max_iters: int = 200) -> np.ndarray:
    """
    Binary hysteresis on magnitude image:
    - seed = mag >= hi_thr
    - allowed = mag >= lo_thr
    - grow seeds inside allowed until convergence
    """
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
    return cur



def save_png(p: Path, img: np.ndarray):
    ensure_dir(p.parent)
    cv2.imwrite(str(p), img)


def to_u8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    
    # 1. Establish the ceiling (the brightest defect/edge response)
    ceiling = np.percentile(x, 99) + 1e-6
    
    # 2. Raise the floor to the median. 
    # This natively kills background noise without needing a mask.
    floor = np.percentile(x, 50) 
    
    x = x - floor
    x = x / (ceiling - floor)
    
    # 3. Clip anything that fell below the median to absolute 0
    x = np.clip(x, 0, 1)
    
    return (255 * x).astype(np.uint8)


def dog(gray: np.ndarray, s1: float, s2: float) -> np.ndarray:
    g1 = cv2.GaussianBlur(gray, (0, 0), s1)
    g2 = cv2.GaussianBlur(gray, (0, 0), s2)
    return (g1.astype(np.float32) - g2.astype(np.float32))


def sobel_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag

def weld_zone_from_texture(gray: np.ndarray,
                           win: int = 31,
                           keep_pctl: float = 75.0,
                           close_w: int = 101,
                           close_h: int = 31,
                           dilate: int = 9) -> np.ndarray:
    """
    Weld-zone mask based on local variance (texture energy).
    - Works when plate is smooth and weld is textured.
    """
    if win % 2 == 0:
        win += 1

    g = gray.astype(np.float32)

    # local mean and mean of squares
    mean = cv2.blur(g, (win, win))
    mean2 = cv2.blur(g * g, (win, win))
    var = mean2 - mean * mean
    var = np.clip(var, 0, None)

    # normalize variance to 8-bit
    v = var - np.percentile(var, 1)
    v = v / (np.percentile(var, 99) + 1e-6)
    v = np.clip(v, 0, 1)
    v_u8 = (255 * v).astype(np.uint8)

    # threshold by percentile (keep strongest texture)
    thr = np.percentile(v_u8, keep_pctl)
    bw = (v_u8 >= thr).astype(np.uint8) * 255

    # connect the band
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    # fill holes
    h, w = bw.shape
    flood = bw.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    bw = cv2.bitwise_or(bw, holes)

    # keep largest CC
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = 1 + int(np.argmax(areas))
        out = np.zeros_like(bw)
        out[labels == idx] = 255
        bw = out

    # expand a bit to include shoulders
    if dilate > 0:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        bw = cv2.dilate(bw, kd, iterations=1)

    return bw, v_u8

def reconstruct_like(gray: np.ndarray, ksize: int = 19) -> np.ndarray:
    """
    Practical substitute for morphological grayscale reconstruction using only OpenCV:
    - Use opening-by-reconstruction would require iterative geodesic dilation.
    - Here we approximate: large-kernel opening to estimate "background texture",
      then subtract to suppress slow variations.
    This is not exact reconstruction, but it behaves similarly for our purpose.
    """
    if ksize % 2 == 0:
        ksize += 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k, iterations=1)
    # suppress irrelevant variations
    out = cv2.subtract(gray, opened)
    return out


def canny_edges(gray: np.ndarray, t1: int, t2: int, close: bool = True) -> np.ndarray:
    e = cv2.Canny(gray, t1, t2)
    if close:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, k, iterations=1)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, default="images/phase_3_keyhole")
    ap.add_argument("--pattern", type=str, default="*/keyhole/05_roi_keyhole_blacked.png")
    ap.add_argument("--out_root", type=str, default="images/phase_3_1_edges")
    ap.add_argument("--micro_pctl", type=float, default=99.3)   # stricter = higher
    ap.add_argument("--micro_open", type=int, default=3)        # kernel size
    ap.add_argument("--micro_hi_pctl", type=float, default=99.2)  # seed threshold
    ap.add_argument("--micro_lo_pctl", type=float, default=97.0)  # grow threshold
    ap.add_argument("--micro_grow_iters", type=int, default=200)
    ap.add_argument("--micro_min_area", type=int, default=60)

    # Macro params
    ap.add_argument("--macro_blur_sigma", type=float, default=2.0)
    ap.add_argument("--dog_s1", type=float, default=1.0)
    ap.add_argument("--dog_s2", type=float, default=4.0)
    ap.add_argument("--macro_canny1", type=int, default=40)
    ap.add_argument("--macro_canny2", type=int, default=120)

    # Micro params
    ap.add_argument("--recon_ksize", type=int, default=19)
    ap.add_argument("--micro_canny1", type=int, default=25)
    ap.add_argument("--micro_canny2", type=int, default=90)

    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    inputs = sorted(in_root.glob(args.pattern))
    if not inputs:
        raise SystemExit(f"No inputs found: {in_root}/{args.pattern}")

    for p in inputs:
        stem = p.parents[1].name  # <stem>/keyhole/file.png
        od = out_root / stem
        ensure_dir(od)

        gray = imread_gray(p)
        H, W = gray.shape
        save_png(od / "01_traverse_input.png", gray)

        # --------------------------
        # Stream A: Macro edges
        # --------------------------
        macro_dir = od / "macro"
        ensure_dir(macro_dir)

        blur = cv2.GaussianBlur(gray, (0, 0), args.macro_blur_sigma)
        save_png(macro_dir / "01_blur.png", blur)

        d = dog(blur, args.dog_s1, args.dog_s2)
        d_u8 = to_u8(np.abs(d))
        save_png(macro_dir / "02_dog.png", d_u8)

        weld_zone, v_u8 = weld_zone_from_texture(gray, win=31, keep_pctl=75.0, close_w=101, close_h=31, dilate=9)
        save_png(macro_dir / "04_texture_energy.png", v_u8)
        save_png(macro_dir / "05_weld_zone_mask.png", weld_zone)

        macro_edges = canny_edges(d_u8, args.macro_canny1, args.macro_canny2, close=True)
        # ADD THIS LINE:
        macro_edges = cv2.bitwise_and(macro_edges, weld_zone)
        save_png(macro_dir / "03_edges.png", macro_edges)

        # --------------------------
        # Stream B: Micro edges
        # --------------------------
        micro_dir = od / "micro"
        ensure_dir(micro_dir)

        recon = reconstruct_like(gray, ksize=args.recon_ksize)
        save_png(micro_dir / "01_reconstruct_like.png", recon)

        sm = sobel_mag(recon)
        sm_u8 = to_u8(sm)
        save_png(micro_dir / "02_sobel.png", sm_u8)

        # micro: hysteresis region mask from Sobel magnitude
        hi_thr = np.percentile(sm_u8, args.micro_hi_pctl)
        lo_thr = np.percentile(sm_u8, args.micro_lo_pctl)

        micro_mask = hysteresis_grow(sm_u8, lo_thr=lo_thr, hi_thr=hi_thr, max_iters=args.micro_grow_iters)
        micro_mask = cv2.bitwise_and(micro_mask, weld_zone)

        # clean speckles + remove tiny CCs
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        micro_mask = cv2.morphologyEx(micro_mask, cv2.MORPH_OPEN, k, iterations=1)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(micro_mask, connectivity=8)
        clean = np.zeros_like(micro_mask)
        for i in range(1, num):
            if int(stats[i, cv2.CC_STAT_AREA]) >= args.micro_min_area:
                clean[labels == i] = 255

        save_png(micro_dir / "03_micro_mask.png", clean)

        seed_img = ((sm_u8 >= hi_thr).astype(np.uint8) * 255)
        allowed_img = ((sm_u8 >= lo_thr).astype(np.uint8) * 255)

        seed_img = cv2.bitwise_and(seed_img, weld_zone)
        allowed_img = cv2.bitwise_and(allowed_img, weld_zone)

        save_png(micro_dir / "03a_micro_seed.png", seed_img)
        save_png(micro_dir / "03b_micro_allowed.png", allowed_img)


        meta = {
            "input": str(p),
            "shape": [int(H), int(W)],
            "params": {
                "macro_blur_sigma": args.macro_blur_sigma,
                "dog_s1": args.dog_s1,
                "dog_s2": args.dog_s2,
                "macro_canny": [args.macro_canny1, args.macro_canny2],
                "recon_ksize": args.recon_ksize,
                "micro_canny": [args.micro_canny1, args.micro_canny2],
            }
        }
        (od / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Done. Wrote edge outputs to: {out_root}")


if __name__ == "__main__":
    main()