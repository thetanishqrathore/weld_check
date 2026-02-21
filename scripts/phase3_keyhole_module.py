import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def imread_bgr(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return img


def save_png(p: Path, img: np.ndarray):
    ensure_dir(p.parent)
    cv2.imwrite(str(p), img)


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def edges_for_circle(gray: np.ndarray) -> np.ndarray:
    # Stable edges: slight blur + Canny
    g = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
    e = cv2.Canny(g, 60, 160)
    # Close small gaps so circle is more continuous
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, k, iterations=1)
    return e


def hough_circle_detect(gray_roi: np.ndarray):
    """
    Returns (cx, cy, r, score) or None.
    We use HoughCircles on a blurred image; parameters are tuned for robustness.
    """
    g = cv2.GaussianBlur(gray_roi, (0, 0), sigmaX=1.5, sigmaY=1.5)

    h, w = g.shape
    # radius bounds relative to ROI height/width
    min_r = int(max(12, 0.12 * min(h, w)))
    max_r = int(max(min_r + 5, 0.48 * min(h, w)))

    circles = cv2.HoughCircles(
        g,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, h // 4),
        param1=160,      # Canny high threshold inside Hough
        param2=30,       # accumulator threshold (lower = more detections)
        minRadius=min_r,
        maxRadius=max_r
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)

    # Score candidates: prefer circles nearer the left side and with strong edge support
    # Edge support: count Canny edges on the circle perimeter band
    edges = edges_for_circle(gray_roi)

    best = None
    best_score = -1e9
    for (cx, cy, r) in circles:
        if r <= 0:
            continue
        # Penalize circles too close to ROI border (partial circles)
        if cx - r < 2 or cy - r < 2 or cx + r > w - 2 or cy + r > h - 2:
            border_pen = -0.5
        else:
            border_pen = 0.0

        # Edge support on a thin ring
        ring = np.zeros_like(edges)
        cv2.circle(ring, (cx, cy), r, 255, 2)
        support = float(np.sum((edges > 0) & (ring > 0))) / (2 * np.pi * r + 1e-6)

        # Left bias (keyhole on left): smaller cx is better
        left_bias = 1.0 - (cx / max(1, w))

        score = 2.0 * support + 0.8 * left_bias + border_pen

        if score > best_score:
            best_score = score
            best = (int(cx), int(cy), int(r), float(score))

    return best


def ellipse_fallback_from_contours(gray_roi: np.ndarray):
    """
    Fallback: find big contour in edges and fit ellipse.
    Returns (cx, cy, a, b, angle, score) in ROI coords or None.
    """
    e = edges_for_circle(gray_roi)

    # Find contours on edges
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None

    h, w = gray_roi.shape
    # Candidate: largest contour by area after fill
    best = None
    best_score = -1e9

    for c in cnts:
        if len(c) < 20:
            continue

        # Approx filled area via bounding box & contour area
        area = cv2.contourArea(c)
        if area < 200:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        # Heuristic: keyhole should occupy decent fraction of the left window height
        if bh < 0.2 * h:
            continue

        if len(c) >= 5:
            ell = cv2.fitEllipse(c)
            (cx, cy), (MA, ma), angle = ell  # MA, ma are full axis lengths
            a = MA / 2.0
            b = ma / 2.0
            if a < 5 or b < 5:
                continue
            axis_ratio = max(a, b) / (min(a, b) + 1e-6)

            # Score: prefer near-circular, large-ish, left-ish
            left_bias = 1.0 - (cx / max(1, w))
            circular_score = 1.0 - min(1.0, abs(axis_ratio - 1.0))  # 1 if perfect circle
            size_score = min(1.0, (a * b) / (0.25 * h * h + 1e-6))

            score = 1.2 * circular_score + 0.8 * size_score + 0.6 * left_bias

            if score > best_score:
                best_score = score
                best = (float(cx), float(cy), float(a), float(b), float(angle), float(score))

    return best


def draw_overlay(roi_bgr: np.ndarray, result: dict, x_off: int):
    """
    Draw detected circle/ellipse on full ROI overlay.
    x_off is the horizontal offset of the keyhole search window relative to ROI.
    """
    out = roi_bgr.copy()
    if result["method"] == "hough_circle":
        cx = int(result["cx"] + x_off)
        cy = int(result["cy"])
        r = int(result["r"])
        cv2.circle(out, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 2, (0, 0, 255), 3)
    else:
        cx = float(result["cx"] + x_off)
        cy = float(result["cy"])
        a = float(result["a"]) * 2.0
        b = float(result["b"]) * 2.0
        angle = float(result["angle"])
        cv2.ellipse(out, ((cx, cy), (a, b), angle), (0, 255, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 2, (0, 0, 255), 3)
    return out


def make_keyhole_mask(h: int, w: int, result: dict, x_off: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    if result["method"] == "hough_circle":
        cx = int(result["cx"] + x_off)
        cy = int(result["cy"])
        r = int(result["r"])
        cv2.circle(mask, (cx, cy), r, 255, -1)
    else:
        cx = float(result["cx"] + x_off)
        cy = float(result["cy"])
        a = float(result["a"]) * 2.0
        b = float(result["b"]) * 2.0
        angle = float(result["angle"])
        cv2.ellipse(mask, ((cx, cy), (a, b), angle), 255, -1)

    # Slight dilation to include rim material
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.dilate(mask, k, iterations=1)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2_dir", type=str, default="images/phase_2_weld_roi_boundaries")
    ap.add_argument("--out_dir", type=str, default="images/phase_3_keyhole")
    ap.add_argument("--roi_name", type=str, default="03_roi.png")  # file inside each <stem> folder
    ap.add_argument("--left_frac", type=float, default=0.20)
    ap.add_argument("--axis_ratio_max", type=float, default=2.0)  # paper-ish criterion
    args = ap.parse_args()

    phase2_dir = Path(args.phase2_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    roi_paths = sorted(phase2_dir.glob(f"*/{args.roi_name}"))
    if not roi_paths:
        raise SystemExit(f"No ROI files found: {phase2_dir}/*/{args.roi_name}")

    for roi_path in roi_paths:
        stem = roi_path.parent.name
        od = out_dir / stem / "keyhole"
        ensure_dir(od)

        roi_bgr = imread_bgr(roi_path)
        roi_gray = to_gray(roi_bgr)
        H, W = roi_gray.shape

        # Left terminus search window
        win_w = max(32, int(args.left_frac * W))
        x0, x1 = 0, win_w
        gray_win = roi_gray[:, x0:x1]

        save_png(od / "01_search_roi.png", gray_win)

        edges = edges_for_circle(gray_win)
        save_png(od / "02_edges.png", edges)

        result = None

        # 1) Hough circle
        hc = hough_circle_detect(gray_win)
        if hc is not None:
            cx, cy, r, score = hc
            # Axis ratio = 1 for circle; always valid under <=2
            result = {
                "method": "hough_circle",
                "cx": int(cx),
                "cy": int(cy),
                "r": int(r),
                "score": float(score),
                "axis_ratio": 1.0,
                "area_px": float(np.pi * r * r),
            }

        # 2) Fallback ellipse fit
        if result is None:
            ell = ellipse_fallback_from_contours(gray_win)
            if ell is not None:
                cx, cy, a, b, angle, score = ell
                axis_ratio = float(max(a, b) / (min(a, b) + 1e-6))
                result = {
                    "method": "ellipse_fit",
                    "cx": float(cx),
                    "cy": float(cy),
                    "a": float(a),   # semi-axis
                    "b": float(b),   # semi-axis
                    "angle": float(angle),
                    "score": float(score),
                    "axis_ratio": axis_ratio,
                    "area_px": float(np.pi * a * b),
                }

        # If still none, write a meta that keyhole detection failed
        meta = {
            "input_roi": str(roi_path),
            "roi_shape": [int(H), int(W)],
            "left_window": {"x0": x0, "x1": x1, "left_frac": args.left_frac},
            "detected": False,
            "result": None,
            "accepted": False,
            "reason": None,
        }

        if result is None:
            meta["reason"] = "No circle/ellipse found"
            (od / "meta.json").write_text(json.dumps(meta, indent=2))
            continue

        # Acceptance based on axis ratio criterion (paper-inspired)
        accepted = bool(result.get("axis_ratio", 99.0) <= args.axis_ratio_max)
        meta["detected"] = True
        meta["result"] = result
        meta["accepted"] = accepted
        if not accepted:
            meta["reason"] = f"axis_ratio>{args.axis_ratio_max}"

        # Create overlay on full ROI
        overlay = draw_overlay(roi_bgr, result, x_off=x0)
        save_png(od / "03_circle_overlay.png", overlay)

        # Create mask on full ROI
        mask = make_keyhole_mask(H, W, result, x_off=x0)
        save_png(od / "04_keyhole_mask.png", mask)

        # Black out keyhole region (this is traverse input later)
        roi_black = roi_gray.copy()
        roi_black[mask > 0] = 0
        save_png(od / "05_roi_keyhole_blacked.png", roi_black)

        (od / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Done. Wrote Phase 3 keyhole outputs to: {out_dir}")


if __name__ == "__main__":
    main()