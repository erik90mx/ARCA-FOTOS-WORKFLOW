#!/bin/bash
# arca_panorama.sh - Genera panorámicas a partir de fotos de grupos de graduados
# Uso:
#   ./arca_panorama.sh                        # Procesa todos los CBTIS
#   ./arca_panorama.sh 78                     # Procesa solo CBTIS 78
#   ./arca_panorama.sh 78 MAT_A_RH_NEGRO     # Procesa solo ese grupo del CBTIS 78

set -euo pipefail

BASE_DIR="${ARCA_BASE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
YEAR_DIR="$BASE_DIR/2026"
CBTIS_FILTER="${1:-}"
GROUP_FILTER="${2:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()   { echo -e "${RED}[ERROR]${NC} $1"; }

if [ ! -d "$YEAR_DIR" ]; then
    log_err "No existe el directorio $YEAR_DIR"
    exit 1
fi

# Use venv python if available (has OpenCV installed)
if [ -f "$BASE_DIR/gui_venv/bin/python3" ]; then
    PYTHON="$BASE_DIR/gui_venv/bin/python3"
elif [ -f "$BASE_DIR/venv/bin/python3" ]; then
    PYTHON="$BASE_DIR/venv/bin/python3"
else
    PYTHON="python3"
fi

$PYTHON -c "import cv2" 2>/dev/null || { log_err "OpenCV no está instalado. Instala con: pip install opencv-python"; exit 1; }

# --- Python stitcher incrustado ---
PYTHON_STITCHER=$(cat << 'PYEOF'
import cv2
import numpy as np
import sys
import json
import gc
import os

# Resolution for SIFT feature detection / transform computation
WORK_DIM = int(os.environ.get("ARCA_WORK_DIM", "4000"))

def get_available_mem_mb():
    """Get available system memory in MB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 8000  # assume 8 GB if unknown

def log(msg):
    print(f"[INFO] py: {msg}", file=sys.stderr, flush=True)

def auto_rotate(img):
    h, w = img.shape[:2]
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def downscale(img, max_dim):
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img, 1.0
    scale = max_dim / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

def scale_transform(T, from_scale, to_scale):
    """Scale a 3x3 transform matrix computed at from_scale to work at to_scale."""
    ratio = to_scale / from_scale
    S = np.array([[ratio, 0, 0], [0, ratio, 0], [0, 0, 1]], dtype=np.float64)
    S_inv = np.array([[1/ratio, 0, 0], [0, 1/ratio, 0], [0, 0, 1]], dtype=np.float64)
    return S @ T @ S_inv

# =====================================================================
# VALIDACIÓN
# =====================================================================
def is_valid_panorama(pano, single_h, single_w, n_imgs):
    h, w = pano.shape[:2]
    if h > single_h * 1.10:
        return False
    if w < single_w * 1.2:
        return False
    if n_imgs >= 3 and w <= h:
        return False
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    fill_ratio = np.count_nonzero(gray > 5) / (h * w)
    del gray
    return fill_ratio >= 0.92

# =====================================================================
# INTENTO 1: OpenCV Stitcher SCANS
# =====================================================================
def try_stitcher(imgs, single_h, single_w, n_imgs, conf_thresh, reg_resol=0.6):
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    stitcher.setCompositingResol(-1)
    stitcher.setRegistrationResol(reg_resol)
    stitcher.setPanoConfidenceThresh(conf_thresh)
    try:
        status, pano = stitcher.stitch(imgs)
    except cv2.error:
        del stitcher
        gc.collect()
        return None
    if status != cv2.Stitcher_OK:
        del stitcher
        gc.collect()
        return None
    indices = stitcher.component()
    del stitcher
    if len(indices) != n_imgs:
        del pano
        gc.collect()
        return None
    if not is_valid_panorama(pano, single_h, single_w, n_imgs):
        del pano
        gc.collect()
        return None
    return pano

# =====================================================================
# INTENTO 2: SIFT + Affine (compute on work-res, render on full-res)
# =====================================================================
def strip_overlap_score(img_left, img_right, overlap_pct=0.35):
    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]
    h = min(h_l, h_r)
    ow = int(min(w_l, w_r) * overlap_pct)
    if ow < 10:
        return 0.0
    strip_l = cv2.cvtColor(img_left[:h, w_l-ow:], cv2.COLOR_BGR2GRAY).astype(np.float64)
    strip_r = cv2.cvtColor(img_right[:h, :ow], cv2.COLOR_BGR2GRAY).astype(np.float64)
    hann = np.hanning(h).reshape(-1,1) * np.hanning(ow).reshape(1,-1)
    fa = np.fft.fft2(strip_l * hann)
    fb = np.fft.fft2(strip_r * hann)
    del strip_l, strip_r, hann
    cross = fa * np.conj(fb)
    del fa, fb
    cross /= np.abs(cross) + 1e-10
    corr = np.fft.ifft2(cross).real
    del cross
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    pdy, pdx = peak
    if pdy > h//2: pdy -= h
    if pdx > ow//2: pdx -= ow
    confidence = corr[peak] / (np.mean(np.abs(corr)) + 1e-10)
    real_overlap = ow - pdx
    dy_penalty = 1.0 if abs(pdy) < h*0.02 else 0.5 if abs(pdy) < h*0.05 else 0.1
    ov_valid = 0.1*min(w_l,w_r) < real_overlap < 0.7*min(w_l,w_r)
    return confidence * dy_penalty * (1.0 if ov_valid else 0.01)

def find_spatial_order(work_imgs):
    from itertools import permutations
    n = len(work_imgs)
    if n == 1:
        return [0]
    best_score = -1
    best_perm = list(range(n))
    for perm in permutations(range(n)):
        total = 0.0
        for k in range(len(perm) - 1):
            total += strip_overlap_score(work_imgs[perm[k]], work_imgs[perm[k+1]])
        if total > best_score:
            best_score = total
            best_perm = list(perm)
    return best_perm

def _match_features(img_left, img_right):
    """Extract SIFT features and find good matches. Returns (src_pts, dst_pts) or None."""
    sift = cv2.SIFT_create(nfeatures=3000)
    gl = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(gl, None)
    kp2, des2 = sift.detectAndCompute(gr, None)
    del gl, gr, sift
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    del des1, des2
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    del matches
    if len(good) < 10:
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    del kp1, kp2, good
    return src_pts, dst_pts

def compute_pairwise_transform(img_left, img_right):
    """Compute transform mapping img_right into img_left's coordinate space.
    Tries affine first (less distortion), falls back to homography."""
    result = _match_features(img_left, img_right)
    if result is None:
        return None
    src_pts, dst_pts = result
    # Try affine first — preserves parallel lines, no perspective distortion
    A, mask_a = cv2.estimateAffine2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if A is not None and mask_a.ravel().sum() >= 8:
        # Promote 2x3 affine to 3x3 for uniform handling
        T = np.vstack([A, [0, 0, 1]]).astype(np.float64)
        return T
    # Fallback: full homography
    H, mask_h = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)
    if H is not None and mask_h.ravel().sum() >= 8:
        return H
    return None

def compute_transforms_lowres(work_imgs, order):
    """Compute transforms relative to center image (no chaining)."""
    n = len(order)
    center = n // 2
    transforms = [None] * n
    transforms[center] = np.eye(3, dtype=np.float64)

    # Left of center
    for i in range(center - 1, -1, -1):
        T = compute_pairwise_transform(work_imgs[order[center]], work_imgs[order[i]])
        if T is not None:
            transforms[i] = T
        else:
            T = compute_pairwise_transform(work_imgs[order[i + 1]], work_imgs[order[i]])
            if T is None:
                return None
            transforms[i] = transforms[i + 1] @ T

    # Right of center
    for i in range(center + 1, n):
        T = compute_pairwise_transform(work_imgs[order[center]], work_imgs[order[i]])
        if T is not None:
            transforms[i] = T
        else:
            T = compute_pairwise_transform(work_imgs[order[i - 1]], work_imgs[order[i]])
            if T is None:
                return None
            transforms[i] = transforms[i - 1] @ T

    return transforms

def find_optimal_seam(cost_map, step=5):
    """Find optimal vertical seam through cost_map.

    step: max horizontal pixels the seam can move per row.
    Larger step lets the seam navigate around obstacles (faces) faster.
    """
    h, w = cost_map.shape
    dp = cost_map.astype(np.float64).copy()
    for r in range(1, h):
        prev = dp[r - 1].copy()
        # Build min over neighborhood of width 2*step+1
        expanded = np.full(w, 1e18, dtype=np.float64)
        for s in range(-step, step + 1):
            shifted = np.roll(prev, -s)
            if s < 0:
                shifted[s:] = 1e18
            elif s > 0:
                shifted[:s] = 1e18
            np.minimum(expanded, shifted, out=expanded)
        dp[r] += expanded
    seam = np.zeros(h, dtype=int)
    seam[-1] = np.argmin(dp[-1])
    for r in range(h - 2, -1, -1):
        c = seam[r + 1]
        c0 = max(0, c - step); c1 = min(w, c + step + 1)
        seam[r] = c0 + np.argmin(dp[r, c0:c1])
    return seam

def render_fullres(image_paths, order, work_transforms, work_scale):
    """Load full-res images one at a time, warp with scaled-up transforms, composite."""
    # Scale transforms from work-res to full-res
    if work_scale >= 1.0:
        full_scale = 1.0
    else:
        full_scale = 1.0 / work_scale  # e.g. if work_scale=0.33, full_scale=3.0

    full_transforms = []
    for T in work_transforms:
        T_full = scale_transform(T, 1.0, full_scale)
        full_transforms.append(T_full)

    # Compute canvas size by projecting corners of each full-res image
    img_dims = []
    for idx in order:
        img = cv2.imread(image_paths[idx])
        img = auto_rotate(img)
        img_dims.append((img.shape[0], img.shape[1]))
        del img
        gc.collect()

    all_corners = []
    for i, (h, w) in enumerate(img_dims):
        corners = np.float64([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]])
        mapped = (full_transforms[i] @ corners.T).T
        mapped = mapped[:, :2] / mapped[:, 2:3]
        all_corners.append(mapped)

    all_corners = np.vstack(all_corners)
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    log(f"Canvas full-res: {canvas_w}x{canvas_h}")

    # Warp and composite one image at a time
    result = None
    result_mask = None

    for i, idx in enumerate(order):
        log(f"Warp+composite imagen {i+1}/{len(order)} (full-res)...")
        img = cv2.imread(image_paths[idx])
        img = auto_rotate(img)

        # Shift transform so canvas starts at (0,0)
        T_shift = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
        T = T_shift @ full_transforms[i]

        warped = cv2.warpPerspective(img, T, (canvas_w, canvas_h),
                                     flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask, T, (canvas_w, canvas_h))
        del img, mask
        gc.collect()

        if result is None:
            result = warped
            result_mask = warped_mask
            continue

        # Seam-cut composite
        overlap = (result_mask > 0) & (warped_mask > 0)
        if np.any(overlap):
            rows_ov = np.any(overlap, axis=1)
            cols_ov = np.any(overlap, axis=0)
            r0, r1 = np.where(rows_ov)[0][[0, -1]]
            c0, c1 = np.where(cols_ov)[0][[0, -1]]
            ov_h = r1 - r0 + 1
            ov_w = c1 - c0 + 1

            region_l = result[r0:r1+1, c0:c1+1]
            region_r = warped[r0:r1+1, c0:c1+1]

            # Cost = pixel difference + gradient penalty
            # Gradient penalty pushes seam through uniform areas (gaps
            # between people) rather than through edges (faces, bodies)
            diff = cv2.absdiff(region_l, region_r)
            cost = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float64)
            del diff

            gray_l = cv2.cvtColor(region_l, cv2.COLOR_BGR2GRAY).astype(np.float64)
            gray_r = cv2.cvtColor(region_r, cv2.COLOR_BGR2GRAY).astype(np.float64)
            gx_l = cv2.Sobel(gray_l, cv2.CV_64F, 1, 0, ksize=3)
            gy_l = cv2.Sobel(gray_l, cv2.CV_64F, 0, 1, ksize=3)
            gx_r = cv2.Sobel(gray_r, cv2.CV_64F, 1, 0, ksize=3)
            gy_r = cv2.Sobel(gray_r, cv2.CV_64F, 0, 1, ksize=3)
            grad = np.sqrt(np.maximum(gx_l**2 + gy_l**2, gx_r**2 + gy_r**2))
            del gray_l, gray_r, gx_l, gy_l, gx_r, gy_r
            # Normalize gradient to same scale as diff cost and add
            g_max = grad.max()
            if g_max > 0:
                cost += grad * (cost.max() / g_max) * 0.5
            del grad

            ov_region = overlap[r0:r1+1, c0:c1+1]
            cost[~ov_region] = 1e6

            seam = find_optimal_seam(cost)
            del cost
            seam_mask = np.zeros((ov_h, ov_w), dtype=np.float32)
            for r in range(ov_h):
                seam_mask[r, seam[r]:] = 1.0
            del seam
            # Adaptive sigma: ~3% of overlap width, minimum 15
            sigma = max(15, int(ov_w * 0.03))
            seam_mask = cv2.GaussianBlur(seam_mask, (0, 0), sigma)
            sm3 = np.stack([seam_mask]*3, axis=-1)
            blended = region_l.astype(np.float32) * (1.0 - sm3) + \
                      region_r.astype(np.float32) * sm3
            del sm3, seam_mask
            # Only apply blending within actual overlap pixels (not entire bbox)
            ov3 = np.stack([ov_region]*3, axis=-1)
            result[r0:r1+1, c0:c1+1] = np.where(
                ov3,
                np.clip(blended, 0, 255).astype(np.uint8),
                result[r0:r1+1, c0:c1+1]
            )
            del blended, ov3, ov_region

        del overlap
        new_only = (warped_mask > 0) & (result_mask == 0)
        result[new_only] = warped[new_only]
        result_mask = np.maximum(result_mask, warped_mask)
        del warped, warped_mask, new_only
        gc.collect()

    # Crop: trim black borders using fill-ratio per row/column
    # Works for both rectangular panoramas and trapezoidal perspective warps
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    content = gray > 5
    del gray

    row_fill = content.sum(axis=1) / canvas_w
    col_fill = content.sum(axis=0) / canvas_h

    # Progressive trim: keep rows/cols with fill above threshold
    # Start gentle, increase until crop region has good fill ratio
    best_crop = None
    for thresh in [0.3, 0.2, 0.1, 0.01]:
        good_rows = np.where(row_fill >= thresh)[0]
        good_cols = np.where(col_fill >= thresh)[0]
        if len(good_rows) < 100 or len(good_cols) < 100:
            continue
        ct, cb = good_rows[0], good_rows[-1]
        cl, cr = good_cols[0], good_cols[-1]
        crop_w = cr - cl + 1
        crop_h = cb - ct + 1
        crop_fill = content[ct:cb+1, cl:cr+1].sum() / (crop_w * crop_h)
        if crop_fill >= 0.85:
            best_crop = (ct, cb, cl, cr)
            break
        # Also try: if this threshold doesn't give 85% fill,
        # remember it as fallback if it's the best so far
        if best_crop is None:
            best_crop = (ct, cb, cl, cr)

    del content
    if best_crop is not None:
        ct, cb, cl, cr = best_crop
        crop_w = cr - cl + 1
        crop_h = cb - ct + 1
        log(f"Crop: {crop_w}x{crop_h} from {canvas_w}x{canvas_h}")
        if crop_w > 100 and crop_h > 100:
            result = result[ct:cb+1, cl:cr+1]

    return result

# =====================================================================
# PIPELINE PRINCIPAL
# =====================================================================
def stitch_panorama(image_paths, output_path):
    # Read original dimensions and validate
    full_dims = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(json.dumps({"ok": False, "error": f"No se pudo leer: {p}"}))
            return False
        img = auto_rotate(img)
        full_dims.append((img.shape[0], img.shape[1]))
        del img

    n = len(full_dims)
    fh, fw = full_dims[0]
    gc.collect()

    if n == 1:
        img = cv2.imread(image_paths[0])
        img = auto_rotate(img)
        cv2.imwrite(output_path, img, [cv2.IMWRITE_WEBP_QUALITY, 101])
        print(json.dumps({"ok": True, "method": "single"}))
        return True

    # Estimate memory for full-res SCANS: ~10x raw image size
    raw_mb = sum(h * w * 3 for h, w in full_dims) / 1024 / 1024
    scans_est_mb = raw_mb * 10
    avail_mb = get_available_mem_mb()
    safe_for_fullres = scans_est_mb < avail_mb * 0.7  # leave 30% headroom

    log(f"{n} fotos, {fw}x{fh}, raw={raw_mb:.0f} MB, avail={avail_mb:.0f} MB")
    log(f"SCANS full-res estimado: {scans_est_mb:.0f} MB → {'OK' if safe_for_fullres else 'riesgoso'}")

    stitch_err = ""

    # --- Intento 1: SCANS full-res (si hay memoria suficiente) ---
    if safe_for_fullres:
        full_imgs = []
        for p in image_paths:
            img = cv2.imread(p)
            img = auto_rotate(img)
            full_imgs.append(img)

        for thresh in [1.0, 0.5, 0.3]:
            log(f"SCANS full-res (t={thresh}, reg=0.3)...")
            pano = try_stitcher(full_imgs, fh, fw, n, thresh, reg_resol=0.3)
            if pano is not None:
                del full_imgs
                cv2.imwrite(output_path, pano, [cv2.IMWRITE_WEBP_QUALITY, 101])
                h, w = pano.shape[:2]
                print(json.dumps({"ok": True, "method": f"scans(t={thresh})", "size": f"{w}x{h}", "imgs": n}))
                return True
            gc.collect()

        del full_imgs
        gc.collect()
        stitch_err = "SCANS full-res failed"
        log("SCANS full-res falló en todos los thresholds")
    else:
        # SCANS at work-res as feasibility test, then try full-res with lowest threshold
        work_imgs = []
        work_scale = 1.0
        for p in image_paths:
            img = cv2.imread(p)
            img = auto_rotate(img)
            small, s = downscale(img, 2000)
            if s < 1.0:
                work_scale = s
            work_imgs.append(small)
            del img
            gc.collect()

        scans_feasible = False
        for thresh in [1.0, 0.5, 0.3]:
            wh, ww = work_imgs[0].shape[:2]
            pano = try_stitcher(work_imgs, wh, ww, n, thresh)
            if pano is not None:
                del pano
                scans_feasible = True
                log(f"SCANS OK en work-res (t={thresh}). Intentando full-res con reg=0.2...")
                break

        del work_imgs
        gc.collect()

        if scans_feasible:
            full_imgs = []
            for p in image_paths:
                img = cv2.imread(p)
                img = auto_rotate(img)
                full_imgs.append(img)

            # Try with very low registration resolution to save memory
            for reg in [0.2, 0.15]:
                for thresh in [1.0, 0.5, 0.3]:
                    log(f"SCANS full-res (t={thresh}, reg={reg})...")
                    pano = try_stitcher(full_imgs, fh, fw, n, thresh, reg_resol=reg)
                    if pano is not None:
                        del full_imgs
                        cv2.imwrite(output_path, pano, [cv2.IMWRITE_WEBP_QUALITY, 101])
                        h, w = pano.shape[:2]
                        print(json.dumps({"ok": True, "method": f"scans(t={thresh},r={reg})", "size": f"{w}x{h}", "imgs": n}))
                        return True
                    gc.collect()

            del full_imgs
            gc.collect()
            stitch_err = "SCANS full-res failed (low mem mode)"
            log("SCANS full-res falló incluso con reg baja")
        else:
            stitch_err = "SCANS failed at all thresholds"

    # --- Intento 2: SIFT at WORK_DIM → render full-res ---
    log(f"SIFT+Affine (features a {WORK_DIM}px, render full-res)...")
    work_imgs = []
    work_scale = 1.0
    for p in image_paths:
        img = cv2.imread(p)
        img = auto_rotate(img)
        small, s = downscale(img, WORK_DIM)
        if s < 1.0:
            work_scale = s
        work_imgs.append(small)
        del img
        gc.collect()

    order = find_spatial_order(work_imgs)
    log(f"Orden espacial: {order}")
    transforms = compute_transforms_lowres(work_imgs, order)
    del work_imgs
    gc.collect()

    if transforms is not None:
        result = render_fullres(image_paths, order, transforms, work_scale)
        if result is not None:
            rh, rw = result.shape[:2]
            # Light validation: only reject clearly broken results
            # After auto_rotate images are portrait, panorama should be landscape
            single_w_rot = min(fh, fw)  # width of a single portrait image
            sift_ok = True
            if n >= 3 and rw <= rh:
                sift_ok = False  # 3+ images should produce landscape panorama
                log(f"SIFT: resultado portrait ({rw}x{rh}) con {n} fotos, descartando")
            elif rw < single_w_rot:
                sift_ok = False  # panorama narrower than a single image
                log(f"SIFT: resultado más angosto ({rw}) que una foto ({single_w_rot}), descartando")
            if sift_ok:
                cv2.imwrite(output_path, result, [cv2.IMWRITE_WEBP_QUALITY, 101])
                order_str = "-".join(str(i) for i in order)
                print(json.dumps({"ok": True, "method": "sift_affine", "size": f"{rw}x{rh}", "imgs": n, "order": order_str, "stitch_error": stitch_err}))
                return True
        del result
        gc.collect()

    log("SIFT falló, usando concatenación directa full-res...")

    # --- Fallback: concatenar full-res ---
    result = None
    for i, idx in enumerate(order):
        img = cv2.imread(image_paths[idx])
        img = auto_rotate(img)
        if result is None:
            result = img
            continue
        h1, h2 = result.shape[0], img.shape[0]
        target_h = max(h1, h2)
        if h1 != target_h:
            s = target_h / h1
            result = cv2.resize(result, (int(result.shape[1]*s), target_h), interpolation=cv2.INTER_LANCZOS4)
        if h2 != target_h:
            s = target_h / h2
            img = cv2.resize(img, (int(img.shape[1]*s), target_h), interpolation=cv2.INTER_LANCZOS4)
        result = np.hstack([result, img])
        del img
        gc.collect()

    cv2.imwrite(output_path, result, [cv2.IMWRITE_WEBP_QUALITY, 101])
    h, w = result.shape[:2]
    order_str = "-".join(str(i) for i in order)
    print(json.dumps({"ok": True, "method": "concat", "size": f"{w}x{h}", "imgs": n, "order": order_str, "stitch_error": stitch_err}))
    return True

if __name__ == "__main__":
    output = sys.argv[1]
    images = sys.argv[2:]
    stitch_panorama(images, output)
PYEOF
)

# --- Procesamiento ---
process_group() {
    local cbtis="$1"
    local group_dir="$2"
    local group_name
    group_name="$(basename "$group_dir")"
    local unidas_dir
    unidas_dir="$(dirname "$group_dir")/UNIDAS"

    # Recopilar fotos jpg/jpeg/png ordenadas por nombre
    mapfile -t photos < <(find "$group_dir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort)

    if [ ${#photos[@]} -eq 0 ]; then
        log_warn "Sin fotos en $group_name (CBTIS $cbtis)"
        return
    fi

    # Extraer IDs de los nombres de archivo (IMG_1184.jpg → 1184)
    local ids=()
    for photo in "${photos[@]}"; do
        fname="$(basename "$photo")"
        # Extraer números del nombre, quitando extensión y prefijos
        id=$(echo "$fname" | sed 's/\.[^.]*$//' | grep -oP '\d+' | tail -1)
        ids+=("$id")
    done

    # Construir nombre de salida: IDs-IDS_CBTIS_GRUPO.webp
    local ids_str
    ids_str=$(IFS=-; echo "${ids[*]}")
    local output_name="${ids_str}_${cbtis}_${group_name}.webp"
    local output_path="$unidas_dir/$output_name"

    # Verificar si ya existe
    if [ -f "$output_path" ]; then
        log_warn "Ya existe: $output_name — omitiendo"
        return
    fi

    mkdir -p "$unidas_dir"

    log_info "Procesando CBTIS $cbtis / $group_name (${#photos[@]} fotos)..."

    # Ejecutar Python stitcher
    # stderr (logs [INFO]) se muestra directamente; solo stdout (JSON) se captura
    local result
    result=$($PYTHON -c "$PYTHON_STITCHER" "$output_path" "${photos[@]}" 2> >(while IFS= read -r line; do log_info "$line"; done))

    # Extract only the last JSON line from result (ignore any non-JSON output)
    local json_line
    json_line=$(echo "$result" | grep -E '^\{' | tail -1)

    local ok
    ok=$(echo "$json_line" | $PYTHON -c "import sys,json; print(json.loads(sys.stdin.read()).get('ok', False))" 2>/dev/null || echo "False")

    if [ "$ok" = "True" ]; then
        local method
        method=$(echo "$json_line" | $PYTHON -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('method','?')); e=d.get('stitch_error',''); print(f' (fallback: {e})' if e else '')" 2>/dev/null | tr '\n' ' ')
        local size
        size=$(du -h "$output_path" | cut -f1)
        log_ok "$output_name ($size) — método: $method"
    else
        log_err "Fallo al procesar $group_name: $json_line"
    fi
}

total=0
ok_count=0

for cbtis_dir in "$YEAR_DIR"/*/; do
    [ ! -d "$cbtis_dir" ] && continue
    cbtis="$(basename "$cbtis_dir")"

    # Filtrar por CBTIS si se especificó
    if [ -n "$CBTIS_FILTER" ] && [ "$cbtis" != "$CBTIS_FILTER" ]; then
        continue
    fi

    log_info "=== CBTIS $cbtis ==="

    for group_dir in "$cbtis_dir"/*/; do
        [ ! -d "$group_dir" ] && continue
        group_name="$(basename "$group_dir")"

        # Saltar directorio UNIDAS
        [ "$group_name" = "UNIDAS" ] && continue

        # Filtrar por grupo si se especificó
        if [ -n "$GROUP_FILTER" ] && [ "$group_name" != "$GROUP_FILTER" ]; then
            continue
        fi

        process_group "$cbtis" "$group_dir"
        total=$((total + 1))
    done
done

echo ""
log_info "Procesamiento completo. Total grupos: $total"
