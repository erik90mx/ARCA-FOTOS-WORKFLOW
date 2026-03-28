#!/usr/bin/env python3
"""ARCA Remover de Fondo - CLI para remover fondo de panorámicas.

Pipeline:
  1. BiRefNet (state-of-the-art) en GPU - máscara de foreground
  2. Guided Filter (He et al. 2013) + sigmoid - refine edge + anti-staircase
  3. Flood-fill desde bordes - elimina fondo externo conectado
  4. Alpha suave con interior opaco (> 0.95 → 1.0)
  5. Feathering (Gaussian blur sigma=1.2) - difuminado natural de bordes
  6. Descontaminación de color en bordes (elimina fringe del fondo)
  7. Auto-crop + salida webp lossless

Uso:
  python3 arca_remover.py <workspace> [cbtis] [filename]

Requiere: torch, torchvision, transformers, Pillow, scipy, numpy
GPU: NVIDIA con CUDA (recomendado 6+ GB VRAM)
"""

import gc
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import label, binary_erosion, binary_dilation

SUPPORTED_EXTENSIONS = (".webp", ".jpg", ".jpeg", ".png")

MODEL_ID = "ZhengPeng7/BiRefNet"


def log(msg):
    print(f"[INFO] rm: {msg}", file=sys.stderr, flush=True)


# ── Motor BiRefNet ──────────────────────────────────────────────────────


class BiRefNetEngine:
    """BiRefNet para máscaras de foreground con bordes de alta calidad."""

    def __init__(self, model_id=MODEL_ID, device="cuda", resolution=1024):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.resolution = resolution
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms as T

        log(f"Cargando modelo {model_id} en {self.device}...")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model.to(self.device).eval()
        self.use_fp16 = False
        if self.device.type == "cuda":
            try:
                self.model.half()
                self.use_fp16 = True
            except Exception:
                pass

        self.transform = T.Compose([
            T.Resize((resolution, resolution)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        log(f"BiRefNet OK {'(FP16 GPU)' if self.use_fp16 else '(FP32 CPU)'}")

    @torch.no_grad()
    def predict_mask(self, img_pil):
        """Devuelve máscara float32 [0..1], tamaño original."""
        orig_size = img_pil.size
        inp = self.transform(img_pil.convert("RGB")).unsqueeze(0).to(self.device)
        if self.use_fp16:
            inp = inp.half()
        preds = self.model(inp)[-1]
        pred = torch.sigmoid(preds[0, 0]).float().cpu().numpy()
        mask = Image.fromarray((pred * 255).astype(np.uint8), "L")
        mask = mask.resize(orig_size, Image.LANCZOS)
        return np.array(mask).astype(np.float32) / 255.0


# ── Procesamiento ───────────────────────────────────────────────────────


def _rgb_to_saturation(img_np):
    """Calcula saturación HSV por pixel (0-1)."""
    r = img_np[:, :, 0].astype(np.float32) / 255.0
    g = img_np[:, :, 1].astype(np.float32) / 255.0
    b = img_np[:, :, 2].astype(np.float32) / 255.0
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    sat = np.zeros_like(cmax)
    mask = cmax > 0
    sat[mask] = delta[mask] / cmax[mask]
    return sat


def _guided_filter(guide, src, radius, eps):
    """Guided filter (He et al. 2013) — edge-aware mask refinement.

    Uses the original photo as guide to snap mask edges to real photo edges.
    Like Photoshop's "Refine Edge". No staircase, no blur halo.

    guide: float32 (H,W) luminance of original photo, [0..1]
    src:   float32 (H,W) upscaled BiRefNet mask, [0..1]
    radius: window radius (16-32 for ~8x upscale)
    eps:   regularization (0.001-0.01, smaller = sharper edges)
    """
    from scipy.ndimage import uniform_filter
    size = 2 * radius + 1

    mean_I = uniform_filter(guide, size=size, mode='reflect')
    mean_p = uniform_filter(src, size=size, mode='reflect')
    mean_Ip = uniform_filter(guide * src, size=size, mode='reflect')
    mean_II = uniform_filter(guide * guide, size=size, mode='reflect')

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = uniform_filter(a, size=size, mode='reflect')
    mean_b = uniform_filter(b, size=size, mode='reflect')

    return mean_a * guide + mean_b


def _color_decontaminate(img_np, alpha_np):
    """Replace RGB of semi-transparent edge pixels with nearest opaque fg color.

    This removes the background color 'fringe' visible on transparent edges.
    For each pixel with alpha in (10, 245), replace its RGB with a weighted
    average of nearby fully-opaque pixels.
    """
    h, w = alpha_np.shape
    result = img_np.copy()
    # Build mask of semi-transparent pixels (the edge band)
    edge = (alpha_np > 10) & (alpha_np < 245)
    if not edge.any():
        return result

    # Build mask of fully opaque interior pixels
    opaque = alpha_np >= 245

    # Blur the opaque colors to get local fg color estimate
    # Weight by alpha to ignore background
    weight = np.where(opaque, 1.0, 0.0).astype(np.float32)
    from scipy.ndimage import uniform_filter
    radius = 7
    r_sum = uniform_filter(img_np[:, :, 0].astype(np.float32) * weight, size=radius * 2 + 1, mode='constant')
    g_sum = uniform_filter(img_np[:, :, 1].astype(np.float32) * weight, size=radius * 2 + 1, mode='constant')
    b_sum = uniform_filter(img_np[:, :, 2].astype(np.float32) * weight, size=radius * 2 + 1, mode='constant')
    w_sum = uniform_filter(weight, size=radius * 2 + 1, mode='constant')

    valid = w_sum > 0.01
    edge_valid = edge & valid
    result[edge_valid, 0] = np.clip(r_sum[edge_valid] / w_sum[edge_valid], 0, 255).astype(np.uint8)
    result[edge_valid, 1] = np.clip(g_sum[edge_valid] / w_sum[edge_valid], 0, 255).astype(np.uint8)
    result[edge_valid, 2] = np.clip(b_sum[edge_valid] / w_sum[edge_valid], 0, 255).astype(np.uint8)
    return result


def remove_bg(img, engine):
    """Remueve el fondo usando BiRefNet + flood-fill + alpha suave por distance transform."""
    from scipy.ndimage import find_objects
    w, h = img.size
    img_rgb = img.convert("RGB")
    img_np = np.array(img_rgb)

    # 1) BiRefNet máscara + guided filter refinement
    log("  [1/6] BiRefNet máscara...")
    from scipy.ndimage import gaussian_filter
    br_raw = engine.predict_mask(img_rgb)
    upscale = max(w, h) / engine.resolution

    # Guided filter: snap mask edges to real photo edges (like Photoshop Refine Edge)
    # Uses original photo luminance as guide — eliminates staircase WITHOUT blur
    # Then sigmoid sharpening compresses transition to ~3px (anti-alias, no halo)
    log("  [2/6] Guided filter (refine edge)...")
    lum = (0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]).astype(np.float32) / 255.0
    gf_radius = max(4, int(upscale))
    br_gf = np.clip(_guided_filter(lum, br_raw.astype(np.float32), radius=gf_radius, eps=0.01), 0, 1)
    # Sigmoid sharpening: guided filter aligns edge position (no staircase),
    # sigmoid controls edge WIDTH (narrow transition, no halo)
    br = 1.0 / (1.0 + np.exp(-14.0 * (br_gf - 0.5)))

    # 3) Flood-fill: quitar fondo externo conectado a bordes
    log("  [3/6] Flood-fill fondo externo...")
    binary = (br > 0.50).astype(np.uint8)
    bg = 1 - binary
    labeled_bg, _ = label(bg)
    border_labels = set()
    border_labels.update(labeled_bg[0, :].flatten())
    border_labels.update(labeled_bg[-1, :].flatten())
    border_labels.update(labeled_bg[:, 0].flatten())
    border_labels.update(labeled_bg[:, -1].flatten())
    border_labels.discard(0)
    ext_bg = np.zeros_like(binary)
    for lbl in border_labels:
        ext_bg[labeled_bg == lbl] = 1
    fg = (1 - ext_bg).astype(np.uint8)

    # 2b) Remove large interior bg holes (connected-component, NOT global threshold)
    # Only removes LARGE contiguous low-confidence regions (wall/ceiling/floor)
    # Keeps faces and small features even if BiRefNet gives low confidence
    interior_low = fg & (br < 0.35).astype(np.uint8)
    if interior_low.any():
        il_labeled, il_num = label(interior_low)
        if il_num > 0:
            il_sizes = np.bincount(il_labeled.ravel())
            fg_area = fg.sum()
            min_hole = max(fg_area * 0.005, 1000)
            il_slices = find_objects(il_labeled)
            holes_removed = 0
            for i, sl in enumerate(il_slices, 1):
                if sl is None or il_sizes[i] < min_hole:
                    continue
                # Only remove if average BiRefNet confidence is very low
                comp = il_labeled[sl] == i
                avg_br = br[sl][comp].mean()
                if avg_br < 0.18:
                    fg[sl][comp] = 0
                    holes_removed += 1
            if holes_removed:
                log(f"  {holes_removed} huecos interiores removidos")

    # 4) Alpha: sigmoid-sharpened mask + feathering for natural blending
    log("  [4/6] Alpha suave...")

    alpha = br.copy()
    alpha[ext_bg > 0] = 0.0
    alpha[fg == 0] = 0.0

    # Sigmoid already made confident areas ~0.99+, clamp to fully opaque
    alpha[alpha > 0.95] = 1.0

    alpha_np = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    # 5) Feather: Gaussian blur on alpha for smooth edge blending
    # Standard technique (Photoshop "Feather") — sigma=1.2 ≈ 3-4px transition
    # Interior (255) and exterior (0) stay unchanged; only boundary softens
    log("  [5/6] Feathering bordes...")
    alpha_np = np.clip(gaussian_filter(alpha_np.astype(np.float32), sigma=1.2), 0, 255).astype(np.uint8)

    # 6) Color decontamination — remove bg color bleed from edge pixels
    log("  [6/6] Descontaminación de color...")
    img_clean = _color_decontaminate(img_np, alpha_np)

    img_rgba = Image.fromarray(img_clean, "RGB").convert("RGBA")
    img_rgba.putalpha(Image.fromarray(alpha_np, "L"))
    return img_rgba


def autocrop(img, padding=10, alpha_threshold=10):
    """Recorta espacio transparente. Retorna (img_cropped, crop_bbox)."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = img.split()[3]
    mask = alpha.point(lambda p: 255 if p > alpha_threshold else 0)
    bbox = mask.getbbox()
    if not bbox:
        return img, (0, 0, img.size[0], img.size[1])
    x1, y1, x2, y2 = bbox
    w, h = img.size
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return img.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)


# ── CLI ─────────────────────────────────────────────────────────────────


def process_cbtis(workspace, cbtis, filename=None):
    """Process panoramas from UNIDAS/ → RECORTADAS/ for a given CBTIS."""
    year_dir = os.path.join(workspace, "2026")
    cbtis_dir = os.path.join(year_dir, cbtis)
    unidas_dir = os.path.join(cbtis_dir, "UNIDAS")
    recortadas_dir = os.path.join(cbtis_dir, "RECORTADAS")

    if not os.path.isdir(unidas_dir):
        print(json.dumps({"ok": False, "error": f"No existe: {unidas_dir}"}))
        return False

    # Collect files to process
    if filename:
        files = [filename] if os.path.isfile(os.path.join(unidas_dir, filename)) else []
        if not files:
            print(json.dumps({"ok": False, "error": f"No existe: {filename}"}))
            return False
    else:
        files = sorted(
            f for f in os.listdir(unidas_dir)
            if os.path.isfile(os.path.join(unidas_dir, f))
            and f.lower().endswith(SUPPORTED_EXTENSIONS)
        )

    if not files:
        print(json.dumps({"ok": False, "error": "No hay panorámicas en UNIDAS/"}))
        return False

    # Skip already processed
    os.makedirs(recortadas_dir, exist_ok=True)
    existing = set(os.listdir(recortadas_dir))
    to_process = []
    for f in files:
        out_name = os.path.splitext(f)[0] + ".webp"
        if out_name not in existing:
            to_process.append(f)
        else:
            log(f"Ya existe: {out_name} — omitiendo")

    if not to_process:
        print(json.dumps({"ok": True, "processed": 0, "skipped": len(files),
                          "message": "Todos ya procesados"}))
        return True

    log(f"CBTIS {cbtis}: {len(to_process)} panorámicas por procesar")

    # Load engine once
    engine = BiRefNetEngine()

    errors = []
    t_start = time.time()

    for i, fname in enumerate(to_process, 1):
        t0 = time.time()
        src_path = os.path.join(unidas_dir, fname)
        out_name = os.path.splitext(fname)[0] + ".webp"
        out_path = os.path.join(recortadas_dir, out_name)

        try:
            log(f"[{i}/{len(to_process)}] {fname}")
            img = Image.open(src_path)
            w_orig, h_orig = img.size
            log(f"  Tamaño: {w_orig}x{h_orig}")

            result = remove_bg(img, engine)
            result, crop_bbox = autocrop(result, padding=10)
            del img

            result.save(out_path, format="WEBP", lossless=True, quality=100)

            # Save crop metadata for refine tool alignment
            meta_path = os.path.splitext(out_path)[0] + ".meta.json"
            with open(meta_path, "w") as mf:
                json.dump({"crop_bbox": list(crop_bbox),
                           "original_size": [w_orig, h_orig]}, mf)

            elapsed = time.time() - t0
            w_out, h_out = result.size
            fsize = round(os.path.getsize(out_path) / 1024 / 1024, 1)
            log(f"  OK → {out_name} ({w_out}x{h_out}, {fsize}MB) [{elapsed:.1f}s]")
            del result
        except Exception as exc:
            elapsed = time.time() - t0
            errors.append({"file": fname, "error": str(exc)})
            log(f"  ERROR: {exc} [{elapsed:.1f}s]")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    t_total = time.time() - t_start
    processed = len(to_process) - len(errors)

    result = {
        "ok": len(errors) == 0,
        "cbtis": cbtis,
        "processed": processed,
        "errors": len(errors),
        "skipped": len(files) - len(to_process),
        "total_time": round(t_total, 1),
        "avg_time": round(t_total / len(to_process), 1) if to_process else 0,
    }
    if errors:
        result["error_details"] = errors

    print(json.dumps(result))
    return len(errors) == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Uso: {sys.argv[0]} <workspace> [cbtis] [filename]")
        sys.exit(1)

    workspace = sys.argv[1]
    cbtis_filter = sys.argv[2] if len(sys.argv) > 2 else None
    file_filter = sys.argv[3] if len(sys.argv) > 3 else None

    if cbtis_filter:
        process_cbtis(workspace, cbtis_filter, file_filter)
    else:
        # Process all CBTIS
        year_dir = os.path.join(workspace, "2026")
        if not os.path.isdir(year_dir):
            print(json.dumps({"ok": False, "error": f"No existe: {year_dir}"}))
            sys.exit(1)

        for cbtis in sorted(os.listdir(year_dir)):
            cbtis_dir = os.path.join(year_dir, cbtis)
            unidas_dir = os.path.join(cbtis_dir, "UNIDAS")
            if os.path.isdir(unidas_dir) and cbtis.isdigit():
                process_cbtis(workspace, cbtis)
