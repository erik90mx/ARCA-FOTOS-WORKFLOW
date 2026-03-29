#!/usr/bin/env python3
"""ARCA Panorama GUI - Web interface for managing panorama batch jobs.

Runs as a Flask server in WSL2, opens in Windows browser.
Processes photos from any path (including Windows filesystem via /mnt/).
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import deque
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_file
from PIL import Image
from scipy.ndimage import gaussian_filter

APP_DIR = Path(__file__).resolve().parent
SCRIPT = APP_DIR / "arca_panorama.sh"
REMOVER_SCRIPT = APP_DIR / "arca_remover.py"
CONFIG_FILE = APP_DIR / "arca_gui_config.json"

app = Flask(__name__)

# --- Configuration ---
config = {
    "workspace": str(APP_DIR),  # default: same dir as script
}


def load_config():
    global config
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            config.update(saved)
        except Exception:
            pass


def save_config():
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_year_dir() -> Path:
    return Path(config["workspace"]) / "2026"


def win_to_wsl(winpath: str) -> str:
    """Convert Windows path like C:\\Users\\... to /mnt/c/Users/..."""
    p = winpath.strip().replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return p


# --- ODS Parser ---

ODS_NS = {
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
}
# Column indices after expanding repeats
_COL_FOTOS = 1
_COL_TN = 3
_COL_GP = 4
_COL_AB = 5
_COL_COLOR = 7
_COL_OBS = 8


def _cell_text(cell) -> str:
    """Extract text content from an ODS table-cell via <text:p>."""
    p = cell.find("text:p", ODS_NS)
    return (p.text or "").strip() if p is not None else ""


def _expand_row_cells(row) -> list[str]:
    """Expand repeated cells in a row, returning list of text values (capped at 20 cols)."""
    result: list[str] = []
    for cell in row.findall("table:table-cell", ODS_NS):
        repeat = int(cell.get(
            "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}number-columns-repeated", "1"
        ))
        text = _cell_text(cell)
        # Cap repeat to avoid inflating with 996-cell padding
        repeat = min(repeat, 20 - len(result)) if len(result) < 20 else 0
        for _ in range(max(repeat, 0)):
            result.append(text)
            if len(result) >= 20:
                break
    return result


def parse_ods_groups(ods_path: str) -> dict[str, list[dict]]:
    """Parse DATOS_2026.ods and return groups per CBTIS.

    Returns {"105": [{"fotografias": "1730-1709-1737", "tn": "MAT", ...}], ...}
    """
    result: dict[str, list[dict]] = {}
    with zipfile.ZipFile(ods_path) as z:
        content = z.read("content.xml")
    root = ET.fromstring(content)
    body = root.find(".//office:body/office:spreadsheet", ODS_NS)
    if body is None:
        return result

    for table in body.findall("table:table", ODS_NS):
        sheet_name = table.get(
            "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}name", ""
        )
        # Skip lookup sheet
        if sheet_name.startswith("####"):
            continue
        # Extract CBTIS number from sheet name like "105 Z.S"
        m = re.search(r"(\d+)", sheet_name)
        if not m:
            continue
        cbtis_num = m.group(1)
        groups: list[dict] = []

        rows = table.findall("table:table-row", ODS_NS)
        for row in rows[1:]:  # skip header row
            cells = _expand_row_cells(row)
            if len(cells) <= _COL_FOTOS:
                continue
            fotos_str = cells[_COL_FOTOS]
            if not fotos_str:
                continue
            tn = cells[_COL_TN] if len(cells) > _COL_TN else ""
            gp = cells[_COL_GP] if len(cells) > _COL_GP else ""
            ab = cells[_COL_AB] if len(cells) > _COL_AB else ""
            color = cells[_COL_COLOR] if len(cells) > _COL_COLOR else ""
            obs = cells[_COL_OBS] if len(cells) > _COL_OBS else ""
            if not tn:
                continue
            groups.append({
                "fotografias": fotos_str,
                "tn": tn, "gp": gp, "ab": ab,
                "color": color, "obs": obs,
            })

        if groups:
            result[cbtis_num] = groups
    return result


def parse_fotografias(foto_str: str) -> list[str]:
    """Parse FOTOGRAFIAS field into list of image basenames (no extension).

    "1730-1709-1737" -> ["IMG_1730", "IMG_1709", "IMG_1737"]
    "-1357-1368"     -> ["IMG_1357", "IMG_1368"]
    "2799"           -> ["IMG_2799"]
    """
    parts = foto_str.split("-")
    return [f"IMG_{p.strip()}" for p in parts if p.strip()]


def make_group_dirname(group: dict) -> str:
    """Build directory name from group dict: TN_GP_ab_COLOR."""
    return f"{group['tn']}_{group['gp']}_{group['ab']}_{group['color']}"


def build_file_index(directory: Path) -> dict[str, str]:
    """Build a case-insensitive index {lowercase_name: actual_name} for a directory."""
    index: dict[str, str] = {}
    if not directory.is_dir():
        return index
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            index[f.name.lower()] = f.name
    return index


def find_photo_on_disk(file_index: dict[str, str], img_name: str) -> str | None:
    """Find a photo file by base name (e.g. 'IMG_1730') in pre-built index.

    Tries .jpg, .JPG, .jpeg, .png case-insensitively.
    """
    for ext in (".jpg", ".jpeg", ".png"):
        key = (img_name + ext).lower()
        if key in file_index:
            return file_index[key]
    return None


# --- Process state ---
process_lock = threading.Lock()
current_process: subprocess.Popen | None = None
log_buffer: deque[dict] = deque(maxlen=10000)
run_id = 0

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TAG_RE = re.compile(r"^\[(INFO|OK|WARN|ERROR)\]\s*")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def classify_line(raw: str) -> dict:
    clean = strip_ansi(raw).strip()
    if not clean:
        return {"text": clean, "level": "info"}
    m = TAG_RE.match(clean)
    if m:
        tag = m.group(1).lower()
        level = {"info": "info", "ok": "ok", "warn": "warn", "error": "error"}.get(tag, "info")
        return {"text": clean, "level": level}
    if clean.startswith("{"):
        try:
            data = json.loads(clean)
            level = "ok" if data.get("ok") else "error"
            return {"text": clean, "level": level}
        except json.JSONDecodeError:
            pass
    return {"text": clean, "level": "info"}


def scan_groups() -> list[dict]:
    """Scan the 2026/ directory and return status of all CBTIS/groups."""
    results = []
    year_dir = get_year_dir()
    if not year_dir.is_dir():
        return results
    for cbtis_dir in sorted(year_dir.iterdir()):
        if not cbtis_dir.is_dir():
            continue
        cbtis = cbtis_dir.name
        if not cbtis.isdigit():
            continue
        unidas_dir = cbtis_dir / "UNIDAS"
        recortadas_dir = cbtis_dir / "RECORTADAS"
        existing = set()
        if unidas_dir.is_dir():
            for f in unidas_dir.iterdir():
                if f.suffix == ".webp":
                    existing.add(f.name)
        cutout_existing = set()
        borlas_existing = set()
        togas_existing = set()
        sombras_existing = set()
        fixes_existing = set()
        if recortadas_dir.is_dir():
            for f in recortadas_dir.iterdir():
                if f.suffix == ".webp":
                    cutout_existing.add(f.name)
                elif f.name.endswith(".borlas.json") and f.name.startswith("."):
                    borlas_existing.add(f.name)
                elif f.name.endswith(".sombras.json") and f.name.startswith("."):
                    sombras_existing.add(f.name)
                elif f.name.endswith(".fixes.json"):
                    fixes_existing.add(f.name)
                elif f.name.endswith(".togas.json") and f.name.startswith("."):
                    togas_existing.add(f.name)
        # Load workflow flags
        workflow_path = cbtis_dir / ".workflow.json"
        workflow = {}
        if workflow_path.exists():
            try:
                with open(workflow_path) as wf:
                    workflow = json.load(wf)
            except Exception:
                pass
        for group_dir in sorted(cbtis_dir.iterdir()):
            if not group_dir.is_dir() or group_dir.name in ("UNIDAS", "RECORTADAS"):
                continue
            group = group_dir.name
            photos = sorted(
                p.name for p in group_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            photo_count = len(photos)
            completed = any(f"_{cbtis}_{group}.webp" in name for name in existing)
            # Find output file for preview
            output_file = ""
            if completed:
                for name in existing:
                    if f"_{cbtis}_{group}.webp" in name:
                        output_file = name
                        break
            # Check if cutout exists for this group's output
            cutout_file = ""
            has_borlas = False
            has_togas = False
            has_sombras = False
            has_fixes = False
            if output_file:
                cutout_name = os.path.splitext(output_file)[0] + ".webp"
                if cutout_name in cutout_existing:
                    cutout_file = cutout_name
                    stem = os.path.splitext(cutout_name)[0]
                    if f".{stem}.borlas.json" in borlas_existing:
                        has_borlas = True
                    if f".{stem}.togas.json" in togas_existing:
                        has_togas = True
                    if f".{stem}.sombras.json" in sombras_existing:
                        has_sombras = True
                    if f"{stem}.fixes.json" in fixes_existing:
                        has_fixes = True
            grp_wf = workflow.get(group, {})
            results.append({
                "cbtis": cbtis,
                "group": group,
                "photos": photo_count,
                "photo_names": photos,
                "completed": completed,
                "output": output_file,
                "cutout": cutout_file,
                "has_borlas": has_borlas,
                "has_togas": has_togas,
                "has_sombras": has_sombras,
                "has_fixes": has_fixes,
                "workflow": {
                    "pano_done": grp_wf.get("pano_done", False),
                    "recorte_done": grp_wf.get("recorte_done", False),
                    "borlas_done": grp_wf.get("borlas_done", False),
                    "togas_done": grp_wf.get("togas_done", False),
                    "sombras_done": grp_wf.get("sombras_done", False),
                    "fixes_done": grp_wf.get("fixes_done", False),
                },
            })
    return results


stop_requested = False


def _run_one(cmd: list[str], rid: int) -> int:
    """Run one arca_panorama.sh invocation, streaming output. Returns exit code."""
    global current_process

    env = os.environ.copy()
    env["ARCA_BASE_DIR"] = config["workspace"]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=config["workspace"],
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
        env=env,
    )
    with process_lock:
        current_process = proc

    for line in proc.stdout:
        entry = classify_line(line)
        entry["ts"] = time.time()
        entry["run"] = rid
        with process_lock:
            log_buffer.append(entry)

    proc.wait()
    return proc.returncode


def run_script(cbtis: str = "", group: str = ""):
    """Run arca_panorama.sh for a single target."""
    run_batch([{"cbtis": cbtis, "group": group}])


def run_batch(items: list[dict]):
    """Run arca_panorama.sh sequentially for a list of {cbtis, group} items."""
    global current_process, run_id, stop_requested
    stop_requested = False

    with process_lock:
        run_id += 1
        rid = run_id
        log_buffer.clear()

    total = len(items)

    for idx, item in enumerate(items, 1):
        if stop_requested:
            with process_lock:
                log_buffer.append({
                    "text": f"=== Detenido por el usuario ({idx-1}/{total} completados) ===",
                    "level": "warn", "ts": time.time(), "run": rid,
                })
            break

        cbtis = item.get("cbtis", "")
        group = item.get("group", "")

        cmd = ["bash", str(SCRIPT)]
        if cbtis:
            cmd.append(cbtis)
            if group:
                cmd.append(group)

        label = f"CBTIS {cbtis}" if cbtis else "TODOS"
        if group:
            label += f" / {group}"

        with process_lock:
            log_buffer.append({
                "text": f"=== [{idx}/{total}] Iniciando: {label} ===",
                "level": "info", "ts": time.time(), "run": rid,
            })

        try:
            exit_code = _run_one(cmd, rid)
            level = "ok" if exit_code == 0 else "error"
            with process_lock:
                log_buffer.append({
                    "text": f"=== [{idx}/{total}] Terminado: {label} (código: {exit_code}) ===",
                    "level": level, "ts": time.time(), "run": rid,
                })
        except Exception as e:
            with process_lock:
                log_buffer.append({
                    "text": f"=== [{idx}/{total}] Error fatal: {e} ===",
                    "level": "error", "ts": time.time(), "run": rid,
                })

    with process_lock:
        current_process = None
        log_buffer.append({
            "text": f"=== Batch finalizado ({total} grupos) ===",
            "level": "ok", "ts": time.time(), "run": rid,
        })


# --- Routes ---

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/config", methods=["GET"])
def api_config_get():
    return jsonify(config)


@app.route("/api/config", methods=["POST"])
def api_config_set():
    data = request.get_json()
    workspace = data.get("workspace", "").strip()
    if not workspace:
        return jsonify({"error": "Ruta vacía"}), 400
    # Auto-convert Windows paths
    workspace = win_to_wsl(workspace)
    if not Path(workspace).is_dir():
        return jsonify({"error": f"Directorio no existe: {workspace}"}), 400
    year_dir = Path(workspace) / "2026"
    if not year_dir.is_dir():
        return jsonify({"error": f"No se encontró subcarpeta 2026/ en {workspace}"}), 400
    config["workspace"] = workspace
    save_config()
    return jsonify({"ok": True, "workspace": workspace})


@app.route("/api/groups")
def api_groups():
    return jsonify(scan_groups())


@app.route("/api/group/workflow/<cbtis>/<group>", methods=["GET"])
def api_workflow_get(cbtis, group):
    """Get workflow done flags for a group."""
    year_dir = get_year_dir()
    wf_path = year_dir / cbtis / ".workflow.json"
    if wf_path.exists():
        try:
            with open(wf_path) as f:
                data = json.load(f)
            return jsonify(data.get(group, {}))
        except Exception:
            pass
    return jsonify({})


@app.route("/api/group/workflow/<cbtis>/<group>", methods=["POST"])
def api_workflow_set(cbtis, group):
    """Set workflow done flags for a group. Body: {section: "pano"|"recorte"|"borlas", done: bool}"""
    year_dir = get_year_dir()
    cbtis_dir = year_dir / cbtis
    if not cbtis_dir.is_dir():
        return jsonify({"error": "CBTIS no encontrado"}), 404
    wf_path = cbtis_dir / ".workflow.json"
    data = {}
    if wf_path.exists():
        try:
            with open(wf_path) as f:
                data = json.load(f)
        except Exception:
            pass
    section = request.json.get("section", "")
    done = bool(request.json.get("done", False))
    valid = {"pano": "pano_done", "recorte": "recorte_done", "borlas": "borlas_done", "togas": "togas_done", "sombras": "sombras_done", "fixes": "fixes_done"}
    if section not in valid:
        return jsonify({"error": "Sección inválida"}), 400
    if group not in data:
        data[group] = {}
    data[group][valid[section]] = done
    with open(wf_path, "w") as f:
        json.dump(data, f, indent=2)
    return jsonify({"ok": True, "workflow": data[group]})


@app.route("/api/run")
@app.route("/api/run/<cbtis>")
@app.route("/api/run/<cbtis>/<group>")
def api_run(cbtis="", group=""):
    with process_lock:
        if current_process and current_process.poll() is None:
            return jsonify({"error": "Ya hay un proceso en ejecución"}), 409

    t = threading.Thread(target=run_script, args=(cbtis, group), daemon=True)
    t.start()
    return jsonify({"ok": True, "cbtis": cbtis, "group": group})


@app.route("/api/run_batch", methods=["POST"])
def api_run_batch():
    """Run multiple specific groups. Body: {"items": [{"cbtis":"15","group":"MAT_A_..."},..]}"""
    with process_lock:
        if current_process and current_process.poll() is None:
            return jsonify({"error": "Ya hay un proceso en ejecución"}), 409

    data = request.get_json()
    items = data.get("items", [])
    if not items:
        return jsonify({"error": "No se seleccionaron grupos"}), 400

    t = threading.Thread(target=run_batch, args=(items,), daemon=True)
    t.start()
    return jsonify({"ok": True, "count": len(items)})


@app.route("/api/stop")
def api_stop():
    global stop_requested
    stop_requested = True
    with process_lock:
        if current_process and current_process.poll() is None:
            try:
                os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            return jsonify({"ok": True, "message": "Proceso detenido"})
    return jsonify({"ok": False, "message": "No hay proceso activo"}), 404


@app.route("/api/status")
def api_status():
    with process_lock:
        running = current_process is not None and current_process.poll() is None
    return jsonify({"running": running, "run_id": run_id, "log_count": len(log_buffer)})


@app.route("/api/logs")
def api_logs_stream():
    """SSE endpoint for real-time log streaming."""
    def generate():
        last_sent = 0
        while True:
            with process_lock:
                buf = list(log_buffer)
            if len(buf) > last_sent:
                for entry in buf[last_sent:]:
                    yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
                last_sent = len(buf)
            time.sleep(0.3)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/logs/export")
def api_logs_export():
    """Export current log as a text file."""
    with process_lock:
        lines = [e["text"] for e in log_buffer]
    content = "\n".join(lines)
    return Response(content, mimetype="text/plain",
                    headers={"Content-Disposition": "attachment; filename=arca_panorama.log"})


@app.route("/api/preview/<cbtis>/<filename>")
def api_preview(cbtis, filename):
    """Serve a panorama preview image."""
    year_dir = get_year_dir()
    filepath = year_dir / cbtis / "UNIDAS" / filename
    if filepath.exists() and filepath.suffix == ".webp":
        return send_file(str(filepath), mimetype="image/webp")
    return jsonify({"error": "Not found"}), 404


# --- Organize endpoints ---

def _get_ods_path() -> Path:
    return Path(config["workspace"]) / "DATOS_2026.ods"


def _is_organized(cbtis_dir: Path) -> bool:
    """A CBTIS dir is 'organized' if it has subdirs matching TN_GP_AB_COLOR pattern."""
    if not cbtis_dir.is_dir():
        return False
    pattern = re.compile(r"^[A-Z]+_[A-Z]_[A-Z]+_[A-Z]+$")
    for item in cbtis_dir.iterdir():
        if item.is_dir() and item.name != "UNIDAS" and pattern.match(item.name):
            return True
    return False


@app.route("/api/organize/status")
def api_organize_status():
    """Return organize status for all CBTIS found in ODS."""
    ods_path = _get_ods_path()
    if not ods_path.exists():
        return jsonify({"error": "DATOS_2026.ods no encontrado"}), 404

    ods_groups = parse_ods_groups(str(ods_path))
    year_dir = get_year_dir()
    result = []

    for cbtis_num in sorted(ods_groups.keys(), key=int):
        cbtis_dir = year_dir / cbtis_num
        groups = ods_groups[cbtis_num]
        total_photos = sum(len(parse_fotografias(g["fotografias"])) for g in groups)
        organized = _is_organized(cbtis_dir)

        # Count loose photos in root (not in subdirs)
        loose_photos = 0
        if cbtis_dir.is_dir():
            for f in cbtis_dir.iterdir():
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    loose_photos += 1

        result.append({
            "cbtis": cbtis_num,
            "organized": organized,
            "groups_in_ods": len(groups),
            "photos_in_ods": total_photos,
            "loose_photos": loose_photos,
            "dir_exists": cbtis_dir.is_dir(),
        })

    return jsonify(result)


@app.route("/api/organize/preview/<cbtis>")
def api_organize_preview(cbtis):
    """Dry-run: show what dirs would be created and what photos moved."""
    ods_path = _get_ods_path()
    if not ods_path.exists():
        return jsonify({"error": "DATOS_2026.ods no encontrado"}), 404

    ods_groups = parse_ods_groups(str(ods_path))
    if cbtis not in ods_groups:
        return jsonify({"error": f"CBTIS {cbtis} no encontrado en ODS"}), 404

    year_dir = get_year_dir()
    cbtis_dir = year_dir / cbtis

    if _is_organized(cbtis_dir):
        return jsonify({"error": f"CBTIS {cbtis} ya está organizado"}), 409

    # Build file index from root of cbtis dir
    file_index = build_file_index(cbtis_dir)

    actions = []
    missing_photos = []
    total_photos = 0

    for group in ods_groups[cbtis]:
        dirname = make_group_dirname(group)
        dir_path = cbtis_dir / dirname
        dir_exists = dir_path.is_dir()
        foto_names = parse_fotografias(group["fotografias"])
        total_photos += len(foto_names)

        moves = []
        for img_base in foto_names:
            actual_file = find_photo_on_disk(file_index, img_base)
            if actual_file:
                moves.append({"from": actual_file, "to": f"{dirname}/{actual_file}"})
            else:
                missing_photos.append({"group": dirname, "expected": img_base})

        actions.append({
            "dirname": dirname,
            "dir_exists": dir_exists,
            "photos": len(foto_names),
            "moves": moves,
            "obs": group.get("obs", ""),
        })

    # Detect photos on disk not in any ODS group
    all_ods_files = set()
    for group in ods_groups[cbtis]:
        for img_base in parse_fotografias(group["fotografias"]):
            actual = find_photo_on_disk(file_index, img_base)
            if actual:
                all_ods_files.add(actual)
    extra_photos = [f for f in file_index.values() if f not in all_ods_files]

    return jsonify({
        "cbtis": cbtis,
        "actions": actions,
        "total_groups": len(actions),
        "total_photos": total_photos,
        "photos_found": total_photos - len(missing_photos),
        "missing_photos": missing_photos,
        "extra_photos": extra_photos,
    })


@app.route("/api/organize/execute/<cbtis>", methods=["POST"])
def api_organize_execute(cbtis):
    """Create dirs and move photos for a CBTIS."""
    with process_lock:
        if current_process and current_process.poll() is None:
            return jsonify({"error": "Hay un proceso de panorámica en ejecución"}), 409

    ods_path = _get_ods_path()
    if not ods_path.exists():
        return jsonify({"error": "DATOS_2026.ods no encontrado"}), 404

    ods_groups = parse_ods_groups(str(ods_path))
    if cbtis not in ods_groups:
        return jsonify({"error": f"CBTIS {cbtis} no encontrado en ODS"}), 404

    year_dir = get_year_dir()
    cbtis_dir = year_dir / cbtis

    if _is_organized(cbtis_dir):
        return jsonify({"error": f"CBTIS {cbtis} ya está organizado"}), 409

    if not cbtis_dir.is_dir():
        return jsonify({"error": f"Directorio {cbtis_dir} no existe"}), 404

    file_index = build_file_index(cbtis_dir)

    dirs_created = 0
    photos_moved = 0
    errors = []

    for group in ods_groups[cbtis]:
        dirname = make_group_dirname(group)
        dir_path = cbtis_dir / dirname

        # Create directory
        if not dir_path.exists():
            try:
                os.makedirs(str(dir_path))
                dirs_created += 1
            except OSError as e:
                errors.append(f"No se pudo crear {dirname}: {e}")
                continue

        # Move photos
        for img_base in parse_fotografias(group["fotografias"]):
            actual_file = find_photo_on_disk(file_index, img_base)
            if not actual_file:
                errors.append(f"Foto no encontrada: {img_base} (grupo {dirname})")
                continue
            src = cbtis_dir / actual_file
            dst = dir_path / actual_file
            if not src.exists():
                errors.append(f"Archivo ya movido o no existe: {actual_file}")
                continue
            try:
                shutil.move(str(src), str(dst))
                photos_moved += 1
            except OSError as e:
                errors.append(f"Error moviendo {actual_file}: {e}")

    return jsonify({
        "ok": True,
        "cbtis": cbtis,
        "dirs_created": dirs_created,
        "photos_moved": photos_moved,
        "errors": errors,
    })


# --- Cutout (background removal) endpoints ---

cutout_process = None
cutout_lock = threading.Lock()
cutout_log_buffer: deque[dict] = deque(maxlen=5000)
cutout_run_id = 0


@app.route("/api/cutout/status")
def api_cutout_status():
    """Return cutout status per CBTIS: how many panoramas, how many cutouts."""
    year_dir = get_year_dir()
    if not year_dir.is_dir():
        return jsonify([])

    results = []
    for cbtis_dir in sorted(year_dir.iterdir()):
        if not cbtis_dir.is_dir() or not cbtis_dir.name.isdigit():
            continue
        cbtis = cbtis_dir.name
        unidas_dir = cbtis_dir / "UNIDAS"
        recortadas_dir = cbtis_dir / "RECORTADAS"

        panoramas = []
        if unidas_dir.is_dir():
            panoramas = sorted(
                f.name for f in unidas_dir.iterdir()
                if f.suffix.lower() in (".webp", ".jpg", ".jpeg", ".png")
            )
        cutouts = set()
        if recortadas_dir.is_dir():
            cutouts = {
                f.name for f in recortadas_dir.iterdir()
                if f.suffix.lower() in (".webp", ".png")
            }

        items = []
        for p in panoramas:
            out_name = os.path.splitext(p)[0] + ".webp"
            items.append({
                "filename": p,
                "cutout": out_name if out_name in cutouts else "",
                "done": out_name in cutouts,
            })

        if panoramas:
            results.append({
                "cbtis": cbtis,
                "total": len(panoramas),
                "done": sum(1 for i in items if i["done"]),
                "items": items,
            })
    return jsonify(results)


@app.route("/api/cutout/run/<cbtis>", methods=["POST"])
@app.route("/api/cutout/run/<cbtis>/<filename>", methods=["POST"])
def api_cutout_run(cbtis, filename=None):
    """Start background removal for a CBTIS (or single file)."""
    global cutout_process, cutout_run_id

    with cutout_lock:
        if cutout_process and cutout_process.poll() is None:
            return jsonify({"error": "Ya hay un proceso de recorte en ejecución"}), 409

    with process_lock:
        if current_process and current_process.poll() is None:
            return jsonify({"error": "Hay un proceso de panorámica en ejecución"}), 409

    # Find python with torch
    python_path = str(APP_DIR / "gui_venv" / "bin" / "python3")
    if not Path(python_path).exists():
        python_path = "python3"

    cmd = [python_path, str(REMOVER_SCRIPT), config["workspace"], cbtis]
    if filename:
        cmd.append(filename)

    def run_cutout():
        global cutout_process, cutout_run_id
        with cutout_lock:
            cutout_run_id += 1
            rid = cutout_run_id
            cutout_log_buffer.clear()

        env = os.environ.copy()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=config["workspace"],
            text=True,
            bufsize=1,
        )
        with cutout_lock:
            cutout_process = proc

        # Read stderr (logs) in a thread
        def read_stderr():
            for line in proc.stderr:
                entry = classify_line(line)
                entry["ts"] = time.time()
                entry["run"] = rid
                with cutout_lock:
                    cutout_log_buffer.append(entry)

        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()

        # Read stdout (JSON result)
        stdout_data = proc.stdout.read()
        proc.wait()
        stderr_thread.join(timeout=5)

        # Parse result
        level = "ok" if proc.returncode == 0 else "error"
        for line in stdout_data.strip().split("\n"):
            if line.strip():
                entry = classify_line(line)
                entry["ts"] = time.time()
                entry["run"] = rid
                with cutout_lock:
                    cutout_log_buffer.append(entry)

        with cutout_lock:
            cutout_log_buffer.append({
                "text": f"=== Recorte finalizado (CBTIS {cbtis}) ===",
                "level": level, "ts": time.time(), "run": rid,
            })
            cutout_process = None

    t = threading.Thread(target=run_cutout, daemon=True)
    t.start()
    return jsonify({"ok": True, "cbtis": cbtis, "filename": filename})


@app.route("/api/cutout/status/live")
def api_cutout_live_status():
    """Return current cutout process status and recent logs."""
    with cutout_lock:
        running = cutout_process is not None and cutout_process.poll() is None
        logs = [dict(e) for e in cutout_log_buffer]
    return jsonify({"running": running, "logs": logs})


@app.route("/api/cutout/preview/<cbtis>/<filename>")
def api_cutout_preview(cbtis, filename):
    """Serve a cutout image for preview."""
    year_dir = get_year_dir()
    file_path = year_dir / cbtis / "RECORTADAS" / filename
    if not file_path.exists():
        return "Not found", 404
    return send_file(str(file_path), mimetype="image/webp")


# --- Refine endpoints ---

_align_cache = {}
_resp_cache = {}    # {key: (mtime, bytes, mimetype)}
_file_locks = {}    # per-file threading locks for concurrent access
_file_locks_lock = threading.Lock()


def _get_file_lock(path):
    """Get or create a threading lock for a file path."""
    key = str(path)
    with _file_locks_lock:
        if key not in _file_locks:
            _file_locks[key] = threading.Lock()
        return _file_locks[key]


def _cached_response(cache_key, file_path, generate_fn):
    """Return cached encoded response bytes, regenerating if file changed."""
    mtime = os.path.getmtime(str(file_path))
    cached = _resp_cache.get(cache_key)
    if cached and cached[0] == mtime:
        buf = BytesIO(cached[1])
        return send_file(buf, mimetype=cached[2])
    data_bytes, mimetype = generate_fn()
    _resp_cache[cache_key] = (mtime, data_bytes, mimetype)
    buf = BytesIO(data_bytes)
    return send_file(buf, mimetype=mimetype)


def _compute_alignment(cbtis, filename):
    """Compute cutout→panorama alignment. Uses .meta.json or template matching."""
    cache_key = f"{cbtis}/{filename}"
    if cache_key in _align_cache:
        return _align_cache[cache_key]

    year_dir = get_year_dir()
    cut_path = year_dir / cbtis / "RECORTADAS" / filename

    # Find matching panorama in UNIDAS/
    stem = os.path.splitext(filename)[0]
    unidas_dir = year_dir / cbtis / "UNIDAS"
    pano_path = None
    for ext in (".webp", ".jpg", ".jpeg", ".png"):
        p = unidas_dir / (stem + ext)
        if p.exists():
            pano_path = p
            break
    if not pano_path or not cut_path.exists():
        return None

    # Try .meta.json first
    meta_path = year_dir / cbtis / "RECORTADAS" / (stem + ".meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        bbox = meta["crop_bbox"]
        orig = meta["original_size"]
        result = {
            "crop_x": bbox[0], "crop_y": bbox[1],
            "cut_w": bbox[2] - bbox[0], "cut_h": bbox[3] - bbox[1],
            "pano_w": orig[0], "pano_h": orig[1],
            "pano_file": pano_path.name
        }
        _align_cache[cache_key] = result
        return result

    # Fallback: template matching at reduced scale
    pano = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)
    cut_rgba = cv2.imread(str(cut_path), cv2.IMREAD_UNCHANGED)
    if pano is None or cut_rgba is None:
        return None

    # Use only opaque region of cutout for matching
    if cut_rgba.shape[2] == 4:
        alpha = cut_rgba[:, :, 3]
        cut_rgb = cut_rgba[:, :, :3]
        # Mask out transparent areas
        mask = (alpha > 200).astype(np.uint8) * 255
    else:
        cut_rgb = cut_rgba
        mask = np.ones(cut_rgb.shape[:2], dtype=np.uint8) * 255

    scale = 0.15
    pano_s = cv2.resize(pano, None, fx=scale, fy=scale)
    cut_s = cv2.resize(cut_rgb, None, fx=scale, fy=scale)
    mask_s = cv2.resize(mask, None, fx=scale, fy=scale)

    res = cv2.matchTemplate(pano_s, cut_s, cv2.TM_CCOEFF_NORMED, mask=mask_s)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val < 0.5:
        return None

    cx = int(max_loc[0] / scale)
    cy = int(max_loc[1] / scale)
    ch, cw = cut_rgba.shape[:2]
    ph, pw = pano.shape[:2]

    # Sub-pixel refinement: re-match a center patch at full res in small window
    try:
        patch_sz = min(400, cw // 2, ch // 2)
        if patch_sz >= 64:
            pc_y = ch // 2 - patch_sz // 2
            pc_x = cw // 2 - patch_sz // 2
            tpl = cut_rgb[pc_y:pc_y + patch_sz, pc_x:pc_x + patch_sz]
            tpl_m = mask[pc_y:pc_y + patch_sz, pc_x:pc_x + patch_sz]
            margin = 12
            sx1 = max(0, cx + pc_x - margin)
            sy1 = max(0, cy + pc_y - margin)
            sx2 = min(pw, cx + pc_x + patch_sz + margin)
            sy2 = min(ph, cy + pc_y + patch_sz + margin)
            win = pano[sy1:sy2, sx1:sx2]
            if win.shape[0] >= tpl.shape[0] and win.shape[1] >= tpl.shape[1]:
                res2 = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED, mask=tpl_m)
                _, v2, _, loc2 = cv2.minMaxLoc(res2)
                if v2 > 0.7:
                    cx = sx1 + loc2[0] - pc_x
                    cy = sy1 + loc2[1] - pc_y
    except Exception:
        pass  # Keep coarse result on any error

    result = {
        "crop_x": cx, "crop_y": cy,
        "cut_w": cw, "cut_h": ch,
        "pano_w": pw, "pano_h": ph,
        "pano_file": pano_path.name
    }
    _align_cache[cache_key] = result
    return result


@app.route("/api/cutout/align/<cbtis>/<filename>")
def api_cutout_align(cbtis, filename):
    """Return alignment between cutout and panorama."""
    result = _compute_alignment(cbtis, filename)
    if not result:
        return jsonify({"ok": False, "error": "No se pudo alinear"}), 404
    return jsonify({"ok": True, **result})


@app.route("/api/cutout/pano-region/<cbtis>/<filename>")
def api_cutout_pano_region(cbtis, filename):
    """Serve panorama region matching cutout bounds, scaled down."""
    maxw = int(request.args.get("maxw", 3000))
    align = _compute_alignment(cbtis, filename)
    if not align:
        return "Alignment failed", 404

    year_dir = get_year_dir()
    pano_path = year_dir / cbtis / "UNIDAS" / align["pano_file"]
    if not pano_path.exists():
        return "Panorama not found", 404

    cache_key = f"pano:{cbtis}:{filename}:{maxw}"

    def generate():
        # cv2 is 2-3x faster than PIL for JPEG decode
        pano_bgr = cv2.imread(str(pano_path))
        if pano_bgr is None:
            pano_pil = Image.open(str(pano_path)).convert("RGB")
            pano_bgr = cv2.cvtColor(np.array(pano_pil), cv2.COLOR_RGB2BGR)
        ph, pw = pano_bgr.shape[:2]
        cx, cy = align["crop_x"], align["crop_y"]
        cw, ch = align["cut_w"], align["cut_h"]
        cx2 = min(cx + cw, pw)
        cy2 = min(cy + ch, ph)
        region = pano_bgr[cy:cy2, cx:cx2]
        rh, rw = region.shape[:2]
        if rw > maxw:
            ratio = maxw / rw
            region = cv2.resize(region, (maxw, int(rh * ratio)), interpolation=cv2.INTER_AREA)
        # JPEG for proxy (tiny, fast), WebP for hi-res
        if maxw <= 1500:
            ok, enc = cv2.imencode(".jpg", region, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return enc.tobytes(), "image/jpeg"
        else:
            ok, enc = cv2.imencode(".webp", region, [cv2.IMWRITE_WEBP_QUALITY, 85])
            return enc.tobytes(), "image/webp"

    return _cached_response(cache_key, pano_path, generate)


@app.route("/api/cutout/preview-scaled/<cbtis>/<filename>")
def api_cutout_preview_scaled(cbtis, filename):
    """Serve cutout scaled down for canvas editing."""
    maxw = int(request.args.get("maxw", 3000))
    year_dir = get_year_dir()
    file_path = year_dir / cbtis / "RECORTADAS" / filename
    if not file_path.exists():
        return "Not found", 404

    cache_key = f"cut:{cbtis}:{filename}:{maxw}"

    def generate():
        # Load with cv2 for speed, handle alpha channel
        raw = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            img = Image.open(str(file_path)).convert("RGBA")
            raw = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
        h, w = raw.shape[:2]
        if w > maxw:
            ratio = maxw / w
            raw = cv2.resize(raw, (maxw, int(h * ratio)), interpolation=cv2.INTER_AREA)
        # WebP with alpha; lossy for proxy, lossless for full-res
        if maxw <= 1500:
            ok, enc = cv2.imencode(".webp", raw, [cv2.IMWRITE_WEBP_QUALITY, 90])
        else:
            ok, enc = cv2.imencode(".webp", raw, [cv2.IMWRITE_WEBP_QUALITY, 101])  # 101=lossless
        return enc.tobytes(), "image/webp"

    return _cached_response(cache_key, file_path, generate)


@app.route("/api/cutout/refine/<cbtis>/<filename>", methods=["POST"])
def api_cutout_refine(cbtis, filename):
    """Apply correction strokes to cutout at full resolution."""
    data = request.get_json()
    if not data or "strokes" not in data:
        return jsonify({"ok": False, "error": "No strokes"}), 400

    strokes = data["strokes"]
    if not strokes:
        return jsonify({"ok": True, "message": "No changes"})

    year_dir = get_year_dir()
    cut_path = year_dir / cbtis / "RECORTADAS" / filename
    if not cut_path.exists():
        return jsonify({"ok": False, "error": "Cutout not found"}), 404

    align = _compute_alignment(cbtis, filename)
    if not align:
        return jsonify({"ok": False, "error": "Alignment failed"}), 500

    # Lock file to prevent concurrent read/write corruption
    flock = _get_file_lock(cut_path)
    with flock:
        return _apply_refine(cut_path, cbtis, filename, strokes, align, year_dir)


def _apply_refine(cut_path, cbtis, filename, strokes, align, year_dir):
    # Load cutout at full resolution (try cv2 as fallback if PIL fails)
    try:
        cut_img = Image.open(str(cut_path)).convert("RGBA")
        cut_np = np.array(cut_img)
    except Exception:
        # File might be corrupted from interrupted save — try cv2
        raw = cv2.imread(str(cut_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return jsonify({"ok": False, "error": "Archivo corrupto, re-procesar"}), 500
        if raw.shape[2] == 4:
            cut_np = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
        else:
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            cut_np = np.dstack([rgb, np.full(raw.shape[:2], 255, dtype=np.uint8)])
    h, w = cut_np.shape[:2]

    # Load panorama region at matching size
    pano_path = year_dir / cbtis / "UNIDAS" / align["pano_file"]
    pano_full = Image.open(str(pano_path)).convert("RGB")
    cx, cy = align["crop_x"], align["crop_y"]
    cw, ch = align["cut_w"], align["cut_h"]
    pano_region = pano_full.crop((cx, cy, min(cx + cw, pano_full.width),
                                  min(cy + ch, pano_full.height)))
    # Resize to match cutout exactly (in case of slight mismatch)
    if pano_region.size != (w, h):
        pano_region = pano_region.resize((w, h), Image.LANCZOS)
    pano_np = np.array(pano_region)

    # Render strokes into restore/erase masks
    restore_mask = np.zeros((h, w), dtype=np.uint8)
    erase_mask = np.zeros((h, w), dtype=np.uint8)

    for stroke in strokes:
        mode = stroke.get("mode", "restore")
        mask = restore_mask if mode == "restore" else erase_mask
        pts = [(int(p["x"] * w), int(p["y"] * h)) for p in stroke.get("points", [])]
        if not pts:
            continue

        if stroke.get("type") == "lasso":
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.fillPoly(mask, [pts_arr], 255, cv2.LINE_AA)
        else:  # brush
            radius = max(1, int(stroke.get("radius", 0.01) * max(w, h)))
            for pt in pts:
                cv2.circle(mask, pt, radius, 255, -1, cv2.LINE_AA)
            # Connect points with lines for smooth strokes
            if len(pts) > 1:
                for j in range(len(pts) - 1):
                    cv2.line(mask, pts[j], pts[j + 1], 255, radius * 2, cv2.LINE_AA)

    # Feather mask edges to match auto-processed edge quality.
    # Auto-process uses guided filter (~8px radius) + gaussian sigma=1.2 at full-res.
    # Keep feathering subtle to avoid diffuse/blurry edges.
    feather_sigma = max(0.8, min(w, h) / 2000.0)  # ~1.3 at 2600px, ~1.75 at 3500px
    if restore_mask.any():
        rm_f = gaussian_filter(restore_mask.astype(np.float32), sigma=feather_sigma)
        restore_mask = np.clip(rm_f, 0, 255).astype(np.uint8)
    if erase_mask.any():
        em_f = gaussian_filter(erase_mask.astype(np.float32), sigma=feather_sigma)
        erase_mask = np.clip(em_f, 0, 255).astype(np.uint8)

    # Apply restore: blend panorama pixels where mask > 0
    if restore_mask.any():
        blend = restore_mask.astype(np.float32) / 255.0
        for c in range(3):
            cut_np[:, :, c] = np.clip(
                cut_np[:, :, c] * (1 - blend) + pano_np[:, :, c] * blend,
                0, 255
            ).astype(np.uint8)
        # Alpha: blend toward opaque
        cut_np[:, :, 3] = np.clip(
            cut_np[:, :, 3].astype(np.float32) * (1 - blend) + 255.0 * blend,
            0, 255
        ).astype(np.uint8)

    # Apply erase: set alpha to 0 where mask > 0
    if erase_mask.any():
        blend = erase_mask.astype(np.float32) / 255.0
        cut_np[:, :, 3] = np.clip(
            cut_np[:, :, 3].astype(np.float32) * (1 - blend),
            0, 255
        ).astype(np.uint8)

    # Save refined cutout (atomic: write to temp, then rename)
    import tempfile
    result = Image.fromarray(cut_np, "RGBA")
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".webp", dir=str(cut_path.parent))
    try:
        with os.fdopen(tmp_fd, "wb") as tmp_f:
            result.save(tmp_f, format="WEBP", lossless=True, quality=100)
        os.replace(tmp_path, str(cut_path))
    except Exception:
        os.unlink(tmp_path)
        raise

    # Invalidate cached previews for this cutout
    for k in list(_resp_cache.keys()):
        if k.startswith(f"cut:{cbtis}:{filename}:"):
            del _resp_cache[k]

    # Save stroke history
    stem = os.path.splitext(filename)[0]
    refine_path = year_dir / cbtis / "RECORTADAS" / (stem + ".refine.json")
    history = []
    if refine_path.exists():
        with open(refine_path) as f:
            history = json.load(f).get("history", [])
    history.append({"ts": time.time(), "strokes": strokes})
    with open(refine_path, "w") as f:
        json.dump({"history": history, "align": align}, f)

    return jsonify({"ok": True, "message": f"{len(strokes)} correcciones aplicadas"})


def _write_psd(f, layers):
    """Write a PSD file with named RGBA layers and visibility flags.

    layers: list of {name: str, data: np.array(H,W,4 uint8), visible: bool}
    """
    import struct
    pk = struct.pack

    h, w = layers[0]["data"].shape[:2]
    n = len(layers)

    # --- File Header ---
    f.write(b"8BPS")
    f.write(pk(">H", 1))           # version
    f.write(b"\x00" * 6)           # reserved
    f.write(pk(">H", 4))           # channels (RGBA)
    f.write(pk(">II", h, w))       # height, width
    f.write(pk(">HH", 8, 3))      # depth=8, mode=RGB

    # --- Color Mode / Image Resources ---
    f.write(pk(">I", 0))
    f.write(pk(">I", 0))

    # --- Layer and Mask Information ---
    section_pos = f.tell()
    f.write(pk(">I", 0))           # placeholder: section length

    info_pos = f.tell()
    f.write(pk(">I", 0))           # placeholder: layer info length
    f.write(pk(">h", -n))          # negative = merged alpha is transparency

    # -- Layer records --
    for lyr in layers:
        f.write(pk(">iiii", 0, 0, h, w))   # bounds
        f.write(pk(">H", 4))               # 4 channels

        ch_size = 2 + h * w                 # compression(2) + raw pixels
        for ch_id in (-1, 0, 1, 2):         # alpha, R, G, B
            f.write(pk(">hI", ch_id, ch_size))

        f.write(b"8BIMnorm")               # blend sig + mode
        f.write(pk(">BB", 255, 0))          # opacity=255, clipping=base

        flags = 0x00 if lyr["visible"] else 0x02   # bit1: 0=visible, 1=hidden
        f.write(pk(">BB", flags, 0))        # flags + filler

        # Extra data: mask(4) + blend-ranges(4) + pascal-name
        name_b = lyr["name"].encode("latin-1", errors="replace")[:255]
        pascal_len = 1 + len(name_b)
        pascal_pad = (4 - pascal_len % 4) % 4
        extra = 4 + 4 + pascal_len + pascal_pad
        f.write(pk(">I", extra))
        f.write(pk(">I", 0))               # no layer mask
        f.write(pk(">I", 0))               # no blend ranges
        f.write(pk(">B", len(name_b)))
        f.write(name_b)
        f.write(b"\x00" * pascal_pad)

    # -- Channel image data --
    ch_order = [3, 0, 1, 2]                # alpha, R, G, B
    for lyr in layers:
        d = lyr["data"]
        for ci in ch_order:
            f.write(pk(">H", 0))           # compression=raw
            f.write(d[:, :, ci].tobytes())

    # Pad layer info to even length
    info_end = f.tell()
    info_len = info_end - info_pos - 4
    if info_len % 2:
        f.write(b"\x00")
        info_end += 1
        info_len += 1

    # Write lengths
    section_len = info_end - section_pos - 4
    f.seek(info_pos)
    f.write(pk(">I", info_len))
    f.seek(section_pos)
    f.write(pk(">I", section_len))
    f.seek(info_end)

    # --- Composite Image Data ---
    composite = layers[0]["data"]           # top layer as composite
    f.write(pk(">H", 0))                   # compression=raw
    for ci in (0, 1, 2, 3):                # R, G, B, A
        f.write(composite[:, :, ci].tobytes())


@app.route("/api/cutout/export-psd/<cbtis>/<filename>")
def api_cutout_export_psd(cbtis, filename):
    """Export cutout + panorama region as PSD with 2 named layers."""
    year_dir = get_year_dir()
    cut_path = year_dir / cbtis / "RECORTADAS" / filename
    if not cut_path.exists():
        return "Not found", 404

    align = _compute_alignment(cbtis, filename)
    if not align:
        return "Alignment failed", 500

    # Load cutout (RGBA)
    cutout = np.array(Image.open(str(cut_path)).convert("RGBA"))
    h, w = cutout.shape[:2]

    # Load panorama region at matching size
    pano_path = year_dir / cbtis / "UNIDAS" / align["pano_file"]
    pano_full = Image.open(str(pano_path)).convert("RGB")
    cx, cy = align["crop_x"], align["crop_y"]
    cw, ch = align["cut_w"], align["cut_h"]
    pano_region = pano_full.crop((cx, cy, min(cx + cw, pano_full.width),
                                  min(cy + ch, pano_full.height)))
    if pano_region.size != (w, h):
        pano_region = pano_region.resize((w, h), Image.LANCZOS)
    # Convert to RGBA with full opacity
    pano_np = np.array(pano_region)
    pano_rgba = np.dstack([pano_np, np.full((h, w), 255, dtype=np.uint8)])

    layers = [
        {"name": "Recortada", "data": cutout, "visible": True},
        {"name": "Panoramica Original", "data": pano_rgba, "visible": False},
    ]

    buf = BytesIO()
    _write_psd(buf, layers)
    buf.seek(0)

    stem = os.path.splitext(filename)[0]
    return send_file(buf, mimetype="image/vnd.adobe.photoshop", as_attachment=True,
                     download_name=f"{stem}.psd")


@app.route("/api/cutout/export-tiff/<cbtis>/<filename>")
def api_cutout_export_tiff(cbtis, filename):
    """Export cutout + panorama region as Photoshop-compatible layered TIFF.

    Uses Adobe tag #37724 (ImageSourceData) to embed PSD layer data inside
    the TIFF. This gives real layers with proper transparency in:
    - Affinity Photo
    - Adobe Photoshop
    - Krita
    - GIMP (opens as flat but preserves composite)
    """
    import tifffile
    import imagecodecs
    from psdtags import (
        PsdBlendMode, PsdChannel, PsdChannelId, PsdClippingType,
        PsdColorSpaceType, PsdCompressionType, PsdEmpty, PsdFilterMask,
        PsdFormat, PsdKey, PsdLayer, PsdLayerFlag, PsdLayerMask,
        PsdLayers, PsdRectangle, PsdString, PsdUserMask,
        TiffImageSourceData, overlay,
    )

    year_dir = get_year_dir()
    cut_path = year_dir / cbtis / "RECORTADAS" / filename
    if not cut_path.exists():
        return "Not found", 404

    align = _compute_alignment(cbtis, filename)
    if not align:
        return "Alignment failed", 500

    # --- Load cutout RGBA ---
    cut_pil = Image.open(str(cut_path))
    if cut_pil.mode != "RGBA":
        cut_pil = cut_pil.convert("RGBA")
    cut_np = np.array(cut_pil, dtype=np.uint8)  # (H, W, 4) straight alpha
    h, w = cut_np.shape[:2]

    # --- Load panorama region at matching size ---
    pano_path = year_dir / cbtis / "UNIDAS" / align["pano_file"]
    pano_full = Image.open(str(pano_path)).convert("RGB")
    cx, cy = align["crop_x"], align["crop_y"]
    cw, ch_ = align["cut_w"], align["cut_h"]
    pano_region = pano_full.crop((cx, cy, min(cx + cw, pano_full.width),
                                  min(cy + ch_, pano_full.height)))
    if pano_region.size != (w, h):
        pano_region = pano_region.resize((w, h), Image.LANCZOS)
    pano_np = np.array(pano_region, dtype=np.uint8)  # (H, W, 3)
    # Convert to RGBA with full opacity
    pano_rgba = np.dstack([pano_np, np.full((h, w), 255, dtype=np.uint8)])

    # --- Build Photoshop layer data (tag #37724) ---
    image_source_data = TiffImageSourceData(
        name='ARCA Export',
        psdformat=PsdFormat.LE32BIT,
        layers=PsdLayers(
            key=PsdKey.LAYER,
            has_transparency=True,
            layers=[
                # Bottom layer: Panorama (hidden by default)
                PsdLayer(
                    name='Panoramica',
                    rectangle=PsdRectangle(0, 0, h, w),
                    channels=[
                        PsdChannel(channelid=PsdChannelId.CHANNEL0,
                                   compression=PsdCompressionType.ZIP_PREDICTED,
                                   data=pano_np[:, :, 0]),
                        PsdChannel(channelid=PsdChannelId.CHANNEL1,
                                   compression=PsdCompressionType.ZIP_PREDICTED,
                                   data=pano_np[:, :, 1]),
                        PsdChannel(channelid=PsdChannelId.CHANNEL2,
                                   compression=PsdCompressionType.ZIP_PREDICTED,
                                   data=pano_np[:, :, 2]),
                    ],
                    mask=PsdLayerMask(),
                    opacity=255,
                    blendmode=PsdBlendMode.NORMAL,
                    blending_ranges=(),
                    clipping=PsdClippingType.BASE,
                    flags=PsdLayerFlag.PHOTOSHOP5 | PsdLayerFlag.TRANSPARENCY_PROTECTED | PsdLayerFlag.VISIBLE,
                    info=[PsdString(PsdKey.UNICODE_LAYER_NAME, 'Panoramica Original')],
                ),
                # Top layer: Cutout with transparency
                PsdLayer(
                    name='Recortada',
                    rectangle=PsdRectangle(0, 0, h, w),
                    channels=[
                        PsdChannel(channelid=PsdChannelId.TRANSPARENCY_MASK,
                                   compression=PsdCompressionType.ZIP_PREDICTED,
                                   data=cut_np[:, :, 3]),
                        PsdChannel(channelid=PsdChannelId.CHANNEL0,
                                   compression=PsdCompressionType.ZIP_PREDICTED,
                                   data=cut_np[:, :, 0]),
                        PsdChannel(channelid=PsdChannelId.CHANNEL1,
                                   compression=PsdCompressionType.ZIP_PREDICTED,
                                   data=cut_np[:, :, 1]),
                        PsdChannel(channelid=PsdChannelId.CHANNEL2,
                                   compression=PsdCompressionType.ZIP_PREDICTED,
                                   data=cut_np[:, :, 2]),
                    ],
                    mask=PsdLayerMask(),
                    opacity=255,
                    blendmode=PsdBlendMode.NORMAL,
                    blending_ranges=(),
                    clipping=PsdClippingType.BASE,
                    flags=PsdLayerFlag.PHOTOSHOP5,
                    info=[PsdString(PsdKey.UNICODE_LAYER_NAME, 'Recortada')],
                ),
            ],
        ),
        usermask=PsdUserMask(
            colorspace=PsdColorSpaceType.RGB,
            components=(65535, 0, 0, 0), opacity=50,
        ),
        info=[PsdEmpty(PsdKey.PATTERNS)],
    )

    # Flattened composite for non-layer-aware readers
    composite = overlay(
        (pano_rgba, (0, 0)),
        (cut_np, (0, 0)),
        shape=(h, w),
    )

    # --- Write layered TIFF to memory buffer ---
    buf = BytesIO()
    tifffile.imwrite(
        buf,
        composite,
        photometric='rgb',
        compression='adobe_deflate',
        resolution=((720000, 10000), (720000, 10000)),
        resolutionunit='inch',
        metadata=None,
        extratags=[
            image_source_data.tifftag(maxworkers=2),
            (34675, 7, None, imagecodecs.cms_profile('srgb'), True),
        ],
    )
    buf.seek(0)
    return send_file(buf, mimetype="image/tiff", as_attachment=True,
                     download_name=f"{os.path.splitext(filename)[0]}_capas.tif")


# --- Borlas (graduation tassels) endpoints ---

BORLAS_DIR = APP_DIR / "BORLAS"


@app.route("/api/borlas/list")
def api_borlas_list():
    """List available borla colors and suggest one based on group name."""
    colors = []
    if BORLAS_DIR.is_dir():
        for f in sorted(BORLAS_DIR.iterdir()):
            if f.suffix.lower() == ".webp" and f.stem.startswith("B_"):
                colors.append(f.stem[2:])  # "B_DORADO.webp" -> "DORADO"
    suggested = request.args.get("group", "")
    # Extract color from group name: last _-separated segment of the filename stem
    suggested_color = ""
    if suggested:
        parts = suggested.replace(".webp", "").split("_")
        for p in reversed(parts):
            if p.upper() in colors:
                suggested_color = p.upper()
                break
    return jsonify({"colors": colors, "suggested": suggested_color})


def _borlas_state_path(cbtis, filename):
    """Path to borla state JSON file."""
    stem = os.path.splitext(filename)[0]
    return get_year_dir() / cbtis / "RECORTADAS" / f".{stem}.borlas.json"


@app.route("/api/borlas/state/<cbtis>/<filename>")
def api_borlas_state_get(cbtis, filename):
    """Load saved borla state."""
    p = _borlas_state_path(cbtis, filename)
    if p.exists():
        with open(p) as f:
            return jsonify(json.load(f))
    return jsonify(None)


@app.route("/api/borlas/state/<cbtis>/<filename>", methods=["POST"])
def api_borlas_state_save(cbtis, filename):
    """Save borla state (positions, rotations, visibility, scale)."""
    data = request.get_json(force=True)
    p = _borlas_state_path(cbtis, filename)
    with open(p, "w") as f:
        json.dump(data, f)
    return jsonify({"ok": True})


@app.route("/api/borlas/detect-faces/<cbtis>/<filename>")
def api_borlas_detect_faces(cbtis, filename):
    """Detect faces in cutout image using YuNet + Haar cascade (merged)."""
    import cv2

    year_dir = get_year_dir()
    cut_path = year_dir / cbtis / "RECORTADAS" / filename
    if not cut_path.exists():
        return jsonify({"ok": False, "error": "Archivo no encontrado"}), 404

    img = cv2.imread(str(cut_path))
    if img is None:
        return jsonify({"ok": False, "error": "No se pudo leer la imagen"}), 500
    h, w = img.shape[:2]

    # Downscale for faster detection — 1500px width is plenty for face detection
    MAX_DET_W = 1500
    if w > MAX_DET_W:
        det_scale = MAX_DET_W / w
        det_img = cv2.resize(img, (MAX_DET_W, int(h * det_scale)), interpolation=cv2.INTER_AREA)
    else:
        det_scale = 1.0
        det_img = img
    dh, dw = det_img.shape[:2]

    yunet_faces = []
    haar_faces = []

    # YuNet detection (on downscaled image)
    model_path = str(APP_DIR / "models" / "face_detection_yunet_2023mar.onnx")
    try:
        detector = cv2.FaceDetectorYN.create(model_path, "", (dw, dh),
                                             score_threshold=0.55,
                                             nms_threshold=0.3,
                                             top_k=80)
        _, detections = detector.detect(det_img)
        if detections is not None:
            for det in detections:
                fx, fy, fw, fh = det[0], det[1], det[2], det[3]
                yunet_faces.append({
                    "x": float(fx / dw), "y": float(fy / dh),
                    "w": float(fw / dw), "h": float(fh / dh),
                    "conf": round(float(det[-1]), 3)
                })
    except Exception:
        pass

    # Always run Haar cascade too for better coverage (on downscaled image)
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(det_img, cv2.COLOR_BGR2GRAY)
        min_face = max(20, dw // 30)
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                         minNeighbors=4,
                                         minSize=(min_face, min_face))
        for (fx, fy, fw, fh) in rects:
            haar_faces.append({
                "x": float(fx / dw), "y": float(fy / dh),
                "w": float(fw / dw), "h": float(fh / dh),
                "conf": 0.7
            })
    except Exception:
        pass

    # Merge: start with YuNet, add Haar faces that don't overlap existing ones
    faces = list(yunet_faces)

    def _iou(a, b):
        """Intersection-over-union of two face rects (normalized coords)."""
        ax1, ay1 = a["x"], a["y"]
        ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
        bx1, by1 = b["x"], b["y"]
        bx2, by2 = bx1 + b["w"], by1 + b["h"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = a["w"] * a["h"] + b["w"] * b["h"] - inter
        return inter / union if union > 0 else 0

    for hf in haar_faces:
        # Add Haar face only if it doesn't overlap any existing face (IoU < 0.3)
        if all(_iou(hf, ef) < 0.3 for ef in faces):
            faces.append(hf)

    # Sort left to right
    faces.sort(key=lambda f: f["x"])
    return jsonify({"ok": True, "faces": faces, "img_w": w, "img_h": h})


@app.route("/api/borlas/image/<color>")
def api_borlas_image(color):
    """Serve a borla image, optionally scaled."""
    path = BORLAS_DIR / f"B_{color.upper()}.webp"
    if not path.exists():
        return "Not found", 404
    maxw = request.args.get("maxw", type=int)
    if maxw and maxw < 4000:
        img = Image.open(str(path))
        if img.width > maxw:
            ratio = maxw / img.width
            img = img.resize((maxw, int(img.height * ratio)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, "WEBP", quality=90)
        buf.seek(0)
        return send_file(buf, mimetype="image/webp")
    return send_file(str(path), mimetype="image/webp")


@app.route("/api/borlas/export-tiff/<cbtis>/<filename>", methods=["POST"])
def api_borlas_export_tiff(cbtis, filename):
    """Export cutout + borlas as Photoshop-compatible layered TIFF with BORLAS group."""
    import tifffile
    import imagecodecs
    from psdtags import (
        PsdBlendMode, PsdChannel, PsdChannelId, PsdClippingType,
        PsdColorSpaceType, PsdCompressionType, PsdEmpty, PsdFilterMask,
        PsdFormat, PsdKey, PsdLayer, PsdLayerFlag, PsdLayerMask,
        PsdLayers, PsdRectangle, PsdSectionDividerSetting,
        PsdSectionDividerType, PsdString, PsdUserMask,
        TiffImageSourceData, overlay,
    )

    data = request.get_json(force=True)
    borlas_data = data.get("borlas", [])

    year_dir = get_year_dir()
    cut_path = year_dir / cbtis / "RECORTADAS" / filename
    if not cut_path.exists():
        return jsonify({"error": "Not found"}), 404

    align = _compute_alignment(cbtis, filename)
    if not align:
        return jsonify({"error": "Alignment failed"}), 500

    # Load cutout RGBA
    cut_pil = Image.open(str(cut_path))
    if cut_pil.mode != "RGBA":
        cut_pil = cut_pil.convert("RGBA")
    cut_np = np.array(cut_pil, dtype=np.uint8)
    h, w = cut_np.shape[:2]

    # Load panorama region
    pano_path = year_dir / cbtis / "UNIDAS" / align["pano_file"]
    pano_full = Image.open(str(pano_path)).convert("RGB")
    cx, cy = align["crop_x"], align["crop_y"]
    cw_, ch_ = align["cut_w"], align["cut_h"]
    pano_region = pano_full.crop((cx, cy, min(cx + cw_, pano_full.width),
                                  min(cy + ch_, pano_full.height)))
    if pano_region.size != (w, h):
        pano_region = pano_region.resize((w, h), Image.LANCZOS)
    pano_np = np.array(pano_region, dtype=np.uint8)
    pano_rgba = np.dstack([pano_np, np.full((h, w), 255, dtype=np.uint8)])

    # --- Render borla layers ---
    borla_layers_psd = []
    borla_layers_rgba = []  # (rgba_array, (x_offset, y_offset)) for composite

    for i, bd in enumerate(borlas_data):
        if not bd.get("visible", True):
            continue
        color = bd.get("color", "DORADO")
        bpath = BORLAS_DIR / f"B_{color}.webp"
        if not bpath.exists():
            continue

        borla_img = Image.open(str(bpath)).convert("RGBA")
        # Scale borla: bd["scale"] is the target width in pixels (at full image resolution)
        target_w = max(10, int(bd.get("scale", 50)))
        ratio = target_w / borla_img.width
        target_h = int(borla_img.height * ratio)
        borla_img = borla_img.resize((target_w, target_h), Image.LANCZOS)

        # Apply per-borla mask if present (brush erase/restore)
        mask_data = bd.get("mask")
        if mask_data and mask_data.startswith("data:image/"):
            import base64
            b64 = mask_data.split(",", 1)[1]
            mask_bytes = base64.b64decode(b64)
            mask_pil = Image.open(BytesIO(mask_bytes)).convert("L")
            mask_pil = mask_pil.resize((target_w, target_h), Image.LANCZOS)
            # Multiply borla alpha with mask
            borla_arr = np.array(borla_img)
            mask_arr = np.array(mask_pil, dtype=np.float32) / 255.0
            borla_arr[:, :, 3] = (borla_arr[:, :, 3].astype(np.float32) * mask_arr).astype(np.uint8)
            borla_img = Image.fromarray(borla_arr)

        # Rotate around top-center pivot
        rotation = bd.get("rotation", 0)
        if abs(rotation) > 0.5:
            # PIL rotate: expand=True to fit rotated image, center is the pivot
            # We need pivot at (w/2, 0) — top center
            # Translate so pivot is at center, rotate, translate back
            bw, bh = borla_img.size
            # Create larger canvas to avoid clipping
            pad = max(bw, bh)
            padded = Image.new("RGBA", (bw + pad * 2, bh + pad * 2), (0, 0, 0, 0))
            padded.paste(borla_img, (pad, pad), borla_img)
            # Rotate around the top-center of the original image position
            pivot_x = pad + bw // 2
            pivot_y = pad
            rotated = padded.rotate(-rotation, center=(pivot_x, pivot_y),
                                    expand=False, resample=Image.BICUBIC)
            # Crop back to tight bounding box
            bbox = rotated.getbbox()
            if bbox:
                rotated = rotated.crop(bbox)
                # Adjust position for the crop offset
                dx = bbox[0] - pad
                dy = bbox[1] - pad
            else:
                dx, dy = 0, 0
                rotated = borla_img
            borla_img = rotated
        else:
            dx, dy = 0, 0

        # Position: bd["x"], bd["y"] are pixel coords of borla top-center in full-res image
        bx = int(bd.get("x", 0)) - borla_img.width // 2 + dx
        by = int(bd.get("y", 0)) + dy

        # Clip to image bounds and create layer rectangle
        # Determine the visible region within the full canvas
        lx1 = max(0, bx)
        ly1 = max(0, by)
        lx2 = min(w, bx + borla_img.width)
        ly2 = min(h, by + borla_img.height)
        if lx2 <= lx1 or ly2 <= ly1:
            continue

        # Crop borla to visible region
        crop_x1 = lx1 - bx
        crop_y1 = ly1 - by
        crop_x2 = crop_x1 + (lx2 - lx1)
        crop_y2 = crop_y1 + (ly2 - ly1)
        borla_cropped = np.array(borla_img, dtype=np.uint8)[crop_y1:crop_y2, crop_x1:crop_x2]
        lh, lw = borla_cropped.shape[:2]

        borla_layers_rgba.append((borla_cropped, (ly1, lx1)))  # overlay() expects (y, x)

        # --- Render hilo (thread) for this borla ---
        hilo_pts = bd.get("hilo")
        if hilo_pts and len(hilo_pts) >= 2:
            import cv2 as cv
            hilo_color_str = data.get("hiloColor", "#d4a017")
            hilo_size = data.get("hiloSize", 3)
            hilo_feather = data.get("hiloFeather", 1)
            hilo_flow = data.get("hiloFlow", 100) / 100.0
            # Parse hex/rgb color
            hc = hilo_color_str.lstrip('#')
            if hc.startswith('rgb'):
                import re
                rgb_vals = re.findall(r'\d+', hc)
                hr, hg, hb = int(rgb_vals[0]), int(rgb_vals[1]), int(rgb_vals[2])
            else:
                hr = int(hc[0:2], 16) if len(hc) >= 6 else 212
                hg = int(hc[2:4], 16) if len(hc) >= 6 else 160
                hb = int(hc[4:6], 16) if len(hc) >= 6 else 23
            # Hilo points are in borla-local coords (relative to pivot b.x, b.y)
            # Scale = bd["scale"] is borla width in pixels
            borla_orig = Image.open(str(bpath)).convert("RGBA")
            borla_ar = borla_orig.height / borla_orig.width
            hilo_scale = target_w  # same scale as borla width
            rotation_rad = bd.get("rotation", 0) * np.pi / 180
            cos_r, sin_r = np.cos(rotation_rad), np.sin(rotation_rad)
            pivot_x_img = int(bd.get("x", 0))
            pivot_y_img = int(bd.get("y", 0))
            # Convert local coords → image coords (rotate around pivot)
            img_pts = []
            for pt in hilo_pts:
                lx_ = pt["x"] * (hilo_scale / target_w) if target_w > 0 else pt["x"]
                ly_ = pt["y"] * (hilo_scale / target_w) if target_w > 0 else pt["y"]
                # Rotate back to image space
                rx = lx_ * cos_r - ly_ * sin_r
                ry = lx_ * sin_r + ly_ * cos_r
                img_pts.append((int(pivot_x_img + rx), int(pivot_y_img + ry)))
            if len(img_pts) >= 2:
                # Compute bounding box
                xs = [p[0] for p in img_pts]
                ys = [p[1] for p in img_pts]
                line_w = max(1, int(hilo_size * (hilo_scale / 50))) + int(hilo_feather * 2) + 4
                hx1 = max(0, min(xs) - line_w)
                hy1 = max(0, min(ys) - line_w)
                hx2 = min(w, max(xs) + line_w)
                hy2 = min(h, max(ys) + line_w)
                hh_ = hy2 - hy1
                hw_ = hx2 - hx1
                if hw_ > 0 and hh_ > 0:
                    hilo_canvas = np.zeros((hh_, hw_, 4), dtype=np.uint8)
                    local_pts = np.array([(p[0] - hx1, p[1] - hy1) for p in img_pts], dtype=np.int32)
                    thickness = max(1, int(hilo_size * (hilo_scale / 50)))
                    cv.polylines(hilo_canvas[:, :, :3], [local_pts], False,
                                 (hr, hg, hb), thickness, cv.LINE_AA)
                    # Set alpha where color was drawn
                    gray = cv.cvtColor(hilo_canvas[:, :, :3], cv.COLOR_RGB2GRAY)
                    hilo_canvas[:, :, 3] = np.clip(gray * 3, 0, 255).astype(np.uint8)
                    # Apply feather
                    if hilo_feather > 0.1:
                        from scipy.ndimage import gaussian_filter
                        alpha_f = gaussian_filter(hilo_canvas[:, :, 3].astype(np.float32),
                                                  sigma=hilo_feather)
                        hilo_canvas[:, :, 3] = np.clip(alpha_f, 0, 255).astype(np.uint8)
                    # Apply flow (opacity)
                    hilo_canvas[:, :, 3] = (hilo_canvas[:, :, 3].astype(np.float32) * hilo_flow).astype(np.uint8)
                    # Apply hilo mask if present
                    hilo_mask_data = bd.get("hiloMask")
                    if hilo_mask_data and hilo_mask_data.startswith("data:image/"):
                        import base64 as b64m
                        b64str = hilo_mask_data.split(",", 1)[1]
                        mask_bytes = b64m.b64decode(b64str)
                        mask_pil = Image.open(BytesIO(mask_bytes)).convert("L")
                        mask_pil = mask_pil.resize((hw_, hh_), Image.LANCZOS)
                        mask_arr = np.array(mask_pil, dtype=np.float32) / 255.0
                        hilo_canvas[:, :, 3] = (hilo_canvas[:, :, 3].astype(np.float32) * mask_arr).astype(np.uint8)
                    borla_layers_rgba.append((hilo_canvas, (hy1, hx1)))
                    borla_layers_psd.append(PsdLayer(
                        name=f'Hilo_{i+1}',
                        rectangle=PsdRectangle(hy1, hx1, hy1 + hh_, hx1 + hw_),
                        channels=[
                            PsdChannel(channelid=PsdChannelId.TRANSPARENCY_MASK,
                                       compression=PsdCompressionType.ZIP_PREDICTED,
                                       data=hilo_canvas[:, :, 3]),
                            PsdChannel(channelid=PsdChannelId.CHANNEL0,
                                       compression=PsdCompressionType.ZIP_PREDICTED,
                                       data=hilo_canvas[:, :, 0]),
                            PsdChannel(channelid=PsdChannelId.CHANNEL1,
                                       compression=PsdCompressionType.ZIP_PREDICTED,
                                       data=hilo_canvas[:, :, 1]),
                            PsdChannel(channelid=PsdChannelId.CHANNEL2,
                                       compression=PsdCompressionType.ZIP_PREDICTED,
                                       data=hilo_canvas[:, :, 2]),
                        ],
                        mask=PsdLayerMask(), opacity=255,
                        blendmode=PsdBlendMode.NORMAL, blending_ranges=(),
                        clipping=PsdClippingType.BASE,
                        flags=PsdLayerFlag.PHOTOSHOP5,
                        info=[PsdString(PsdKey.UNICODE_LAYER_NAME, f'Hilo {i+1}')],
                    ))

        # PSD layer with tight rectangle (not full canvas)
        borla_layers_psd.append(PsdLayer(
            name=f'Borla_{i+1}',
            rectangle=PsdRectangle(ly1, lx1, ly1 + lh, lx1 + lw),
            channels=[
                PsdChannel(channelid=PsdChannelId.TRANSPARENCY_MASK,
                           compression=PsdCompressionType.ZIP_PREDICTED,
                           data=borla_cropped[:, :, 3]),
                PsdChannel(channelid=PsdChannelId.CHANNEL0,
                           compression=PsdCompressionType.ZIP_PREDICTED,
                           data=borla_cropped[:, :, 0]),
                PsdChannel(channelid=PsdChannelId.CHANNEL1,
                           compression=PsdCompressionType.ZIP_PREDICTED,
                           data=borla_cropped[:, :, 1]),
                PsdChannel(channelid=PsdChannelId.CHANNEL2,
                           compression=PsdCompressionType.ZIP_PREDICTED,
                           data=borla_cropped[:, :, 2]),
            ],
            mask=PsdLayerMask(),
            opacity=255,
            blendmode=PsdBlendMode.NORMAL,
            blending_ranges=(),
            clipping=PsdClippingType.BASE,
            flags=PsdLayerFlag.PHOTOSHOP5,
            info=[PsdString(PsdKey.UNICODE_LAYER_NAME, f'Borla {i+1}')],
        ))

    # Build layer list: pano (hidden) -> cutout -> group-end -> borla layers -> group-start
    # PSD layer order is bottom-to-top in the layers list
    all_layers = [
        # Bottom: Panorama (hidden)
        PsdLayer(
            name='Panoramica',
            rectangle=PsdRectangle(0, 0, h, w),
            channels=[
                PsdChannel(channelid=PsdChannelId.CHANNEL0,
                           compression=PsdCompressionType.ZIP_PREDICTED,
                           data=pano_np[:, :, 0]),
                PsdChannel(channelid=PsdChannelId.CHANNEL1,
                           compression=PsdCompressionType.ZIP_PREDICTED,
                           data=pano_np[:, :, 1]),
                PsdChannel(channelid=PsdChannelId.CHANNEL2,
                           compression=PsdCompressionType.ZIP_PREDICTED,
                           data=pano_np[:, :, 2]),
            ],
            mask=PsdLayerMask(), opacity=255,
            blendmode=PsdBlendMode.NORMAL, blending_ranges=(),
            clipping=PsdClippingType.BASE,
            flags=PsdLayerFlag.PHOTOSHOP5 | PsdLayerFlag.TRANSPARENCY_PROTECTED | PsdLayerFlag.VISIBLE,
            info=[PsdString(PsdKey.UNICODE_LAYER_NAME, 'Panoramica Original')],
        ),
        # Cutout
        PsdLayer(
            name='Recortada',
            rectangle=PsdRectangle(0, 0, h, w),
            channels=[
                PsdChannel(channelid=PsdChannelId.TRANSPARENCY_MASK,
                           compression=PsdCompressionType.ZIP_PREDICTED,
                           data=cut_np[:, :, 3]),
                PsdChannel(channelid=PsdChannelId.CHANNEL0,
                           compression=PsdCompressionType.ZIP_PREDICTED,
                           data=cut_np[:, :, 0]),
                PsdChannel(channelid=PsdChannelId.CHANNEL1,
                           compression=PsdCompressionType.ZIP_PREDICTED,
                           data=cut_np[:, :, 1]),
                PsdChannel(channelid=PsdChannelId.CHANNEL2,
                           compression=PsdCompressionType.ZIP_PREDICTED,
                           data=cut_np[:, :, 2]),
            ],
            mask=PsdLayerMask(), opacity=255,
            blendmode=PsdBlendMode.NORMAL, blending_ranges=(),
            clipping=PsdClippingType.BASE,
            flags=PsdLayerFlag.PHOTOSHOP5,
            info=[PsdString(PsdKey.UNICODE_LAYER_NAME, 'Recortada')],
        ),
    ]

    # Group folder: bounding divider (bottom marker) -> borla layers -> open folder (top marker)
    if borla_layers_psd:
        # Bounding section divider (group END marker, comes first in list = bottom)
        all_layers.append(PsdLayer(
            name='</BORLAS>',
            rectangle=PsdRectangle(0, 0, 0, 0),
            channels=[], mask=PsdLayerMask(), opacity=255,
            blendmode=PsdBlendMode.NORMAL, blending_ranges=(),
            clipping=PsdClippingType.BASE,
            flags=PsdLayerFlag.PHOTOSHOP5,
            info=[PsdSectionDividerSetting(
                kind=PsdSectionDividerType.BOUNDING_SECTION_DIVIDER)],
        ))
        # Individual borla layers
        all_layers.extend(borla_layers_psd)
        # Open folder (group START marker, comes last in list = top)
        all_layers.append(PsdLayer(
            name='BORLAS',
            rectangle=PsdRectangle(0, 0, 0, 0),
            channels=[], mask=PsdLayerMask(), opacity=255,
            blendmode=PsdBlendMode.NORMAL, blending_ranges=(),
            clipping=PsdClippingType.BASE,
            flags=PsdLayerFlag.PHOTOSHOP5,
            info=[
                PsdSectionDividerSetting(
                    kind=PsdSectionDividerType.OPEN_FOLDER,
                    blendmode=PsdBlendMode.PASS_THROUGH),
                PsdString(PsdKey.UNICODE_LAYER_NAME, 'BORLAS'),
            ],
        ))

    image_source_data = TiffImageSourceData(
        name='ARCA Export',
        psdformat=PsdFormat.LE32BIT,
        layers=PsdLayers(
            key=PsdKey.LAYER,
            has_transparency=True,
            layers=all_layers,
        ),
        usermask=PsdUserMask(
            colorspace=PsdColorSpaceType.RGB,
            components=(65535, 0, 0, 0), opacity=50,
        ),
        info=[PsdEmpty(PsdKey.PATTERNS)],
    )

    # Composite: pano + cutout + borlas
    comp_args = [(pano_rgba, (0, 0)), (cut_np, (0, 0))]
    for (barr, (bx, by)) in borla_layers_rgba:
        comp_args.append((barr, (bx, by)))
    composite = overlay(*comp_args, shape=(h, w))

    buf = BytesIO()
    tifffile.imwrite(
        buf,
        composite,
        photometric='rgb',
        compression='adobe_deflate',
        resolution=((720000, 10000), (720000, 10000)),
        resolutionunit='inch',
        metadata=None,
        extratags=[
            image_source_data.tifftag(maxworkers=1),
            (34675, 7, None, imagecodecs.cms_profile('srgb'), True),
        ],
    )
    buf.seek(0)
    return send_file(buf, mimetype="image/tiff", as_attachment=True,
                     download_name=f"{os.path.splitext(filename)[0]}_borlas.tif")


# ─── TOGAS (caída de togas) ───────────────────────────────────────────────────
CAIDA_DIR = APP_DIR / "CAIDA_TOGAS"


@app.route("/api/togas/list")
def api_togas_list():
    """List available toga-fall variants."""
    variants = []
    if CAIDA_DIR.is_dir():
        for f in sorted(CAIDA_DIR.iterdir()):
            if f.suffix.lower() == ".webp" and f.stem.startswith("CAIDA_"):
                variants.append(f.stem)  # "CAIDA_1", "CAIDA_2", etc.
    return jsonify({"variants": variants})


@app.route("/api/togas/image/<variant>")
def api_togas_image(variant):
    """Serve a toga-fall image, optionally scaled."""
    path = CAIDA_DIR / f"{variant}.webp"
    if not path.exists():
        return "Not found", 404
    maxw = request.args.get("maxw", type=int)
    if maxw and maxw < 4000:
        img = Image.open(str(path))
        if img.width > maxw:
            ratio = maxw / img.width
            img = img.resize((maxw, int(img.height * ratio)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, "WEBP", lossless=True)
        buf.seek(0)
        return send_file(buf, mimetype="image/webp")
    return send_file(str(path), mimetype="image/webp")


@app.route("/api/togas/detect-seated/<cbtis>/<filename>")
def api_togas_detect_seated(cbtis, filename):
    """Detect faces and identify front-row (seated) people."""
    import cv2

    year_dir = get_year_dir()
    cut_path = year_dir / cbtis / "RECORTADAS" / filename
    if not cut_path.exists():
        return jsonify({"ok": False, "error": "Not found"}), 404

    img = cv2.imread(str(cut_path))
    if img is None:
        return jsonify({"ok": False, "error": "Cannot read image"}), 500
    h, w = img.shape[:2]

    # Downscale for faster detection
    MAX_DET_W = 1500
    if w > MAX_DET_W:
        det_scale = MAX_DET_W / w
        det_img = cv2.resize(img, (MAX_DET_W, int(h * det_scale)), interpolation=cv2.INTER_AREA)
    else:
        det_scale = 1.0
        det_img = img
    dh, dw = det_img.shape[:2]

    faces = []

    # YuNet detection (on downscaled image)
    model_path = str(APP_DIR / "models" / "face_detection_yunet_2023mar.onnx")
    try:
        detector = cv2.FaceDetectorYN.create(model_path, "", (dw, dh),
                                             score_threshold=0.60,
                                             nms_threshold=0.3, top_k=80)
        _, detections = detector.detect(det_img)
        if detections is not None:
            for det in detections:
                fx, fy, fw, fh = det[0], det[1], det[2], det[3]
                faces.append({
                    "x": float(fx / dw), "y": float(fy / dh),
                    "w": float(fw / dw), "h": float(fh / dh),
                    "conf": round(float(det[-1]), 3)
                })
    except Exception:
        pass

    # Haar fallback (on downscaled image)
    if len(faces) < 2:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(det_img, cv2.COLOR_BGR2GRAY)
            min_face = max(20, dw // 30)
            rects = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                             minNeighbors=4,
                                             minSize=(min_face, min_face))
            if len(rects) > len(faces):
                faces = []
                for (fx, fy, fw, fh) in rects:
                    faces.append({
                        "x": float(fx / dw), "y": float(fy / dh),
                        "w": float(fw / dw), "h": float(fh / dh),
                        "conf": 0.8
                    })
        except Exception:
            pass

    faces.sort(key=lambda f: f["x"])

    # --- Identify front row (seated) ---
    # Heuristic: sort by Y center, find ALL significant gaps, take faces
    # below the LAST gap (= bottom-most cluster = front/seated row)
    seated_indices = list(range(len(faces)))  # default: all
    if len(faces) >= 3:
        by_y = sorted(range(len(faces)), key=lambda i: faces[i]["y"] + faces[i]["h"] / 2)
        centers_y = [faces[i]["y"] + faces[i]["h"] / 2 for i in by_y]
        avg_face_h = sum(f["h"] for f in faces) / len(faces)
        threshold = avg_face_h * 0.5
        # Find the LAST significant gap (not the largest)
        last_gap_idx = -1
        for j in range(1, len(centers_y)):
            gap = centers_y[j] - centers_y[j - 1]
            if gap > threshold:
                last_gap_idx = j
        if last_gap_idx > 0:
            front_set = set(by_y[last_gap_idx:])
            seated_indices = [i for i in range(len(faces)) if i in front_set]

    return jsonify({
        "ok": True,
        "faces": faces,
        "seated_indices": seated_indices,
        "img_w": w, "img_h": h,
    })


def _togas_state_path(cbtis, filename):
    stem = os.path.splitext(filename)[0]
    return get_year_dir() / cbtis / "RECORTADAS" / f".{stem}.togas.json"


@app.route("/api/togas/state/<cbtis>/<filename>")
def api_togas_state_get(cbtis, filename):
    p = _togas_state_path(cbtis, filename)
    if p.exists():
        with open(p) as f:
            return jsonify(json.load(f))
    return jsonify(None)


@app.route("/api/togas/state/<cbtis>/<filename>", methods=["POST"])
def api_togas_state_set(cbtis, filename):
    p = _togas_state_path(cbtis, filename)
    data = request.get_json(force=True)
    with open(p, "w") as f:
        json.dump(data, f)
    return jsonify({"ok": True})


@app.route("/api/togas/export-tiff/<cbtis>/<filename>", methods=["POST"])
def api_togas_export_tiff(cbtis, filename):
    """Export cutout + borlas group + togas group as Photoshop-compatible layered TIFF."""
    import tifffile
    import imagecodecs
    from psdtags import (
        PsdBlendMode, PsdChannel, PsdChannelId, PsdClippingType,
        PsdColorSpaceType, PsdCompressionType, PsdEmpty, PsdFilterMask,
        PsdFormat, PsdKey, PsdLayer, PsdLayerFlag, PsdLayerMask,
        PsdLayers, PsdRectangle, PsdSectionDividerSetting,
        PsdSectionDividerType, PsdString, PsdUserMask,
        TiffImageSourceData, overlay,
    )
    import base64

    data = request.get_json(force=True)
    borlas_data = data.get("borlas", [])
    togas_data = data.get("togas", [])
    toga_group_tf = data.get("togaGroupTransform", {})
    img_tf = data.get("imgTransform", {})
    img_rot = img_tf.get("rotation", 0)
    img_tx = img_tf.get("x", 0)
    img_ty = img_tf.get("y", 0)

    year_dir = get_year_dir()
    cut_path = year_dir / cbtis / "RECORTADAS" / filename
    if not cut_path.exists():
        return jsonify({"error": "Not found"}), 404

    align = _compute_alignment(cbtis, filename)
    if not align:
        return jsonify({"error": "Alignment failed"}), 500

    # Load cutout RGBA
    cut_pil = Image.open(str(cut_path))
    if cut_pil.mode != "RGBA":
        cut_pil = cut_pil.convert("RGBA")
    orig_w, orig_h = cut_pil.size
    img_cx, img_cy = orig_w / 2.0, orig_h / 2.0

    # Load panorama region
    pano_path = year_dir / cbtis / "UNIDAS" / align["pano_file"]
    pano_full = Image.open(str(pano_path)).convert("RGB")
    cx, cy = align["crop_x"], align["crop_y"]
    cw_, ch_ = align["cut_w"], align["cut_h"]
    pano_region = pano_full.crop((cx, cy, min(cx + cw_, pano_full.width),
                                  min(cy + ch_, pano_full.height)))
    if pano_region.size != (orig_w, orig_h):
        pano_region = pano_region.resize((orig_w, orig_h), Image.LANCZOS)

    # --- Apply imgTransform: rotate cutout+panorama, adjust element positions ---
    import math as _m
    exp_ox, exp_oy = 0.0, 0.0  # expansion offset from rotation
    irad = _m.radians(img_rot)
    cos_r, sin_r = _m.cos(irad), _m.sin(irad)

    if abs(img_rot) > 0.1:
        # Rotate cutout with expand=True (PIL rotate is CCW, negate for CW)
        cut_pil = cut_pil.rotate(-img_rot, expand=True, resample=Image.BICUBIC)
        # Rotate panorama (RGBA so transparent corners)
        pano_rgba_pil = pano_region.convert("RGBA")
        pano_rgba_pil = pano_rgba_pil.rotate(-img_rot, expand=True,
                                              resample=Image.BICUBIC)
        new_w, new_h = cut_pil.size
        exp_ox = (new_w - orig_w) / 2.0
        exp_oy = (new_h - orig_h) / 2.0
        # Build numpy arrays from rotated images
        cut_np = np.array(cut_pil, dtype=np.uint8)
        pano_np_raw = np.array(pano_rgba_pil, dtype=np.uint8)
        pano_np = pano_np_raw[:, :, :3]
        pano_rgba = pano_np_raw
        h, w = cut_np.shape[:2]
    else:
        cut_np = np.array(cut_pil, dtype=np.uint8)
        pano_np = np.array(pano_region, dtype=np.uint8)
        h, w = cut_np.shape[:2]
        pano_rgba = np.dstack([pano_np, np.full((h, w), 255, dtype=np.uint8)])

    # --- Transform borla positions: rotate around original center + expansion offset ---
    # Borlas are in cutout space. The cutout is rotated on server, so borla anchors
    # must undergo the same rotation to stay aligned with the cutout.
    for bd in borlas_data:
        bx, by = bd.get("x", 0), bd.get("y", 0)
        if abs(img_rot) > 0.1:
            dx, dy = bx - img_cx, by - img_cy
            bd["x"] = img_cx + dx * cos_r - dy * sin_r + exp_ox
            bd["y"] = img_cy + dx * sin_r + dy * cos_r + exp_oy
            bd["rotation"] = bd.get("rotation", 0) + img_rot
        else:
            bd["x"] = bx + exp_ox
            bd["y"] = by + exp_oy

    # --- Transform toga positions: inverse imgTransform translation + expansion ---
    # On screen, togas are in base coords while cutout is shifted by imgTf.
    # In TIFF, cutout is at origin (rotated). Toga position formula:
    #   tiff_x = toga_x - cos(r)*tx + sin(r)*ty + exp_ox
    #   tiff_y = toga_y - sin(r)*tx - cos(r)*ty + exp_oy
    for td in togas_data:
        tx_, ty_ = td.get("x", 0), td.get("y", 0)
        td["x"] = tx_ - cos_r * img_tx + sin_r * img_ty + exp_ox
        td["y"] = ty_ - sin_r * img_tx - cos_r * img_ty + exp_oy

    # --- Expand canvas to fit all layers (prevent clipping) ---
    _min_x, _min_y = 0, 0
    _max_x, _max_y = w, h

    # Estimate toga ref_ar for bounds calculation
    _toga_ars = []
    for _td in togas_data:
        _tp = CAIDA_DIR / f"{_td.get('variant', 'CAIDA_1')}.webp"
        if _tp.exists():
            with Image.open(str(_tp)) as _tmp:
                if _tmp.width > 0:
                    _toga_ars.append(_tmp.height / _tmp.width)
    _ref_ar = sum(_toga_ars) / len(_toga_ars) if _toga_ars else 1.5
    _gsy = toga_group_tf.get("scaleY", 1.0)

    for _td in togas_data:
        _tw = _td.get("scale", 50) * _td.get("scaleX", 1.0)
        _th = _td.get("scale", 50) * _ref_ar * _gsy
        _tx, _ty = _td.get("x", 0), _td.get("y", 0)
        _rot = abs(_td.get("rotation", 0))
        _margin = 1.0 + 0.2 * (_rot / 45.0) if _rot > 1 else 1.0
        _half_w = _tw * _margin / 2
        _full_h = _th * _margin
        _min_x = min(_min_x, _tx - _half_w)
        _max_x = max(_max_x, _tx + _half_w)
        _min_y = min(_min_y, _ty)
        _max_y = max(_max_y, _ty + _full_h)

    for _bd in borlas_data:
        _bs = _bd.get("scale", 50)
        _bx, _by = _bd.get("x", 0), _bd.get("y", 0)
        _min_x = min(_min_x, _bx - _bs)
        _max_x = max(_max_x, _bx + _bs)
        _min_y = min(_min_y, _by - _bs // 2)
        _max_y = max(_max_y, _by + _bs * 2)

    # Compute padding needed
    pad_left = max(0, int(_m.ceil(-_min_x)))
    pad_top = max(0, int(_m.ceil(-_min_y)))
    pad_right = max(0, int(_m.ceil(_max_x - w)))
    pad_bottom = max(0, int(_m.ceil(_max_y - h)))

    if pad_left or pad_top or pad_right or pad_bottom:
        new_w = w + pad_left + pad_right
        new_h = h + pad_top + pad_bottom
        # Expand cutout
        new_cut = np.zeros((new_h, new_w, 4), dtype=np.uint8)
        new_cut[pad_top:pad_top + h, pad_left:pad_left + w] = cut_np
        cut_np = new_cut
        # Expand panorama
        new_pano = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        new_pano[pad_top:pad_top + h, pad_left:pad_left + w] = pano_np
        pano_np = new_pano
        new_pano_rgba = np.zeros((new_h, new_w, 4), dtype=np.uint8)
        new_pano_rgba[pad_top:pad_top + h, pad_left:pad_left + w] = pano_rgba
        pano_rgba = new_pano_rgba
        # Offset all element positions
        for _bd in borlas_data:
            _bd["x"] = _bd.get("x", 0) + pad_left
            _bd["y"] = _bd.get("y", 0) + pad_top
        for _td in togas_data:
            _td["x"] = _td.get("x", 0) + pad_left
            _td["y"] = _td.get("y", 0) + pad_top
        w, h = new_w, new_h

    def _render_layer(img_pil, bd, pivot_top_center=True):
        """Render a single layer: scale, mask, rotate, position. Returns (cropped_np, lx1, ly1) or None."""
        target_w = max(4, int(bd.get("scale", 50)))
        ratio = target_w / img_pil.width
        target_h = int(img_pil.height * ratio)
        resized = img_pil.resize((target_w, target_h), Image.LANCZOS)

        # Flip horizontal
        if bd.get("flipH"):
            resized = resized.transpose(Image.FLIP_LEFT_RIGHT)

        # Scale X individually (for togas)
        sx = bd.get("scaleX", 1.0)
        if abs(sx - 1.0) > 0.01:
            new_w = max(4, int(resized.width * sx))
            resized = resized.resize((new_w, resized.height), Image.LANCZOS)

        # Apply per-layer mask (alpha channel = opacity: 255=show, 0=erased)
        # Uses BICUBIC to match browser's smooth drawImage interpolation
        mask_data = bd.get("mask")
        if mask_data and isinstance(mask_data, str) and mask_data.startswith("data:image/"):
            b64 = mask_data.split(",", 1)[1]
            mask_pil = Image.open(BytesIO(base64.b64decode(b64))).convert("RGBA")
            mask_alpha = mask_pil.split()[3]  # extract alpha channel
            mask_orig_w = mask_alpha.width
            mask_alpha = mask_alpha.resize(resized.size, Image.BICUBIC)
            # Adaptive Gaussian blur: when mask is downscaled significantly,
            # feather gradients get compressed. Blur softens them to match
            # the smooth appearance seen on screen at zoom.
            downscale_ratio = mask_orig_w / max(1, resized.width)
            if downscale_ratio > 1.5:
                sigma = 0.35 * (downscale_ratio - 1)
                marr_blur = gaussian_filter(
                    np.array(mask_alpha, dtype=np.float32), sigma=sigma)
                mask_alpha = Image.fromarray(
                    np.clip(marr_blur, 0, 255).astype(np.uint8))
            arr = np.array(resized)
            marr = np.array(mask_alpha, dtype=np.float32) / 255.0
            arr[:, :, 3] = np.clip(np.round(
                arr[:, :, 3].astype(np.float32) * marr), 0, 255).astype(np.uint8)
            resized = Image.fromarray(arr)

        # Rotate
        rotation = bd.get("rotation", 0)
        dx, dy = 0, 0
        if abs(rotation) > 0.5:
            bw, bh = resized.size
            pad = max(bw, bh)
            padded = Image.new("RGBA", (bw + pad * 2, bh + pad * 2), (0, 0, 0, 0))
            padded.paste(resized, (pad, pad), resized)
            if pivot_top_center:
                pivot_x, pivot_y = pad + bw // 2, pad
            else:
                pivot_x, pivot_y = pad + bw // 2, pad + bh // 2
            rotated = padded.rotate(-rotation, center=(pivot_x, pivot_y),
                                    expand=False, resample=Image.BICUBIC)
            bbox = rotated.getbbox()
            if bbox:
                rotated = rotated.crop(bbox)
                dx = bbox[0] - pad
                dy = bbox[1] - pad
            else:
                rotated = resized
            resized = rotated

        # Position
        bx = int(bd.get("x", 0)) - resized.width // 2 + dx
        by = int(bd.get("y", 0)) + dy

        # Clip to canvas
        lx1, ly1 = max(0, bx), max(0, by)
        lx2 = min(w, bx + resized.width)
        ly2 = min(h, by + resized.height)
        if lx2 <= lx1 or ly2 <= ly1:
            return None
        crop_x1, crop_y1 = lx1 - bx, ly1 - by
        cropped = np.array(resized, dtype=np.uint8)[crop_y1:ly2 - ly1 + crop_y1,
                                                     crop_x1:lx2 - lx1 + crop_x1]
        return cropped, lx1, ly1

    def _make_psd_layer(name, arr, lx1, ly1):
        lh, lw = arr.shape[:2]
        return PsdLayer(
            name=name,
            rectangle=PsdRectangle(ly1, lx1, ly1 + lh, lx1 + lw),
            channels=[
                PsdChannel(channelid=PsdChannelId.TRANSPARENCY_MASK,
                           compression=PsdCompressionType.ZIP_PREDICTED,
                           data=arr[:, :, 3]),
                PsdChannel(channelid=PsdChannelId.CHANNEL0,
                           compression=PsdCompressionType.ZIP_PREDICTED,
                           data=arr[:, :, 0]),
                PsdChannel(channelid=PsdChannelId.CHANNEL1,
                           compression=PsdCompressionType.ZIP_PREDICTED,
                           data=arr[:, :, 1]),
                PsdChannel(channelid=PsdChannelId.CHANNEL2,
                           compression=PsdCompressionType.ZIP_PREDICTED,
                           data=arr[:, :, 2]),
            ],
            mask=PsdLayerMask(), opacity=255,
            blendmode=PsdBlendMode.NORMAL, blending_ranges=(),
            clipping=PsdClippingType.BASE,
            flags=PsdLayerFlag.PHOTOSHOP5,
            info=[PsdString(PsdKey.UNICODE_LAYER_NAME, name)],
        )

    def _make_group(name, layer_list, rgba_list):
        """Build PSD group markers + layers. Returns (psd_layers, rgba_overlays)."""
        if not layer_list:
            return [], []
        group_psd = []
        # Bounding divider (group END = bottom of group in PSD order)
        group_psd.append(PsdLayer(
            name=f'</{name}>', rectangle=PsdRectangle(0, 0, 0, 0),
            channels=[], mask=PsdLayerMask(), opacity=255,
            blendmode=PsdBlendMode.NORMAL, blending_ranges=(),
            clipping=PsdClippingType.BASE, flags=PsdLayerFlag.PHOTOSHOP5,
            info=[PsdSectionDividerSetting(
                kind=PsdSectionDividerType.BOUNDING_SECTION_DIVIDER)],
        ))
        group_psd.extend(layer_list)
        # Open folder (group START = top of group in PSD order)
        group_psd.append(PsdLayer(
            name=name, rectangle=PsdRectangle(0, 0, 0, 0),
            channels=[], mask=PsdLayerMask(), opacity=255,
            blendmode=PsdBlendMode.NORMAL, blending_ranges=(),
            clipping=PsdClippingType.BASE, flags=PsdLayerFlag.PHOTOSHOP5,
            info=[
                PsdSectionDividerSetting(
                    kind=PsdSectionDividerType.OPEN_FOLDER,
                    blendmode=PsdBlendMode.PASS_THROUGH),
                PsdString(PsdKey.UNICODE_LAYER_NAME, name),
            ],
        ))
        return group_psd, rgba_list

    # --- Render borla layers ---
    borla_psd, borla_rgba = [], []
    for i, bd in enumerate(borlas_data):
        if not bd.get("visible", True):
            continue
        color = bd.get("color", "DORADO")
        bpath = BORLAS_DIR / f"B_{color}.webp"
        if not bpath.exists():
            continue
        borla_img = Image.open(str(bpath)).convert("RGBA")
        result = _render_layer(borla_img, bd)
        if result:
            arr, lx1, ly1 = result
            borla_rgba.append((arr, (ly1, lx1)))
            borla_psd.append(_make_psd_layer(f'Borla_{i+1}', arr, lx1, ly1))

    # --- Render toga layers ---
    # Compute uniform reference AR (avg h/w of all variants) so all togas have same height
    toga_ars = []
    for td in togas_data:
        tp = CAIDA_DIR / f"{td.get('variant', 'CAIDA_1')}.webp"
        if tp.exists():
            with Image.open(str(tp)) as tmp:
                if tmp.width > 0:
                    toga_ars.append(tmp.height / tmp.width)
    ref_ar = sum(toga_ars) / len(toga_ars) if toga_ars else 1.5
    gsy = toga_group_tf.get("scaleY", 1.0)

    toga_psd, toga_rgba = [], []
    for i, td in enumerate(togas_data):
        variant = td.get("variant", "CAIDA_1")
        tpath = CAIDA_DIR / f"{variant}.webp"
        if not tpath.exists():
            continue
        toga_img = Image.open(str(tpath)).convert("RGBA")
        # Normalize height to uniform ref_ar * gsy so all togas match vertically
        target_h = max(4, int(toga_img.width * ref_ar * gsy))
        if target_h != toga_img.height:
            toga_img = toga_img.resize((toga_img.width, target_h), Image.LANCZOS)
        result = _render_layer(toga_img, td, pivot_top_center=True)
        if result:
            arr, lx1, ly1 = result
            toga_rgba.append((arr, (ly1, lx1)))
            toga_psd.append(_make_psd_layer(f'Toga_{i+1}', arr, lx1, ly1))

    # --- Build full layer stack ---
    all_layers = [
        # Panorama (hidden)
        PsdLayer(
            name='Panoramica', rectangle=PsdRectangle(0, 0, h, w),
            channels=[
                PsdChannel(channelid=PsdChannelId.CHANNEL0,
                           compression=PsdCompressionType.ZIP_PREDICTED, data=pano_np[:, :, 0]),
                PsdChannel(channelid=PsdChannelId.CHANNEL1,
                           compression=PsdCompressionType.ZIP_PREDICTED, data=pano_np[:, :, 1]),
                PsdChannel(channelid=PsdChannelId.CHANNEL2,
                           compression=PsdCompressionType.ZIP_PREDICTED, data=pano_np[:, :, 2]),
            ],
            mask=PsdLayerMask(), opacity=255,
            blendmode=PsdBlendMode.NORMAL, blending_ranges=(),
            clipping=PsdClippingType.BASE,
            flags=PsdLayerFlag.PHOTOSHOP5 | PsdLayerFlag.TRANSPARENCY_PROTECTED | PsdLayerFlag.VISIBLE,
            info=[PsdString(PsdKey.UNICODE_LAYER_NAME, 'Panoramica Original')],
        ),
        # Cutout
        PsdLayer(
            name='Recortada', rectangle=PsdRectangle(0, 0, h, w),
            channels=[
                PsdChannel(channelid=PsdChannelId.TRANSPARENCY_MASK,
                           compression=PsdCompressionType.ZIP_PREDICTED, data=cut_np[:, :, 3]),
                PsdChannel(channelid=PsdChannelId.CHANNEL0,
                           compression=PsdCompressionType.ZIP_PREDICTED, data=cut_np[:, :, 0]),
                PsdChannel(channelid=PsdChannelId.CHANNEL1,
                           compression=PsdCompressionType.ZIP_PREDICTED, data=cut_np[:, :, 1]),
                PsdChannel(channelid=PsdChannelId.CHANNEL2,
                           compression=PsdCompressionType.ZIP_PREDICTED, data=cut_np[:, :, 2]),
            ],
            mask=PsdLayerMask(), opacity=255,
            blendmode=PsdBlendMode.NORMAL, blending_ranges=(),
            clipping=PsdClippingType.BASE, flags=PsdLayerFlag.PHOTOSHOP5,
            info=[PsdString(PsdKey.UNICODE_LAYER_NAME, 'Recortada')],
        ),
    ]

    # BORLAS group
    grp_psd, _ = _make_group('BORLAS', borla_psd, borla_rgba)
    all_layers.extend(grp_psd)

    # TOGAS group
    grp_psd2, _ = _make_group('TOGAS', toga_psd, toga_rgba)
    all_layers.extend(grp_psd2)

    image_source_data = TiffImageSourceData(
        name='ARCA Export',
        psdformat=PsdFormat.LE32BIT,
        layers=PsdLayers(key=PsdKey.LAYER, has_transparency=True, layers=all_layers),
        usermask=PsdUserMask(
            colorspace=PsdColorSpaceType.RGB,
            components=(65535, 0, 0, 0), opacity=50),
        info=[PsdEmpty(PsdKey.PATTERNS)],
    )

    # Composite for TIFF thumbnail
    comp_args = [(pano_rgba, (0, 0)), (cut_np, (0, 0))]
    for (arr, pos) in borla_rgba:
        comp_args.append((arr, pos))
    for (arr, pos) in toga_rgba:
        comp_args.append((arr, pos))
    composite = overlay(*comp_args, shape=(h, w))

    buf = BytesIO()
    tifffile.imwrite(
        buf, composite, photometric='rgb', compression='adobe_deflate',
        resolution=((720000, 10000), (720000, 10000)), resolutionunit='inch',
        metadata=None,
        extratags=[
            image_source_data.tifftag(maxworkers=1),
            (34675, 7, None, imagecodecs.cms_profile('srgb'), True),
        ],
    )
    buf.seek(0)
    return send_file(buf, mimetype="image/tiff", as_attachment=True,
                     download_name=f"{os.path.splitext(filename)[0]}_togas.tif")


# ─── SOMBRAS (shadows) ───────────────────────────────────────────────────────

def _sombras_state_path(cbtis, filename):
    stem = os.path.splitext(filename)[0]
    return get_year_dir() / cbtis / "RECORTADAS" / f".{stem}.sombras.json"


@app.route("/api/sombras/state/<cbtis>/<filename>")
def api_sombras_state_get(cbtis, filename):
    p = _sombras_state_path(cbtis, filename)
    if p.exists():
        with open(p) as f:
            return jsonify(json.load(f))
    return jsonify(None)


@app.route("/api/sombras/state/<cbtis>/<filename>", methods=["POST"])
def api_sombras_state_set(cbtis, filename):
    p = _sombras_state_path(cbtis, filename)
    data = request.get_json(force=True)
    with open(p, "w") as f:
        json.dump(data, f)
    return jsonify({"ok": True})


@app.route("/api/sombras/snapshot/<cbtis>/<filename>", methods=["POST"])
def api_sombras_snapshot_set(cbtis, filename):
    """Save a rendered snapshot from the Sombras canvas."""
    stem = os.path.splitext(filename)[0]
    snap_path = get_year_dir() / cbtis / "RECORTADAS" / f".{stem}.sombras_snapshot.webp"
    if 'snapshot' in request.files:
        request.files['snapshot'].save(str(snap_path))
    else:
        snap_path.write_bytes(request.data)
    return jsonify({"ok": True})


@app.route("/api/sombras/snapshot/<cbtis>/<filename>")
def api_sombras_snapshot_get(cbtis, filename):
    """Return saved Sombras snapshot for Fixes section."""
    stem = os.path.splitext(filename)[0]
    snap_path = get_year_dir() / cbtis / "RECORTADAS" / f".{stem}.sombras_snapshot.webp"
    if not snap_path.exists():
        return jsonify({"error": "No snapshot saved. Open Sombras section first."}), 404
    maxw = request.args.get("maxw", 99999, type=int)
    img = Image.open(str(snap_path))
    if img.width > maxw:
        ratio = maxw / img.width
        img = img.resize((maxw, int(img.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, "WEBP", quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/webp")


@app.route("/api/sombras/export-tiff/<cbtis>/<filename>", methods=["POST"])
def api_sombras_export_tiff(cbtis, filename):
    """Export EVERYTHING: pano + sombras-below + cutout + borlas + sombras-above + togas as layered TIFF."""
    import traceback as _tb
    try:
        return _sombras_export_tiff_impl(cbtis, filename)
    except Exception as e:
        print(f"[TIFF EXPORT ERROR] {e}")
        _tb.print_exc()
        return jsonify({"error": str(e)}), 500

def _sombras_export_tiff_impl(cbtis, filename):
    """Full TIFF export: pano + sombras-below + cutout + borlas + hilos + togas + sombras-above.
    Reuses the same transform/rendering logic as the Togas TIFF export."""
    import tifffile
    import imagecodecs
    import cv2 as cv
    import math as _m
    import base64
    from scipy.ndimage import gaussian_filter
    from psdtags import (
        PsdBlendMode, PsdChannel, PsdChannelId, PsdClippingType,
        PsdColorSpaceType, PsdCompressionType, PsdEmpty,
        PsdFormat, PsdKey, PsdLayer, PsdLayerFlag, PsdLayerMask,
        PsdLayers, PsdRectangle, PsdSectionDividerSetting,
        PsdSectionDividerType, PsdString, PsdUserMask,
        TiffImageSourceData, overlay,
    )

    data = request.get_json(force=True)
    sombras_data = data.get("sombras", [])
    borlas_data = data.get("borlas", [])
    togas_data = data.get("togas", [])
    toga_group_tf = data.get("togaGroupTransform", {})
    img_tf = data.get("imgTransform", {})
    img_rot = img_tf.get("rotation", 0)
    img_tx = img_tf.get("x", 0)
    img_ty = img_tf.get("y", 0)
    hilo_color_hex = data.get("hiloColor", "#d4a017")
    hilo_size = data.get("hiloSize", 3)
    hilo_feather = data.get("hiloFeather", 1)
    hilo_flow = data.get("hiloFlow", 100) / 100.0
    toga_ref_ar = data.get("togaRefAR", 1.5)

    year_dir = get_year_dir()
    cut_path = year_dir / cbtis / "RECORTADAS" / filename
    if not cut_path.exists():
        return jsonify({"error": "Not found"}), 404

    align = _compute_alignment(cbtis, filename)
    if not align:
        return jsonify({"error": "Alignment failed"}), 500

    # --- Load cutout + panorama ---
    cut_pil = Image.open(str(cut_path))
    if cut_pil.mode != "RGBA":
        cut_pil = cut_pil.convert("RGBA")
    orig_w, orig_h = cut_pil.size
    img_cx, img_cy = orig_w / 2.0, orig_h / 2.0

    pano_path = year_dir / cbtis / "UNIDAS" / align["pano_file"]
    pano_full = Image.open(str(pano_path)).convert("RGB")
    cx, cy = align["crop_x"], align["crop_y"]
    cw_, ch_ = align["cut_w"], align["cut_h"]
    pano_region = pano_full.crop((cx, cy, min(cx + cw_, pano_full.width),
                                  min(cy + ch_, pano_full.height)))
    if pano_region.size != (orig_w, orig_h):
        pano_region = pano_region.resize((orig_w, orig_h), Image.LANCZOS)

    # --- Apply imgTransform: rotate cutout+panorama ---
    exp_ox, exp_oy = 0.0, 0.0
    irad = _m.radians(img_rot)
    cos_r, sin_r = _m.cos(irad), _m.sin(irad)

    if abs(img_rot) > 0.1:
        cut_pil = cut_pil.rotate(-img_rot, expand=True, resample=Image.BICUBIC)
        pano_rgba_pil = pano_region.convert("RGBA")
        pano_rgba_pil = pano_rgba_pil.rotate(-img_rot, expand=True, resample=Image.BICUBIC)
        new_w, new_h = cut_pil.size
        exp_ox = (new_w - orig_w) / 2.0
        exp_oy = (new_h - orig_h) / 2.0
        cut_np = np.array(cut_pil, dtype=np.uint8)
        pano_np_raw = np.array(pano_rgba_pil, dtype=np.uint8)
        pano_np = pano_np_raw[:, :, :3]
        pano_rgba = pano_np_raw
        h, w = cut_np.shape[:2]
    else:
        cut_np = np.array(cut_pil, dtype=np.uint8)
        pano_np = np.array(pano_region, dtype=np.uint8)
        h, w = cut_np.shape[:2]
        pano_rgba = np.dstack([pano_np, np.full((h, w), 255, dtype=np.uint8)])

    # --- Transform borla positions ---
    for bd in borlas_data:
        bx, by = bd.get("x", 0), bd.get("y", 0)
        if abs(img_rot) > 0.1:
            dx, dy = bx - img_cx, by - img_cy
            bd["x"] = img_cx + dx * cos_r - dy * sin_r + exp_ox
            bd["y"] = img_cy + dx * sin_r + dy * cos_r + exp_oy
            bd["rotation"] = bd.get("rotation", 0) + img_rot
        else:
            bd["x"] = bx + exp_ox
            bd["y"] = by + exp_oy

    # --- Transform toga positions ---
    for td in togas_data:
        tx_, ty_ = td.get("x", 0), td.get("y", 0)
        td["x"] = tx_ - cos_r * img_tx + sin_r * img_ty + exp_ox
        td["y"] = ty_ - sin_r * img_tx - cos_r * img_ty + exp_oy

    # --- Canvas expansion: all layers + sombras ---
    _min_x, _min_y = 0, 0
    _max_x, _max_y = w, h

    # Sombras bounds
    for sp in sombras_data:
        pts = sp.get("points", [])
        feather = sp.get("feather", {})
        mf = max(feather.get("top", 0), feather.get("bottom", 0),
                 feather.get("left", 0), feather.get("right", 0))
        for pt in pts:
            _min_x = min(_min_x, int(pt["x"]) - mf)
            _min_y = min(_min_y, int(pt["y"]) - mf)
            _max_x = max(_max_x, int(pt["x"]) + mf)
            _max_y = max(_max_y, int(pt["y"]) + mf)

    # Toga bounds
    gsy = toga_group_tf.get("scaleY", 1.0)
    for _td in togas_data:
        _tw = _td.get("scale", 50) * _td.get("scaleX", 1.0)
        _th = _td.get("scale", 50) * toga_ref_ar * gsy
        _tx, _ty = _td.get("x", 0), _td.get("y", 0)
        _rot = abs(_td.get("rotation", 0))
        _margin = 1.0 + 0.2 * (_rot / 45.0) if _rot > 1 else 1.0
        _half_w = _tw * _margin / 2
        _full_h = _th * _margin
        _min_x = min(_min_x, _tx - _half_w)
        _max_x = max(_max_x, _tx + _half_w)
        _min_y = min(_min_y, _ty)
        _max_y = max(_max_y, _ty + _full_h)

    # Borla bounds
    for _bd in borlas_data:
        _bs = _bd.get("scale", 50)
        _bx, _by = _bd.get("x", 0), _bd.get("y", 0)
        _min_x = min(_min_x, _bx - _bs)
        _max_x = max(_max_x, _bx + _bs)
        _min_y = min(_min_y, _by - _bs // 2)
        _max_y = max(_max_y, _by + _bs * 2)

    pad_left = max(0, int(_m.ceil(-_min_x)))
    pad_top = max(0, int(_m.ceil(-_min_y)))
    pad_right = max(0, int(_m.ceil(_max_x - w)))
    pad_bottom = max(0, int(_m.ceil(_max_y - h)))

    if pad_left or pad_top or pad_right or pad_bottom:
        new_w = w + pad_left + pad_right
        new_h = h + pad_top + pad_bottom
        new_cut = np.zeros((new_h, new_w, 4), dtype=np.uint8)
        new_cut[pad_top:pad_top + h, pad_left:pad_left + w] = cut_np
        cut_np = new_cut
        new_pano = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        new_pano[pad_top:pad_top + h, pad_left:pad_left + w] = pano_np
        pano_np = new_pano
        new_pano_rgba = np.zeros((new_h, new_w, 4), dtype=np.uint8)
        new_pano_rgba[pad_top:pad_top + h, pad_left:pad_left + w] = pano_rgba
        pano_rgba = new_pano_rgba
        for _bd in borlas_data:
            _bd["x"] = _bd.get("x", 0) + pad_left
            _bd["y"] = _bd.get("y", 0) + pad_top
        for _td in togas_data:
            _td["x"] = _td.get("x", 0) + pad_left
            _td["y"] = _td.get("y", 0) + pad_top
        w, h = new_w, new_h

    # --- Helpers (same as togas export) ---
    def _render_layer(img_pil, bd, pivot_top_center=True):
        target_w = max(4, int(bd.get("scale", 50)))
        ratio = target_w / img_pil.width
        target_h = int(img_pil.height * ratio)
        resized = img_pil.resize((target_w, target_h), Image.LANCZOS)
        if bd.get("flipH"):
            resized = resized.transpose(Image.FLIP_LEFT_RIGHT)
        sx = bd.get("scaleX", 1.0)
        if abs(sx - 1.0) > 0.01:
            new_w2 = max(4, int(resized.width * sx))
            resized = resized.resize((new_w2, resized.height), Image.LANCZOS)
        mask_data = bd.get("mask")
        if mask_data and isinstance(mask_data, str) and mask_data.startswith("data:image/"):
            b64 = mask_data.split(",", 1)[1]
            mask_pil = Image.open(BytesIO(base64.b64decode(b64))).convert("RGBA")
            mask_alpha = mask_pil.split()[3]
            mask_orig_w = mask_alpha.width
            mask_alpha = mask_alpha.resize(resized.size, Image.BICUBIC)
            downscale_ratio = mask_orig_w / max(1, resized.width)
            if downscale_ratio > 1.5:
                sigma = 0.35 * (downscale_ratio - 1)
                marr_blur = gaussian_filter(
                    np.array(mask_alpha, dtype=np.float32), sigma=sigma)
                mask_alpha = Image.fromarray(
                    np.clip(marr_blur, 0, 255).astype(np.uint8))
            arr = np.array(resized)
            marr = np.array(mask_alpha, dtype=np.float32) / 255.0
            arr[:, :, 3] = np.clip(np.round(
                arr[:, :, 3].astype(np.float32) * marr), 0, 255).astype(np.uint8)
            resized = Image.fromarray(arr)
        rotation = bd.get("rotation", 0)
        dx, dy = 0, 0
        if abs(rotation) > 0.5:
            bw, bh = resized.size
            pad = max(bw, bh)
            padded = Image.new("RGBA", (bw + pad * 2, bh + pad * 2), (0, 0, 0, 0))
            padded.paste(resized, (pad, pad), resized)
            if pivot_top_center:
                pivot_x, pivot_y = pad + bw // 2, pad
            else:
                pivot_x, pivot_y = pad + bw // 2, pad + bh // 2
            rotated = padded.rotate(-rotation, center=(pivot_x, pivot_y),
                                    expand=False, resample=Image.BICUBIC)
            bbox = rotated.getbbox()
            if bbox:
                rotated = rotated.crop(bbox)
                dx = bbox[0] - pad
                dy = bbox[1] - pad
            else:
                rotated = resized
            resized = rotated
        bx = int(bd.get("x", 0)) - resized.width // 2 + dx
        by = int(bd.get("y", 0)) + dy
        lx1, ly1 = max(0, bx), max(0, by)
        lx2 = min(w, bx + resized.width)
        ly2 = min(h, by + resized.height)
        if lx2 <= lx1 or ly2 <= ly1:
            return None
        crop_x1, crop_y1 = lx1 - bx, ly1 - by
        cropped = np.array(resized, dtype=np.uint8)[crop_y1:ly2 - ly1 + crop_y1,
                                                     crop_x1:lx2 - lx1 + crop_x1]
        return cropped, lx1, ly1

    def _make_psd_layer(name, arr, lx1, ly1, blendmode=PsdBlendMode.NORMAL):
        lh, lw = arr.shape[:2]
        return PsdLayer(
            name=name,
            rectangle=PsdRectangle(ly1, lx1, ly1 + lh, lx1 + lw),
            channels=[
                PsdChannel(channelid=PsdChannelId.TRANSPARENCY_MASK,
                           compression=PsdCompressionType.ZIP_PREDICTED, data=arr[:, :, 3]),
                PsdChannel(channelid=PsdChannelId.CHANNEL0,
                           compression=PsdCompressionType.ZIP_PREDICTED, data=arr[:, :, 0]),
                PsdChannel(channelid=PsdChannelId.CHANNEL1,
                           compression=PsdCompressionType.ZIP_PREDICTED, data=arr[:, :, 1]),
                PsdChannel(channelid=PsdChannelId.CHANNEL2,
                           compression=PsdCompressionType.ZIP_PREDICTED, data=arr[:, :, 2]),
            ],
            mask=PsdLayerMask(), opacity=255,
            blendmode=blendmode, blending_ranges=(),
            clipping=PsdClippingType.BASE, flags=PsdLayerFlag.PHOTOSHOP5,
            info=[PsdString(PsdKey.UNICODE_LAYER_NAME, name)],
        )

    def _add_group(layers_list, psd_layers, group_name):
        if not psd_layers:
            return
        layers_list.append(PsdLayer(
            name=f'</{group_name}>', rectangle=PsdRectangle(0, 0, 0, 0),
            channels=[], mask=PsdLayerMask(), opacity=255,
            blendmode=PsdBlendMode.NORMAL, blending_ranges=(),
            clipping=PsdClippingType.BASE, flags=PsdLayerFlag.PHOTOSHOP5,
            info=[PsdSectionDividerSetting(kind=PsdSectionDividerType.BOUNDING_SECTION_DIVIDER)],
        ))
        layers_list.extend(psd_layers)
        layers_list.append(PsdLayer(
            name=group_name, rectangle=PsdRectangle(0, 0, 0, 0),
            channels=[], mask=PsdLayerMask(), opacity=255,
            blendmode=PsdBlendMode.NORMAL, blending_ranges=(),
            clipping=PsdClippingType.BASE, flags=PsdLayerFlag.PHOTOSHOP5,
            info=[
                PsdSectionDividerSetting(kind=PsdSectionDividerType.OPEN_FOLDER,
                                         blendmode=PsdBlendMode.PASS_THROUGH),
                PsdString(PsdKey.UNICODE_LAYER_NAME, group_name),
            ],
        ))

    # --- Render shadow layers ---
    shadow_below_psd, shadow_below_rgba = [], []
    shadow_above_psd, shadow_above_rgba = [], []

    def _round_poly_pts(raw_pts, radius, pad_l, pad_t):
        coords = [(int(pt["x"]) + pad_l, int(pt["y"]) + pad_t) for pt in raw_pts]
        if radius < 1 or len(coords) < 3:
            return np.array(coords, dtype=np.int32)
        n = len(coords)
        result = []
        for i in range(n):
            prev = coords[(i-1) % n]
            curr = coords[i]
            nxt = coords[(i+1) % n]
            dx1 = prev[0]-curr[0]; dy1 = prev[1]-curr[1]
            dx2 = nxt[0]-curr[0]; dy2 = nxt[1]-curr[1]
            d1 = _m.sqrt(dx1*dx1+dy1*dy1)
            d2 = _m.sqrt(dx2*dx2+dy2*dy2)
            if d1 < 1 or d2 < 1:
                result.append(curr); continue
            r = min(radius, d1*0.4, d2*0.4)
            t1x = curr[0]+dx1/d1*r; t1y = curr[1]+dy1/d1*r
            t2x = curr[0]+dx2/d2*r; t2y = curr[1]+dy2/d2*r
            steps = max(4, int(r/8))
            for s in range(steps+1):
                t = s/steps
                bx = (1-t)**2*t1x + 2*(1-t)*t*curr[0] + t*t*t2x
                by = (1-t)**2*t1y + 2*(1-t)*t*curr[1] + t*t*t2y
                result.append((int(round(bx)), int(round(by))))
        return np.array(result, dtype=np.int32)

    for si, sp in enumerate(sombras_data):
        pts = sp.get("points", [])
        if len(pts) < 3:
            continue
        feather = sp.get("feather", {})
        opacity = sp.get("opacity", 0.5)
        layer_pos = sp.get("layer", "below")
        hardness_val = sp.get("hardness", 30) / 100.0
        mf = max(feather.get("top", 0), feather.get("bottom", 0),
                 feather.get("left", 0), feather.get("right", 0))
        corner_r = mf * max(0.15, (1 - hardness_val) * 0.6)
        np_pts = _round_poly_pts(pts, corner_r, pad_left, pad_top)
        mask = np.zeros((h, w), dtype=np.float32)
        cv.fillPoly(mask, [np_pts], 1.0, cv.LINE_AA)
        ft = feather.get("top", 0)
        fb = feather.get("bottom", 0)
        fl = feather.get("left", 0)
        fr = feather.get("right", 0)
        if fl > 0 or fr > 0 or ft > 0 or fb > 0:
            # 2-pass Gaussian blur to match client CSS blur exactly:
            # Pass 1: horizontal blur (sigma = max(left, right))
            # Pass 2: vertical blur (sigma = max(top, bottom))
            # CSS blur extends both inward and outward from the edge.
            sigma_boost = 1 + (1 - hardness_val) * 0.5
            h_sigma = max(fl, fr) * sigma_boost
            v_sigma = max(ft, fb) * sigma_boost
            if h_sigma > 0.5:
                kw = int(h_sigma * 6) | 1
                mask = cv.GaussianBlur(mask, (kw, 1), h_sigma)
            if v_sigma > 0.5:
                kh = int(v_sigma * 6) | 1
                mask = cv.GaussianBlur(mask, (1, kh), v_sigma)
        alpha = np.clip(mask * opacity * 255, 0, 255).astype(np.uint8)
        ys, xs = np.where(alpha > 0)
        if len(ys) == 0:
            continue
        ly1, ly2 = int(ys.min()), int(ys.max()) + 1
        lx1, lx2 = int(xs.min()), int(xs.max()) + 1
        alpha_crop = alpha[ly1:ly2, lx1:lx2]
        lh, lw = alpha_crop.shape
        layer_rgba = np.zeros((lh, lw, 4), dtype=np.uint8)
        layer_rgba[:, :, 3] = alpha_crop
        psd_layer = _make_psd_layer(f'Sombra {si+1}', layer_rgba, lx1, ly1,
                                    blendmode=PsdBlendMode.MULTIPLY)
        if layer_pos == "below":
            shadow_below_psd.append(psd_layer)
            shadow_below_rgba.append((layer_rgba, (ly1, lx1)))
        else:
            shadow_above_psd.append(psd_layer)
            shadow_above_rgba.append((layer_rgba, (ly1, lx1)))

    # --- Render borla layers (same approach as togas export) ---
    borla_psd, borla_comp = [], []
    for i, bd in enumerate(borlas_data):
        if not bd.get("visible", True):
            continue
        color = bd.get("color", "DORADO")
        bpath = BORLAS_DIR / f"B_{color}.webp"
        if not bpath.exists():
            continue
        borla_img = Image.open(str(bpath)).convert("RGBA")
        result = _render_layer(borla_img, bd)
        if result:
            arr, lx1, ly1 = result
            borla_comp.append((arr, (ly1, lx1)))
            borla_psd.append(_make_psd_layer(f'Borla {i+1}', arr, lx1, ly1))

    # --- Render hilo (thread) layers ---
    hilo_psd, hilo_comp = [], []
    # Parse hilo color hex to BGR for cv2
    hc = hilo_color_hex.lstrip('#')
    hilo_rgb = tuple(int(hc[i:i+2], 16) for i in (0, 2, 4)) if len(hc) >= 6 else (212, 160, 23)
    for i, bd in enumerate(borlas_data):
        if not bd.get("visible", True):
            continue
        hilo_pts = bd.get("hilo")
        if not hilo_pts or len(hilo_pts) < 2:
            continue
        # Compute hilo bounding box
        hxs = [p["x"] for p in hilo_pts]
        hys = [p["y"] for p in hilo_pts]
        pad_h = hilo_size * 2 + hilo_feather * 2 + 4
        hx1 = int(bd.get("x", 0) + min(hxs) - pad_h)
        hy1 = int(bd.get("y", 0) + min(hys) - pad_h)
        hx2 = int(bd.get("x", 0) + max(hxs) + pad_h)
        hy2 = int(bd.get("y", 0) + max(hys) + pad_h)
        # Rotate hilo points with borla transform
        bx_c, by_c = bd.get("x", 0), bd.get("y", 0)
        b_rot = bd.get("rotation", 0)
        b_rad = _m.radians(b_rot)
        b_cos, b_sin = _m.cos(b_rad), _m.sin(b_rad)
        canvas_pts = []
        for p in hilo_pts:
            # Hilo coords are relative to borla position
            rx, ry = p["x"], p["y"]
            # Rotate around borla anchor
            if abs(b_rot) > 0.1:
                rx2 = rx * b_cos - ry * b_sin
                ry2 = rx * b_sin + ry * b_cos
                rx, ry = rx2, ry2
            canvas_pts.append((int(bx_c + rx), int(by_c + ry)))
        if len(canvas_pts) < 2:
            continue
        # Compute tight bbox
        cxs = [p[0] for p in canvas_pts]
        cys = [p[1] for p in canvas_pts]
        pad_px = hilo_size + hilo_feather * 2 + 6
        lx1 = max(0, min(cxs) - pad_px)
        ly1 = max(0, min(cys) - pad_px)
        lx2 = min(w, max(cxs) + pad_px)
        ly2 = min(h, max(cys) + pad_px)
        if lx2 <= lx1 or ly2 <= ly1:
            continue
        lw_, lh_ = lx2 - lx1, ly2 - ly1
        # Draw hilo on a small RGBA canvas
        hilo_canvas = np.zeros((lh_, lw_, 4), dtype=np.uint8)
        local_pts = np.array([(x - lx1, y - ly1) for x, y in canvas_pts], dtype=np.int32)
        cv.polylines(hilo_canvas, [local_pts], False,
                     (hilo_rgb[0], hilo_rgb[1], hilo_rgb[2], 255),
                     thickness=max(1, hilo_size), lineType=cv.LINE_AA)
        # Apply feather (Gaussian blur on alpha)
        if hilo_feather > 0.5:
            alpha_f = hilo_canvas[:, :, 3].astype(np.float32)
            alpha_f = gaussian_filter(alpha_f, sigma=hilo_feather)
            # Re-normalize: keep max alpha at original level
            mx = alpha_f.max()
            if mx > 0:
                alpha_f = alpha_f / mx * 255
            hilo_canvas[:, :, 3] = np.clip(alpha_f, 0, 255).astype(np.uint8)
        # Apply hilo mask if present
        hilo_mask_data = bd.get("hiloMask")
        if hilo_mask_data and isinstance(hilo_mask_data, str) and hilo_mask_data.startswith("data:image/"):
            b64m = hilo_mask_data.split(",", 1)[1]
            hm_pil = Image.open(BytesIO(base64.b64decode(b64m))).convert("L")
            hm_pil = hm_pil.resize((lw_, lh_), Image.BICUBIC)
            hm_arr = np.array(hm_pil, dtype=np.float32) / 255.0
            hilo_canvas[:, :, 3] = np.clip(
                hilo_canvas[:, :, 3].astype(np.float32) * hm_arr, 0, 255).astype(np.uint8)
        # Apply flow (opacity)
        if hilo_flow < 0.99:
            hilo_canvas[:, :, 3] = np.clip(
                hilo_canvas[:, :, 3].astype(np.float32) * hilo_flow, 0, 255).astype(np.uint8)
        hilo_comp.append((hilo_canvas, (ly1, lx1)))
        hilo_psd.append(_make_psd_layer(f'Hilo {i+1}', hilo_canvas, lx1, ly1))

    # --- Render toga layers (proper ref_ar + gsy + groupTf) ---
    toga_psd, toga_comp = [], []
    for i, td in enumerate(togas_data):
        variant = td.get("variant", "CAIDA_1")
        tpath = CAIDA_DIR / f"{variant}.webp"
        if not tpath.exists():
            continue
        toga_img = Image.open(str(tpath)).convert("RGBA")
        # Normalize height to uniform ref_ar * gsy
        target_h = max(4, int(toga_img.width * toga_ref_ar * gsy))
        if target_h != toga_img.height:
            toga_img = toga_img.resize((toga_img.width, target_h), Image.LANCZOS)
        # Apply mask
        mask_data = td.get("mask")
        if mask_data and isinstance(mask_data, str) and mask_data.startswith("data:image/"):
            b64 = mask_data.split(",", 1)[1]
            mask_pil = Image.open(BytesIO(base64.b64decode(b64))).convert("RGBA")
            mask_alpha = mask_pil.split()[3]
            mask_alpha = mask_alpha.resize(toga_img.size, Image.BICUBIC)
            arr = np.array(toga_img)
            marr = np.array(mask_alpha, dtype=np.float32) / 255.0
            arr[:, :, 3] = np.clip(np.round(
                arr[:, :, 3].astype(np.float32) * marr), 0, 255).astype(np.uint8)
            toga_img = Image.fromarray(arr)
        result = _render_layer(toga_img, td, pivot_top_center=True)
        if result:
            arr, lx1, ly1 = result
            toga_comp.append((arr, (ly1, lx1)))
            toga_psd.append(_make_psd_layer(f'Toga {i+1}', arr, lx1, ly1))

    # --- Build PSD layer stack (bottom-to-top) ---
    all_layers = [
        PsdLayer(
            name='Panoramica', rectangle=PsdRectangle(0, 0, h, w),
            channels=[
                PsdChannel(channelid=PsdChannelId.CHANNEL0, compression=PsdCompressionType.ZIP_PREDICTED, data=pano_rgba[:, :, 0]),
                PsdChannel(channelid=PsdChannelId.CHANNEL1, compression=PsdCompressionType.ZIP_PREDICTED, data=pano_rgba[:, :, 1]),
                PsdChannel(channelid=PsdChannelId.CHANNEL2, compression=PsdCompressionType.ZIP_PREDICTED, data=pano_rgba[:, :, 2]),
            ],
            mask=PsdLayerMask(), opacity=255,
            blendmode=PsdBlendMode.NORMAL, blending_ranges=(),
            clipping=PsdClippingType.BASE,
            flags=PsdLayerFlag.PHOTOSHOP5 | PsdLayerFlag.TRANSPARENCY_PROTECTED | PsdLayerFlag.VISIBLE,
            info=[PsdString(PsdKey.UNICODE_LAYER_NAME, 'Panoramica Original')],
        ),
    ]

    _add_group(all_layers, shadow_below_psd, 'SOMBRAS DEBAJO')

    # Cutout
    all_layers.append(PsdLayer(
        name='Recortada', rectangle=PsdRectangle(0, 0, h, w),
        channels=[
            PsdChannel(channelid=PsdChannelId.TRANSPARENCY_MASK, compression=PsdCompressionType.ZIP_PREDICTED, data=cut_np[:, :, 3]),
            PsdChannel(channelid=PsdChannelId.CHANNEL0, compression=PsdCompressionType.ZIP_PREDICTED, data=cut_np[:, :, 0]),
            PsdChannel(channelid=PsdChannelId.CHANNEL1, compression=PsdCompressionType.ZIP_PREDICTED, data=cut_np[:, :, 1]),
            PsdChannel(channelid=PsdChannelId.CHANNEL2, compression=PsdCompressionType.ZIP_PREDICTED, data=cut_np[:, :, 2]),
        ],
        mask=PsdLayerMask(), opacity=255,
        blendmode=PsdBlendMode.NORMAL, blending_ranges=(),
        clipping=PsdClippingType.BASE, flags=PsdLayerFlag.PHOTOSHOP5,
        info=[PsdString(PsdKey.UNICODE_LAYER_NAME, 'Recortada')],
    ))

    _add_group(all_layers, borla_psd, 'BORLAS')
    _add_group(all_layers, hilo_psd, 'HILOS')
    _add_group(all_layers, toga_psd, 'TOGAS')
    _add_group(all_layers, shadow_above_psd, 'SOMBRAS ARRIBA')

    image_source_data = TiffImageSourceData(
        name='ARCA Export',
        psdformat=PsdFormat.LE32BIT,
        layers=PsdLayers(key=PsdKey.LAYER, has_transparency=True, layers=all_layers),
        usermask=PsdUserMask(
            colorspace=PsdColorSpaceType.RGB,
            components=(65535, 0, 0, 0), opacity=50),
        info=[PsdEmpty(PsdKey.PATTERNS)],
    )

    # Flat composite for TIFF thumbnail
    comp_args = [(pano_rgba, (0, 0))]
    for a in shadow_below_rgba: comp_args.append(a)
    comp_args.append((cut_np, (0, 0)))
    for a in borla_comp: comp_args.append(a)
    for a in hilo_comp: comp_args.append(a)
    for a in toga_comp: comp_args.append(a)
    for a in shadow_above_rgba: comp_args.append(a)
    composite = overlay(*comp_args, shape=(h, w))

    buf = BytesIO()
    tifffile.imwrite(
        buf, composite, photometric='rgb', compression='adobe_deflate',
        resolution=((720000, 10000), (720000, 10000)), resolutionunit='inch',
        metadata=None,
        extratags=[
            image_source_data.tifftag(maxworkers=1),
            (34675, 7, None, imagecodecs.cms_profile('srgb'), True),
        ],
    )
    buf.seek(0)
    return send_file(buf, mimetype="image/tiff", as_attachment=True,
                     download_name=f"{os.path.splitext(filename)[0]}_final.tif")


# --- Fixes endpoints ---

def _parse_source_photos(filename):
    """Parse panorama filename to extract source photo IDs and group name.
    '2513-2522-2540_15_MAT_A_CONTA_PLATEADO.webp' →
    ids=['2513','2522','2540'], cbtis_from_name='15', group='MAT_A_CONTA_PLATEADO'
    """
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_", 2)  # ['2513-2522-2540', '15', 'MAT_A_CONTA_PLATEADO']
    if len(parts) < 3:
        return [], ""
    ids = [x for x in parts[0].split("-") if x.strip()]
    group = parts[2]
    return ids, group


@app.route("/api/fixes/source-photos/<cbtis>/<filename>")
def api_fixes_source_photos(cbtis, filename):
    """Return list of source photos that compose this panorama."""
    ids, group = _parse_source_photos(filename)
    if not ids:
        return jsonify({"ok": False, "error": "Cannot parse filename"}), 400
    year_dir = get_year_dir()
    photos = []
    for pid in ids:
        photo_name = f"IMG_{pid}.jpg"
        photo_path = year_dir / cbtis / group / photo_name
        photos.append({"id": pid, "name": photo_name, "exists": photo_path.exists()})
    return jsonify({"ok": True, "photos": photos, "group": group})


@app.route("/api/fixes/source-image/<cbtis>/<group>/<photo_id>")
def api_fixes_source_image(cbtis, group, photo_id):
    """Serve a source photo, optionally scaled."""
    year_dir = get_year_dir()
    photo_path = year_dir / cbtis / group / f"IMG_{photo_id}.jpg"
    if not photo_path.exists():
        return jsonify({"error": "Not found"}), 404
    maxw = request.args.get("maxw", 99999, type=int)
    img = Image.open(str(photo_path))
    if img.width > maxw:
        ratio = maxw / img.width
        img = img.resize((maxw, int(img.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, "WEBP", quality=85)
    buf.seek(0)
    return send_file(buf, mimetype="image/webp")


@app.route("/api/fixes/composite-base/<cbtis>/<filename>")
def api_fixes_composite_base(cbtis, filename):
    """Render full composite: pano + sombras-below + cutout + borlas + hilos + togas + sombras-above."""
    import traceback
    try:
        return _fixes_composite_base_impl(cbtis, filename)
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "composite render failed", "detail": traceback.format_exc()}), 500

def _fixes_composite_base_impl(cbtis, filename):
    import cv2 as cv
    import math as _m
    import base64 as b64mod

    year_dir = get_year_dir()
    cut_path = year_dir / cbtis / "RECORTADAS" / filename
    if not cut_path.exists():
        return jsonify({"error": "Not found"}), 404

    maxw = request.args.get("maxw", 99999, type=int)

    # --- Load cutout ---
    cut_pil = Image.open(str(cut_path))
    if cut_pil.mode != "RGBA":
        cut_pil = cut_pil.convert("RGBA")
    orig_w, orig_h = cut_pil.size

    # --- Load states ---
    borlas_data, togas_data, sombras_list = [], [], []
    img_tf = {}
    toga_group_tf = {}
    hilo_color_hex, hilo_size, hilo_feather, hilo_flow = "#d4a017", 3, 1, 1.0

    borlas_sp = _borlas_state_path(cbtis, filename)
    if borlas_sp.exists():
        try:
            with open(borlas_sp) as f:
                bs = json.load(f)
            borlas_data = bs.get("borlas", [])
            hilo_color_hex = bs.get("hiloColor", "#d4a017")
            hilo_size = bs.get("hiloSize", 3)
            hilo_feather = bs.get("hiloFeather", 1)
            hilo_flow = bs.get("hiloFlow", 100) / 100.0
        except Exception:
            pass

    togas_sp = _togas_state_path(cbtis, filename)
    if togas_sp.exists():
        try:
            with open(togas_sp) as f:
                ts = json.load(f)
            togas_data = ts.get("togas", [])
            toga_group_tf = ts.get("groupTf", {})
            img_tf = ts.get("imgTf", {})
        except Exception:
            pass

    sombras_sp = _sombras_state_path(cbtis, filename)
    if sombras_sp.exists():
        try:
            with open(sombras_sp) as f:
                ss = json.load(f)
            sombras_list = ss.get("polygons", [])
        except Exception:
            pass

    # --- Apply imgTransform (rotate cutout + pano) ---
    img_rot = img_tf.get("rotation", 0)
    img_tx = img_tf.get("x", 0)
    img_ty = img_tf.get("y", 0)
    irad = _m.radians(img_rot)
    cos_r, sin_r = _m.cos(irad), _m.sin(irad)
    img_cx, img_cy = orig_w / 2.0, orig_h / 2.0
    exp_ox, exp_oy = 0.0, 0.0

    if abs(img_rot) > 0.1:
        cut_pil = cut_pil.rotate(-img_rot, expand=True, resample=Image.BICUBIC)
        new_w, new_h = cut_pil.size
        exp_ox = (new_w - orig_w) / 2.0
        exp_oy = (new_h - orig_h) / 2.0
    else:
        new_w, new_h = orig_w, orig_h

    canvas_w, canvas_h = new_w, new_h

    # --- Expand canvas to fit shadows that extend beyond image bounds ---
    shadow_expand_l, shadow_expand_t, shadow_expand_r, shadow_expand_b = 0, 0, 0, 0
    for sp in sombras_list:
        pts = sp.get("points", [])
        if len(pts) < 3:
            continue
        feather = sp.get("feather", {})
        mf = max(feather.get("top", 0), feather.get("bottom", 0),
                 feather.get("left", 0), feather.get("right", 0))
        pad = int(mf * 2 + 20)
        for pt in pts:
            px, py = pt.get("x", 0), pt.get("y", 0)
            # Relative to image bounds (after rotation expansion)
            if px - pad < -exp_ox:
                shadow_expand_l = max(shadow_expand_l, int(-exp_ox - (px - pad)))
            if py - pad < -exp_oy:
                shadow_expand_t = max(shadow_expand_t, int(-exp_oy - (py - pad)))
            if px + pad > new_w - exp_ox:
                shadow_expand_r = max(shadow_expand_r, int((px + pad) - (new_w - exp_ox)))
            if py + pad > new_h - exp_oy:
                shadow_expand_b = max(shadow_expand_b, int((py + pad) - (new_h - exp_oy)))
    if shadow_expand_l or shadow_expand_t or shadow_expand_r or shadow_expand_b:
        canvas_w += shadow_expand_l + shadow_expand_r
        canvas_h += shadow_expand_t + shadow_expand_b
        exp_ox += shadow_expand_l
        exp_oy += shadow_expand_t

    # --- Create composite canvas (transparent — no panorama, same as Sombras view) ---
    composite = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # --- Helper: render a layer image with transforms ---
    def _render_layer_pil(img_pil, bd, pivot_top=True):
        target_w = max(4, int(bd.get("scale", 50)))
        ratio = target_w / img_pil.width
        target_h = int(img_pil.height * ratio)
        resized = img_pil.resize((target_w, target_h), Image.LANCZOS)
        if bd.get("flipH"):
            resized = resized.transpose(Image.FLIP_LEFT_RIGHT)
        sx = bd.get("scaleX", 1.0)
        if abs(sx - 1.0) > 0.01:
            nw2 = max(4, int(resized.width * sx))
            resized = resized.resize((nw2, resized.height), Image.LANCZOS)
        mask_data = bd.get("mask")
        if mask_data and isinstance(mask_data, str) and mask_data.startswith("data:image/"):
            b64str = mask_data.split(",", 1)[1]
            mask_pil = Image.open(BytesIO(b64mod.b64decode(b64str))).convert("RGBA")
            mask_alpha = mask_pil.split()[3].resize(resized.size, Image.BICUBIC)
            arr = np.array(resized)
            marr = np.array(mask_alpha, dtype=np.float32) / 255.0
            arr[:, :, 3] = np.clip(arr[:, :, 3].astype(np.float32) * marr, 0, 255).astype(np.uint8)
            resized = Image.fromarray(arr)
        rotation = bd.get("rotation", 0)
        dx, dy = 0, 0
        if abs(rotation) > 0.5:
            bw, bh = resized.size
            pad = max(bw, bh)
            padded = Image.new("RGBA", (bw + pad * 2, bh + pad * 2), (0, 0, 0, 0))
            padded.paste(resized, (pad, pad), resized)
            if pivot_top:
                pivot_x, pivot_y = pad + bw // 2, pad
            else:
                pivot_x, pivot_y = pad + bw // 2, pad + bh // 2
            rotated = padded.rotate(-rotation, center=(pivot_x, pivot_y),
                                    expand=False, resample=Image.BICUBIC)
            bbox = rotated.getbbox()
            if bbox:
                rotated = rotated.crop(bbox)
                dx = bbox[0] - pad
                dy = bbox[1] - pad
            else:
                rotated = resized
            resized = rotated
        return resized, dx, dy

    # --- Helper: safe paste on composite ---
    def _paste(target, layer, px, py):
        """Paste layer onto target, handling out-of-bounds."""
        tw, th = target.size
        lw, lh = layer.size
        # Crop to canvas bounds
        sx = max(0, -px)
        sy = max(0, -py)
        ex = min(lw, tw - px)
        ey = min(lh, th - py)
        if ex <= sx or ey <= sy:
            return
        cropped = layer.crop((sx, sy, ex, ey))
        target.paste(cropped, (px + sx, py + sy), cropped)

    # --- Render sombras (shadows) ---
    def _render_shadow(sp):
        pts = sp.get("points", [])
        if len(pts) < 3:
            return None, None
        feather = sp.get("feather", {})
        opacity = sp.get("opacity", 0.5)
        hardness_val = sp.get("hardness", 30) / 100.0
        ft = feather.get("top", 0)
        fb = feather.get("bottom", 0)
        fl = feather.get("left", 0)
        fr = feather.get("right", 0)
        mf = max(ft, fb, fl, fr)
        # Rounded corners
        corner_r = mf * max(0.15, (1 - hardness_val) * 0.6)
        coords = [(int(p["x"]), int(p["y"])) for p in pts]
        if corner_r >= 1 and len(coords) >= 3:
            n = len(coords)
            rounded = []
            for i in range(n):
                prev = coords[(i-1) % n]
                curr = coords[i]
                nxt = coords[(i+1) % n]
                dx1 = prev[0]-curr[0]; dy1 = prev[1]-curr[1]
                dx2 = nxt[0]-curr[0]; dy2 = nxt[1]-curr[1]
                d1 = _m.sqrt(dx1*dx1+dy1*dy1)
                d2 = _m.sqrt(dx2*dx2+dy2*dy2)
                if d1 < 1 or d2 < 1:
                    rounded.append(curr); continue
                r = min(corner_r, d1*0.4, d2*0.4)
                t1x = curr[0]+dx1/d1*r; t1y = curr[1]+dy1/d1*r
                t2x = curr[0]+dx2/d2*r; t2y = curr[1]+dy2/d2*r
                steps = max(4, int(r/8))
                for s in range(steps+1):
                    t = s/steps
                    bx = (1-t)**2*t1x + 2*(1-t)*t*curr[0] + t*t*t2x
                    by = (1-t)**2*t1y + 2*(1-t)*t*curr[1] + t*t*t2y
                    rounded.append((int(round(bx)), int(round(by))))
            np_pts = np.array(rounded, dtype=np.int32)
        else:
            np_pts = np.array(coords, dtype=np.int32)
        # Tight bbox with padding for feather (allow negative coords for expansion)
        pad_f = int(mf * 2 + 10)
        xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
        bx1 = min(xs) - pad_f; by1 = min(ys) - pad_f
        bx2 = max(xs) + pad_f; by2 = max(ys) + pad_f
        bw, bh = bx2 - bx1, by2 - by1
        if bw <= 0 or bh <= 0:
            return None, None
        # Offset points to local bbox
        local_pts = np_pts - np.array([bx1, by1])
        mask = np.zeros((bh, bw), dtype=np.float32)
        cv.fillPoly(mask, [local_pts], 1.0, cv.LINE_AA)
        # 2-pass Gaussian blur (matches client CSS blur)
        if fl > 0 or fr > 0 or ft > 0 or fb > 0:
            sigma_boost = 1 + (1 - hardness_val) * 0.5
            h_sigma = max(fl, fr) * sigma_boost
            v_sigma = max(ft, fb) * sigma_boost
            if h_sigma > 0.5:
                kw = int(h_sigma * 6) | 1
                mask = cv.GaussianBlur(mask, (kw, 1), h_sigma)
            if v_sigma > 0.5:
                kh = int(v_sigma * 6) | 1
                mask = cv.GaussianBlur(mask, (1, kh), v_sigma)
        alpha = np.clip(mask * opacity * 255, 0, 255).astype(np.uint8)
        shadow_rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
        shadow_rgba[:, :, 3] = alpha
        return Image.fromarray(shadow_rgba), (bx1, by1)

    shadows_below, shadows_above = [], []
    for sp in sombras_list:
        img, pos = _render_shadow(sp)
        if img is not None and pos is not None:
            layer_pos = sp.get("layer", "below")
            if layer_pos == "below":
                shadows_below.append((img, pos))
            else:
                shadows_above.append((img, pos))

    # --- Composite: shadows below (offset for canvas expansion) ---
    for simg, (sx, sy) in shadows_below:
        _paste(composite, simg, sx + shadow_expand_l, sy + shadow_expand_t)

    # --- Composite: cutout (offset for canvas expansion) ---
    _paste(composite, cut_pil, shadow_expand_l, shadow_expand_t)

    # --- Composite: borlas ---
    for bd in borlas_data:
        if not bd.get("visible", True):
            continue
        color = bd.get("color", "DORADO")
        bpath = BORLAS_DIR / f"B_{color}.webp"
        if not bpath.exists():
            continue
        borla_img = Image.open(str(bpath)).convert("RGBA")
        # Transform borla position with imgTransform
        bx, by = bd.get("x", 0), bd.get("y", 0)
        b_rot = bd.get("rotation", 0)
        if abs(img_rot) > 0.1:
            dx, dy = bx - img_cx, by - img_cy
            bx = img_cx + dx * cos_r - dy * sin_r + exp_ox
            by = img_cy + dx * sin_r + dy * cos_r + exp_oy
            b_rot += img_rot
        else:
            bx += exp_ox
            by += exp_oy
        bd_tf = dict(bd, x=bx, y=by, rotation=b_rot)
        resized, dx, dy = _render_layer_pil(borla_img, bd_tf)
        px = int(bx) - resized.width // 2 + dx
        py = int(by) + dy
        _paste(composite, resized, px, py)

    # --- Composite: hilos ---
    hc = hilo_color_hex.lstrip('#')
    hilo_rgb = tuple(int(hc[i:i+2], 16) for i in (0, 2, 4)) if len(hc) >= 6 else (212, 160, 23)
    for bd in borlas_data:
        if not bd.get("visible", True):
            continue
        hilo_pts = bd.get("hilo")
        if not hilo_pts or len(hilo_pts) < 2:
            continue
        bx_c, by_c = bd.get("x", 0), bd.get("y", 0)
        b_rot = bd.get("rotation", 0)
        if abs(img_rot) > 0.1:
            dx, dy = bx_c - img_cx, by_c - img_cy
            bx_c = img_cx + dx * cos_r - dy * sin_r + exp_ox
            by_c = img_cy + dx * sin_r + dy * cos_r + exp_oy
            b_rot += img_rot
        else:
            bx_c += exp_ox
            by_c += exp_oy
        b_rad = _m.radians(b_rot)
        b_cos, b_sin = _m.cos(b_rad), _m.sin(b_rad)
        canvas_pts = []
        for p in hilo_pts:
            rx, ry = p["x"], p["y"]
            if abs(b_rot) > 0.1:
                rx2 = rx * b_cos - ry * b_sin
                ry2 = rx * b_sin + ry * b_cos
                rx, ry = rx2, ry2
            canvas_pts.append((int(bx_c + rx), int(by_c + ry)))
        if len(canvas_pts) < 2:
            continue
        cxs = [p[0] for p in canvas_pts]; cys = [p[1] for p in canvas_pts]
        pad_px = hilo_size + hilo_feather * 2 + 6
        lx1 = max(0, min(cxs) - pad_px); ly1 = max(0, min(cys) - pad_px)
        lx2 = min(canvas_w, max(cxs) + pad_px); ly2 = min(canvas_h, max(cys) + pad_px)
        if lx2 <= lx1 or ly2 <= ly1:
            continue
        lw_, lh_ = lx2 - lx1, ly2 - ly1
        hilo_canvas = np.zeros((lh_, lw_, 4), dtype=np.uint8)
        local_pts = np.array([(x - lx1, y - ly1) for x, y in canvas_pts], dtype=np.int32)
        cv.polylines(hilo_canvas, [local_pts], False,
                     (hilo_rgb[0], hilo_rgb[1], hilo_rgb[2], 255),
                     thickness=max(1, hilo_size), lineType=cv.LINE_AA)
        if hilo_feather > 0.5:
            alpha_f = hilo_canvas[:, :, 3].astype(np.float32)
            alpha_f = gaussian_filter(alpha_f, sigma=hilo_feather)
            mx = alpha_f.max()
            if mx > 0:
                alpha_f = alpha_f / mx * 255
            hilo_canvas[:, :, 3] = np.clip(alpha_f, 0, 255).astype(np.uint8)
        if hilo_flow < 0.99:
            hilo_canvas[:, :, 3] = np.clip(
                hilo_canvas[:, :, 3].astype(np.float32) * hilo_flow, 0, 255).astype(np.uint8)
        _paste(composite, Image.fromarray(hilo_canvas), lx1, ly1)

    # --- Composite: togas (with groupTf — matching JS shDraw logic) ---
    # In JS shDraw, togas are drawn OUTSIDE the imgTransform group.
    # GroupTf rotation is orbital around (imgCx + gx, imgCy + gy).
    # Individual toga delta from center is (t.x - imgCx, t.y - imgCy).
    gx = toga_group_tf.get("x", 0)
    gy_tf = toga_group_tf.get("y", 0)
    g_rot = toga_group_tf.get("rotation", 0)
    g_scaleY = toga_group_tf.get("scaleY", 1.0)
    g_rad = _m.radians(g_rot)
    g_cos, g_sin = _m.cos(g_rad), _m.sin(g_rad)
    # Load all toga images and compute uniform ref_ar (average, like JS shTogaRefAR)
    toga_imgs_cache = {}
    toga_ars = []
    for td in togas_data:
        variant = td.get("variant", "")
        if variant not in toga_imgs_cache:
            tpath = CAIDA_DIR / f"{variant}.webp"
            if tpath.exists():
                toga_imgs_cache[variant] = Image.open(str(tpath)).convert("RGBA")
        toga_src = toga_imgs_cache.get(variant)
        if toga_src:
            toga_ars.append(toga_src.height / toga_src.width)
    uniform_ref_ar = sum(toga_ars) / len(toga_ars) if toga_ars else 1.5

    for td in togas_data:
        toga_src = toga_imgs_cache.get(td.get("variant", ""))
        if not toga_src:
            continue
        # Position: match JS shDraw transform exactly
        # JS: delta = (t.x - imgW/2, t.y - imgH/2), rotated around (imgCx+gx, imgCy+gy)
        # final = rotCenter + Rotate(delta) = (imgCx+gx, imgCy+gy) + R(t.x-imgCx, t.y-imgCy)
        tx = td.get("x", 0)
        ty = td.get("y", 0)
        dx_ = tx - img_cx
        dy_ = ty - img_cy
        if abs(g_rad) > 0.001:
            rx = dx_ * g_cos - dy_ * g_sin
            ry = dx_ * g_sin + dy_ * g_cos
        else:
            rx, ry = dx_, dy_
        rot_cx = img_cx + gx
        rot_cy = img_cy + gy_tf
        px = rot_cx + rx
        py = rot_cy + ry
        # Only add canvas expansion offset (NO imgTransform rotation/translation)
        px += exp_ox
        py += exp_oy
        t_rot = td.get("rotation", 0) + g_rot
        # Uniform height: resize source to (width, width * uniform_ref_ar * scaleY)
        # Then _render_layer_pil scales to target_w=scale, preserving aspect ratio
        # Result: final_w=scale, final_h = scale * uniform_ref_ar * scaleY (matches JS)
        target_h = max(4, int(toga_src.width * uniform_ref_ar * g_scaleY))
        if target_h != toga_src.height:
            toga_for_render = toga_src.resize((toga_src.width, target_h), Image.LANCZOS)
        else:
            toga_for_render = toga_src.copy()
        td_tf = {
            "scale": td.get("scale", 50), "scaleX": td.get("scaleX", 1.0),
            "flipH": td.get("flipH", False), "rotation": t_rot,
            "mask": td.get("mask"), "x": px, "y": py,
        }
        resized, dx, dy = _render_layer_pil(toga_for_render, td_tf)
        paste_x = int(px) - resized.width // 2 + dx
        paste_y = int(py) + dy
        _paste(composite, resized, paste_x, paste_y)

    # --- Composite: shadows above (offset for canvas expansion) ---
    for simg, (sx, sy) in shadows_above:
        _paste(composite, simg, sx + shadow_expand_l, sy + shadow_expand_t)

    # Scale down if needed
    if composite.width > maxw:
        ratio = maxw / composite.width
        composite = composite.resize((maxw, int(composite.height * ratio)), Image.LANCZOS)

    buf = BytesIO()
    if composite.width <= 800:
        composite.save(buf, "WEBP", quality=90)
    else:
        composite.save(buf, "WEBP", quality=85)
    buf.seek(0)
    return send_file(buf, mimetype="image/webp")


@app.route("/api/fixes/extract/<cbtis>/<filename>", methods=["POST"])
def api_fixes_extract(cbtis, filename):
    """Extract a selected region from a source photo as RGBA with feathered mask."""
    import traceback
    try:
        return _fixes_extract_impl(cbtis, filename)
    except Exception:
        traceback.print_exc()
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500

def _fixes_extract_impl(cbtis, filename):
    import cv2 as cv
    from scipy.ndimage import gaussian_filter

    data = request.get_json()
    if not data:
        return jsonify({"ok": False, "error": "No data"}), 400

    photo_id = data.get("photo_id", "")
    group = data.get("group", "")
    strokes = data.get("strokes", [])
    if not photo_id or not group or not strokes:
        return jsonify({"ok": False, "error": "Missing photo_id, group, or strokes"}), 400

    year_dir = get_year_dir()
    photo_path = year_dir / cbtis / group / f"IMG_{photo_id}.jpg"
    if not photo_path.exists():
        return jsonify({"ok": False, "error": "Source photo not found"}), 404

    img = Image.open(str(photo_path)).convert("RGB")
    # Apply rotation if provided
    src_rotation = data.get("rotation", 0)
    if src_rotation == 90:
        img = img.transpose(Image.ROTATE_270)
    elif src_rotation == 180:
        img = img.transpose(Image.ROTATE_180)
    elif src_rotation == 270:
        img = img.transpose(Image.ROTATE_90)
    w, h = img.size
    img_np = np.array(img, dtype=np.uint8)

    # Build selection mask from strokes
    mask = np.zeros((h, w), dtype=np.uint8)
    for stroke in strokes:
        mode = stroke.get("mode", "restore")
        pts = [(int(p["x"] * w), int(p["y"] * h)) for p in stroke.get("points", [])]
        if not pts:
            continue
        if stroke.get("type") == "lasso":
            pts_arr = np.array(pts, dtype=np.int32)
            if mode == "erase":
                cv.fillPoly(mask, [pts_arr], 0, cv.LINE_AA)
            else:
                cv.fillPoly(mask, [pts_arr], 255, cv.LINE_AA)
        else:
            radius = max(1, int(stroke.get("radius", 0.01) * max(w, h)))
            val = 0 if mode == "erase" else 255
            for pt in pts:
                cv.circle(mask, pt, radius, val, -1, cv.LINE_AA)
            if len(pts) > 1:
                for j in range(len(pts) - 1):
                    cv.line(mask, pts[j], pts[j + 1], val, radius * 2, cv.LINE_AA)

    # Feather edges
    feather_sigma = max(1.0, min(w, h) / 1500.0)
    mask_f = gaussian_filter(mask.astype(np.float32), sigma=feather_sigma)
    mask_u8 = np.clip(mask_f, 0, 255).astype(np.uint8)

    # Find bounding box of mask
    ys, xs = np.where(mask_u8 > 0)
    if len(ys) == 0:
        return jsonify({"ok": False, "error": "Empty selection"}), 400

    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    # Create RGBA crop
    crop_rgb = img_np[y1:y2, x1:x2]
    crop_alpha = mask_u8[y1:y2, x1:x2]
    crop_rgba = np.dstack([crop_rgb, crop_alpha])

    # Save as WebP lossless
    extract_img = Image.fromarray(crop_rgba, "RGBA")

    # Save extraction to disk for later use in TIFF export
    fix_dir = get_year_dir() / cbtis / "RECORTADAS"
    stem = os.path.splitext(filename)[0]
    fix_id = data.get("fix_id", 0)
    fix_path = fix_dir / f"{stem}.fix_{fix_id}.webp"
    extract_img.save(str(fix_path), "WEBP", lossless=True)

    buf = BytesIO()
    extract_img.save(buf, "WEBP", lossless=True)
    buf.seek(0)

    # Return image + metadata
    from flask import make_response
    resp = make_response(send_file(buf, mimetype="image/webp"))
    resp.headers["X-Fix-Bbox"] = json.dumps({
        "x": x1 / w, "y": y1 / h,
        "w": (x2 - x1) / w, "h": (y2 - y1) / h,
        "px": x1, "py": y1, "pw": x2 - x1, "ph": y2 - y1,
        "src_w": w, "src_h": h
    })
    return resp


def _fixes_state_path(cbtis, filename):
    stem = os.path.splitext(filename)[0]
    return get_year_dir() / cbtis / "RECORTADAS" / f"{stem}.fixes.json"


@app.route("/api/fixes/state/<cbtis>/<filename>")
def api_fixes_state_get(cbtis, filename):
    sp = _fixes_state_path(cbtis, filename)
    if sp.exists():
        with open(sp) as f:
            return jsonify(json.load(f))
    return jsonify({"fixes": []})


@app.route("/api/fixes/state/<cbtis>/<filename>", methods=["POST"])
def api_fixes_state_post(cbtis, filename):
    data = request.get_json()
    sp = _fixes_state_path(cbtis, filename)
    with open(sp, "w") as f:
        json.dump(data, f)
    return jsonify({"ok": True})


# --- HTML Template ---

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARCA Panorama</title>
<style>
  :root {
    --bg: #1a1b26; --bg2: #24283b; --bg3: #2f3348;
    --fg: #c0caf5; --fg2: #a9b1d6; --fg3: #565f89;
    --red: #f7768e; --green: #9ece6a; --yellow: #e0af68;
    --cyan: #7dcfff; --blue: #7aa2f7; --purple: #bb9af7;
    --border: #3b4261;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--fg);
    display: flex; flex-direction: column; height: 100vh;
  }

  /* Header */
  header {
    background: var(--bg2); border-bottom: 1px solid var(--border);
    padding: 10px 20px; display: flex; align-items: center; gap: 12px;
    flex-shrink: 0;
  }
  header h1 { font-size: 18px; font-weight: 600; color: var(--blue); letter-spacing: 1px; }
  header .subtitle { font-size: 12px; color: var(--fg3); }
  header .status {
    margin-left: auto; padding: 4px 12px; border-radius: 12px;
    font-size: 12px; font-weight: 600;
  }
  .status.idle { background: var(--bg3); color: var(--fg3); }
  .status.running { background: #9ece6a22; color: var(--green); animation: pulse 1.5s infinite; }
  @keyframes pulse { 50% { opacity: 0.6; } }

  /* Path bar */
  .path-bar {
    padding: 6px 20px; background: var(--bg2); border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 8px; flex-shrink: 0;
  }
  .path-bar label { font-size: 11px; color: var(--fg3); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  .path-bar input {
    flex: 1; background: var(--bg); border: 1px solid var(--border); color: var(--fg);
    padding: 4px 10px; border-radius: 4px; font-size: 13px;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
  }
  .path-bar input:focus { outline: none; border-color: var(--blue); }
  .path-bar button {
    padding: 4px 12px; border-radius: 4px; border: 1px solid var(--border);
    background: var(--bg3); color: var(--fg); font-size: 12px; cursor: pointer;
  }
  .path-bar button:hover { border-color: var(--blue); color: var(--blue); }
  .path-bar .hint { font-size: 10px; color: var(--fg3); }

  /* Main layout */
  .main { display: flex; flex: 1; overflow: hidden; }

  /* Sidebar */
  .sidebar {
    width: 340px; min-width: 200px; max-width: 60vw; background: var(--bg2);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column; overflow: hidden;
    position: relative; flex-shrink: 0;
  }
  .sidebar-resize {
    position: absolute; top: 0; right: -3px; width: 6px; height: 100%;
    cursor: col-resize; z-index: 10;
  }
  .sidebar-resize:hover, .sidebar-resize.active {
    background: var(--blue); opacity: 0.5;
  }
  .sidebar-header {
    padding: 10px 16px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 8px;
  }
  .sidebar-header h2 { font-size: 14px; color: var(--fg2); }
  .sidebar-count {
    font-size: 11px; background: var(--bg3); color: var(--fg3);
    padding: 2px 8px; border-radius: 10px;
  }
  .group-list { flex: 1; overflow-y: auto; padding: 8px; }
  .cbtis-section { margin-bottom: 12px; }
  .cbtis-label {
    font-size: 11px; font-weight: 700; color: var(--purple);
    text-transform: uppercase; letter-spacing: 1px;
    padding: 4px 8px; margin-bottom: 4px;
    display: flex; align-items: center; gap: 8px;
  }
  .cbtis-actions { margin-left: auto; }
  .cbtis-actions button {
    font-size: 10px; padding: 2px 8px; border-radius: 4px;
    background: var(--blue); color: var(--bg); border: none;
    cursor: pointer; font-weight: 600;
  }
  .cbtis-actions button:hover { opacity: 0.8; }

  .group-item {
    display: flex; align-items: center; padding: 6px 8px;
    border-radius: 6px; cursor: pointer; gap: 8px; font-size: 13px;
    transition: background 0.15s;
  }
  .group-item:hover { background: var(--bg3); }
  .group-item.selected { background: var(--bg3); outline: 1px solid var(--blue); }
  .group-item .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .dot.done { background: var(--green); }
  .dot.pending { background: var(--yellow); }
  .group-item .name { flex: 1; color: var(--fg2); font-family: monospace; font-size: 11px; }
  .group-item .photos { color: var(--fg3); font-size: 11px; white-space: nowrap; }
  .group-item .run-btn {
    opacity: 0; border: none; background: var(--blue); color: var(--bg);
    font-size: 10px; padding: 2px 8px; border-radius: 4px;
    cursor: pointer; font-weight: 600; transition: opacity 0.15s;
  }
  .group-item:hover .run-btn { opacity: 1; }
  /* Status badges */
  .grp-badges { display: flex; gap: 2px; align-items: center; flex-shrink: 0; }
  .grp-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 16px; height: 16px; border-radius: 3px; font-size: 9px;
    font-weight: 700; line-height: 1; border: 1px solid transparent;
    cursor: pointer; transition: transform 0.1s, filter 0.1s;
  }
  .grp-badge:not(.off):hover { transform: scale(1.25); filter: brightness(1.3); }
  .grp-badge.off { background: var(--bg3); color: var(--fg3); opacity: 0.4; cursor: default; }
  .grp-badge.pano { background: rgba(122,162,247,0.2); color: var(--blue); border-color: rgba(122,162,247,0.3); }
  .grp-badge.recorte { background: rgba(187,154,247,0.2); color: var(--purple); border-color: rgba(187,154,247,0.3); }
  .grp-badge.borlas { background: rgba(125,207,255,0.2); color: var(--cyan); border-color: rgba(125,207,255,0.3); }
  .grp-badge.togas { background: rgba(224,175,104,0.2); color: var(--yellow); border-color: rgba(224,175,104,0.3); }
  .grp-badge.sombras { background: rgba(86,95,137,0.2); color: var(--fg3); border-color: rgba(86,95,137,0.3); }
  .grp-badge.fixes { background: rgba(187,154,247,0.2); color: #bb9af7; border-color: rgba(187,154,247,0.3); }
  .grp-badge.pending { border-style: dashed; opacity: 0.7; }
  .grp-badge.done { position: relative; }
  .grp-badge.done::after {
    content: ''; position: absolute; bottom: -2px; right: -2px;
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--green); border: 1px solid var(--bg1);
  }
  /* Done toggle button in toolbars */
  .rf-btn.done-toggle { border-style: dashed; }
  .rf-btn.done-toggle.is-done { border-color: var(--green); color: var(--green); background: rgba(158,206,106,0.15); border-style: solid; }

  /* Content area */
  .content { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

  /* Tabs */
  .tabs {
    display: flex; background: var(--bg2); border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .tab {
    padding: 8px 16px; font-size: 13px; color: var(--fg3);
    cursor: pointer; border-bottom: 2px solid transparent;
    transition: all 0.15s;
  }
  .tab:hover { color: var(--fg2); }
  .tab.active { color: var(--blue); border-bottom-color: var(--blue); }

  /* Toolbar */
  .toolbar {
    padding: 8px 16px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 8px; flex-shrink: 0;
    background: var(--bg2);
  }
  .toolbar button {
    padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--bg3); color: var(--fg); font-size: 13px;
    cursor: pointer; transition: all 0.15s; font-weight: 500;
  }
  .toolbar button:hover { border-color: var(--blue); color: var(--blue); }
  .toolbar button.primary {
    background: var(--blue); color: var(--bg); border-color: var(--blue); font-weight: 600;
  }
  .toolbar button.primary:hover { opacity: 0.85; }
  .toolbar button.danger {
    background: transparent; border-color: var(--red); color: var(--red);
  }
  .toolbar button.danger:hover { background: #f7768e22; }
  .toolbar button:disabled { opacity: 0.3; cursor: not-allowed; }
  .toolbar .spacer { flex: 1; }
  .filter-group { display: flex; align-items: center; gap: 4px; }
  .filter-group label { font-size: 12px; color: var(--fg3); cursor: pointer; }
  .filter-group input[type="checkbox"] { accent-color: var(--blue); }

  /* Log viewer */
  .tab-content { display: none; flex: 1; flex-direction: column; overflow: hidden; }
  .tab-content.active { display: flex; }
  .log-container { flex: 1; overflow: hidden; position: relative; }
  .log-view {
    height: 100%; overflow-y: auto; padding: 12px 16px;
    font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace;
    font-size: 13px; line-height: 1.6;
  }
  .log-line { white-space: pre-wrap; word-break: break-all; }
  .log-line.info { color: var(--cyan); }
  .log-line.ok { color: var(--green); }
  .log-line.warn { color: var(--yellow); }
  .log-line.error { color: var(--red); font-weight: 600; }
  .log-empty {
    position: absolute; inset: 0; display: flex;
    align-items: center; justify-content: center;
    color: var(--fg3); font-size: 14px; flex-direction: column; gap: 8px;
  }

  /* Preview panel */
  .preview-panel {
    flex: 1; overflow: hidden; padding: 0;
    display: flex; flex-direction: column; align-items: center;
  }
  .preview-controls {
    padding: 8px 16px; display: flex; gap: 8px; align-items: center;
    flex-shrink: 0; width: 100%; border-bottom: 1px solid var(--border);
    background: var(--bg2);
  }
  .preview-controls .zoom-label { font-size: 11px; color: var(--fg3); font-family: monospace; min-width: 50px; }
  .preview-controls button {
    padding: 2px 10px; border-radius: 4px; border: 1px solid var(--border);
    background: var(--bg3); color: var(--fg); font-size: 12px; cursor: pointer;
  }
  .preview-controls button:hover { border-color: var(--blue); color: var(--blue); }
  .preview-viewport {
    flex: 1; overflow: hidden; position: relative; cursor: grab;
    width: 100%; touch-action: none;
  }
  .preview-viewport.dragging { cursor: grabbing; }
  .preview-viewport img {
    position: absolute; transform-origin: 0 0;
    border-radius: 4px; user-select: none; -webkit-user-drag: none;
  }
  .preview-panel .no-preview { color: var(--fg3); font-size: 14px; margin-top: 40px; }
  .preview-info {
    font-size: 12px; color: var(--fg3); text-align: center;
    font-family: monospace; padding: 6px; flex-shrink: 0;
  }

  /* Refine mode */
  .refine-toolbar {
    padding: 6px 12px; display: flex; align-items: center; gap: 6px;
    background: var(--bg2); border-bottom: 1px solid var(--border);
    flex-wrap: wrap; flex-shrink: 0; align-self: stretch;
  }
  .refine-toolbar .rf-sep {
    width: 1px; height: 24px; background: var(--border); margin: 0 4px;
  }
  .rf-btn {
    padding: 4px 10px; border: 1px solid var(--border); border-radius: 4px;
    background: var(--bg3); color: var(--fg); cursor: pointer; font-size: 12px;
    display: flex; align-items: center; gap: 4px; white-space: nowrap;
  }
  .rf-btn:hover { background: var(--border); }
  .rf-btn.active { border-color: var(--blue); background: rgba(122,162,247,0.15); color: var(--blue); }
  .rf-btn.restore.active { border-color: var(--green); background: rgba(158,206,106,0.15); color: var(--green); }
  .rf-btn.erase.active { border-color: var(--red); background: rgba(247,118,142,0.15); color: var(--red); }
  .rf-btn.primary { background: var(--blue); color: #fff; border-color: var(--blue); }
  .rf-btn.primary:hover { background: #6b92e0; }
  .rf-btn.danger { background: transparent; color: var(--red); border-color: var(--red); }
  .rf-size-slider {
    width: 100px; accent-color: var(--blue); cursor: pointer;
  }
  .rf-size-label {
    font-size: 11px; color: var(--fg3); min-width: 30px;
  }
  .rf-canvas-wrap {
    flex: 1; position: relative; overflow: hidden; cursor: crosshair;
    background: repeating-conic-gradient(#3a3a4a 0% 25%, #2a2a3a 0% 50%) 50% / 20px 20px;
    align-self: stretch; touch-action: none;
  }
  .rf-canvas-wrap.panning { cursor: grab; }
  .rf-canvas-wrap.dragging { cursor: grabbing; }
  .rf-canvas-wrap canvas {
    position: absolute; top: 0; left: 0; transform-origin: 0 0;
    image-rendering: auto;
  }
  .rf-loading {
    position: absolute; inset: 0; display: flex; align-items: center;
    justify-content: center; background: rgba(0,0,0,0.5); color: var(--fg);
    font-size: 16px; z-index: 10;
  }
  .preview-panel:fullscreen, .preview-panel.faux-fullscreen {
    background: var(--bg1); display: flex; flex-direction: column;
    width: 100% !important; height: 100% !important;
  }
  .preview-panel:fullscreen .rf-canvas-wrap, .preview-panel.faux-fullscreen .rf-canvas-wrap,
  .preview-panel:fullscreen .tg-list, .preview-panel.faux-fullscreen .tg-list {
    flex: 1;
  }
  .preview-panel.faux-fullscreen {
    position: fixed !important; top: 0; left: 0;
    width: 100vw !important; height: 100vh !important;
    z-index: 9999;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--bg3); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--border); }

  /* Summary bar */
  .summary-bar {
    padding: 6px 16px; border-top: 1px solid var(--border);
    background: var(--bg2); font-size: 12px; color: var(--fg3);
    display: flex; gap: 16px; flex-shrink: 0;
  }
  .summary-bar span { display: flex; align-items: center; gap: 4px; }
  .summary-bar .cnt { font-weight: 600; }
  .summary-bar .cnt.green { color: var(--green); }
  .summary-bar .cnt.yellow { color: var(--yellow); }
  .summary-bar .cnt.red { color: var(--red); }

  /* Toast */
  .toast {
    position: fixed; bottom: 60px; right: 20px; padding: 10px 20px;
    background: var(--bg3); border: 1px solid var(--border); border-radius: 8px;
    color: var(--fg); font-size: 13px; opacity: 0; transition: opacity 0.3s;
    z-index: 100; pointer-events: none;
  }
  .toast.show { opacity: 1; }
  .toast.error { border-color: var(--red); color: var(--red); }
  .toast.success, .toast.ok { border-color: var(--green); color: var(--green); }
  .toast.warn { border-color: var(--yellow); color: var(--yellow); }

  /* Search */
  .search-box {
    padding: 6px 16px; border-bottom: 1px solid var(--border);
  }
  .search-box input {
    width: 100%; background: var(--bg); border: 1px solid var(--border);
    color: var(--fg); padding: 4px 10px; border-radius: 4px; font-size: 12px;
  }
  .search-box input:focus { outline: none; border-color: var(--blue); }
  .search-box input::placeholder { color: var(--fg3); }

  /* Selection bar */
  .selection-bar {
    padding: 4px 8px; display: flex; gap: 4px;
    border-bottom: 1px solid var(--border);
  }
  .selection-bar button {
    font-size: 10px; padding: 2px 8px; border-radius: 4px;
    background: var(--bg3); color: var(--fg3); border: 1px solid var(--border);
    cursor: pointer;
  }
  .selection-bar button:hover { color: var(--fg); border-color: var(--blue); }

  /* Checkbox in group items */
  .group-item .sel-check {
    width: 14px; height: 14px; accent-color: var(--blue);
    cursor: pointer; flex-shrink: 0;
  }
  .group-item.checked { background: #7aa2f711; }

  /* Organize tab */
  .org-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; padding: 16px; overflow-y: auto; }
  .org-card {
    background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; display: flex; flex-direction: column; gap: 8px;
  }
  .org-card h3 { font-size: 15px; color: var(--fg); font-weight: 600; }
  .org-card .org-badge {
    display: inline-block; padding: 2px 10px; border-radius: 10px;
    font-size: 11px; font-weight: 600; width: fit-content;
  }
  .org-badge.done { background: #9ece6a22; color: var(--green); }
  .org-badge.pending { background: #e0af6822; color: var(--yellow); }
  .org-card .org-stats { font-size: 12px; color: var(--fg3); display: flex; flex-direction: column; gap: 2px; }
  .org-card .org-actions { display: flex; gap: 6px; margin-top: 4px; }
  .org-card .org-actions button {
    padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--bg3); color: var(--fg); font-size: 12px;
    cursor: pointer; font-weight: 500;
  }
  .org-card .org-actions button:hover { border-color: var(--blue); color: var(--blue); }
  .org-card .org-actions button.primary {
    background: var(--blue); color: var(--bg); border-color: var(--blue);
  }
  .org-card .org-actions button.primary:hover { opacity: 0.85; }
  .org-card .org-actions button:disabled { opacity: 0.3; cursor: not-allowed; }

  .org-preview { padding: 16px; overflow-y: auto; }
  .org-preview h3 { font-size: 15px; margin-bottom: 12px; color: var(--fg); }
  .org-preview table {
    width: 100%; border-collapse: collapse; font-size: 12px;
    font-family: monospace;
  }
  .org-preview th {
    text-align: left; padding: 6px 10px; background: var(--bg3);
    color: var(--fg3); font-weight: 600; border-bottom: 1px solid var(--border);
  }
  .org-preview td { padding: 6px 10px; border-bottom: 1px solid var(--border); color: var(--fg2); }
  .org-preview .missing { color: var(--red); }
  .org-preview .extra { color: var(--yellow); }
  .org-preview .ok { color: var(--green); }
  .org-summary {
    margin: 12px 0; padding: 10px 16px; background: var(--bg3);
    border-radius: 6px; font-size: 13px; display: flex; gap: 20px;
  }
  .org-summary span { display: flex; gap: 4px; }
  .org-back-btn {
    padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--bg3); color: var(--fg); font-size: 12px;
    cursor: pointer; margin-bottom: 12px;
  }
  .org-back-btn:hover { border-color: var(--blue); color: var(--blue); }
  .org-confirm-bar {
    margin-top: 12px; padding: 12px; background: var(--bg3);
    border-radius: 6px; display: flex; align-items: center; gap: 12px;
  }
  .org-confirm-bar button {
    padding: 8px 20px; border-radius: 6px; border: none;
    background: var(--green); color: var(--bg); font-size: 13px;
    font-weight: 600; cursor: pointer;
  }
  .org-confirm-bar button:hover { opacity: 0.85; }
  .org-confirm-bar button:disabled { opacity: 0.3; cursor: not-allowed; }
  .org-confirm-bar .warn { font-size: 12px; color: var(--yellow); }
  .org-result { margin-top: 12px; padding: 12px; border-radius: 6px; font-size: 13px; }
  .org-result.success { background: #9ece6a22; color: var(--green); }
  .org-result.error { background: #f7768e22; color: var(--red); }

  .cbtis-actions .cut-cbtis-btn {
    font-size: 10px; padding: 2px 8px; border-radius: 4px;
    background: var(--yellow); color: var(--bg); border: none;
    cursor: pointer; font-weight: 600; margin-left: 2px;
  }
  .cbtis-actions .cut-cbtis-btn:hover { opacity: 0.8; }

  /* Preview toggle buttons */
  .preview-toggle {
    display: flex; gap: 6px; margin-bottom: 8px;
  }
  .preview-toggle button {
    padding: 4px 14px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--bg3); color: var(--fg); font-size: 12px;
    cursor: pointer; font-weight: 500;
  }
  .preview-toggle button.active { background: var(--blue); color: var(--bg); border-color: var(--blue); }
  .preview-toggle button:hover { border-color: var(--blue); }

  /* --- Borlas Mode --- */
  .borlas-toolbar {
    display: flex; align-items: center; gap: 6px; padding: 6px 10px;
    background: var(--bg2); border-bottom: 1px solid var(--border);
    flex-wrap: wrap; flex-shrink: 0;
  }
  .borlas-toolbar label { font-size: 12px; color: var(--fg3); white-space: nowrap; }
  .borlas-toolbar select {
    background: var(--bg3); color: var(--fg); border: 1px solid var(--border);
    border-radius: 4px; padding: 3px 6px; font-size: 12px;
  }
  .borlas-toolbar input[type="number"] {
    background: var(--bg3); color: var(--fg); border: 1px solid var(--border);
    border-radius: 4px; padding: 3px 4px; font-size: 12px; width: 48px; text-align: center;
  }
  .bl-sep { width: 1px; height: 22px; background: var(--border); margin: 0 4px; flex-shrink: 0; }
  .bl-list {
    display: flex; gap: 4px; padding: 6px 10px; background: var(--bg1);
    border-top: 1px solid var(--border); overflow-x: auto; flex-shrink: 0;
  }
  .bl-item {
    display: flex; align-items: center; gap: 4px; padding: 3px 8px;
    background: var(--bg3); border: 1px solid var(--border); border-radius: 4px;
    font-size: 11px; white-space: nowrap; cursor: pointer;
  }
  .bl-item.selected { border-color: var(--blue); background: #7aa2f722; }
  .bl-item.hidden { opacity: 0.35; text-decoration: line-through; }
  .bl-item button {
    background: none; border: none; color: var(--fg3); cursor: pointer;
    padding: 0 2px; font-size: 13px;
  }
  .bl-item button:hover { color: var(--fg); }
  .bl-color-dot {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    border: 1px solid var(--border); vertical-align: middle; margin-right: 3px;
  }
  /* --- Togas --- */
  .togas-toolbar {
    display: flex; align-items: center; gap: 6px; padding: 6px 10px;
    background: var(--bg2); border-bottom: 1px solid var(--border);
    flex-wrap: wrap; flex-shrink: 0;
  }
  .togas-toolbar label { font-size: 12px; color: var(--fg3); white-space: nowrap; }
  .togas-toolbar input[type="number"] {
    background: var(--bg3); color: var(--fg); border: 1px solid var(--border);
    border-radius: 4px; padding: 3px 4px; font-size: 12px; width: 48px; text-align: center;
  }
  .tg-sep { width: 1px; height: 22px; background: var(--border); margin: 0 4px; flex-shrink: 0; }
  .tg-list {
    display: flex; gap: 4px; padding: 6px 10px; background: var(--bg1);
    border-top: 1px solid var(--border); overflow-x: auto; flex-shrink: 0;
  }
  .tg-item {
    display: flex; align-items: center; gap: 4px; padding: 3px 8px;
    background: var(--bg3); border: 1px solid var(--border); border-radius: 4px;
    font-size: 11px; white-space: nowrap; cursor: pointer;
  }
  .tg-item.selected { border-color: var(--cyan); background: #7dcfff22; }
  .tg-item button {
    background: none; border: none; color: var(--fg3); cursor: pointer;
    padding: 0 2px; font-size: 13px;
  }
  .tg-item button:hover { color: var(--fg); }
  .tg-guide-line {
    position: absolute; left: 0; right: 0; height: 2px;
    background: var(--cyan); opacity: 0.6; cursor: ns-resize; z-index: 5;
  }
  .tg-guide-line:hover { opacity: 1; height: 3px; }
  .tg-seated-marker {
    position: absolute; width: 8px; height: 8px; border-radius: 50%;
    background: var(--yellow); border: 2px solid var(--bg1);
    transform: translate(-50%, -50%); pointer-events: none; z-index: 4;
  }
  /* --- Fixes section --- */
  .fx-source-thumb { width:56px; height:38px; object-fit:cover; border-radius:4px; cursor:pointer; border:2px solid var(--bg3); transition:border-color .15s; }
  .fx-source-thumb:hover { border-color: var(--fg3); }
  .fx-source-thumb.active { border-color: #bb9af7; box-shadow: 0 0 4px #bb9af7; }
  .fx-layer-item { display:flex; align-items:center; gap:4px; padding:3px 5px; font-size:11px; border-bottom:1px solid var(--bg3); cursor:pointer; }
  .fx-layer-item:hover { background: rgba(187,154,247,0.08); }
  .fx-layer-item.selected { background: rgba(187,154,247,0.18); }
  .fx-layer-panel { max-height:140px; overflow-y:auto; background: var(--bg2); border:1px solid var(--bg3); border-radius:4px; margin-top:4px; }
  .fx-sub-banner { background: rgba(187,154,247,0.15); color: #bb9af7; padding:3px 8px; font-size:11px; border-radius:4px; text-align:center; }
</style>
</head>
<body>
<header>
  <h1>ARCA Panorama</h1>
  <span class="subtitle">Generador de panorámicas por lotes</span>
  <div id="statusBadge" class="status idle">Inactivo</div>
</header>

<div class="path-bar">
  <label>Workspace</label>
  <input type="text" id="pathInput" placeholder="Ruta al directorio de trabajo (ej: C:\Users\erik\ARCA o /mnt/c/Users/erik/ARCA)">
  <button onclick="setWorkspace()">Aplicar</button>
  <span class="hint">Windows: C:\ruta... → auto-convierte a /mnt/c/ruta...</span>
</div>

<div class="main">
  <div class="sidebar" id="sidebar">
    <div class="sidebar-resize" id="sidebarResize"></div>
    <div class="sidebar-header">
      <h2>Grupos</h2>
      <span id="groupCount" class="sidebar-count">-</span>
      <span id="selCount" class="sidebar-count" style="display:none;background:var(--blue);color:var(--bg)">0 sel</span>
    </div>
    <div class="selection-bar">
      <button onclick="toggleSelectAll()" id="btnSelAll" title="Seleccionar/Deseleccionar todos">Todo</button>
      <button onclick="selectPending()" title="Seleccionar pendientes">Pendientes</button>
      <button onclick="clearSelection()" title="Limpiar selección">Limpiar</button>
    </div>
    <div class="search-box">
      <input type="text" id="searchInput" placeholder="Buscar grupo..." oninput="filterGroups()">
    </div>
    <div id="groupList" class="group-list"></div>
  </div>

  <div class="content">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('log')">Log</div>
      <div class="tab" onclick="switchTab('preview')">Vista previa</div>
      <div class="tab" onclick="switchTab('organize')">Organizar</div>
    </div>

    <div id="tabLog" class="tab-content active">
      <div class="toolbar">
        <button class="primary" onclick="runSelected()" id="btnRunSel" disabled>Procesar seleccionados (<span id="btnSelNum">0</span>)</button>
        <button onclick="runAll()" id="btnRunAll">Procesar TODOS</button>
        <button class="danger" onclick="stopProcess()" id="btnStop" disabled>Detener</button>
        <button onclick="refreshGroups()">Actualizar</button>
        <div class="spacer"></div>
        <div class="filter-group">
          <input type="checkbox" id="chkErrors" onchange="applyFilter()">
          <label for="chkErrors">Solo errores/warn</label>
        </div>
        <div class="filter-group">
          <input type="checkbox" id="chkAutoScroll" checked>
          <label for="chkAutoScroll">Auto-scroll</label>
        </div>
        <button onclick="exportLog()">Exportar log</button>
        <button onclick="clearLog()">Limpiar</button>
      </div>

      <div class="log-container">
        <div id="logView" class="log-view"></div>
        <div id="logEmpty" class="log-empty">
          <span>Sin registros</span>
          <span style="font-size:12px">Selecciona un grupo o presiona "Procesar TODOS" para iniciar</span>
        </div>
      </div>
    </div>

    <div id="tabPreview" class="tab-content">
      <div class="preview-panel" id="previewPanel">
        <span class="no-preview">Selecciona un grupo completado para ver la panorámica</span>
      </div>
    </div>

    <div id="tabOrganize" class="tab-content">
      <div id="orgContent" class="org-grid">
        <div style="padding:16px;color:var(--fg3)">Cargando...</div>
      </div>
    </div>

    <div class="summary-bar">
      <span>Completados: <span class="cnt green" id="sumDone">0</span></span>
      <span>Pendientes: <span class="cnt yellow" id="sumPending">0</span></span>
      <span>Errores en log: <span class="cnt red" id="sumErrors">0</span></span>
      <span>Total fotos: <span class="cnt" id="sumPhotos">0</span></span>
    </div>
  </div>
</div>

<div id="toast" class="toast"></div>

<script>
let groups = [];
let logEntries = [];
let errorsOnly = false;
let evtSource = null;
let errorCount = 0;
let searchFilter = '';
let selected = new Set(); // "cbtis/group" keys

// --- Toast ---
function showToast(msg, type='info') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show ' + type;
  // In fullscreen/faux-fullscreen, move toast inside the visible container
  const fsEl = document.fullscreenElement || document.querySelector('.faux-fullscreen');
  if (fsEl && !fsEl.contains(t)) {
    fsEl.appendChild(t);
  } else if (!fsEl && t.parentElement !== document.body) {
    document.body.appendChild(t);
  }
  clearTimeout(t._tid);
  t._tid = setTimeout(() => t.className = 'toast', 3000);
}

// --- Config ---
async function loadConfig() {
  const res = await fetch('/api/config');
  const cfg = await res.json();
  document.getElementById('pathInput').value = cfg.workspace || '';
}

async function setWorkspace() {
  const path = document.getElementById('pathInput').value.trim();
  if (!path) return showToast('Ingresa una ruta', 'error');
  const res = await fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({workspace: path})
  });
  const data = await res.json();
  if (res.ok) {
    showToast('Workspace actualizado: ' + data.workspace, 'success');
    document.getElementById('pathInput').value = data.workspace;
    refreshGroups();
  } else {
    showToast(data.error, 'error');
  }
}

// --- Groups ---
async function refreshGroups() {
  try {
    const res = await fetch('/api/groups');
    groups = await res.json();
    renderGroups();
    updateSummary();
  } catch(e) {
    showToast('Error al cargar grupos', 'error');
  }
}

function filterGroups() {
  searchFilter = document.getElementById('searchInput').value.toLowerCase();
  renderGroups();
}

function renderGroups() {
  const container = document.getElementById('groupList');
  const byCbtis = {};
  const filtered = groups.filter(g =>
    !searchFilter || g.group.toLowerCase().includes(searchFilter) || g.cbtis.includes(searchFilter)
  );
  filtered.forEach(g => {
    if (!byCbtis[g.cbtis]) byCbtis[g.cbtis] = [];
    byCbtis[g.cbtis].push(g);
  });

  let html = '';
  for (const [cbtis, items] of Object.entries(byCbtis)) {
    const done = items.filter(i => i.completed).length;
    const allDone = items.filter(i => i.workflow && i.workflow.pano_done && i.workflow.recorte_done && i.workflow.borlas_done && i.workflow.togas_done && i.workflow.sombras_done).length;
    const hasPendingCutouts = items.some(i => i.completed && i.output && !i.cutout);
    const cutCbtisBtn = hasPendingCutouts
      ? `<button class="cut-cbtis-btn" onclick="runCutout('${cbtis}')" title="Recortar fondo de todas las panorámicas">&#9988;</button>`
      : '';
    html += `<div class="cbtis-section">
      <div class="cbtis-label">
        CBTIS ${cbtis}
        <span style="font-size:10px;color:var(--fg3);font-weight:400">${done}/${items.length}${allDone ? ` (${allDone} listos)` : ''}</span>
        <div class="cbtis-actions">
          <button onclick="runCbtis('${cbtis}')">Procesar</button>
          ${cutCbtisBtn}
        </div>
      </div>`;
    items.forEach(g => {
      const wf = g.workflow || {};
      // Status badges: P (pano), R (recorte), B (borlas), T (togas), S (sombras)
      const pCls = g.completed ? (wf.pano_done ? 'pano done' : 'pano') : 'off';
      const rCls = g.cutout ? (wf.recorte_done ? 'recorte done' : 'recorte') : (g.completed && g.output ? 'recorte pending' : 'off');
      const bCls = g.has_borlas ? (wf.borlas_done ? 'borlas done' : 'borlas') : (g.cutout ? 'borlas' : 'off');
      const tCls = g.has_togas ? (wf.togas_done ? 'togas done' : 'togas') : (g.cutout ? 'togas' : 'off');
      const sCls = g.has_sombras ? (wf.sombras_done ? 'sombras done' : 'sombras') : (g.cutout ? 'sombras' : 'off');
      const fCls = wf.fixes_done ? 'fixes done' : (g.cutout ? 'fixes' : 'off');
      // P R B T S F badges — clickable to navigate to each section
      const pClick = g.completed && g.output
        ? `event.stopPropagation();showPreview('${g.cbtis}','${g.output}','${g.group}')`
        : '';
      const rClick = g.cutout
        ? `event.stopPropagation();showCutoutPreview('${g.cbtis}','${g.cutout}','${g.group}')`
        : (g.completed && g.output ? `event.stopPropagation();runCutout('${g.cbtis}','${g.output}')` : '');
      const bClick = g.cutout
        ? `event.stopPropagation();showCutoutPreview('${g.cbtis}','${g.cutout}','${g.group}',true)`
        : '';
      const tClick = g.cutout
        ? `event.stopPropagation();switchTab('preview');enterTogasMode('${g.cbtis}','${g.cutout}')`
        : '';
      const sClick = g.cutout
        ? `event.stopPropagation();switchTab('preview');enterSombrasMode('${g.cbtis}','${g.cutout}')`
        : '';
      const fClick = g.cutout
        ? `event.stopPropagation();switchTab('preview');enterFixesMode('${g.cbtis}','${g.cutout}')`
        : '';
      const badges = `<span class="grp-badges">` +
        `<span class="grp-badge ${pCls}" title="${g.completed ? (wf.pano_done ? 'Panorámica lista' : 'Panorámica disponible') : 'Sin panorámica'}"${pClick ? ` onclick="${pClick}"` : ''}>P</span>` +
        `<span class="grp-badge ${rCls}" title="${g.cutout ? (wf.recorte_done ? 'Recorte listo' : 'Recorte disponible') : (g.completed && g.output ? 'Recortar fondo' : 'Sin recorte')}"${rClick ? ` onclick="${rClick}"` : ''}>R</span>` +
        `<span class="grp-badge ${bCls}" title="${g.has_borlas ? (wf.borlas_done ? 'Borlas listas' : 'Borlas colocadas') : (g.cutout ? 'Ir a borlas' : 'Sin borlas')}"${bClick ? ` onclick="${bClick}"` : ''}>B</span>` +
        `<span class="grp-badge ${tCls}" title="${g.has_togas ? (wf.togas_done ? 'Togas listas' : 'Togas colocadas') : (g.cutout ? 'Ir a togas' : 'Sin togas')}"${tClick ? ` onclick="${tClick}"` : ''}>T</span>` +
        `<span class="grp-badge ${sCls}" title="${g.has_sombras ? (wf.sombras_done ? 'Sombras listas' : 'Sombras colocadas') : (g.cutout ? 'Ir a sombras' : 'Sin sombras')}"${sClick ? ` onclick="${sClick}"` : ''}>S</span>` +
        `<span class="grp-badge ${fCls}" title="${wf.fixes_done ? 'Fixes listos' : (g.cutout ? 'Ir a fixes' : 'Sin fixes')}"${fClick ? ` onclick="${fClick}"` : ''}>F</span>` +
        `</span>`;

      const key = g.cbtis + '/' + g.group;
      const checked = selected.has(key);
      html += `<div class="group-item ${checked ? 'checked' : ''}" onclick="toggleSelect('${g.cbtis}','${g.group}')" ondblclick="runGroup('${g.cbtis}','${g.group}')">
        <input type="checkbox" class="sel-check" ${checked ? 'checked' : ''} onclick="event.stopPropagation();toggleSelect('${g.cbtis}','${g.group}')">
        <span class="name">${g.group}</span>
        <span class="photos">${g.photos}f</span>
        ${badges}
        <button class="run-btn" onclick="event.stopPropagation();runGroup('${g.cbtis}','${g.group}')" title="Procesar panorámica">&#9654;</button>
      </div>`;
    });
    html += '</div>';
  }
  if (!html) html = '<div style="padding:16px;color:var(--fg3);font-size:13px;text-align:center">No se encontraron grupos.<br>Verifica la ruta del workspace.</div>';
  container.innerHTML = html;
  document.getElementById('groupCount').textContent = filtered.length;
}

function updateSummary() {
  const done = groups.filter(g => g.completed).length;
  const pending = groups.filter(g => !g.completed).length;
  const photos = groups.reduce((s, g) => s + g.photos, 0);
  document.getElementById('sumDone').textContent = done;
  document.getElementById('sumPending').textContent = pending;
  document.getElementById('sumPhotos').textContent = photos;
}

function selectGroup(cbtis, group, output) {
  if (output) {
    showPreview(cbtis, output, group);
  }
}

// --- Selection ---
function toggleSelect(cbtis, group) {
  const key = cbtis + '/' + group;
  if (selected.has(key)) selected.delete(key);
  else selected.add(key);
  updateSelectionUI();
  renderGroups();
}

function toggleSelectAll() {
  const visible = getVisibleGroups();
  const allSelected = visible.every(g => selected.has(g.cbtis + '/' + g.group));
  if (allSelected) {
    visible.forEach(g => selected.delete(g.cbtis + '/' + g.group));
  } else {
    visible.forEach(g => selected.add(g.cbtis + '/' + g.group));
  }
  updateSelectionUI();
  renderGroups();
}

function selectPending() {
  selected.clear();
  groups.filter(g => !g.completed).forEach(g => selected.add(g.cbtis + '/' + g.group));
  updateSelectionUI();
  renderGroups();
}

function clearSelection() {
  selected.clear();
  updateSelectionUI();
  renderGroups();
}

function getVisibleGroups() {
  return groups.filter(g =>
    !searchFilter || g.group.toLowerCase().includes(searchFilter) || g.cbtis.includes(searchFilter)
  );
}

function updateSelectionUI() {
  const n = selected.size;
  const badge = document.getElementById('selCount');
  const btn = document.getElementById('btnRunSel');
  const num = document.getElementById('btnSelNum');
  if (n > 0) {
    badge.style.display = '';
    badge.textContent = n + ' sel';
    btn.disabled = false;
    num.textContent = n;
  } else {
    badge.style.display = 'none';
    btn.disabled = true;
    num.textContent = '0';
  }
}

async function runSelected() {
  if (selected.size === 0) return;
  const items = [];
  selected.forEach(key => {
    const [cbtis, group] = key.split('/');
    items.push({cbtis, group});
  });
  const res = await fetch('/api/run_batch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({items})
  });
  if (res.ok) { startListening(); switchTab('log'); }
  else { const d = await res.json(); showToast(d.error, 'error'); }
}

// --- Tabs ---
const TAB_MAP = {
  log:      { index: 1, id: 'tabLog' },
  preview:  { index: 2, id: 'tabPreview' },
  organize: { index: 3, id: 'tabOrganize' },
};
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  const info = TAB_MAP[tab] || TAB_MAP.log;
  document.querySelector(`.tabs .tab:nth-child(${info.index})`).classList.add('active');
  document.getElementById(info.id).classList.add('active');
  if (tab === 'organize') loadOrgStatus();
}

// --- Pointer helpers (touchpad + touchscreen free pan) ---
function ptrCenter(ptrs) {
  const ids = Object.keys(ptrs);
  let sx = 0, sy = 0;
  for (const id of ids) { sx += ptrs[id].x; sy += ptrs[id].y; }
  return { x: sx / ids.length, y: sy / ids.length };
}
function ptrDist(ptrs) {
  const ids = Object.keys(ptrs);
  if (ids.length < 2) return 0;
  const a = ptrs[ids[0]], b = ptrs[ids[1]];
  return Math.hypot(b.x - a.x, b.y - a.y);
}

// --- Workflow done toggle ---
async function toggleWorkflowDone(section, btnEl) {
  const panel = document.getElementById('previewPanel');
  const cbtis = panel.dataset.cbtis || blCbtis || (typeof tgCbtis !== 'undefined' ? tgCbtis : '') || (typeof rfCbtis !== 'undefined' ? rfCbtis : '');
  const group = panel.dataset.group || '';
  if (!cbtis || !group) { showToast('No se pudo determinar el grupo', 'error'); return; }
  // Get current state from group data
  const gData = groups.find(g => g.cbtis === cbtis && g.group === group);
  const wf = gData && gData.workflow ? gData.workflow : {};
  const key = section + '_done';
  const newVal = !wf[key];
  try {
    const res = await fetch(`/api/group/workflow/${cbtis}/${encodeURIComponent(group)}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({section, done: newVal})
    });
    if (!res.ok) throw new Error('Error al guardar');
    const data = await res.json();
    // Update local group data
    if (gData) gData.workflow = data.workflow;
    // Update button visual
    if (btnEl) {
      btnEl.classList.toggle('is-done', newVal);
      btnEl.textContent = newVal ? '\u2714 Listo' : '\u2610 Listo';
    }
    renderGroups();
    showToast(newVal ? `${section} marcado como listo` : `${section} desmarcado`, 'ok');
  } catch(e) {
    showToast('Error: ' + e.message, 'error');
  }
}

function getWorkflowDoneState(section) {
  const panel = document.getElementById('previewPanel');
  const cbtis = panel.dataset.cbtis || blCbtis || (typeof tgCbtis !== 'undefined' ? tgCbtis : '') || '';
  const group = panel.dataset.group || '';
  const gData = groups.find(g => g.cbtis === cbtis && g.group === group);
  return gData && gData.workflow && gData.workflow[section + '_done'];
}

// --- Preview with zoom/pan ---
let pvZoom = 1, pvX = 0, pvY = 0, pvDragging = false, pvStartX = 0, pvStartY = 0;

function initPreviewPanel(imgSrc, label, isCutout) {
  const panel = document.getElementById('previewPanel');
  const gData = groups.find(g => g.cbtis === panel.dataset.cbtis && g.group === panel.dataset.group);
  const hasCutout = gData && gData.cutout;
  const hasPano = gData && gData.output;

  let toggleHtml = '';
  if (hasPano && hasCutout) {
    const panoActive = !isCutout ? 'active' : '';
    const cutActive = isCutout ? 'active' : '';
    const cutFile = panel.dataset.cutout || '';
    const refBtn = isCutout && cutFile
      ? `<button onclick="enterRefineMode('${panel.dataset.cbtis}','${cutFile.replace(/'/g, "\\'")}')">Refinar</button>`
      : '';
    const borlasBtn = isCutout && cutFile
      ? `<button onclick="enterBorlasMode('${panel.dataset.cbtis}','${cutFile.replace(/'/g, "\\'")}')">&#127891; Borlas</button>`
      : '';
    const togasBtn = isCutout && cutFile
      ? `<button onclick="enterTogasMode('${panel.dataset.cbtis}','${cutFile.replace(/'/g, "\\'")}')">&#127891; Togas</button>`
      : '';
    const pvSection = isCutout ? 'recorte' : 'pano';
    const pvDoneState = getWorkflowDoneState(pvSection);
    const pvDoneBtn = `<button class="rf-btn done-toggle ${pvDoneState ? 'is-done' : ''}" onclick="toggleWorkflowDone('${pvSection}', this)">${pvDoneState ? '\u2714 Listo' : '\u2610 Listo'}</button>`;
    toggleHtml = `<div class="preview-toggle">
      <button class="${panoActive}" onclick="togglePreviewType(this, 'pano')">Panorámica</button>
      <button class="${cutActive}" onclick="togglePreviewType(this, 'cutout')">Recortada</button>
      ${refBtn}
      ${borlasBtn}
      ${togasBtn}
      ${pvDoneBtn}
    </div>`;
  }

  const checkBg = isCutout
    ? 'background: repeating-conic-gradient(#3a3a4a 0% 25%, #2a2a3a 0% 50%) 50% / 20px 20px;'
    : '';

  panel.innerHTML = `
    <div class="preview-controls">
      ${toggleHtml}
      <div style="flex:1"></div>
      <button onclick="pvZoomTo(0.5)">50%</button>
      <button onclick="pvZoomTo(1)">100%</button>
      <button onclick="pvZoomFit()">Ajustar</button>
      <span class="zoom-label" id="zoomLabel">100%</span>
    </div>
    <div class="preview-viewport" id="pvViewport" style="${checkBg}">
      <img id="pvImg" src="${imgSrc}" alt="preview" draggable="false">
    </div>
    <div class="preview-info" id="pvLabel">${label}</div>
  `;

  pvZoom = 1; pvX = 0; pvY = 0;
  const img = document.getElementById('pvImg');
  img.onload = () => pvZoomFit();
  if (img.complete) pvZoomFit();

  // Mouse events on viewport
  const vp = document.getElementById('pvViewport');
  vp.onwheel = (e) => {
    e.preventDefault();
    const rect = vp.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (e.ctrlKey) {
      // Ctrl+scroll or trackpad pinch → zoom
      const oldZoom = pvZoom;
      const delta = -e.deltaY * (e.deltaMode === 1 ? 20 : 1);
      const factor = Math.pow(1.01, delta);
      pvZoom = Math.max(0.05, Math.min(20, pvZoom * factor));
      pvX = mx - (mx - pvX) * (pvZoom / oldZoom);
      pvY = my - (my - pvY) * (pvZoom / oldZoom);
    } else {
      // Plain scroll (trackpad 2-finger) → pan
      pvX -= e.deltaX;
      pvY -= e.deltaY;
    }
    pvApply();
  };
  vp.onmousedown = (e) => {
    // Any click → free pan (no axis restriction)
    if (e.button === 0 || e.button === 1 || e.button === 2) {
      e.preventDefault();
      pvDragging = true;
      pvStartX = e.clientX - pvX;
      pvStartY = e.clientY - pvY;
      vp.style.cursor = 'grabbing';
    }
  };
  vp.oncontextmenu = (e) => e.preventDefault();
  vp.style.cursor = 'grab';
  window.addEventListener('mousemove', pvMouseMove);
  window.addEventListener('mouseup', pvMouseUp);

  // --- Pointer Events: touchpad + touchscreen free pan (no axis lock) ---
  let pvPtrs = {}, pvPtrCenter = null, pvPtrDist = 0, pvPtrSingle = null;

  vp.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse') return;  // mouse uses existing handlers
    vp.setPointerCapture(e.pointerId);
    pvPtrs[e.pointerId] = { x: e.clientX, y: e.clientY };
    const cnt = Object.keys(pvPtrs).length;
    if (cnt >= 2) {
      pvPtrCenter = ptrCenter(pvPtrs);
      pvPtrDist = ptrDist(pvPtrs);
      pvPtrSingle = null;
    } else {
      pvPtrSingle = { x: e.clientX, y: e.clientY };
    }
  });

  vp.addEventListener('pointermove', (e) => {
    if (e.pointerType === 'mouse' || !pvPtrs[e.pointerId]) return;
    pvPtrs[e.pointerId] = { x: e.clientX, y: e.clientY };
    const ids = Object.keys(pvPtrs);
    if (ids.length >= 2 && pvPtrCenter) {
      const c = ptrCenter(pvPtrs);
      const d = ptrDist(pvPtrs);
      // Free pan
      pvX += c.x - pvPtrCenter.x;
      pvY += c.y - pvPtrCenter.y;
      // Pinch zoom
      if (pvPtrDist > 0 && d > 0) {
        const rect = vp.getBoundingClientRect();
        const mx = c.x - rect.left, my = c.y - rect.top;
        const oldZoom = pvZoom;
        pvZoom = Math.max(0.05, Math.min(20, pvZoom * (d / pvPtrDist)));
        pvX = mx - (mx - pvX) * (pvZoom / oldZoom);
        pvY = my - (my - pvY) * (pvZoom / oldZoom);
      }
      pvPtrCenter = c;
      pvPtrDist = d;
      pvApply();
    } else if (ids.length === 1 && pvPtrSingle) {
      // Single touch pointer = pan
      pvX += e.clientX - pvPtrSingle.x;
      pvY += e.clientY - pvPtrSingle.y;
      pvPtrSingle = { x: e.clientX, y: e.clientY };
      pvApply();
    }
  });

  const pvPtrClean = (e) => {
    delete pvPtrs[e.pointerId];
    if (Object.keys(pvPtrs).length < 2) { pvPtrCenter = null; pvPtrDist = 0; }
    if (Object.keys(pvPtrs).length === 0) pvPtrSingle = null;
  };
  vp.addEventListener('pointerup', pvPtrClean);
  vp.addEventListener('pointercancel', pvPtrClean);
}

function pvMouseMove(e) {
  if (!pvDragging) return;
  pvX = e.clientX - pvStartX;
  pvY = e.clientY - pvStartY;
  pvApply();
}
function pvMouseUp() {
  pvDragging = false;
  const vp = document.getElementById('pvViewport');
  if (vp) { vp.classList.remove('dragging'); vp.style.cursor = 'grab'; }
}

function pvApply() {
  const img = document.getElementById('pvImg');
  if (!img) return;
  img.style.transform = `translate(${pvX}px, ${pvY}px) scale(${pvZoom})`;
  const label = document.getElementById('zoomLabel');
  if (label) label.textContent = Math.round(pvZoom * 100) + '%';
}

function pvZoomTo(z) {
  const vp = document.getElementById('pvViewport');
  const img = document.getElementById('pvImg');
  if (!vp || !img) return;
  const rect = vp.getBoundingClientRect();
  const cx = rect.width / 2, cy = rect.height / 2;
  const oldZoom = pvZoom;
  pvZoom = z;
  pvX = cx - (cx - pvX) * (pvZoom / oldZoom);
  pvY = cy - (cy - pvY) * (pvZoom / oldZoom);
  pvApply();
}

function pvZoomFit() {
  const vp = document.getElementById('pvViewport');
  const img = document.getElementById('pvImg');
  if (!vp || !img || !img.naturalWidth) return;
  const rect = vp.getBoundingClientRect();
  const scaleX = rect.width / img.naturalWidth;
  const scaleY = rect.height / img.naturalHeight;
  pvZoom = Math.min(scaleX, scaleY) * 0.95;
  pvX = (rect.width - img.naturalWidth * pvZoom) / 2;
  pvY = (rect.height - img.naturalHeight * pvZoom) / 2;
  pvApply();
}

function showPreview(cbtis, filename, group) {
  switchTab('preview');
  const panel = document.getElementById('previewPanel');
  panel.dataset.cbtis = cbtis;
  panel.dataset.group = group;
  panel.dataset.pano = filename;
  const gData = groups.find(g => g.cbtis === cbtis && g.group === group);
  panel.dataset.cutout = gData && gData.cutout ? gData.cutout : '';
  initPreviewPanel(`/api/preview/${cbtis}/${filename}`, `${filename} (panorámica)`, false);
}

// --- Run / Stop ---
async function runAll() {
  const res = await fetch('/api/run');
  if (res.ok) { startListening(); switchTab('log'); }
  else { const d = await res.json(); showToast(d.error, 'error'); }
}

async function runCbtis(cbtis) {
  const res = await fetch('/api/run/' + cbtis);
  if (res.ok) { startListening(); switchTab('log'); }
  else { const d = await res.json(); showToast(d.error, 'error'); }
}

async function runGroup(cbtis, group) {
  const res = await fetch('/api/run/' + cbtis + '/' + group);
  if (res.ok) { startListening(); switchTab('log'); }
  else { const d = await res.json(); showToast(d.error, 'error'); }
}

async function stopProcess() {
  const res = await fetch('/api/stop');
  if (res.ok) showToast('Proceso detenido', 'success');
}

// --- Log streaming ---
function startListening() {
  if (evtSource) evtSource.close();
  logEntries = [];
  errorCount = 0;
  document.getElementById('logView').innerHTML = '';
  document.getElementById('logEmpty').style.display = 'none';
  document.getElementById('sumErrors').textContent = '0';

  evtSource = new EventSource('/api/logs');
  evtSource.onmessage = (e) => {
    const entry = JSON.parse(e.data);
    logEntries.push(entry);
    if (entry.level === 'error') {
      errorCount++;
      document.getElementById('sumErrors').textContent = errorCount;
    }
    appendLogLine(entry);
    if (entry.text.includes('Batch finalizado') || entry.text.startsWith('=== Proceso terminado') || entry.text.startsWith('=== Error fatal')) {
      setRunning(false);
      refreshGroups();
    }
  };
  evtSource.onerror = () => {
    // Reconnect handled by browser, just update UI
  };
  setRunning(true);
}

function appendLogLine(entry) {
  if (errorsOnly && entry.level !== 'error' && entry.level !== 'warn') return;
  const logView = document.getElementById('logView');
  const div = document.createElement('div');
  div.className = 'log-line ' + entry.level;
  div.textContent = entry.text;
  logView.appendChild(div);
  if (document.getElementById('chkAutoScroll').checked) {
    logView.scrollTop = logView.scrollHeight;
  }
}

function applyFilter() {
  errorsOnly = document.getElementById('chkErrors').checked;
  const logView = document.getElementById('logView');
  logView.innerHTML = '';
  logEntries.forEach(e => appendLogLine(e));
}

function clearLog() {
  logEntries = [];
  errorCount = 0;
  document.getElementById('logView').innerHTML = '';
  document.getElementById('logEmpty').style.display = 'flex';
  document.getElementById('sumErrors').textContent = '0';
}

function exportLog() {
  window.open('/api/logs/export', '_blank');
}

function setRunning(running) {
  const badge = document.getElementById('statusBadge');
  const btnRun = document.getElementById('btnRunAll');
  const btnStop = document.getElementById('btnStop');
  if (running) {
    badge.textContent = 'Ejecutando...';
    badge.className = 'status running';
    btnRun.disabled = true;
    btnStop.disabled = false;
  } else {
    badge.textContent = 'Inactivo';
    badge.className = 'status idle';
    btnRun.disabled = false;
    btnStop.disabled = true;
  }
}

// --- Organize ---
async function loadOrgStatus() {
  const container = document.getElementById('orgContent');
  try {
    const res = await fetch('/api/organize/status');
    if (!res.ok) {
      const d = await res.json();
      container.innerHTML = `<div style="padding:16px;color:var(--red)">${d.error || 'Error al cargar'}</div>`;
      return;
    }
    const items = await res.json();
    if (!items.length) {
      container.innerHTML = '<div style="padding:16px;color:var(--fg3)">No se encontraron CBTIS en DATOS_2026.ods</div>';
      return;
    }
    container.className = 'org-grid';
    container.innerHTML = items.map(c => {
      const badge = c.organized
        ? '<span class="org-badge done">Organizado</span>'
        : '<span class="org-badge pending">Pendiente</span>';
      const btns = c.organized
        ? '<button disabled>Ya organizado</button>'
        : `<button onclick="loadOrgPreview('${c.cbtis}')">Vista previa</button>`;
      return `<div class="org-card">
        <h3>CBTIS ${c.cbtis} ${badge}</h3>
        <div class="org-stats">
          <span>Grupos en ODS: <strong>${c.groups_in_ods}</strong></span>
          <span>Fotos en ODS: <strong>${c.photos_in_ods}</strong></span>
          <span>Fotos sueltas: <strong>${c.loose_photos}</strong></span>
          ${!c.dir_exists ? '<span style="color:var(--red)">Directorio no existe</span>' : ''}
        </div>
        <div class="org-actions">${btns}</div>
      </div>`;
    }).join('');
  } catch(e) {
    container.innerHTML = `<div style="padding:16px;color:var(--red)">Error: ${e.message}</div>`;
  }
}

async function loadOrgPreview(cbtis) {
  const container = document.getElementById('orgContent');
  container.className = 'org-preview';
  container.innerHTML = '<div style="color:var(--fg3)">Analizando...</div>';
  try {
    const res = await fetch('/api/organize/preview/' + cbtis);
    const data = await res.json();
    if (!res.ok) {
      container.innerHTML = `<div style="color:var(--red)">${data.error}</div>`;
      return;
    }

    let html = `<button class="org-back-btn" onclick="loadOrgStatus()">&#8592; Volver</button>`;
    html += `<h3>CBTIS ${cbtis} &mdash; Vista previa de organización</h3>`;

    html += `<div class="org-summary">
      <span>Grupos: <strong>${data.total_groups}</strong></span>
      <span>Fotos en ODS: <strong>${data.total_photos}</strong></span>
      <span class="ok">Encontradas: <strong>${data.photos_found}</strong></span>
      <span class="missing">Faltantes: <strong>${data.missing_photos.length}</strong></span>
      <span class="extra">Extra en disco: <strong>${data.extra_photos.length}</strong></span>
    </div>`;

    html += '<table><thead><tr><th>Directorio</th><th>Fotos</th><th>Acciones</th><th>Obs</th></tr></thead><tbody>';
    data.actions.forEach(a => {
      const status = a.dir_exists ? '<span class="extra">existe</span>' : '<span class="ok">crear</span>';
      const moveList = a.moves.map(m => m.from).join(', ');
      html += `<tr><td>${a.dirname} (${status})</td><td>${a.photos}</td><td>${moveList || '<span class="missing">sin fotos</span>'}</td><td>${a.obs}</td></tr>`;
    });
    html += '</tbody></table>';

    if (data.missing_photos.length > 0) {
      html += '<h3 style="margin-top:12px;color:var(--red)">Fotos faltantes</h3><table><thead><tr><th>Grupo</th><th>Esperada</th></tr></thead><tbody>';
      data.missing_photos.forEach(m => {
        html += `<tr><td>${m.group}</td><td class="missing">${m.expected}</td></tr>`;
      });
      html += '</tbody></table>';
    }

    if (data.extra_photos.length > 0) {
      html += '<h3 style="margin-top:12px;color:var(--yellow)">Fotos extra (no en ODS)</h3><div style="font-size:12px;color:var(--fg3);padding:4px 0">';
      html += data.extra_photos.join(', ');
      html += '</div>';
    }

    html += `<div class="org-confirm-bar">
      <button onclick="executeOrganize('${cbtis}')" id="orgConfirmBtn">Confirmar y mover fotos</button>
      ${data.missing_photos.length > 0 ? '<span class="warn">Hay fotos faltantes. Se moverán las que existan.</span>' : ''}
    </div>`;
    html += '<div id="orgResult"></div>';

    container.innerHTML = html;
  } catch(e) {
    container.innerHTML = `<div style="color:var(--red)">Error: ${e.message}</div>`;
  }
}

async function executeOrganize(cbtis) {
  const btn = document.getElementById('orgConfirmBtn');
  if (btn) btn.disabled = true;
  const resultDiv = document.getElementById('orgResult');
  resultDiv.innerHTML = '<div style="color:var(--fg3)">Organizando...</div>';
  try {
    const res = await fetch('/api/organize/execute/' + cbtis, {method: 'POST'});
    const data = await res.json();
    if (!res.ok) {
      resultDiv.innerHTML = `<div class="org-result error">${data.error}</div>`;
      if (btn) btn.disabled = false;
      return;
    }
    let msg = `Directorios creados: ${data.dirs_created} | Fotos movidas: ${data.photos_moved}`;
    if (data.errors.length > 0) {
      msg += `<br><br><strong>Errores (${data.errors.length}):</strong><br>` + data.errors.map(e => `- ${e}`).join('<br>');
      resultDiv.innerHTML = `<div class="org-result error">${msg}</div>`;
    } else {
      resultDiv.innerHTML = `<div class="org-result success">${msg}</div>`;
    }
    // Refresh sidebar groups
    refreshGroups();
  } catch(e) {
    resultDiv.innerHTML = `<div class="org-result error">Error: ${e.message}</div>`;
    if (btn) btn.disabled = false;
  }
}

// --- Cutout (background removal) ---
let cutoutPollTimer = null;

async function runCutout(cbtis, filename) {
  const url = filename
    ? `/api/cutout/run/${cbtis}/${encodeURIComponent(filename)}`
    : `/api/cutout/run/${cbtis}`;
  try {
    const res = await fetch(url, { method: 'POST' });
    if (!res.ok) {
      const d = await res.json();
      showToast(d.error, 'error');
      return;
    }
    showToast(`Recorte iniciado: CBTIS ${cbtis}${filename ? ' / ' + filename : ''}`, 'success');
    switchTab('log');
    // Poll cutout logs into the main log view
    if (cutoutPollTimer) clearInterval(cutoutPollTimer);
    let lastCount = 0;
    cutoutPollTimer = setInterval(async () => {
      try {
        const r = await fetch('/api/cutout/status/live');
        const live = await r.json();
        // Append new log entries to main log
        const newLogs = live.logs.slice(lastCount);
        newLogs.forEach(l => {
          logEntries.push(l);
          appendLogLine(l);
        });
        lastCount = live.logs.length;
        document.getElementById('logEmpty').style.display = 'none';
        if (!live.running) {
          clearInterval(cutoutPollTimer);
          cutoutPollTimer = null;
          setRunning(false);
          refreshGroups();
        }
      } catch {}
    }, 1500);
    setRunning(true);
  } catch(e) {
    showToast('Error al iniciar recorte: ' + e.message, 'error');
  }
}

function showCutoutPreview(cbtis, filename, group, goToBorlas) {
  switchTab('preview');
  const panel = document.getElementById('previewPanel');
  panel.dataset.cbtis = cbtis;
  panel.dataset.group = group;
  const gData = groups.find(g => g.cbtis === cbtis && g.group === group);
  panel.dataset.pano = gData && gData.output ? gData.output : '';
  panel.dataset.cutout = filename;
  if (goToBorlas) {
    enterBorlasMode(cbtis, filename);
  } else {
    initPreviewPanel(`/api/cutout/preview/${cbtis}/${encodeURIComponent(filename)}`, `${filename} (recortada)`, true);
  }
}

function togglePreviewType(btn, type) {
  const panel = document.getElementById('previewPanel');
  const cbtis = panel.dataset.cbtis;
  const group = panel.dataset.group;
  const pano = panel.dataset.pano;
  const cutout = panel.dataset.cutout;

  // Update active button
  btn.parentElement.querySelectorAll('button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  if (type === 'pano' && pano) {
    initPreviewPanel(`/api/preview/${cbtis}/${pano}`, `${pano} (panorámica)`, false);
  } else if (type === 'cutout' && cutout) {
    initPreviewPanel(`/api/cutout/preview/${cbtis}/${encodeURIComponent(cutout)}`, `${cutout} (recortada)`, true);
  }
}

// --- Refine Mode ---
let rfMode = false, rfStrokes = [], rfRedoStack = [];
let rfTool = 'brush', rfAction = 'restore', rfBrushSize = 15;
let rfCutImg = null, rfPanoImg = null, rfAlign = null;
let rfCbtis = '', rfFilename = '';
let rfWorkCanvas = null, rfWorkCtx = null;
let rfDisplayCanvas = null, rfDisplayCtx = null;
let rfCursorCanvas = null, rfCursorCtx = null;
let rfDrawQueued = false;  // rAF throttle flag
let rfPanning = false, rfDrawing = false, rfLassoPoints = [];
let spaceHeld = false;  // Space+drag = free pan (Photoshop-style)
let rfCurStroke = null, rfAnimFrame = 0;
let rfMousePos = null;         // {x, y} normalized 0-1 for brush cursor
let rfDraggingNode = -1;       // index of lasso node being dragged, -1 = none
let rfScaleW = 1, rfScaleH = 1;  // ratio: full-res / canvas-res
let rfImgW = 1, rfImgH = 1;     // logical image dimensions (full-res, constant)
let rfLassoFreehand = false;   // true when click-dragging in lasso mode
let rfLassoDownXY = null;      // {cx, cy} screen coords of mousedown for freehand threshold
let rfHiResLoading = false;    // true while hi-res images loading in background
let rfAutoSaveTimer = null;    // debounce timer for auto-save
let rfAutoSaving = false;      // true while auto-save is in flight

async function enterRefineMode(cbtis, filename) {
  rfCbtis = cbtis; rfFilename = filename;
  rfStrokes = []; rfRedoStack = []; rfLassoPoints = [];

  const panel = document.getElementById('previewPanel');
  panel.innerHTML = '<div class="rf-loading">Cargando editor...</div>';

  // Fetch alignment
  try {
    const aRes = await fetch(`/api/cutout/align/${cbtis}/${encodeURIComponent(filename)}`);
    const aData = await aRes.json();
    if (!aData.ok) { showToast('Error de alineación: ' + aData.error, 'error'); exitRefineMode(); return; }
    rfAlign = aData;
  } catch (e) { showToast('Error: ' + e.message, 'error'); exitRefineMode(); return; }

  // LOD: Load small proxy first for instant display
  const encName = encodeURIComponent(filename);
  const [cutImg, panoImg] = await Promise.all([
    loadImg(`/api/cutout/preview-scaled/${cbtis}/${encName}?maxw=500`),
    loadImg(`/api/cutout/pano-region/${cbtis}/${encName}?maxw=500`)
  ]);
  rfCutImg = cutImg; rfPanoImg = panoImg;
  // Logical dimensions: always full-res, constant across LOD swaps
  rfImgW = rfAlign.cut_w;
  rfImgH = rfAlign.cut_h;
  rfScaleW = rfImgW / cutImg.width;
  rfScaleH = rfImgH / cutImg.height;

  // Create hidden work canvas (compositing at loaded resolution)
  rfWorkCanvas = document.createElement('canvas');
  rfWorkCanvas.width = cutImg.width;
  rfWorkCanvas.height = cutImg.height;
  rfWorkCtx = rfWorkCanvas.getContext('2d');

  // Build UI
  panel.innerHTML = `
    <div class="refine-toolbar">
      <button class="rf-btn active" onclick="rfSetTool(this,'brush')" title="Brocha (B)">&#9998; Brocha</button>
      <button class="rf-btn" onclick="rfSetTool(this,'lasso')" title="Lazo (L)">&#9711; Lazo</button>
      <button class="rf-btn" onclick="rfSetTool(this,'hand')" title="Mano - Panear (H)">&#9995; Mano</button>
      <div class="rf-sep"></div>
      <button class="rf-btn restore active" onclick="rfSetAction(this,'restore')" title="Restaurar (R)">&#10003; Restaurar</button>
      <button class="rf-btn erase" onclick="rfSetAction(this,'erase')" title="Borrar (E)">&#10007; Borrar</button>
      <div class="rf-sep"></div>
      <span class="rf-size-label">Tam:</span>
      <input type="range" class="rf-size-slider" min="2" max="80" value="${rfBrushSize}"
             oninput="rfBrushSize=+this.value; document.getElementById('rfSizeVal').textContent=this.value">
      <span class="rf-size-label" id="rfSizeVal">${rfBrushSize}</span>
      <div class="rf-sep"></div>
      <button class="rf-btn" onclick="rfUndo()" title="Deshacer (Ctrl+Z)">&#8617; Deshacer</button>
      <button class="rf-btn" onclick="rfRedo()" title="Rehacer (Ctrl+Y)">&#8618; Rehacer</button>
      <div style="flex:1"></div>
      <span id="rfSaveStatus" class="rf-size-label" style="margin-right:6px"></span>
      <button class="rf-btn" onclick="rfApply()" title="Guardar ahora (Ctrl+S)">&#10003; Guardar</button>
      <button class="rf-btn" onclick="rfExportTiff()" title="Exportar TIFF con capas (Affinity/Photoshop)">&#8615; TIFF</button>
      <button class="rf-btn done-toggle ${getWorkflowDoneState('recorte') ? 'is-done' : ''}" onclick="toggleWorkflowDone('recorte', this)" title="Marcar recorte como terminado">${getWorkflowDoneState('recorte') ? '\u2714 Listo' : '\u2610 Listo'}</button>
      <button class="rf-btn" onclick="exitRefineMode(); enterBorlasMode('${cbtis}','${filename.replace(/'/g, "\\'")}')" title="Colocar borlas de graduación">&#127891; Borlas</button>
      <button class="rf-btn danger" onclick="exitRefineMode()">&#10005; Cancelar</button>
      <div class="rf-sep"></div>
      <button class="rf-btn" onclick="rfZoomFit()">Ajustar</button>
      <span class="rf-size-label" id="rfZoomLabel">100%</span>
      <button class="rf-btn" onclick="rfToggleFullscreen()" id="rfFullscreenBtn" title="Pantalla completa (F)">&#9974; Completa</button>
    </div>
    <div class="rf-canvas-wrap" id="rfWrap">
      <canvas id="rfDisplay"></canvas>
      <canvas id="rfCursor" style="position:absolute;top:0;left:0;pointer-events:none"></canvas>
    </div>
    <div class="preview-info">${filename} (refinando)</div>
  `;

  rfDisplayCanvas = document.getElementById('rfDisplay');
  rfDisplayCtx = rfDisplayCanvas.getContext('2d');
  rfCursorCanvas = document.getElementById('rfCursor');
  rfCursorCtx = rfCursorCanvas.getContext('2d');
  rfMode = true;

  // Wait for browser to compute layout before sizing canvas
  setTimeout(() => {
    const wrap = document.getElementById('rfWrap');
    if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    rfDisplayCanvas.width = Math.max(rect.width, 200);
    rfDisplayCanvas.height = Math.max(rect.height, 200);
    rfCursorCanvas.width = rfDisplayCanvas.width;
    rfCursorCanvas.height = rfDisplayCanvas.height;

    // Initial compositing
    rfRecomposite();
    pvZoom = 1; pvX = 0; pvY = 0;
    rfZoomFit();

    // Events
    rfBindEvents(wrap);

    // LOD: Load full-res in background
    rfHiResLoading = true;
    const encN = encodeURIComponent(rfFilename);
    Promise.all([
      loadImg(`/api/cutout/preview-scaled/${rfCbtis}/${encN}?maxw=99999`),
      loadImg(`/api/cutout/pano-region/${rfCbtis}/${encN}?maxw=99999`)
    ]).then(([cutHi, panoHi]) => {
      rfHiResLoading = false;
      if (!rfMode) return;
      // Swap images — no zoom/pan changes needed because display
      // always uses rfImgW/rfImgH (constant logical dimensions).
      // The work canvas changes resolution, browser interpolates the rest.
      rfCutImg = cutHi; rfPanoImg = panoHi;
      rfWorkCanvas.width = cutHi.width;
      rfWorkCanvas.height = cutHi.height;
      rfScaleW = rfImgW / cutHi.width;
      rfScaleH = rfImgH / cutHi.height;
      rfRecomposite();
      rfDraw();
    }).catch(() => { rfHiResLoading = false; });
  }, 50);
}

function loadImg(src) {
  return new Promise((resolve, reject) => {
    const img = new window.Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to load: ' + src));
    img.src = src;
  });
}

async function exitRefineMode() {
  // Flush any pending auto-save before exiting
  if (rfAutoSaveTimer) { clearTimeout(rfAutoSaveTimer); rfAutoSaveTimer = null; }
  if (rfStrokes.length > 0 && !rfAutoSaving) {
    await rfAutoSave();
  }
  rfMode = false;
  if (rfAnimFrame) { cancelAnimationFrame(rfAnimFrame); rfAnimFrame = 0; }
  // Exit fullscreen if active
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  const panel = document.getElementById('previewPanel');
  if (panel) panel.classList.remove('faux-fullscreen');
  if (rfCbtis && rfFilename) {
    initPreviewPanel(
      `/api/cutout/preview/${rfCbtis}/${encodeURIComponent(rfFilename)}?t=${Date.now()}`,
      `${rfFilename} (recortada)`, true
    );
  } else {
    panel.innerHTML = '<div style="color:var(--fg3);text-align:center;padding:40px">Sin vista previa</div>';
  }
  refreshGroups();
}

function rfRecomposite() {
  if (!rfWorkCtx || !rfCutImg) { console.error('rfRecomposite: missing ctx or image'); return; }
  // Redraw work canvas: cutout + strokes applied
  rfWorkCtx.clearRect(0, 0, rfWorkCanvas.width, rfWorkCanvas.height);
  rfWorkCtx.drawImage(rfCutImg, 0, 0);

  for (const s of rfStrokes) {
    rfApplyStrokeToCanvas(s);
  }
}

function rfFeatherRadius() {
  // Match server-side: max(0.8, min(w,h)/2000) — subtle feathering
  return Math.max(0.8, Math.min(rfWorkCanvas.width, rfWorkCanvas.height) / 2000.0);
}

function rfApplyStrokeToCanvas(stroke) {
  const ctx = rfWorkCtx;
  const w = rfWorkCanvas.width, h = rfWorkCanvas.height;
  const feather = rfFeatherRadius();

  // Create feathered mask on temporary canvas
  const tmpC = document.createElement('canvas');
  tmpC.width = w; tmpC.height = h;
  const tmpCtx = tmpC.getContext('2d');
  tmpCtx.fillStyle = '#000';
  rfDrawStrokePath(tmpCtx, stroke, w, h);
  // Apply gaussian-like blur for feathered edges
  tmpCtx.filter = `blur(${feather}px)`;
  tmpCtx.drawImage(tmpC, 0, 0);
  tmpCtx.filter = 'none';

  if (stroke.mode === 'erase') {
    ctx.save();
    ctx.globalCompositeOperation = 'destination-out';
    ctx.drawImage(tmpC, 0, 0);
    ctx.restore();
  } else {
    // Restore: use feathered mask to blend pano pixels
    // Draw pano into a temp canvas, mask it with feathered shape
    const panoC = document.createElement('canvas');
    panoC.width = w; panoC.height = h;
    const panoCtx = panoC.getContext('2d');
    panoCtx.drawImage(rfPanoImg, 0, 0, w, h);
    panoCtx.globalCompositeOperation = 'destination-in';
    panoCtx.drawImage(tmpC, 0, 0);
    // Composite onto work canvas
    ctx.save();
    ctx.globalCompositeOperation = 'source-over';
    ctx.drawImage(panoC, 0, 0);
    ctx.restore();
  }
}

function rfDrawStrokePath(ctx, stroke, w, h) {
  const pts = stroke.points;
  if (!pts || pts.length === 0) return;

  if (stroke.type === 'lasso') {
    ctx.beginPath();
    ctx.moveTo(pts[0].x * w, pts[0].y * h);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x * w, pts[i].y * h);
    ctx.closePath();
    ctx.fill();
  } else {
    const r = stroke.radius * Math.max(w, h);
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, r, 0, Math.PI * 2);
      ctx.fill();
    }
    if (pts.length > 1) {
      ctx.lineWidth = r * 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(pts[0].x * w, pts[0].y * h);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x * w, pts[i].y * h);
      ctx.stroke();
    }
  }
}

function rfBuildClipPath(ctx, stroke, w, h) {
  const pts = stroke.points;
  if (!pts || pts.length === 0) return;

  if (stroke.type === 'lasso') {
    ctx.moveTo(pts[0].x * w, pts[0].y * h);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x * w, pts[i].y * h);
    ctx.closePath();
  } else {
    const r = stroke.radius * Math.max(w, h);
    for (const p of pts) {
      ctx.moveTo(p.x * w + r, p.y * h);
      ctx.arc(p.x * w, p.y * h, r, 0, Math.PI * 2);
    }
    // Also build a thick line path between points
    if (pts.length > 1) {
      for (let i = 0; i < pts.length - 1; i++) {
        const ax = pts[i].x * w, ay = pts[i].y * h;
        const bx = pts[i+1].x * w, by = pts[i+1].y * h;
        const dx = bx - ax, dy = by - ay;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        const nx = -dy / len * r, ny = dx / len * r;
        ctx.moveTo(ax + nx, ay + ny);
        ctx.lineTo(bx + nx, by + ny);
        ctx.lineTo(bx - nx, by - ny);
        ctx.lineTo(ax - nx, ay - ny);
        ctx.closePath();
      }
    }
  }
}

// Lightweight: only redraws the cursor circle on the overlay canvas
function rfDrawCursor() {
  if (!rfCursorCanvas) return;
  const ctx = rfCursorCtx;
  ctx.clearRect(0, 0, rfCursorCanvas.width, rfCursorCanvas.height);
  if (rfTool === 'brush' && rfMousePos && !rfPanning) {
    const iw = rfImgW, ih = rfImgH;
    const bx = rfMousePos.x * iw * pvZoom + pvX;
    const by = rfMousePos.y * ih * pvZoom + pvY;
    const br = rfBrushSize * pvZoom;
    ctx.beginPath();
    ctx.arc(bx, by, br, 0, Math.PI * 2);
    ctx.strokeStyle = rfAction === 'restore' ? 'rgba(158,206,106,0.8)' : 'rgba(247,118,142,0.8)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(bx, by, 2, 0, Math.PI * 2);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fill();
  }
}

// Full content redraw — throttled via requestAnimationFrame
function rfDraw() {
  if (!rfMode || !rfDisplayCanvas) return;
  if (rfDrawQueued) return;  // Already scheduled
  rfDrawQueued = true;
  requestAnimationFrame(() => {
    rfDrawQueued = false;
    rfDrawNow();
    rfDrawCursor();
  });
}

function rfDrawNow() {
  if (!rfMode || !rfDisplayCanvas) return;
  const ctx = rfDisplayCtx;
  const cw = rfDisplayCanvas.width, ch = rfDisplayCanvas.height;
  const iw = rfImgW, ih = rfImgH;

  ctx.clearRect(0, 0, cw, ch);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'medium';
  ctx.translate(pvX, pvY);
  ctx.scale(pvZoom, pvZoom);

  // Draw work canvas scaled to logical size
  ctx.drawImage(rfWorkCanvas, 0, 0, iw, ih);

  // Draw current in-progress stroke preview
  if (rfCurStroke && rfCurStroke.points.length > 0) {
    ctx.save();
    if (rfCurStroke.mode === 'erase') {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      ctx.strokeStyle = 'rgba(0,0,0,0.7)';
    } else {
      ctx.globalCompositeOperation = 'source-over';
      ctx.beginPath();
      rfBuildClipPath(ctx, rfCurStroke, iw, ih);
      ctx.clip();
      ctx.drawImage(rfPanoImg, 0, 0, iw, ih);
    }
    if (rfCurStroke.mode === 'erase') {
      rfDrawStrokePath(ctx, rfCurStroke, iw, ih);
    }
    ctx.restore();
  }

  // Draw lasso in-progress outline + nodes
  if (rfTool === 'lasso' && rfLassoPoints.length > 0) {
    const lassoColor = rfAction === 'restore' ? 'rgba(158,206,106' : 'rgba(247,118,142';
    ctx.strokeStyle = lassoColor + ',0.8)';
    ctx.lineWidth = 2 / pvZoom;
    ctx.setLineDash([6 / pvZoom, 4 / pvZoom]);
    ctx.beginPath();
    ctx.moveTo(rfLassoPoints[0].x * iw, rfLassoPoints[0].y * ih);
    for (let i = 1; i < rfLassoPoints.length; i++)
      ctx.lineTo(rfLassoPoints[i].x * iw, rfLassoPoints[i].y * ih);
    if (rfMousePos) {
      ctx.lineTo(rfMousePos.x * iw, rfMousePos.y * ih);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    const nodeR = 5 / pvZoom;
    for (let i = 0; i < rfLassoPoints.length; i++) {
      const px = rfLassoPoints[i].x * iw, py = rfLassoPoints[i].y * ih;
      const isFirst = (i === 0 && rfLassoPoints.length >= 3);
      const r = isFirst ? nodeR * 1.6 : nodeR;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = isFirst ? 'rgba(255,255,100,0.95)' : lassoColor + ',0.9)';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = (isFirst ? 2.5 : 1.5) / pvZoom;
      ctx.stroke();
    }
  }

  ctx.restore();

  const zl = document.getElementById('rfZoomLabel');
  if (zl) zl.textContent = Math.round(pvZoom * 100) + '%';
}

function rfZoomFit() {
  const wrap = document.getElementById('rfWrap');
  if (!wrap) return;
  const rect = wrap.getBoundingClientRect();
  const scaleX = rect.width / rfImgW;
  const scaleY = rect.height / rfImgH;
  pvZoom = Math.min(scaleX, scaleY) * 0.95;
  pvX = (rect.width - rfImgW * pvZoom) / 2;
  pvY = (rect.height - rfImgH * pvZoom) / 2;
  rfDraw();
}

function rfSetTool(btn, tool) {
  rfTool = tool;
  rfLassoPoints = [];
  rfDraggingNode = -1;
  rfLassoFreehand = false;
  rfLassoDownXY = null;
  rfDrawing = false;
  btn.parentElement.querySelectorAll('.rf-btn').forEach(b => {
    if (b.textContent.includes('Brocha') || b.textContent.includes('Lazo') || b.textContent.includes('Mano'))
      b.classList.remove('active');
  });
  btn.classList.add('active');
  const wrap = document.getElementById('rfWrap');
  if (wrap) wrap.style.cursor = tool === 'hand' ? 'grab' : (tool === 'brush' ? 'none' : 'crosshair');
  rfDraw();
}

function rfSetAction(btn, action) {
  if (rfDrawing) return; // Don't switch mid-stroke
  rfAction = action;
  btn.parentElement.querySelectorAll('.rf-btn.restore,.rf-btn.erase').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  rfDraw();
}

function rfUndo() {
  if (rfStrokes.length === 0) return;
  rfRedoStack.push(rfStrokes.pop());
  rfRecomposite();
  rfDraw();
  rfScheduleAutoSave();
}

function rfRedo() {
  if (rfRedoStack.length === 0) return;
  const s = rfRedoStack.pop();
  rfStrokes.push(s);
  rfApplyStrokeToCanvas(s);
  rfDraw();
  rfScheduleAutoSave();
}

function rfCanvasToImg(e) {
  // Convert mouse event to normalized image coordinates (0-1)
  const rect = rfDisplayCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const imgX = (mx - pvX) / pvZoom / rfImgW;
  const imgY = (my - pvY) / pvZoom / rfImgH;
  return { x: imgX, y: imgY };
}

function rfHitTestLassoNode(e) {
  // Returns index of lasso point within 8px screen distance, or -1
  if (rfLassoPoints.length === 0) return -1;
  const rect = rfDisplayCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const w = rfImgW, h = rfImgH;
  const hitR = 8;  // pixels screen space
  for (let i = 0; i < rfLassoPoints.length; i++) {
    const sx = rfLassoPoints[i].x * w * pvZoom + pvX;
    const sy = rfLassoPoints[i].y * h * pvZoom + pvY;
    const dx = mx - sx, dy = my - sy;
    if (dx * dx + dy * dy <= hitR * hitR) return i;
  }
  return -1;
}

function rfBindEvents(wrap) {
  wrap.onwheel = (e) => {
    e.preventDefault();
    const rect = wrap.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const oldZoom = pvZoom;
    pvZoom = Math.max(0.05, Math.min(pvZoom * (e.deltaY < 0 ? 1.1 : 0.9), 20));
    pvX = mx - (mx - pvX) * (pvZoom / oldZoom);
    pvY = my - (my - pvY) * (pvZoom / oldZoom);
    rfDraw();
  };

  wrap.onmousedown = (e) => {
    e.preventDefault();
    // Hand tool, Space+click, middle button, right-click, or Alt+left = free pan
    if (e.button === 1 || e.button === 2 || (e.button === 0 && e.altKey) || (e.button === 0 && spaceHeld) || (e.button === 0 && rfTool === 'hand')) {
      rfPanning = true;
      pvStartX = e.clientX - pvX;
      pvStartY = e.clientY - pvY;
      wrap.classList.add('dragging');
      wrap.style.cursor = 'grabbing';
      return;
    }
    if (e.button !== 0) return;

    const pos = rfCanvasToImg(e);
    // Allow lasso nodes outside image bounds for flexible angles;
    // only restrict brush tool to within image
    if (rfTool !== 'lasso' && (pos.x < 0 || pos.x > 1 || pos.y < 0 || pos.y > 1)) return;

    if (rfTool === 'lasso') {
      rfLassoFreehand = false;
      // Check if clicking on first node to close selection
      if (rfLassoPoints.length >= 3) {
        const rect = rfDisplayCanvas.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const fx = rfLassoPoints[0].x * rfImgW * pvZoom + pvX;
        const fy = rfLassoPoints[0].y * rfImgH * pvZoom + pvY;
        if ((mx - fx) * (mx - fx) + (my - fy) * (my - fy) <= 12 * 12) {
          // Close lasso
          rfRedoStack = [];
          const stroke = { type: 'lasso', mode: rfAction, points: [...rfLassoPoints], radius: 0 };
          rfStrokes.push(stroke);
          rfApplyStrokeToCanvas(stroke);
          rfLassoPoints = [];
          rfDraw();
          rfScheduleAutoSave();
          return;
        }
      }
      // Check if clicking on an existing node to drag it
      const hitIdx = rfHitTestLassoNode(e);
      if (hitIdx >= 0) {
        rfDraggingNode = hitIdx;
        wrap.style.cursor = 'move';
      } else {
        rfLassoPoints.push(pos);
        // Track for potential freehand drag (don't activate yet)
        rfLassoFreehand = false;
        rfLassoDownXY = { cx: e.clientX, cy: e.clientY, t: Date.now() };
      }
      rfDraw();
    } else {
      // Brush
      rfDrawing = true;
      rfRedoStack = [];
      rfCurStroke = {
        type: 'brush', mode: rfAction,
        points: [pos],
        radius: rfBrushSize / Math.max(rfImgW, rfImgH)
      };
      rfDraw();
    }
  };

  wrap.onmousemove = (e) => {
    // Always track mouse for brush cursor preview
    rfMousePos = rfCanvasToImg(e);

    if (rfPanning) {
      pvX = e.clientX - pvStartX;
      pvY = e.clientY - pvStartY;
      rfDraw();
      return;
    }

    // Lasso node dragging
    if (rfDraggingNode >= 0 && rfTool === 'lasso') {
      rfLassoPoints[rfDraggingNode] = rfMousePos;
      rfDraw();
      return;
    }

    // Lasso freehand: activate only after significant drag (20px + 200ms held)
    if (rfTool === 'lasso' && rfLassoDownXY && (e.buttons & 1)) {
      if (!rfLassoFreehand) {
        const ddx = e.clientX - rfLassoDownXY.cx;
        const ddy = e.clientY - rfLassoDownXY.cy;
        const elapsed = Date.now() - (rfLassoDownXY.t || 0);
        // Require both: 20px distance AND 200ms held down
        if (ddx * ddx + ddy * ddy < 400 || elapsed < 200) return;
        rfLassoFreehand = true;
      }
      const pos = rfMousePos;
      const last = rfLassoPoints[rfLassoPoints.length - 1];
      const dx = (pos.x - last.x) * rfImgW;
      const dy = (pos.y - last.y) * rfImgH;
      if (dx * dx + dy * dy > 9) {
        rfLassoPoints.push(pos);
        rfDraw();
      }
      return;
    }

    if (rfDrawing && rfCurStroke) {
      const pos = rfMousePos;
      const last = rfCurStroke.points[rfCurStroke.points.length - 1];
      const dx = (pos.x - last.x) * rfImgW;
      const dy = (pos.y - last.y) * rfImgH;
      if (dx * dx + dy * dy > 4) {
        rfCurStroke.points.push(pos);
        rfDraw();
      }
      return;
    }

    // Lasso hover: check for node proximity and change cursor
    if (rfTool === 'lasso' && rfLassoPoints.length > 0) {
      // Check proximity to first node (close cursor)
      if (rfLassoPoints.length >= 3) {
        const rect2 = rfDisplayCanvas.getBoundingClientRect();
        const mx2 = e.clientX - rect2.left, my2 = e.clientY - rect2.top;
        const fx2 = rfLassoPoints[0].x * rfImgW * pvZoom + pvX;
        const fy2 = rfLassoPoints[0].y * rfImgH * pvZoom + pvY;
        if ((mx2 - fx2) * (mx2 - fx2) + (my2 - fy2) * (my2 - fy2) <= 12 * 12) {
          wrap.style.cursor = 'pointer';
          rfDraw();
          return;
        }
      }
      const hitIdx = rfHitTestLassoNode(e);
      wrap.style.cursor = hitIdx >= 0 ? 'move' : 'crosshair';
    } else if (rfTool === 'hand') {
      wrap.style.cursor = 'grab';
    } else if (rfTool === 'brush') {
      wrap.style.cursor = 'none';  // hide default cursor, we draw our own
    } else {
      wrap.style.cursor = 'crosshair';
    }

    // Only redraw cursor overlay (cheap) when hovering with brush
    // Full rfDraw() only when lasso has active points
    if (rfTool === 'brush') {
      rfDrawCursor();
    } else if (rfTool === 'lasso' && rfLassoPoints.length > 0) {
      rfDraw();
    }
  };

  wrap.onmouseup = (e) => {
    if (rfPanning) {
      rfPanning = false;
      wrap.classList.remove('dragging');
      wrap.style.cursor = (rfTool === 'hand' || spaceHeld) ? 'grab' : (rfTool === 'brush' ? 'none' : 'crosshair');
      return;
    }
    if (rfDraggingNode >= 0) {
      rfDraggingNode = -1;
      wrap.style.cursor = 'crosshair';
      rfDraw();
      return;
    }
    // Lasso: stop freehand drawing (keep points, don't close)
    if (rfTool === 'lasso' && rfLassoDownXY) {
      rfLassoDownXY = null;
      rfLassoFreehand = false;
      rfDraw();
      return;
    }
    if (rfDrawing && rfCurStroke) {
      rfDrawing = false;
      if (rfCurStroke.points.length > 0) {
        rfStrokes.push(rfCurStroke);
        rfApplyStrokeToCanvas(rfCurStroke);
        rfScheduleAutoSave();
      }
      rfCurStroke = null;
      rfDraw();
    }
  };

  wrap.onmouseleave = () => {
    rfMousePos = null;
    rfDraw();
  };

  wrap.ondblclick = (e) => {
    if (rfTool === 'lasso' && rfLassoPoints.length >= 3) {
      e.preventDefault();
      rfRedoStack = [];
      const stroke = { type: 'lasso', mode: rfAction, points: [...rfLassoPoints], radius: 0 };
      rfStrokes.push(stroke);
      rfApplyStrokeToCanvas(stroke);
      rfLassoPoints = [];
      rfDraw();
      rfScheduleAutoSave();
    }
  };

  wrap.oncontextmenu = (e) => e.preventDefault();

  // Keyboard shortcuts for refine mode
  const rfKeyHandler = (e) => {
    if (!rfMode) { document.removeEventListener('keydown', rfKeyHandler); return; }
    if (e.ctrlKey && e.key === 'z') {
      e.preventDefault();
      // If placing lasso points, undo last point first
      if (rfTool === 'lasso' && rfLassoPoints.length > 0) {
        rfLassoPoints.pop();
        rfDraw();
      } else {
        rfUndo();
      }
      return;
    }
    if (e.ctrlKey && e.key === 'y') { e.preventDefault(); rfRedo(); }
    if (e.ctrlKey && e.key === 's') { e.preventDefault(); rfApply(); }
    if (e.key === 'b' || e.key === 'B') {
      const btn = document.querySelector('.rf-btn[title*="Brocha"]');
      if (btn) rfSetTool(btn, 'brush');
    }
    if (e.key === 'l' || e.key === 'L') {
      const btn = document.querySelector('.rf-btn[title*="Lazo"]');
      if (btn) rfSetTool(btn, 'lasso');
    }
    if (e.key === 'h' || e.key === 'H') {
      const btn = document.querySelector('.rf-btn[title*="Mano"]');
      if (btn) rfSetTool(btn, 'hand');
    }
    if (e.key === 'r' || e.key === 'R') {
      if (rfDrawing) return; // Don't switch mid-stroke
      rfAction = 'restore';
      document.querySelectorAll('.rf-btn.restore,.rf-btn.erase').forEach(b => b.classList.remove('active'));
      const btn = document.querySelector('.rf-btn.restore');
      if (btn) btn.classList.add('active');
      rfDraw();
    }
    if (e.key === 'e' || e.key === 'E') {
      if (rfDrawing) return; // Don't switch mid-stroke
      rfAction = 'erase';
      document.querySelectorAll('.rf-btn.restore,.rf-btn.erase').forEach(b => b.classList.remove('active'));
      const btn = document.querySelector('.rf-btn.erase');
      if (btn) btn.classList.add('active');
      rfDraw();
    }
    if (e.key === 'Escape') {
      // Priority: cancel lasso → exit fullscreen → exit refine
      if (rfLassoPoints.length > 0) {
        rfLassoPoints = [];
        rfLassoFreehand = false;
        rfLassoDownXY = null;
        rfDraw();
      } else if (document.fullscreenElement || document.getElementById('previewPanel')?.classList.contains('faux-fullscreen')) {
        rfToggleFullscreen();
      } else {
        exitRefineMode();
      }
    }
    if (e.key === 'f' || e.key === 'F') { rfToggleFullscreen(); }
    if (e.key === '[') { rfBrushSize = Math.max(2, rfBrushSize - 3); rfUpdateSizeUI(); }
    if (e.key === ']') { rfBrushSize = Math.min(80, rfBrushSize + 3); rfUpdateSizeUI(); }
  };
  document.addEventListener('keydown', rfKeyHandler);

  // --- Pointer Events: touchpad + touchscreen free pan (no axis lock) ---
  // Replaces old touch handlers. Works for both Windows Precision Touchpad
  // (2-finger scroll) and mobile/tablet touchscreens.
  let rfPtrs = {}, rfPtrCenter = null, rfPtrDist = 0, rfPtrPanning = false;

  wrap.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse') return;  // mouse uses existing mousedown handler
    wrap.setPointerCapture(e.pointerId);
    rfPtrs[e.pointerId] = { x: e.clientX, y: e.clientY };
    const cnt = Object.keys(rfPtrs).length;
    if (cnt >= 2) {
      // 2+ fingers → pan/zoom gesture
      rfPtrCenter = ptrCenter(rfPtrs);
      rfPtrDist = ptrDist(rfPtrs);
      rfPtrPanning = true;
      // Cancel any drawing in progress
      if (rfDrawing) { rfDrawing = false; rfCurStroke = null; }
      if (rfLassoFreehand) { rfLassoFreehand = false; rfLassoDownXY = null; }
    } else if (cnt === 1) {
      // Single finger on touchscreen = draw (forward to mouse handler)
      const fakeE = { clientX: e.clientX, clientY: e.clientY, button: 0, altKey: false, preventDefault: () => {} };
      wrap.onmousedown(fakeE);
    }
  });

  wrap.addEventListener('pointermove', (e) => {
    if (e.pointerType === 'mouse' || !rfPtrs[e.pointerId]) return;
    rfPtrs[e.pointerId] = { x: e.clientX, y: e.clientY };
    const ids = Object.keys(rfPtrs);
    if (ids.length >= 2 && rfPtrPanning) {
      const c = ptrCenter(rfPtrs);
      const d = ptrDist(rfPtrs);
      // Free pan (no axis restriction!)
      if (rfPtrCenter) {
        pvX += c.x - rfPtrCenter.x;
        pvY += c.y - rfPtrCenter.y;
      }
      // Pinch zoom
      if (rfPtrDist > 0 && d > 0) {
        const rect = wrap.getBoundingClientRect();
        const mx = c.x - rect.left, my = c.y - rect.top;
        const oldZoom = pvZoom;
        pvZoom = Math.max(0.05, Math.min(20, pvZoom * (d / rfPtrDist)));
        pvX = mx - (mx - pvX) * (pvZoom / oldZoom);
        pvY = my - (my - pvY) * (pvZoom / oldZoom);
      }
      rfPtrCenter = c;
      rfPtrDist = d;
      rfDraw();
    } else if (ids.length === 1 && !rfPtrPanning) {
      // Single finger on touchscreen = draw (forward to mouse handler)
      const fakeE = { clientX: e.clientX, clientY: e.clientY };
      wrap.onmousemove(fakeE);
    }
  });

  const rfPtrClean = (e) => {
    if (e.pointerType === 'mouse') return;
    delete rfPtrs[e.pointerId];
    const cnt = Object.keys(rfPtrs).length;
    if (cnt < 2) { rfPtrCenter = null; rfPtrDist = 0; }
    if (cnt === 0) {
      if (rfPtrPanning) {
        rfPtrPanning = false;
      } else {
        // Single finger ended → forward mouseup
        const fakeE = { clientX: e.clientX, clientY: e.clientY };
        wrap.onmouseup(fakeE);
      }
    }
  };
  wrap.addEventListener('pointerup', rfPtrClean);
  wrap.addEventListener('pointercancel', rfPtrClean);

  // Handle window resize
  const rfResize = () => {
    if (!rfMode) return;
    const rect = wrap.getBoundingClientRect();
    rfDisplayCanvas.width = rect.width;
    rfDisplayCanvas.height = rect.height;
    rfCursorCanvas.width = rect.width;
    rfCursorCanvas.height = rect.height;
    rfDraw();
  };
  window.addEventListener('resize', rfResize);
}

function rfUpdateSizeUI() {
  const slider = document.querySelector('.rf-size-slider');
  const label = document.getElementById('rfSizeVal');
  if (slider) slider.value = rfBrushSize;
  if (label) label.textContent = rfBrushSize;
}

function _toggleRealFullscreen(panel, btnId, resizeCb) {
  if (document.fullscreenElement === panel) {
    document.exitFullscreen().catch(() => {});
  } else {
    panel.requestFullscreen().catch(() => {
      // Fallback to faux-fullscreen if API unavailable
      panel.classList.toggle('faux-fullscreen');
      _onFullscreenResize(panel, btnId, resizeCb);
    });
  }
}

function _onFullscreenResize(panel, btnId, resizeCb) {
  const isFull = document.fullscreenElement === panel || panel.classList.contains('faux-fullscreen');
  const btn = document.getElementById(btnId);
  if (btn) btn.innerHTML = isFull ? '&#9974; Salir' : '&#9974; Completa';
  setTimeout(resizeCb, 80);
}

document.addEventListener('fullscreenchange', () => {
  const panel = document.getElementById('previewPanel');
  if (!panel) return;
  if (rfMode) {
    _onFullscreenResize(panel, 'rfFullscreenBtn', () => {
      const wrap = document.getElementById('rfWrap');
      if (wrap && rfDisplayCanvas && rfMode) {
        const rect = wrap.getBoundingClientRect();
        rfDisplayCanvas.width = rect.width; rfDisplayCanvas.height = rect.height;
        rfCursorCanvas.width = rect.width; rfCursorCanvas.height = rect.height;
        rfZoomFit();
      }
    });
  } else if (blMode) {
    _onFullscreenResize(panel, 'blFullscreenBtn', () => {
      const wrap = document.getElementById('blWrap');
      if (wrap && blCanvas && blMode) {
        const rect = wrap.getBoundingClientRect();
        blCanvas.width = rect.width; blCanvas.height = rect.height;
        blZoomFit();
      }
    });
  } else if (tgMode) {
    _onFullscreenResize(panel, 'tgFullscreenBtn', () => {
      const wrap = document.getElementById('tgWrap');
      if (wrap && tgCanvas && tgMode) {
        const rect = wrap.getBoundingClientRect();
        tgCanvas.width = rect.width; tgCanvas.height = rect.height;
        tgZoomFit();
      }
    });
  }
});

function rfToggleFullscreen() {
  const panel = document.getElementById('previewPanel');
  if (!panel) return;
  _toggleRealFullscreen(panel, 'rfFullscreenBtn', () => {
    const wrap = document.getElementById('rfWrap');
    if (wrap && rfDisplayCanvas && rfMode) {
      const rect = wrap.getBoundingClientRect();
      rfDisplayCanvas.width = rect.width; rfDisplayCanvas.height = rect.height;
      rfCursorCanvas.width = rect.width; rfCursorCanvas.height = rect.height;
      rfZoomFit();
    }
  });
}

async function rfApply() {
  // Cancel any pending auto-save timer — we're saving now
  if (rfAutoSaveTimer) { clearTimeout(rfAutoSaveTimer); rfAutoSaveTimer = null; }
  if (rfStrokes.length === 0) { rfUpdateSaveStatus('saved'); showToast('Todo guardado', 'ok'); return; }
  if (rfAutoSaving) { showToast('Guardando...', 'info'); return; }

  rfAutoSaving = true;
  rfUpdateSaveStatus('saving');
  showToast('Guardando...', 'info');

  // Snapshot how many strokes we're saving — any strokes added during
  // the async save (by the user continuing to edit) must be preserved.
  const savedCount = rfStrokes.length;
  const serverStrokes = rfStrokes.slice(0, savedCount).map(s => ({
    type: s.type, mode: s.mode,
    points: s.points,
    radius: s.radius
  }));

  try {
    const res = await fetch(`/api/cutout/refine/${rfCbtis}/${encodeURIComponent(rfFilename)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ strokes: serverStrokes })
    });
    const data = await res.json();
    if (data.ok) {
      // Remove only the strokes that were saved; keep any added during the fetch
      const newStrokes = rfStrokes.slice(savedCount);
      rfStrokes = newStrokes;
      rfRedoStack = [];
      // Reload the cutout image with cache-bust
      const newCut = await loadImg(`/api/cutout/preview-scaled/${rfCbtis}/${encodeURIComponent(rfFilename)}?maxw=99999&_=${Date.now()}`);
      if (!rfMode) { rfAutoSaving = false; return; }
      rfCutImg = newCut;
      rfWorkCanvas.width = newCut.width;
      rfWorkCanvas.height = newCut.height;
      rfRecomposite();  // Redraws from fresh image + any unsaved strokes
      rfDraw();
      rfUpdateSaveStatus(rfStrokes.length > 0 ? 'pending' : 'saved');
      if (rfStrokes.length === 0) showToast('Guardado correctamente', 'ok');
      refreshGroups();
    } else {
      rfUpdateSaveStatus('error');
      showToast('Error: ' + data.error, 'error');
    }
  } catch (e) {
    rfUpdateSaveStatus('error');
    showToast('Error: ' + e.message, 'error');
  }
  rfAutoSaving = false;
}

function rfScheduleAutoSave() {
  if (rfAutoSaveTimer) clearTimeout(rfAutoSaveTimer);
  if (rfStrokes.length === 0) { rfUpdateSaveStatus('saved'); return; }
  rfUpdateSaveStatus('pending');
  rfAutoSaveTimer = setTimeout(() => { rfAutoSaveTimer = null; rfAutoSave(); }, 2000);
}

async function rfAutoSave() {
  if (rfAutoSaving || rfStrokes.length === 0 || !rfMode) return;
  rfAutoSaving = true;
  rfUpdateSaveStatus('saving');
  const savedCount = rfStrokes.length;
  try {
    const serverStrokes = rfStrokes.slice(0, savedCount).map(s => ({
      type: s.type, mode: s.mode, points: s.points, radius: s.radius
    }));
    const res = await fetch(`/api/cutout/refine/${rfCbtis}/${encodeURIComponent(rfFilename)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ strokes: serverStrokes })
    });
    const data = await res.json();
    if (data.ok && rfMode) {
      const newStrokes = rfStrokes.slice(savedCount);
      rfStrokes = newStrokes;
      rfRedoStack = [];
      const newCut = await loadImg(`/api/cutout/preview-scaled/${rfCbtis}/${encodeURIComponent(rfFilename)}?maxw=99999&_=${Date.now()}`);
      if (!rfMode) { rfAutoSaving = false; return; }
      rfCutImg = newCut;
      rfWorkCanvas.width = newCut.width;
      rfWorkCanvas.height = newCut.height;
      rfRecomposite();
      rfDraw();
      rfUpdateSaveStatus(rfStrokes.length > 0 ? 'pending' : 'saved');
    } else {
      rfUpdateSaveStatus('error');
    }
  } catch (e) {
    rfUpdateSaveStatus('error');
  }
  rfAutoSaving = false;
}

function rfUpdateSaveStatus(state) {
  const el = document.getElementById('rfSaveStatus');
  if (!el) return;
  if (state === 'pending') { el.textContent = '\u25cf Sin guardar'; el.style.color = '#e0af68'; }
  else if (state === 'saving') { el.textContent = '\u231b Guardando...'; el.style.color = '#7dcfff'; }
  else if (state === 'saved') { el.textContent = '\u2713 Guardado'; el.style.color = '#9ece6a'; }
  else if (state === 'error') { el.textContent = '\u2717 Error al guardar'; el.style.color = '#f7768e'; }
  else { el.textContent = ''; }
}

async function rfExportTiff() {
  if (rfStrokes.length > 0) {
    await rfApply();
  }
  const url = `/api/cutout/export-tiff/${rfCbtis}/${encodeURIComponent(rfFilename)}`;
  showToast('Generando TIFF con capas... espera', 'info');
  try {
    const res = await fetch(url);
    if (!res.ok) { showToast('Error al generar TIFF: ' + res.status, 'error'); return; }
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    const stem = rfFilename.replace(/\.[^.]+$/, '');
    a.download = stem + '_capas.tif';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
    showToast('TIFF descargado', 'ok');
  } catch (e) {
    showToast('Error: ' + e.message, 'error');
  }
}

// --- Borlas Mode ---
let blMode = false, blCbtis = '', blFilename = '';
let blCutImg = null, blBorlaImg = null;
let blFaces = [], blBorlas = [];
let blSelectedColor = '', blAvailableColors = [];
let blRotMin = -15, blRotMax = 15;
let blScalePct = 150;  // borla height = face_height * blScalePct/100
let blTool = 'move';  // 'move' | 'add' | 'brush' | 'hilo'
let blBrushSize = 20, blBrushAction = 'erase';  // erase/restore parts of a borla
let blDragging = null, blRotating = null, blScaling = null, blSelected = -1;
let blBrushDrawing = false;
let blHiloDrawing = false;  // true while drawing a hilo path
let blMousePos = null;  // {sx, sy} screen coords for brush cursor
let blUndoStack = [], blRedoStack = [];  // mask snapshots for undo/redo
let blCanvas = null, blCtx = null;
let blImgW = 1, blImgH = 1;
let blShowFaceRects = false;  // briefly show face rectangles after detection
let blFaceRectsTimer = null;
let blBaseScale = 20;  // uniform borla width in px (computed from avg face height)
// Hilo (thread) tool settings — persistent via localStorage
let blHiloSize = 3, blHiloFeather = 1, blHiloFlow = 100;
let blHiloColor = '#d4a017';  // active hilo color (custom or auto)
let blHiloAutoColor = '#d4a017';  // auto-extracted dominant color (backup for reset)
// zoom/pan reuses pvZoom, pvX, pvY from preview system

const BL_COLOR_MAP = {
  AZUL:'#1a6dd4', BLANCO:'#e8e8e8', CAFE:'#6b3a2a', CELESTE:'#7ec8e3',
  DORADO:'#d4a017', MARRON:'#5c3317', PLATEADO:'#b0b0b0', ROJO:'#c0392b',
  VERDE:'#27ae60', GUINDA:'#800020', NEGRO:'#222222'
};

async function enterBorlasMode(cbtis, filename) {
  blCbtis = cbtis; blFilename = filename;
  blBorlas = []; blFaces = []; blSelected = -1; blDragging = null; blScaling = null;
  blShowFaceRects = false;
  if (blFaceRectsTimer) { clearTimeout(blFaceRectsTimer); blFaceRectsTimer = null; }
  // Load persistent rotation range from localStorage
  try {
    const savedRotMin = localStorage.getItem('bl_rotMin');
    const savedRotMax = localStorage.getItem('bl_rotMax');
    if (savedRotMin !== null) blRotMin = +savedRotMin;
    if (savedRotMax !== null) blRotMax = +savedRotMax;
    const hs = localStorage.getItem('bl_hiloSize');
    const hf = localStorage.getItem('bl_hiloFeather');
    const hw = localStorage.getItem('bl_hiloFlow');
    if (hs !== null) blHiloSize = +hs;
    if (hf !== null) blHiloFeather = +hf;
    if (hw !== null) blHiloFlow = +hw;
  } catch(e) {}

  const panel = document.getElementById('previewPanel');
  panel.innerHTML = '<div class="rf-loading">Cargando modo Borlas...</div>';

  // Fetch borla colors + suggested color
  try {
    const lRes = await fetch(`/api/borlas/list?group=${encodeURIComponent(filename)}`);
    const lData = await lRes.json();
    blAvailableColors = lData.colors;
    blSelectedColor = lData.suggested || (lData.colors[0] || 'DORADO');
  } catch (e) {
    showToast('Error cargando borlas: ' + e.message, 'error');
    return;
  }

  // Load cutout image
  const encName = encodeURIComponent(filename);
  try {
    blCutImg = await loadImg(`/api/cutout/preview-scaled/${cbtis}/${encName}?maxw=99999&_=${Date.now()}`);
  } catch (e) {
    showToast('Error cargando imagen: ' + e.message, 'error');
    return;
  }
  blImgW = blCutImg.width;
  blImgH = blCutImg.height;

  // Load borla image
  try {
    blBorlaImg = await loadImg(`/api/borlas/image/${blSelectedColor}?maxw=800`);
    blHiloAutoColor = blExtractDominantColor();
    const customC = localStorage.getItem('bl_hiloCustomColor_' + blSelectedColor);
    blHiloColor = customC || blHiloAutoColor;
  } catch (e) {
    blBorlaImg = null;
  }

  // Build color options
  const colorOpts = blAvailableColors.map(c => {
    const sel = c === blSelectedColor ? 'selected' : '';
    return `<option value="${c}" ${sel}>${c}</option>`;
  }).join('');

  // Build UI
  panel.innerHTML = `
    <div class="borlas-toolbar">
      <button class="rf-btn active" id="blToolMove" onclick="blSetTool('move')" title="Mover borlas (V)">&#9995; Mover</button>
      <button class="rf-btn" id="blToolAdd" onclick="blSetTool('add')" title="Click para agregar borla donde falte (A)">&#10010; Agregar</button>
      <button class="rf-btn" id="blToolBrush" onclick="blSetTool('brush')" title="Brocha: borrar/restaurar partes de borla+hilo (B)">&#9998; Brocha</button>
      <span id="blBrushControls" style="display:none">
        <button class="rf-btn active" id="blBrushErase" onclick="blSetBrushAction('erase')" title="Borrar parte (X alterna)">&#10007; Borrar</button>
        <button class="rf-btn" id="blBrushRestore" onclick="blSetBrushAction('restore')" title="Restaurar parte (X alterna)">&#10003; Restaurar</button>
        <input type="range" class="rf-size-slider" min="2" max="80" value="${blBrushSize}" style="width:60px"
               oninput="blBrushSize=+this.value; document.getElementById('blBrushSizeVal').textContent=this.value">
        <span class="rf-size-label" id="blBrushSizeVal">${blBrushSize}</span>px
      </span>
      <button class="rf-btn" id="blToolHilo" onclick="blSetTool('hilo')" title="Hilo: dibujar hilo/cuerda de borla (H)">&#128697; Hilo</button>
      <span id="blHiloControls" style="display:none">
        <label>Tam:</label>
        <input type="range" class="rf-size-slider" min="1" max="30" value="${blHiloSize}" style="width:50px"
               oninput="blHiloSize=+this.value; document.getElementById('blHiloSizeVal').textContent=this.value; blPersistHiloSettings()">
        <span class="rf-size-label" id="blHiloSizeVal">${blHiloSize}</span>
        <label>Feath:</label>
        <input type="range" class="rf-size-slider" min="0" max="10" step="0.5" value="${blHiloFeather}" style="width:40px"
               oninput="blHiloFeather=+this.value; document.getElementById('blHiloFeatherVal').textContent=this.value; blPersistHiloSettings()">
        <span class="rf-size-label" id="blHiloFeatherVal">${blHiloFeather}</span>
        <label>Flujo:</label>
        <input type="range" class="rf-size-slider" min="10" max="100" value="${blHiloFlow}" style="width:40px"
               oninput="blHiloFlow=+this.value; document.getElementById('blHiloFlowVal').textContent=this.value+'%'; blPersistHiloSettings()">
        <span class="rf-size-label" id="blHiloFlowVal">${blHiloFlow}%</span>
        <span id="blHiloColorSwatch" onclick="document.getElementById('blHiloColorPicker').click()"
              style="display:inline-block;width:16px;height:16px;border-radius:3px;border:1px solid #555;vertical-align:middle;background:${blHiloColor};cursor:pointer"
              title="Click para elegir color personalizado"></span>
        <input type="color" id="blHiloColorPicker" value="${blHiloColorToHex(blHiloColor)}"
               style="width:0;height:0;padding:0;border:0;position:absolute;opacity:0"
               oninput="blSetCustomHiloColor(this.value)">
        <button class="rf-btn" id="blHiloColorReset" onclick="blResetHiloColor()"
                title="Restablecer color auto-detectado"
                style="display:${blHiloColor !== blHiloAutoColor ? 'inline-block' : 'none'};font-size:11px;padding:1px 4px">&#8634;</button>
        <button class="rf-btn" onclick="blHiloDelete()" title="Eliminar hilo de la borla seleccionada (Del)">&#128465; Borrar</button>
      </span>
      <div class="bl-sep"></div>
      <label>Color:</label>
      <select id="blColorSel" onchange="blChangeColor(this.value)">${colorOpts}</select>
      <div class="bl-sep"></div>
      <label>Escala:</label>
      <input type="range" min="5" max="200" value="${blScalePct}" id="blScaleSlider" style="width:70px"
             oninput="blScalePct=+this.value; document.getElementById('blScaleInput').value=this.value; blApplyScale()">
      <input type="number" min="5" max="200" value="${blScalePct}" id="blScaleInput" style="width:42px"
             onchange="blScalePct=+this.value; document.getElementById('blScaleSlider').value=this.value; blApplyScale()">%
      <div class="bl-sep"></div>
      <label>Rot:</label>
      <input type="number" id="blRotMinInput" value="${blRotMin}" min="-180" max="0" oninput="blRotMin=+this.value; blPersistRotRange()">°
      <span style="color:var(--fg3)">a</span>
      <input type="number" id="blRotMaxInput" value="${blRotMax}" min="0" max="180" oninput="blRotMax=+this.value; blPersistRotRange()">°
      <button class="rf-btn" onclick="blRandomizeRotations()" title="Re-aleatorizar rotaciones">&#8635;</button>
      <div class="bl-sep"></div>
      <button class="rf-btn" onclick="blDetectFaces()" title="Re-detectar caras">&#128269; Detectar</button>
      <div style="flex:1"></div>
      <button class="rf-btn primary" onclick="blExportTiff()">&#8615; TIFF</button>
      <button class="rf-btn" onclick="exitBorlasMode(); enterTogasMode('${cbtis}','${filename.replace(/'/g, "\\'")}')" title="Ir a modo Togas">&#127891; Togas</button>
      <button class="rf-btn done-toggle ${getWorkflowDoneState('borlas') ? 'is-done' : ''}" onclick="toggleWorkflowDone('borlas', this)" title="Marcar borlas como terminadas">${getWorkflowDoneState('borlas') ? '\u2714 Listo' : '\u2610 Listo'}</button>
      <button class="rf-btn danger" onclick="exitBorlasMode()">&#10005; Salir</button>
      <div class="bl-sep"></div>
      <button class="rf-btn" onclick="blZoomFit()">Ajustar</button>
      <span class="rf-size-label" id="blZoomLabel">100%</span>
      <button class="rf-btn" onclick="blToggleFullscreen()" id="blFullscreenBtn">&#9974; Completa</button>
    </div>
    <div class="rf-canvas-wrap" id="blWrap" style="touch-action:none">
      <canvas id="blCanvas"></canvas>
    </div>
    <div class="bl-list" id="blList"></div>
    <div class="preview-info">${filename} (borlas)</div>
  `;

  blCanvas = document.getElementById('blCanvas');
  blCtx = blCanvas.getContext('2d');
  blMode = true;

  setTimeout(() => {
    const wrap = document.getElementById('blWrap');
    if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    blCanvas.width = Math.max(rect.width, 200);
    blCanvas.height = Math.max(rect.height, 200);
    pvZoom = 1; pvX = 0; pvY = 0;
    blZoomFit();
    blBindEvents(wrap);
    // Try loading saved state, otherwise auto-detect faces
    blLoadState().then(loaded => {
      if (loaded && blBorlas.length > 0) {
        showToast(`${blBorlas.length} borlas restauradas`, 'ok');
        blRenderList();
        blDraw();
      } else {
        blDetectFaces();
      }
    });
  }, 50);
}

function exitBorlasMode() {
  if (blHiloDrawing) blHiloFinish();
  blSaveState();
  blMode = false;
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  const panel = document.getElementById('previewPanel');
  if (panel) panel.classList.remove('faux-fullscreen');
  if (blCbtis && blFilename) {
    initPreviewPanel(
      `/api/cutout/preview/${blCbtis}/${encodeURIComponent(blFilename)}?t=${Date.now()}`,
      `${blFilename} (recortada)`, true
    );
  }
  refreshGroups();
}

function blSaveState() {
  if (!blCbtis || !blFilename || blBorlas.length === 0) return;
  const state = {
    scalePct: blScalePct, rotMin: blRotMin, rotMax: blRotMax,
    baseScale: blBaseScale,
    selectedColor: blSelectedColor,
    hiloColor: blHiloColor,
    hiloSize: blHiloSize,
    hiloFeather: blHiloFeather,
    hiloFlow: blHiloFlow,
    borlas: blBorlas.map(b => ({
      x: b.x, y: b.y, scale: b.scale,
      rotation: b.rotation, visible: b.visible, color: b.color,
      mask: b.mask ? b.mask.toDataURL('image/png') : null,
      hilo: b.hilo || null,
      hiloMask: b.hiloMask ? b.hiloMask.toDataURL('image/png') : null
    }))
  };
  fetch(`/api/borlas/state/${blCbtis}/${encodeURIComponent(blFilename)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(state)
  }).catch(() => {});
}

async function blLoadState() {
  try {
    const res = await fetch(`/api/borlas/state/${blCbtis}/${encodeURIComponent(blFilename)}`);
    const state = await res.json();
    if (!state || !state.borlas) return false;
    blScalePct = state.scalePct || 150;
    // Rotation range: prefer localStorage (persistent), fall back to per-photo state
    if (localStorage.getItem('bl_rotMin') === null) blRotMin = state.rotMin ?? -15;
    if (localStorage.getItem('bl_rotMax') === null) blRotMax = state.rotMax ?? 15;
    blBaseScale = state.baseScale || 20;
    if (state.selectedColor && blAvailableColors.includes(state.selectedColor)) {
      blSelectedColor = state.selectedColor;
    }
    // Restore hilo settings — color comes from localStorage (global), not per-photo
    // blHiloColor is already set by enterBorlasMode (custom or auto-detect)
    if (state.hiloSize) blHiloSize = state.hiloSize;
    if (state.hiloFeather != null) blHiloFeather = state.hiloFeather;
    if (state.hiloFlow) blHiloFlow = state.hiloFlow;
    blBorlas = state.borlas.map((b, i) => ({
      id: i, faceIdx: -1,
      x: b.x, y: b.y, scale: b.scale,
      rotation: b.rotation, visible: b.visible, color: b.color,
      mask: null, _maskData: b.mask,
      hilo: b.hilo || null,
      hiloMask: null, _hiloMaskData: b.hiloMask || null
    }));
    // Restore masks asynchronously
    for (const b of blBorlas) {
      if (b._maskData) {
        const img = await loadImg(b._maskData);
        const c = document.createElement('canvas');
        c.width = img.width; c.height = img.height;
        c.getContext('2d').drawImage(img, 0, 0);
        b.mask = c;
        delete b._maskData;
      }
      if (b._hiloMaskData) {
        const img = await loadImg(b._hiloMaskData);
        const c = document.createElement('canvas');
        c.width = img.width; c.height = img.height;
        c.getContext('2d').drawImage(img, 0, 0);
        b.hiloMask = c;
        delete b._hiloMaskData;
      }
    }
    return true;
  } catch { return false; }
}

async function blDetectFaces() {
  // Show loading state on button
  const detectBtn = document.querySelector('.borlas-toolbar .rf-btn[onclick*="blDetectFaces"]');
  if (detectBtn) { detectBtn.disabled = true; detectBtn.innerHTML = '&#9203; Detectando...'; }
  showToast('Detectando caras...', 'info');
  try {
    const res = await fetch(`/api/borlas/detect-faces/${blCbtis}/${encodeURIComponent(blFilename)}`);
    const data = await res.json();
    if (detectBtn) { detectBtn.disabled = false; detectBtn.innerHTML = '&#128269; Detectar'; }
    if (!data.ok) { showToast('Error: ' + (data.error || 'respuesta inválida'), 'error'); return; }
    blFaces = data.faces;
    if (blFaces.length === 0) { showToast('No se detectaron caras', 'error'); return; }
    // Compute uniform scale from average face height
    const borlaAR = blBorlaImg ? blBorlaImg.height / blBorlaImg.width : 15;
    const avgFaceH = blFaces.reduce((s, f) => s + f.h * blImgH, 0) / blFaces.length;
    const targetH = avgFaceH * blScalePct / 100;
    blBaseScale = targetH / borlaAR;  // uniform borla width for all
    // Create borla for each face — all same size
    blBorlas = blFaces.map((face, i) => {
      const faceCenterX = (face.x + face.w / 2) * blImgW;
      const faceTop = face.y * blImgH;
      const rotation = blRotMin + Math.random() * (blRotMax - blRotMin);
      return {
        id: i, faceIdx: i,
        x: faceCenterX,  // top-center x (pixels in full-res image)
        y: faceTop - (face.h * blImgH) * 0.3,  // slightly above face top
        scale: blBaseScale,  // uniform borla width in pixels
        rotation: Math.round(rotation * 10) / 10,
        visible: true,
        color: blSelectedColor,
        mask: null  // per-borla erase/restore mask canvas (created on demand)
      };
    });
    showToast(`${blFaces.length} caras detectadas — borlas colocadas`, 'ok');
    // Show face rectangles briefly for visual feedback
    blShowFaceRects = true;
    if (blFaceRectsTimer) clearTimeout(blFaceRectsTimer);
    blFaceRectsTimer = setTimeout(() => { blShowFaceRects = false; blDraw(); }, 3000);
    blSelected = -1;
    blRenderList();
    blDraw();
  } catch (e) {
    if (detectBtn) { detectBtn.disabled = false; detectBtn.innerHTML = '&#128269; Detectar'; }
    showToast('Error detectando: ' + e.message, 'error');
  }
}

function blRandomizeRotations() {
  for (const b of blBorlas) {
    b.rotation = Math.round((blRotMin + Math.random() * (blRotMax - blRotMin)) * 10) / 10;
  }
  blRenderList();
  blDraw();
}

function blPersistRotRange() {
  try {
    localStorage.setItem('bl_rotMin', blRotMin);
    localStorage.setItem('bl_rotMax', blRotMax);
  } catch(e) {}
}

function blHitScaleHandle(sx, sy) {
  // Check if click is on the scale handle of the selected borla (bottom-right corner)
  if (blSelected < 0 || blBorlas.length === 0) return false;
  const b = blBorlas[blSelected];
  if (!b.visible) return false;
  const borlaAR = blBorlaImg ? blBorlaImg.height / blBorlaImg.width : 15;
  const borlaW = b.scale * pvZoom;
  const borlaH = borlaW * borlaAR;
  const scr = blImgToScreen(b.x, b.y);
  // Handle is at local (borlaW/2, borlaH), rotated by b.rotation
  const rad = b.rotation * Math.PI / 180;
  const lx = borlaW / 2, ly = borlaH;
  const hx = scr.x + lx * Math.cos(rad) - ly * Math.sin(rad);
  const hy = scr.y + lx * Math.sin(rad) + ly * Math.cos(rad);
  return Math.hypot(sx - hx, sy - hy) <= 12;
}

function blExtractDominantColor() {
  // Extract the most common opaque color from the borla image
  if (!blBorlaImg) return '#d4a017';
  const c = document.createElement('canvas');
  const sz = 64;
  c.width = sz; c.height = sz;
  const cx = c.getContext('2d');
  cx.drawImage(blBorlaImg, 0, 0, sz, sz);
  const d = cx.getImageData(0, 0, sz, sz).data;
  const counts = {};
  for (let i = 0; i < d.length; i += 4) {
    if (d[i+3] < 128) continue;  // skip transparent pixels
    // Quantize to 4-bit per channel
    const key = ((d[i] >> 4) << 8) | ((d[i+1] >> 4) << 4) | (d[i+2] >> 4);
    counts[key] = (counts[key] || 0) + 1;
  }
  let maxK = 0, maxN = 0;
  for (const k in counts) { if (counts[k] > maxN) { maxN = counts[k]; maxK = +k; } }
  const r = ((maxK >> 8) & 0xf) * 17;
  const g = ((maxK >> 4) & 0xf) * 17;
  const b = (maxK & 0xf) * 17;
  return `rgb(${r},${g},${b})`;
}

function blPersistHiloSettings() {
  try {
    localStorage.setItem('bl_hiloSize', blHiloSize);
    localStorage.setItem('bl_hiloFeather', blHiloFeather);
    localStorage.setItem('bl_hiloFlow', blHiloFlow);
  } catch(e) {}
}

function blSetCustomHiloColor(hexColor) {
  blHiloColor = hexColor;
  // Persist per borla color globally
  try { localStorage.setItem('bl_hiloCustomColor_' + blSelectedColor, hexColor); } catch(e) {}
  const swatch = document.getElementById('blHiloColorSwatch');
  if (swatch) swatch.style.background = hexColor;
  const resetBtn = document.getElementById('blHiloColorReset');
  if (resetBtn) resetBtn.style.display = 'inline-block';
  blDraw();
}

function blResetHiloColor() {
  blHiloColor = blHiloAutoColor;
  try { localStorage.removeItem('bl_hiloCustomColor_' + blSelectedColor); } catch(e) {}
  const swatch = document.getElementById('blHiloColorSwatch');
  if (swatch) swatch.style.background = blHiloColor;
  const picker = document.getElementById('blHiloColorPicker');
  if (picker) picker.value = blHiloColorToHex(blHiloColor);
  const resetBtn = document.getElementById('blHiloColorReset');
  if (resetBtn) resetBtn.style.display = 'none';
  blDraw();
}

function blApplyHiloColorForBorla(borlaColor) {
  // Set hilo color: use custom if stored, otherwise auto-detect
  blHiloAutoColor = blExtractDominantColor();
  const custom = localStorage.getItem('bl_hiloCustomColor_' + borlaColor);
  blHiloColor = custom || blHiloAutoColor;
  blUpdateHiloColorUI();
}

function blUpdateHiloColorUI() {
  const swatch = document.getElementById('blHiloColorSwatch');
  if (swatch) swatch.style.background = blHiloColor;
  const picker = document.getElementById('blHiloColorPicker');
  if (picker) picker.value = blHiloColorToHex(blHiloColor);
  const resetBtn = document.getElementById('blHiloColorReset');
  if (resetBtn) resetBtn.style.display = (blHiloColor !== blHiloAutoColor) ? 'inline-block' : 'none';
}

function blHiloColorToHex(color) {
  // Convert rgb(...) or hex to #rrggbb for input[type=color]
  if (color.startsWith('#')) return color.length === 7 ? color : '#d4a017';
  const m = color.match(/rgb\(\s*(\d+),\s*(\d+),\s*(\d+)\s*\)/);
  if (!m) return '#d4a017';
  const r = (+m[1]).toString(16).padStart(2, '0');
  const g = (+m[2]).toString(16).padStart(2, '0');
  const b = (+m[3]).toString(16).padStart(2, '0');
  return '#' + r + g + b;
}

function blImgToLocal(b, imgX, imgY) {
  // Convert image coords to borla-local coords (un-rotate around pivot)
  const rad = -b.rotation * Math.PI / 180;
  const dx = imgX - b.x, dy = imgY - b.y;
  return {
    x: dx * Math.cos(rad) - dy * Math.sin(rad),
    y: dx * Math.sin(rad) + dy * Math.cos(rad)
  };
}

function blHiloAddVertex(imgX, imgY) {
  // Add a vertex to the hilo of the selected borla (click-to-click line segments)
  if (blSelected < 0) return;
  const b = blBorlas[blSelected];
  const pt = blImgToLocal(b, imgX, imgY);
  if (!b.hilo || !blHiloDrawing) {
    // First vertex — save undo snapshot, start new hilo, clear old mask
    blSaveHiloSnapshot(blSelected);
    b.hilo = [pt];
    b.hiloMask = null;
    blHiloDrawing = true;
  } else {
    // Subsequent vertex — add line segment
    b.hilo.push(pt);
  }
  blDraw();
}

function blHiloFinish() {
  // Finish the current hilo (stop adding vertices)
  if (!blHiloDrawing) return;
  blHiloDrawing = false;
  if (blSelected >= 0) {
    const b = blBorlas[blSelected];
    // Remove hilo if only 1 point (no line segments)
    if (b.hilo && b.hilo.length < 2) {
      b.hilo = null;
    }
  }
  blDraw();
  blSaveState();
}

function blHiloDelete() {
  // Delete the hilo of the selected borla
  if (blSelected < 0) return;
  const b = blBorlas[blSelected];
  if (!b.hilo) { showToast('No hay hilo para borrar', 'warn'); return; }
  blSaveHiloSnapshot(blSelected);
  b.hilo = null;
  b.hiloMask = null;
  blHiloDrawing = false;
  blDraw();
  blSaveState();
  showToast('Hilo eliminado', 'ok');
}

function blBrushOnHilo(b, imgX, imgY) {
  // Brush erase/restore on hilo: paint on hiloMask canvas
  if (!b.hilo || b.hilo.length < 2) return;
  const mask = blGetHiloMask(b);
  if (!mask) return;
  // Convert image coords to borla-local coords
  const rad = -b.rotation * Math.PI / 180;
  const dx = imgX - b.x, dy = imgY - b.y;
  const lx = dx * Math.cos(rad) - dy * Math.sin(rad);
  const ly = dx * Math.sin(rad) + dy * Math.cos(rad);
  // Convert local coords to mask coords
  const bbox = blHiloBBox(b);
  const mx = (lx - bbox.x) / bbox.w * mask.width;
  const my = (ly - bbox.y) / bbox.h * mask.height;
  const radius = blBrushSize * (mask.width / bbox.w);
  const ctx = mask.getContext('2d');
  ctx.globalCompositeOperation = blBrushAction === 'erase' ? 'destination-out' : 'source-over';
  ctx.fillStyle = blBrushAction === 'erase' ? '#000' : '#fff';
  ctx.beginPath();
  ctx.arc(mx, my, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalCompositeOperation = 'source-over';
}

function blHiloBBox(b) {
  // Compute bounding box of hilo path in local coords, with padding
  if (!b.hilo || b.hilo.length === 0) return {x:0, y:0, w:1, h:1};
  const pad = blHiloSize * 2 + blHiloFeather * 2 + 4;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of b.hilo) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  return {x: minX - pad, y: minY - pad, w: (maxX - minX) + pad * 2, h: (maxY - minY) + pad * 2};
}

function blGetHiloMask(b) {
  if (!b.hiloMask && b.hilo && b.hilo.length > 1) {
    const bbox = blHiloBBox(b);
    const mw = Math.max(10, Math.round(bbox.w));
    const mh = Math.max(10, Math.round(bbox.h));
    const c = document.createElement('canvas');
    c.width = Math.min(mw, 500); c.height = Math.min(mh, 500);
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, c.width, c.height);
    b.hiloMask = c;
  }
  return b.hiloMask;
}

async function blChangeColor(color) {
  blSelectedColor = color;
  try {
    blBorlaImg = await loadImg(`/api/borlas/image/${color}?maxw=800`);
    blApplyHiloColorForBorla(color);
  } catch (e) {
    showToast('Error cargando borla', 'error');
    return;
  }
  // Update all borlas to new color
  for (const b of blBorlas) b.color = color;
  blRenderList();
  blDraw();
}

function blRenderList() {
  const list = document.getElementById('blList');
  if (!list) return;
  list.innerHTML = blBorlas.map((b, i) => {
    const selCls = i === blSelected ? 'selected' : '';
    const hidCls = b.visible ? '' : 'hidden';
    const dotColor = BL_COLOR_MAP[b.color] || '#888';
    const toggleIcon = b.visible ? '&#128065;' : '&#128064;';
    const toggleTitle = b.visible ? 'Ocultar' : 'Restaurar';
    const scaleInfo = Math.abs(b.scale - blBaseScale) > 0.5 ? ` ×${(b.scale/blBaseScale).toFixed(1)}` : '';
    return `<div class="bl-item ${selCls} ${hidCls}" onclick="blSelectBorla(${i})">
      <span class="bl-color-dot" style="background:${dotColor}"></span>
      #${i+1} (${Math.round(b.rotation)}°${scaleInfo})
      <button onclick="event.stopPropagation(); blToggleBorla(${i})" title="${toggleTitle}">${toggleIcon}</button>
    </div>`;
  }).join('');
}

function blSelectBorla(idx) {
  blSelected = blSelected === idx ? -1 : idx;
  blRenderList();
  blDraw();
}

function blToggleBorla(idx) {
  blBorlas[idx].visible = !blBorlas[idx].visible;
  blRenderList();
  blDraw();
}

function blSetTool(tool) {
  // Finish any ongoing hilo drawing when switching tools
  if (blHiloDrawing && tool !== 'hilo') blHiloFinish();
  blTool = tool;
  const wrap = document.getElementById('blWrap');
  // Update toolbar buttons
  document.getElementById('blToolMove').classList.toggle('active', tool === 'move');
  document.getElementById('blToolAdd').classList.toggle('active', tool === 'add');
  document.getElementById('blToolBrush').classList.toggle('active', tool === 'brush');
  document.getElementById('blToolHilo').classList.toggle('active', tool === 'hilo');
  // Show/hide controls
  document.getElementById('blBrushControls').style.display = tool === 'brush' ? '' : 'none';
  document.getElementById('blHiloControls').style.display = tool === 'hilo' ? '' : 'none';
  if (tool === 'brush') blSetBrushAction(blBrushAction);
  if (wrap) wrap.style.cursor = tool === 'add' ? 'crosshair' : (tool === 'brush' || tool === 'hilo') ? 'none' : 'default';
  blDraw();
}

function blSetBrushAction(action) {
  blBrushAction = action;
  document.getElementById('blBrushErase').classList.toggle('active', action === 'erase');
  document.getElementById('blBrushRestore').classList.toggle('active', action === 'restore');
}

function blApplyScale() {
  const borlaAR = blBorlaImg ? blBorlaImg.height / blBorlaImg.width : 15;
  // Recompute uniform base scale from average face height
  if (blFaces.length > 0) {
    const avgFaceH = blFaces.reduce((s, f) => s + f.h * blImgH, 0) / blFaces.length;
    blBaseScale = (avgFaceH * blScalePct / 100) / borlaAR;
  } else {
    // No faces detected — use image height as reference
    blBaseScale = (blImgH * 0.08 * blScalePct / 100) / borlaAR;
  }
  // Apply uniform scale to all borlas (reset individual overrides)
  for (const b of blBorlas) {
    b.scale = blBaseScale;
    b.mask = null;  // Reset mask since borla size changed
  }
  blDraw();
}

function blAddBorlaAt(imgX, imgY) {
  const rotation = blRotMin + Math.random() * (blRotMax - blRotMin);
  const newBorla = {
    id: blBorlas.length,
    faceIdx: -1,
    x: imgX,
    y: imgY,
    scale: blBaseScale,  // same uniform scale as all borlas
    rotation: Math.round(rotation * 10) / 10,
    visible: true,
    color: blSelectedColor,
    mask: null
  };
  blBorlas.push(newBorla);
  blSelected = blBorlas.length - 1;
  showToast(`Borla #${blBorlas.length} agregada`, 'ok');
  blRenderList();
  blDraw();
}

function blGetMask(b) {
  // Create per-borla mask canvas on demand (white = visible, black = erased)
  if (!b.mask && blBorlaImg) {
    const mw = Math.max(10, Math.round(b.scale));
    const mh = Math.max(10, Math.round(mw * (blBorlaImg.height / blBorlaImg.width)));
    const c = document.createElement('canvas');
    c.width = mw; c.height = mh;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, mw, mh);
    b.mask = c;
  }
  return b.mask;
}

function blSaveMaskSnapshot(borlaIdx) {
  // Save mask state before brush stroke for undo
  const b = blBorlas[borlaIdx];
  const mask = blGetMask(b);
  if (!mask) return;
  const snapshot = mask.getContext('2d').getImageData(0, 0, mask.width, mask.height);
  // Also save hiloMask if it exists (brush affects hilo too)
  let hiloMaskSnap = null;
  if (b.hiloMask) {
    hiloMaskSnap = b.hiloMask.getContext('2d').getImageData(0, 0, b.hiloMask.width, b.hiloMask.height);
  }
  blUndoStack.push({ type: 'mask', idx: borlaIdx, data: snapshot, hiloMask: hiloMaskSnap, hiloMaskW: b.hiloMask?.width, hiloMaskH: b.hiloMask?.height });
  blRedoStack = [];
  if (blUndoStack.length > 50) blUndoStack.shift();
}

function blSaveHiloSnapshot(borlaIdx) {
  // Save hilo state before hilo edit for undo
  const b = blBorlas[borlaIdx];
  blUndoStack.push({
    type: 'hilo', idx: borlaIdx,
    hilo: b.hilo ? JSON.parse(JSON.stringify(b.hilo)) : null,
    hiloMask: b.hiloMask ? b.hiloMask.getContext('2d').getImageData(0, 0, b.hiloMask.width, b.hiloMask.height) : null,
    hiloMaskW: b.hiloMask?.width, hiloMaskH: b.hiloMask?.height
  });
  blRedoStack = [];
  if (blUndoStack.length > 50) blUndoStack.shift();
}

function _blRestoreSnap(snap, b) {
  if (snap.type === 'mask') {
    const mask = blGetMask(b);
    if (mask && snap.data) mask.getContext('2d').putImageData(snap.data, 0, 0);
    if (snap.hiloMask) {
      if (!b.hiloMask) {
        const c = document.createElement('canvas');
        c.width = snap.hiloMaskW; c.height = snap.hiloMaskH;
        b.hiloMask = c;
      }
      b.hiloMask.getContext('2d').putImageData(snap.hiloMask, 0, 0);
    }
  } else if (snap.type === 'hilo') {
    b.hilo = snap.hilo ? JSON.parse(JSON.stringify(snap.hilo)) : null;
    if (snap.hiloMask) {
      const c = document.createElement('canvas');
      c.width = snap.hiloMaskW; c.height = snap.hiloMaskH;
      c.getContext('2d').putImageData(snap.hiloMask, 0, 0);
      b.hiloMask = c;
    } else {
      b.hiloMask = null;
    }
  }
}

function _blCaptureSnap(snap, b) {
  // Capture current state matching snap type
  if (snap.type === 'mask') {
    const mask = blGetMask(b);
    const cur = mask ? mask.getContext('2d').getImageData(0, 0, mask.width, mask.height) : null;
    let hiloMaskSnap = null;
    if (b.hiloMask) {
      hiloMaskSnap = b.hiloMask.getContext('2d').getImageData(0, 0, b.hiloMask.width, b.hiloMask.height);
    }
    return { type: 'mask', idx: snap.idx, data: cur, hiloMask: hiloMaskSnap, hiloMaskW: b.hiloMask?.width, hiloMaskH: b.hiloMask?.height };
  } else {
    return {
      type: 'hilo', idx: snap.idx,
      hilo: b.hilo ? JSON.parse(JSON.stringify(b.hilo)) : null,
      hiloMask: b.hiloMask ? b.hiloMask.getContext('2d').getImageData(0, 0, b.hiloMask.width, b.hiloMask.height) : null,
      hiloMaskW: b.hiloMask?.width, hiloMaskH: b.hiloMask?.height
    };
  }
}

function blUndo() {
  if (blUndoStack.length === 0) return;
  const snap = blUndoStack.pop();
  const b = blBorlas[snap.idx];
  if (!b) return;
  // If we're in the middle of drawing hilo, cancel it
  if (snap.type === 'hilo' && blHiloDrawing) blHiloDrawing = false;
  // Save current state for redo
  blRedoStack.push(_blCaptureSnap(snap, b));
  // Restore
  _blRestoreSnap(snap, b);
  blDraw();
}

function blRedo() {
  if (blRedoStack.length === 0) return;
  const snap = blRedoStack.pop();
  const b = blBorlas[snap.idx];
  if (!b) return;
  // Save current state for undo
  blUndoStack.push(_blCaptureSnap(snap, b));
  // Restore
  _blRestoreSnap(snap, b);
  blDraw();
}

function blBrushOnMask(b, imgX, imgY) {
  // Paint on the borla's mask at image coordinates, accounting for rotation
  if (!blBorlaImg) return;
  const mask = blGetMask(b);
  if (!mask) return;
  const borlaW = b.scale;
  const borlaH = borlaW * (blBorlaImg.height / blBorlaImg.width);
  // Un-rotate the point around the borla pivot (b.x, b.y = top-center)
  const rad = -b.rotation * Math.PI / 180;
  const dx = imgX - b.x;
  const dy = imgY - b.y;
  const rx = dx * Math.cos(rad) - dy * Math.sin(rad);
  const ry = dx * Math.sin(rad) + dy * Math.cos(rad);
  // Convert un-rotated coords to mask coords
  const localX = (rx + borlaW / 2) / borlaW * mask.width;
  const localY = ry / borlaH * mask.height;
  const radius = blBrushSize * (mask.width / borlaW);
  const ctx = mask.getContext('2d');
  ctx.globalCompositeOperation = blBrushAction === 'erase' ? 'destination-out' : 'source-over';
  ctx.fillStyle = blBrushAction === 'erase' ? '#000' : '#fff';
  ctx.beginPath();
  ctx.arc(localX, localY, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalCompositeOperation = 'source-over';
  // Also affect hilo mask if this borla has a hilo
  if (b.hilo && b.hilo.length > 1) {
    blBrushOnHilo(b, imgX, imgY);
  }
}

function blDraw() {
  if (!blMode || !blCanvas) return;
  const ctx = blCtx;
  const cw = blCanvas.width, ch = blCanvas.height;
  ctx.clearRect(0, 0, cw, ch);

  // Checkerboard background
  ctx.save();
  const pat = ctx.createPattern((() => {
    const c = document.createElement('canvas');
    c.width = 20; c.height = 20;
    const x = c.getContext('2d');
    x.fillStyle = '#3a3a4a'; x.fillRect(0, 0, 10, 10); x.fillRect(10, 10, 10, 10);
    x.fillStyle = '#2a2a3a'; x.fillRect(10, 0, 10, 10); x.fillRect(0, 10, 10, 10);
    return c;
  })(), 'repeat');

  // Transform: zoom + pan
  const scale = pvZoom;
  const ox = (cw - blImgW * scale) / 2 + pvX;
  const oy = (ch - blImgH * scale) / 2 + pvY;

  // Draw checkerboard behind image
  ctx.fillStyle = pat;
  ctx.fillRect(ox, oy, blImgW * scale, blImgH * scale);

  // Draw cutout image
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'medium';
  ctx.drawImage(blCutImg, ox, oy, blImgW * scale, blImgH * scale);

  // Draw face detection rectangles (shown briefly after detection)
  if (blShowFaceRects && blFaces.length > 0) {
    ctx.save();
    ctx.strokeStyle = 'rgba(122, 162, 247, 0.8)';
    ctx.fillStyle = 'rgba(122, 162, 247, 0.12)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    for (const face of blFaces) {
      const fx = ox + face.x * blImgW * scale;
      const fy = oy + face.y * blImgH * scale;
      const fw = face.w * blImgW * scale;
      const fh = face.h * blImgH * scale;
      ctx.fillRect(fx, fy, fw, fh);
      ctx.strokeRect(fx, fy, fw, fh);
    }
    ctx.setLineDash([]);
    // Face count label
    ctx.fillStyle = 'rgba(122, 162, 247, 0.9)';
    ctx.font = 'bold 14px monospace';
    ctx.fillText(`${blFaces.length} caras detectadas`, ox + 8, oy + 20);
    ctx.restore();
  }

  // Draw borlas
  const borlaAR = blBorlaImg ? blBorlaImg.height / blBorlaImg.width : 15;
  for (let i = 0; i < blBorlas.length; i++) {
    const b = blBorlas[i];
    if (!b.visible) continue;

    const borlaW = b.scale * scale;
    const borlaH = borlaW * borlaAR;
    const sx = ox + b.x * scale;
    const sy = oy + b.y * scale;

    ctx.save();
    ctx.translate(sx, sy);
    ctx.rotate(b.rotation * Math.PI / 180);

    if (blBorlaImg) {
      if (b.mask) {
        // Draw borla through mask: composite borla + mask onto temp canvas
        const tw = Math.ceil(borlaW), th = Math.ceil(borlaH);
        if (tw > 0 && th > 0) {
          const tmp = document.createElement('canvas');
          tmp.width = tw; tmp.height = th;
          const tc = tmp.getContext('2d');
          tc.drawImage(blBorlaImg, 0, 0, tw, th);
          // Apply mask: destination-in keeps only where mask is white
          tc.globalCompositeOperation = 'destination-in';
          tc.drawImage(b.mask, 0, 0, tw, th);
          ctx.drawImage(tmp, -borlaW / 2, 0, borlaW, borlaH);
        }
      } else {
        ctx.drawImage(blBorlaImg, -borlaW / 2, 0, borlaW, borlaH);
      }
    } else {
      // Fallback: draw colored indicator when borla image not loaded
      const dotColor = BL_COLOR_MAP[b.color] || '#d4a017';
      ctx.fillStyle = dotColor;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      ctx.arc(0, 8, Math.max(4, borlaW / 2), 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Draw hilo (thread) for this borla — in rotated local space
    if (b.hilo && b.hilo.length > 1) {
      const hiloS = scale;  // hilo points are in image-pixel local coords
      ctx.save();
      ctx.globalAlpha = blHiloFlow / 100;
      ctx.strokeStyle = blHiloColor;
      ctx.lineWidth = Math.max(1, blHiloSize * hiloS);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      if (blHiloFeather > 0.1) {
        ctx.shadowColor = blHiloColor;
        ctx.shadowBlur = blHiloFeather * hiloS;
      }
      // If hilo mask exists, render via temp canvas with mask
      if (b.hiloMask) {
        const bbox = blHiloBBox(b);
        const tw = Math.max(1, Math.ceil(bbox.w * hiloS));
        const th = Math.max(1, Math.ceil(bbox.h * hiloS));
        if (tw > 0 && th > 0 && tw < 4000 && th < 4000) {
          const tmp = document.createElement('canvas');
          tmp.width = tw; tmp.height = th;
          const tc = tmp.getContext('2d');
          tc.strokeStyle = blHiloColor;
          tc.lineWidth = Math.max(1, blHiloSize * hiloS);
          tc.lineCap = 'round'; tc.lineJoin = 'round';
          tc.globalAlpha = blHiloFlow / 100;
          if (blHiloFeather > 0.1) { tc.shadowColor = blHiloColor; tc.shadowBlur = blHiloFeather * hiloS; }
          tc.beginPath();
          tc.moveTo((b.hilo[0].x - bbox.x) * hiloS, (b.hilo[0].y - bbox.y) * hiloS);
          for (let p = 1; p < b.hilo.length; p++) {
            tc.lineTo((b.hilo[p].x - bbox.x) * hiloS, (b.hilo[p].y - bbox.y) * hiloS);
          }
          tc.stroke();
          tc.globalCompositeOperation = 'destination-in';
          tc.drawImage(b.hiloMask, 0, 0, tw, th);
          ctx.globalAlpha = 1;
          ctx.shadowBlur = 0;
          ctx.drawImage(tmp, bbox.x * hiloS, bbox.y * hiloS);
        }
      } else {
        ctx.beginPath();
        ctx.moveTo(b.hilo[0].x * hiloS, b.hilo[0].y * hiloS);
        for (let p = 1; p < b.hilo.length; p++) {
          ctx.lineTo(b.hilo[p].x * hiloS, b.hilo[p].y * hiloS);
        }
        ctx.stroke();
      }
      ctx.restore();
    }

    ctx.restore();

    // Selection highlight (hidden while dragging, shown while rotating)
    if (i === blSelected && !blDragging) {
      ctx.save();
      ctx.translate(sx, sy);
      ctx.rotate(b.rotation * Math.PI / 180);
      ctx.strokeStyle = '#7aa2f7';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(-borlaW / 2 - 2, -2, borlaW + 4, borlaH + 4);
      ctx.setLineDash([]);
      // Rotation handle: line + circle above the selection rect
      const handleY = -22;
      ctx.beginPath();
      ctx.moveTo(0, -2);
      ctx.lineTo(0, handleY);
      ctx.strokeStyle = '#7aa2f7';
      ctx.setLineDash([]);
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, handleY, 6, 0, Math.PI * 2);
      ctx.fillStyle = blRotating ? '#ff9e64' : '#7aa2f7';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      ctx.stroke();
      // Scale handle: square at bottom-right corner
      ctx.beginPath();
      ctx.rect(borlaW / 2 - 2, borlaH - 2, 8, 8);
      ctx.fillStyle = blScaling ? '#e0af68' : '#7dcfff';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      ctx.stroke();
      // Index badge
      ctx.fillStyle = '#7aa2f7';
      ctx.font = 'bold 12px monospace';
      ctx.fillText('#' + (i + 1), -borlaW / 2, -30);
      ctx.restore();
    }
  }

  ctx.restore();

  // Brush cursor
  if (blTool === 'brush' && blMousePos) {
    const r = blBrushSize * pvZoom;
    ctx.save();
    ctx.strokeStyle = blBrushAction === 'erase' ? '#f7768e' : '#9ece6a';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(blMousePos.sx, blMousePos.sy, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // Hilo cursor + rubber-band preview
  if (blTool === 'hilo' && blMousePos) {
    const r = Math.max(2, blHiloSize * pvZoom * 0.5);
    ctx.save();
    ctx.strokeStyle = blHiloColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(blMousePos.sx, blMousePos.sy, r, 0, Math.PI * 2);
    ctx.stroke();
    // Rubber-band line from last vertex to cursor
    if (blHiloDrawing && blSelected >= 0) {
      const b = blBorlas[blSelected];
      if (b && b.hilo && b.hilo.length > 0) {
        const lastPt = b.hilo[b.hilo.length - 1];
        // Convert local coords back to screen coords
        const rad = b.rotation * Math.PI / 180;
        const scale = pvZoom;
        const scr = blImgToScreen(b.x, b.y);
        const sx2 = scr.x + (lastPt.x * Math.cos(rad) - lastPt.y * Math.sin(rad)) * scale;
        const sy2 = scr.y + (lastPt.x * Math.sin(rad) + lastPt.y * Math.cos(rad)) * scale;
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = Math.max(1, blHiloSize * pvZoom);
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.moveTo(sx2, sy2);
        ctx.lineTo(blMousePos.sx, blMousePos.sy);
        ctx.stroke();
        ctx.setLineDash([]);
        // Draw small dots on existing vertices
        ctx.globalAlpha = 0.8;
        ctx.fillStyle = blHiloColor;
        for (const pt of b.hilo) {
          const vsx = scr.x + (pt.x * Math.cos(rad) - pt.y * Math.sin(rad)) * scale;
          const vsy = scr.y + (pt.x * Math.sin(rad) + pt.y * Math.cos(rad)) * scale;
          ctx.beginPath();
          ctx.arc(vsx, vsy, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
    ctx.restore();
  }

  // Zoom label
  const lbl = document.getElementById('blZoomLabel');
  if (lbl) lbl.textContent = Math.round(pvZoom * 100) + '%';
}

function blZoomFit() {
  if (!blCutImg || !blCanvas) return;
  const cw = blCanvas.width, ch = blCanvas.height;
  pvZoom = Math.min(cw / blImgW, ch / blImgH) * 0.95;
  pvX = 0; pvY = 0;
  blDraw();
}

function blToggleFullscreen() {
  const panel = document.getElementById('previewPanel');
  if (!panel) return;
  _toggleRealFullscreen(panel, 'blFullscreenBtn', () => {
    const wrap = document.getElementById('blWrap');
    if (wrap && blCanvas && blMode) {
      const rect = wrap.getBoundingClientRect();
      blCanvas.width = rect.width; blCanvas.height = rect.height;
      blZoomFit();
    }
  });
}

function blImgToScreen(ix, iy) {
  const cw = blCanvas.width, ch = blCanvas.height;
  const scale = pvZoom;
  const ox = (cw - blImgW * scale) / 2 + pvX;
  const oy = (ch - blImgH * scale) / 2 + pvY;
  return { x: ox + ix * scale, y: oy + iy * scale };
}

function blScreenToImg(sx, sy) {
  const cw = blCanvas.width, ch = blCanvas.height;
  const scale = pvZoom;
  const ox = (cw - blImgW * scale) / 2 + pvX;
  const oy = (ch - blImgH * scale) / 2 + pvY;
  return { x: (sx - ox) / scale, y: (sy - oy) / scale };
}

function blHitTest(sx, sy) {
  // Hit test borlas in reverse order (top borla first)
  // Un-rotates the click point around each borla's pivot for accurate testing
  const _ar = blBorlaImg ? blBorlaImg.height / blBorlaImg.width : 15;
  for (let i = blBorlas.length - 1; i >= 0; i--) {
    const b = blBorlas[i];
    if (!b.visible) continue;
    const borlaW = b.scale * pvZoom;
    const borlaH = borlaW * _ar;
    const scr = blImgToScreen(b.x, b.y);  // pivot = top-center on screen
    // Un-rotate click point around the borla pivot
    const rad = -b.rotation * Math.PI / 180;
    const dx = sx - scr.x;
    const dy = sy - scr.y;
    const rx = dx * Math.cos(rad) - dy * Math.sin(rad);
    const ry = dx * Math.sin(rad) + dy * Math.cos(rad);
    // Test against un-rotated borla rectangle (centered at pivot)
    if (rx >= -borlaW / 2 && rx <= borlaW / 2 && ry >= 0 && ry <= borlaH) {
      return i;
    }
  }
  return -1;
}

function blHitRotHandle(sx, sy) {
  // Check if click is on the rotation handle of the selected borla
  if (blSelected < 0 || blBorlas.length === 0) return false;
  const b = blBorlas[blSelected];
  if (!b.visible) return false;
  const scr = blImgToScreen(b.x, b.y);
  // Handle is at local (0, -22), rotated by b.rotation
  // Canvas rotate: local (lx,ly) → screen offset (lx*cos-ly*sin, lx*sin+ly*cos)
  const rad = b.rotation * Math.PI / 180;
  const hx = scr.x + 22 * Math.sin(rad);   // -(-22)*sin = +22*sin
  const hy = scr.y - 22 * Math.cos(rad);    // (-22)*cos  = -22*cos
  const dist = Math.hypot(sx - hx, sy - hy);
  return dist <= 14;  // generous hit radius for small handle
}

function blBindEvents(wrap) {
  let panning = false, panStartX = 0, panStartY = 0, panOX = 0, panOY = 0;

  wrap.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = blCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const oldZ = pvZoom;
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    pvZoom = Math.max(0.05, Math.min(pvZoom * factor, 20));
    // Zoom toward cursor
    const cw = blCanvas.width, ch = blCanvas.height;
    const cx = cw / 2 + pvX;
    const cy = ch / 2 + pvY;
    pvX += (mx - cx) * (1 - pvZoom / oldZ);
    pvY += (my - cy) * (1 - pvZoom / oldZ);
    blDraw();
  }, { passive: false });

  wrap.addEventListener('mousedown', (e) => {
    const rect = blCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Middle click or space+click = pan
    if (e.button === 1 || spaceHeld || (e.button === 2 && e.altKey)) {
      panning = true;
      panStartX = e.clientX; panStartY = e.clientY;
      panOX = pvX; panOY = pvY;
      wrap.style.cursor = 'grabbing';
      e.preventDefault();
      return;
    }

    // Right-click finishes hilo drawing
    if (e.button === 2 && blHiloDrawing) {
      blHiloFinish();
      return;
    }

    if (e.button !== 0) return;

    // Add tool: place borla at click position
    if (blTool === 'add') {
      const img = blScreenToImg(mx, my);
      blAddBorlaAt(img.x, img.y);
      return;
    }

    // Brush tool: paint on selected borla mask
    if (blTool === 'brush' && blSelected >= 0) {
      blSaveMaskSnapshot(blSelected);  // save for undo before stroke
      blBrushDrawing = true;
      const img = blScreenToImg(mx, my);
      blBrushOnMask(blBorlas[blSelected], img.x, img.y);
      blDraw();
      return;
    }

    // Hilo tool: click-to-click line segments on selected borla
    if (blTool === 'hilo' && blSelected >= 0) {
      const img = blScreenToImg(mx, my);
      blHiloAddVertex(img.x, img.y);
      return;
    }

    // Check scale handle (on selected borla, bottom-right)
    if (blSelected >= 0 && blHitScaleHandle(mx, my)) {
      const b = blBorlas[blSelected];
      blScaling = { idx: blSelected, startMy: my, startScale: b.scale };
      wrap.style.cursor = 'ns-resize';
      blDraw();
      return;
    }

    // Check rotation handle (on selected borla, top)
    if (blSelected >= 0 && blHitRotHandle(mx, my)) {
      const b = blBorlas[blSelected];
      const scr = blImgToScreen(b.x, b.y);
      blRotating = { idx: blSelected, startAngle: Math.atan2(mx - scr.x, -(my - scr.y)), startRot: b.rotation };
      wrap.style.cursor = 'crosshair';
      blDraw();
      return;
    }

    // Move tool: hit test borlas
    const hit = blHitTest(mx, my);
    if (hit >= 0) {
      blSelected = hit;
      const b = blBorlas[hit];
      const scr = blImgToScreen(b.x, b.y);
      blDragging = { idx: hit, offX: mx - scr.x, offY: my - scr.y };
      wrap.style.cursor = 'move';
      blRenderList();
      blDraw();
    } else {
      blSelected = -1;
      blRenderList();
      blDraw();
      // Pan if clicked on empty space
      panning = true;
      panStartX = e.clientX; panStartY = e.clientY;
      panOX = pvX; panOY = pvY;
      wrap.style.cursor = 'grabbing';
    }
  });

  wrap.addEventListener('mousemove', (e) => {
    const rect = blCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    blMousePos = { sx: mx, sy: my };

    if (panning) {
      pvX = panOX + (e.clientX - panStartX);
      pvY = panOY + (e.clientY - panStartY);
      blDraw();
      return;
    }
    if (blScaling) {
      const deltaPx = my - blScaling.startMy;
      const deltaScale = deltaPx / pvZoom;  // convert screen px to image px
      const b = blBorlas[blScaling.idx];
      b.scale = Math.max(2, blScaling.startScale + deltaScale / 15);  // divided by AR for smooth feel
      b.mask = null;  // reset mask when resized
      blDraw();
      return;
    }
    if (blRotating) {
      const b = blBorlas[blRotating.idx];
      const scr = blImgToScreen(b.x, b.y);
      const curAngle = Math.atan2(mx - scr.x, -(my - scr.y));
      const delta = (curAngle - blRotating.startAngle) * 180 / Math.PI;
      b.rotation = Math.round((blRotating.startRot + delta) * 10) / 10;
      blDraw();
      return;
    }
    if (blDragging) {
      const img = blScreenToImg(mx - blDragging.offX, my - blDragging.offY);
      blBorlas[blDragging.idx].x = img.x;
      blBorlas[blDragging.idx].y = img.y;
      blDraw();
      return;
    }
    if (blBrushDrawing && blTool === 'brush' && blSelected >= 0) {
      const img = blScreenToImg(mx, my);
      blBrushOnMask(blBorlas[blSelected], img.x, img.y);
      blDraw();
      return;
    }
    // Hilo preview: redraw to show rubber-band line from last vertex to cursor
    if (blTool === 'hilo' && blHiloDrawing && blSelected >= 0) {
      blDraw();
      return;
    }
    // Update brush/hilo cursor
    if (blTool === 'brush' || blTool === 'hilo') blDraw();
  });

  wrap.addEventListener('mouseup', () => {
    const wasDragging = !!blDragging;
    const wasBrushing = blBrushDrawing;
    const wasRotating = !!blRotating;
    const wasScaling = !!blScaling;
    panning = false;
    blBrushDrawing = false;
    if (blDragging) {
      blDragging = null;
      blRenderList();
      blDraw();
    }
    if (blRotating) {
      blRotating = null;
      blRenderList();
      blDraw();
    }
    if (blScaling) {
      blScaling = null;
      blRenderList();
      blDraw();
    }
    wrap.style.cursor = blTool === 'add' ? 'crosshair' : (blTool === 'brush' || blTool === 'hilo') ? 'none' : 'default';
    if (wasDragging || wasBrushing || wasRotating || wasScaling) blSaveState();
  });

  wrap.addEventListener('mouseleave', () => {
    panning = false;
    blDragging = null;
    blRotating = null;
    blScaling = null;
    blBrushDrawing = false;
    blMousePos = null;
    wrap.style.cursor = blTool === 'add' ? 'crosshair' : (blTool === 'brush' || blTool === 'hilo') ? 'none' : 'default';
    blDraw();
  });

  wrap.addEventListener('dblclick', (e) => {
    // Double-click finishes hilo drawing
    if (blTool === 'hilo' && blHiloDrawing) {
      e.preventDefault();
      blHiloFinish();
    }
  });

  wrap.addEventListener('contextmenu', (e) => e.preventDefault());

  // Pointer events for touchpad/touchscreen
  let blPtrs = {}, blPtrCenter = null, blPtrDist = 0;
  wrap.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse') return;
    wrap.setPointerCapture(e.pointerId);
    blPtrs[e.pointerId] = { x: e.clientX, y: e.clientY };
    const cnt = Object.keys(blPtrs).length;
    if (cnt >= 2) {
      blPtrCenter = ptrCenter(blPtrs);
      blPtrDist = ptrDist(blPtrs);
    }
  });
  wrap.addEventListener('pointermove', (e) => {
    if (e.pointerType === 'mouse' || !blPtrs[e.pointerId]) return;
    blPtrs[e.pointerId] = { x: e.clientX, y: e.clientY };
    const cnt = Object.keys(blPtrs).length;
    if (cnt >= 2) {
      const nc = ptrCenter(blPtrs);
      const nd = ptrDist(blPtrs);
      if (blPtrCenter) {
        pvX += nc.x - blPtrCenter.x;
        pvY += nc.y - blPtrCenter.y;
        if (blPtrDist > 0 && nd > 0) {
          const ratio = nd / blPtrDist;
          pvZoom = Math.max(0.05, Math.min(pvZoom * ratio, 20));
        }
        blDraw();
      }
      blPtrCenter = nc; blPtrDist = nd;
    } else if (cnt === 1) {
      if (blPtrCenter) {
        pvX += e.clientX - blPtrs[e.pointerId].x;
        pvY += e.clientY - blPtrs[e.pointerId].y;
        blDraw();
      }
      blPtrCenter = { x: e.clientX, y: e.clientY };
    }
  });
  const blPtrEnd = (e) => {
    delete blPtrs[e.pointerId];
    if (Object.keys(blPtrs).length < 2) { blPtrCenter = null; blPtrDist = 0; }
  };
  wrap.addEventListener('pointerup', blPtrEnd);
  wrap.addEventListener('pointercancel', blPtrEnd);
}

async function blExportTiff() {
  if (blBorlas.filter(b => b.visible).length === 0) {
    showToast('No hay borlas visibles para exportar', 'warn');
    return;
  }
  showToast('Generando TIFF con borlas... espera', 'info');
  try {
    const payload = {
      hiloColor: blHiloColor,
      hiloSize: blHiloSize,
      hiloFeather: blHiloFeather,
      hiloFlow: blHiloFlow,
      borlas: blBorlas.filter(b => b.visible).map(b => {
        let maskData = null;
        if (b.mask) {
          maskData = b.mask.toDataURL('image/png');
        }
        return {
          color: b.color, x: Math.round(b.x), y: Math.round(b.y),
          scale: Math.round(b.scale), rotation: b.rotation, visible: true,
          mask: maskData,
          hilo: b.hilo || null,
          hiloMask: b.hiloMask ? b.hiloMask.toDataURL('image/png') : null
        };
      })
    };
    const res = await fetch(`/api/borlas/export-tiff/${blCbtis}/${encodeURIComponent(blFilename)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) { showToast('Error al generar TIFF: ' + res.status, 'error'); return; }
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    const stem = blFilename.replace(/\.[^.]+$/, '');
    a.download = stem + '_borlas.tif';
    document.body.appendChild(a);
    a.click(); a.remove();
    URL.revokeObjectURL(a.href);
    showToast('TIFF con borlas descargado', 'ok');
  } catch (e) {
    showToast('Error: ' + e.message, 'error');
  }
}

// ─── TOGAS (caída de togas) ──────────────────────────────────────────────────
let tgMode = false, tgCbtis = '', tgFilename = '';
let tgCutImg = null;           // cutout image (same as borlas uses)
let tgBorlaImg = null;         // loaded borla image for compositing
let tgBorlaState = null;       // saved borlas array from borlas state
let tgTogaImgs = {};           // {variant: Image} loaded toga images
let tgTogaVariants = [];       // ["CAIDA_1","CAIDA_2","CAIDA_3"]
let tgRefAR = 1.5;             // reference aspect ratio (h/w) so all togas have same height
let tgTogas = [];              // [{id, variant, x, y, baseX, baseY, scaleX, flipH, scale, mask}]
let tgFaces = [];              // all detected faces
let tgSeatedIdx = [];          // indices into tgFaces for seated people
let tgDetectionDone = false;
let tgGuides = [];             // [{y: imgPixels}] horizontal guide lines
let tgGroupTf = {x:0, y:0, rotation:0, scaleY:1.0}; // toga group transform
let tgImgTf = {x:0, y:0, rotation:0};                 // image group transform (cutout+borlas)
let tgTool = 'move';          // 'move'|'add-seat'|'brush'|'guide'|'img-move'|'toga-group'
let tgBrushSize = 20, tgBrushAction = 'erase';
let tgBrushFeather = 0.5, tgBrushFlow = 1.0;
let tgSelected = -1;
let tgBrushAll = false;      // When true, brush affects all togas simultaneously
let tgDragging = null, tgDraggingGuide = -1, tgBrushDrawing = false;
let tgScaleXDrag = null;  // {idx, side:'left'|'right', startMx, startScaleX}
let tgImgDragging = null, tgImgRotating = false, tgImgRotStart = 0;
let tgGroupDragging = null, tgGroupRotating = false;
let tgMousePos = null;
let tgUndoStack = [], tgRedoStack = [];
let tgCanvas = null, tgCtx = null;
let tgImgW = 1, tgImgH = 1;

async function enterTogasMode(cbtis, filename) {
  tgCbtis = cbtis; tgFilename = filename;
  tgTogas = []; tgFaces = []; tgSeatedIdx = []; tgSelected = -1;
  tgDetectionDone = false; tgGuides = []; tgBrushAll = false;
  tgGroupTf = {x:0, y:0, rotation:0, scaleY:1.0};
  tgImgTf = {x:0, y:0, rotation:0};
  tgImgRotating = false;
  tgUndoStack = []; tgRedoStack = [];

  const panel = document.getElementById('previewPanel');
  // Set panel dataset so toggleWorkflowDone can find the correct group
  panel.dataset.cbtis = cbtis;
  panel.dataset.cutout = filename;
  const _gd = groups.find(g => g.cbtis === cbtis && g.cutout === filename);
  if (_gd) { panel.dataset.group = _gd.group; panel.dataset.pano = _gd.output || ''; }
  panel.innerHTML = '<div class="rf-loading">Cargando modo Togas...</div>';

  // Load toga variants
  try {
    const lRes = await fetch('/api/togas/list');
    const lData = await lRes.json();
    tgTogaVariants = lData.variants;
  } catch (e) { showToast('Error cargando togas: ' + e.message, 'error'); return; }

  // Load cutout image
  const encN = encodeURIComponent(filename);
  try {
    tgCutImg = await loadImg(`/api/cutout/preview-scaled/${cbtis}/${encN}?maxw=99999&_=${Date.now()}`);
  } catch (e) { showToast('Error cargando imagen: ' + e.message, 'error'); return; }
  tgImgW = tgCutImg.width; tgImgH = tgCutImg.height;

  // Load borla state for compositing
  tgBorlaState = null; tgBorlaImg = null;
  try {
    const bsRes = await fetch(`/api/borlas/state/${cbtis}/${encN}`);
    const bs = await bsRes.json();
    if (bs && bs.borlas && bs.borlas.length > 0) {
      tgBorlaState = bs;
      // Load the borla image for the saved color
      const color = bs.selectedColor || bs.borlas[0].color || 'DORADO';
      tgBorlaImg = await loadImg(`/api/borlas/image/${color}?maxw=800`);
    }
  } catch {}

  // Load all toga variant images
  tgTogaImgs = {};
  await Promise.all(tgTogaVariants.map(async v => {
    try { tgTogaImgs[v] = await loadImg(`/api/togas/image/${v}?maxw=600`); } catch {}
  }));

  // Compute uniform reference aspect ratio (avg of all variants) so all togas have same height
  { const ars = Object.values(tgTogaImgs).map(img => img.height / img.width).filter(v => v > 0);
    tgRefAR = ars.length > 0 ? ars.reduce((a,b) => a+b, 0) / ars.length : 1.5; }

  // Build UI
  panel.innerHTML = `
    <div class="togas-toolbar">
      <button class="rf-btn active" id="tgToolMove" onclick="tgSetTool('move')" title="Mover togas individuales en horizontal (V)">&#9995; Mover toga</button>
      <button class="rf-btn" id="tgToolAddSeat" onclick="tgSetTool('add-seat')" title="Click: agregar sentado | Click derecho: quitar (A)">&#10010; Sentado</button>
      <button class="rf-btn" id="tgToolBrush" onclick="tgSetTool('brush')" title="Brocha borrar/restaurar (B)">&#9998; Brocha</button>
      <button class="rf-btn" id="tgToolGuide" onclick="tgSetTool('guide')" title="Click: agregar guia | Click derecho: quitar (G)">&#9473; Guia</button>
      <button class="rf-btn" id="tgToolImgMove" onclick="tgSetTool('img-move')" title="Rectangulo: mover/rotar foto+borlas. NO mueve togas (I)">&#9634; Foto+Borlas</button>
      <button class="rf-btn" id="tgToolTogaGroup" onclick="tgSetTool('toga-group')" title="Mover/rotar/escalar todas las togas juntas (T)">&#9632; Grupo togas</button>
      <span id="tgBrushControls" style="display:none">
        <button class="rf-btn active" id="tgBrushErase" onclick="tgSetBrushAction('erase')">&#10007; Borrar</button>
        <button class="rf-btn" id="tgBrushRestore" onclick="tgSetBrushAction('restore')">&#10003; Restaurar</button>
        <button class="rf-btn" id="tgBrushAllBtn" onclick="tgToggleBrushAll()" title="Brocha afecta a todas o solo la seleccionada">Sel.</button>
        <input type="range" class="rf-size-slider" min="2" max="120" value="${tgBrushSize}" style="width:50px"
               oninput="tgBrushSize=+this.value; document.getElementById('tgBrushSizeVal').textContent=this.value">
        <span class="rf-size-label" id="tgBrushSizeVal">${tgBrushSize}</span>px
        <label>Pluma:</label>
        <input type="range" min="0" max="100" value="${Math.round(tgBrushFeather*100)}" style="width:40px"
               oninput="tgBrushFeather=this.value/100; document.getElementById('tgFeatherVal').textContent=this.value+'%'">
        <span class="rf-size-label" id="tgFeatherVal">${Math.round(tgBrushFeather*100)}%</span>
        <label>Flujo:</label>
        <input type="range" min="5" max="100" value="${Math.round(tgBrushFlow*100)}" style="width:40px"
               oninput="tgBrushFlow=this.value/100; document.getElementById('tgFlowVal').textContent=this.value+'%'">
        <span class="rf-size-label" id="tgFlowVal">${Math.round(tgBrushFlow*100)}%</span>
      </span>
      <span id="tgMoveControls" style="display:none">
        <label>Ancho sel:</label>
        <input type="range" min="20" max="300" value="100" id="tgIndScaleXSlider" style="width:60px"
               oninput="tgSetIndScaleX(this.value/100)">
        <span class="rf-size-label" id="tgIndScaleXVal">100%</span>
      </span>
      <span id="tgGroupControls" style="display:none">
        <label>Esc.V:</label>
        <input type="range" min="20" max="300" value="${Math.round(tgGroupTf.scaleY*100)}" style="width:60px"
               oninput="tgGroupTf.scaleY=this.value/100; document.getElementById('tgGScaleVal').textContent=this.value+'%'; tgDraw()">
        <span class="rf-size-label" id="tgGScaleVal">${Math.round(tgGroupTf.scaleY*100)}%</span>
        <label>Rot:</label>
        <input type="number" min="-180" max="180" value="0" id="tgGRotInput" style="width:48px"
               oninput="tgGroupTf.rotation=+this.value; tgDraw()">°
      </span>
      <span id="tgImgControls" style="display:none">
        <label>Rot:</label>
        <input type="number" min="-10" max="10" value="0" id="tgImgRotInput" style="width:50px" step="0.1"
               oninput="tgImgTf.rotation=+this.value; tgDraw()">°
      </span>
      <div class="tg-sep"></div>
      <label>Ancho togas:</label>
      <input type="range" min="20" max="300" value="100" id="tgScaleXAllSlider" style="width:60px"
             oninput="tgSetAllScaleX(this.value/100)">
      <span class="rf-size-label" id="tgScaleXAllVal">100%</span>
      <div class="tg-sep"></div>
      <button class="rf-btn" onclick="tgDetectSeated()" title="Detectar personas sentadas">&#128269; Detectar</button>
      <div style="flex:1"></div>
      <button class="rf-btn" onclick="tgUndo()" title="Deshacer (Ctrl+Z)">&#8617;</button>
      <button class="rf-btn" onclick="tgRedo()" title="Rehacer (Ctrl+Y)">&#8618;</button>
      <button class="rf-btn primary" onclick="tgExportTiff()">&#8615; TIFF</button>
      <button class="rf-btn done-toggle ${getWorkflowDoneState('togas') ? 'is-done' : ''}" onclick="toggleWorkflowDone('togas', this)">${getWorkflowDoneState('togas') ? '\u2714 Listo' : '\u2610 Listo'}</button>
      <button class="rf-btn danger" onclick="exitTogasMode()">&#10005; Salir</button>
      <div class="tg-sep"></div>
      <button class="rf-btn" onclick="tgZoomFit()">Ajustar</button>
      <span class="rf-size-label" id="tgZoomLabel">100%</span>
      <button class="rf-btn" onclick="tgToggleFullscreen()" id="tgFullscreenBtn">&#9974; Completa</button>
    </div>
    <div class="rf-canvas-wrap" id="tgWrap" style="touch-action:none;position:relative">
      <canvas id="tgCanvas"></canvas>
    </div>
    <div class="tg-list" id="tgList"></div>
    <div class="preview-info">${filename} (togas)</div>
  `;

  tgCanvas = document.getElementById('tgCanvas');
  tgCtx = tgCanvas.getContext('2d');
  tgMode = true;

  setTimeout(() => {
    const wrap = document.getElementById('tgWrap');
    if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    tgCanvas.width = Math.max(rect.width, 200);
    tgCanvas.height = Math.max(rect.height, 200);
    pvZoom = 1; pvX = 0; pvY = 0;
    tgZoomFit();
    tgBindEvents(wrap);
    tgLoadState().then(loaded => {
      if (loaded && tgTogas.length > 0) {
        showToast(`${tgTogas.length} togas restauradas`, 'ok');
        tgRenderList(); tgDraw();
      } else {
        tgDetectSeated();
      }
    });
  }, 50);
}

function exitTogasMode() {
  tgSaveState();
  tgMode = false;
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  const panel = document.getElementById('previewPanel');
  if (panel) panel.classList.remove('faux-fullscreen');
  if (tgCbtis && tgFilename) {
    initPreviewPanel(
      `/api/cutout/preview/${tgCbtis}/${encodeURIComponent(tgFilename)}?t=${Date.now()}`,
      `${tgFilename} (recortada)`, true
    );
  }
  refreshGroups();
}

// --- Detection ---
async function tgDetectSeated() {
  showToast('Detectando personas sentadas...', 'info');
  try {
    const res = await fetch(`/api/togas/detect-seated/${tgCbtis}/${encodeURIComponent(tgFilename)}`);
    const data = await res.json();
    if (!data.ok) { showToast('Error: ' + data.error, 'error'); return; }
    tgFaces = data.faces;
    tgSeatedIdx = data.seated_indices;
    tgDetectionDone = false;
    _tgCreateTogasFromSeated();
    showToast(`${tgFaces.length} caras, ${tgSeatedIdx.length} sentadas`, 'ok');
    tgRenderList(); tgDraw();
  } catch (e) { showToast('Error: ' + e.message, 'error'); }
}

function _tgCreateTogasFromSeated() {
  tgTogas = [];
  if (tgSeatedIdx.length === 0) return;
  const variants = tgTogaVariants;

  // Compute a SINGLE common Y for all togas: average face-bottom of seated faces + offset
  let sumBottom = 0, sumFaceH = 0, sumFaceW = 0;
  tgSeatedIdx.forEach(fi => {
    const f = tgFaces[fi];
    sumBottom += (f.y + f.h) * tgImgH;
    sumFaceH += f.h * tgImgH;
    sumFaceW += f.w * tgImgW;
  });
  const avgBottom = sumBottom / tgSeatedIdx.length;
  const avgFaceH = sumFaceH / tgSeatedIdx.length;
  const avgFaceW = sumFaceW / tgSeatedIdx.length;
  const commonY = avgBottom + avgFaceH * 0.3;  // same Y for ALL togas
  const togaW = avgFaceW * 1.8;

  tgSeatedIdx.forEach((fi, i) => {
    const f = tgFaces[fi];
    const cx = (f.x + f.w / 2) * tgImgW;
    const variant = variants[i % variants.length];
    tgTogas.push({
      id: i, variant,
      x: cx, y: commonY,
      baseX: cx, baseY: commonY,
      scale: togaW, scaleX: 1.0,
      flipH: Math.random() > 0.5,
      rotation: 0, mask: null,
    });
  });
}

function tgMarkDetectionDone() {
  tgDetectionDone = !tgDetectionDone;
  const btn = document.getElementById('tgDetDoneBtn');
  if (btn) btn.innerHTML = tgDetectionDone ? '&#9745; Det. lista' : '&#9744; Det. lista';
  tgSaveState();
  showToast(tgDetectionDone ? 'Deteccion marcada como completa' : 'Deteccion desmarcada', 'ok');
}

// --- Tools ---
function tgSetTool(tool) {
  tgTool = tool;
  const ids = ['tgToolMove','tgToolAddSeat','tgToolBrush','tgToolGuide','tgToolImgMove','tgToolTogaGroup'];
  const tools = ['move','add-seat','brush','guide','img-move','toga-group'];
  ids.forEach((id, i) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('active', tools[i] === tool);
  });
  document.getElementById('tgBrushControls').style.display = tool === 'brush' ? '' : 'none';
  document.getElementById('tgMoveControls').style.display = tool === 'move' ? '' : 'none';
  document.getElementById('tgGroupControls').style.display = tool === 'toga-group' ? '' : 'none';
  document.getElementById('tgImgControls').style.display = tool === 'img-move' ? '' : 'none';
  if (tool === 'move') tgUpdateIndScaleXUI();
  const wrap = document.getElementById('tgWrap');
  if (wrap) {
    if (tool === 'add-seat' || tool === 'guide') wrap.style.cursor = 'crosshair';
    else if (tool === 'brush') wrap.style.cursor = 'none';
    else wrap.style.cursor = 'default';
  }
  tgDraw();
}

function tgSetBrushAction(a) {
  tgBrushAction = a;
  document.getElementById('tgBrushErase').classList.toggle('active', a === 'erase');
  document.getElementById('tgBrushRestore').classList.toggle('active', a === 'restore');
}

function tgSetAllScaleX(val) {
  for (const t of tgTogas) t.scaleX = val;
  const lbl = document.getElementById('tgScaleXAllVal');
  if (lbl) lbl.textContent = Math.round(val * 100) + '%';
  tgDraw();
  tgRenderList();
}

function tgToggleBrushAll() {
  tgBrushAll = !tgBrushAll;
  const btn = document.getElementById('tgBrushAllBtn');
  if (btn) {
    btn.textContent = tgBrushAll ? 'Todas' : 'Sel.';
    btn.classList.toggle('active', tgBrushAll);
  }
}

function tgSetIndScaleX(val) {
  if (tgSelected < 0 || tgSelected >= tgTogas.length) return;
  tgTogas[tgSelected].scaleX = val;
  const lbl = document.getElementById('tgIndScaleXVal');
  if (lbl) lbl.textContent = Math.round(val * 100) + '%';
  tgDraw(); tgRenderList(); tgSaveState();
}

function tgUpdateIndScaleXUI() {
  const slider = document.getElementById('tgIndScaleXSlider');
  const lbl = document.getElementById('tgIndScaleXVal');
  if (tgSelected >= 0 && tgSelected < tgTogas.length) {
    const val = tgTogas[tgSelected].scaleX;
    if (slider) slider.value = Math.round(val * 100);
    if (lbl) lbl.textContent = Math.round(val * 100) + '%';
  }
}

// --- Coordinate helpers ---
function tgImgToScreen(ix, iy) {
  const cw = tgCanvas.width, ch = tgCanvas.height;
  const s = pvZoom;
  const ox = (cw - tgImgW * s) / 2 + pvX;
  const oy = (ch - tgImgH * s) / 2 + pvY;
  // Apply image transform
  const rad = tgImgTf.rotation * Math.PI / 180;
  const icx = tgImgW / 2, icy = tgImgH / 2;
  const dx = ix - icx, dy = iy - icy;
  const rx = dx * Math.cos(rad) - dy * Math.sin(rad) + icx + tgImgTf.x;
  const ry = dx * Math.sin(rad) + dy * Math.cos(rad) + icy + tgImgTf.y;
  return { x: ox + rx * s, y: oy + ry * s };
}

function tgScreenToImg(sx, sy) {
  const cw = tgCanvas.width, ch = tgCanvas.height;
  const s = pvZoom;
  const ox = (cw - tgImgW * s) / 2 + pvX;
  const oy = (ch - tgImgH * s) / 2 + pvY;
  const px = (sx - ox) / s, py = (sy - oy) / s;
  // Inverse image transform
  const rad = -tgImgTf.rotation * Math.PI / 180;
  const icx = tgImgW / 2, icy = tgImgH / 2;
  const dx = px - tgImgTf.x - icx, dy = py - tgImgTf.y - icy;
  const rx = dx * Math.cos(rad) - dy * Math.sin(rad) + icx;
  const ry = dx * Math.sin(rad) + dy * Math.cos(rad) + icy;
  return { x: rx, y: ry };
}

// Convert screen coords to base image space (no image transform — for togas)
function tgScreenToBase(sx, sy) {
  const cw = tgCanvas.width, ch = tgCanvas.height;
  const s = pvZoom;
  const ox = (cw - tgImgW * s) / 2 + pvX;
  const oy = (ch - tgImgH * s) / 2 + pvY;
  return { x: (sx - ox) / s, y: (sy - oy) / s };
}

// --- Canvas drawing ---
function tgDraw() {
  if (!tgMode || !tgCanvas) return;
  const ctx = tgCtx;
  const cw = tgCanvas.width, ch = tgCanvas.height;
  ctx.clearRect(0, 0, cw, ch);

  // Checkerboard
  const pat = ctx.createPattern((() => {
    const c = document.createElement('canvas'); c.width = 20; c.height = 20;
    const x = c.getContext('2d');
    x.fillStyle = '#3a3a4a'; x.fillRect(0, 0, 10, 10); x.fillRect(10, 10, 10, 10);
    x.fillStyle = '#2a2a3a'; x.fillRect(10, 0, 10, 10); x.fillRect(0, 10, 10, 10);
    return c;
  })(), 'repeat');

  const s = pvZoom;
  const ox = (cw - tgImgW * s) / 2 + pvX;
  const oy = (ch - tgImgH * s) / 2 + pvY;

  ctx.save();
  // Image group transform (rotate around image center)
  const imgRad = tgImgTf.rotation * Math.PI / 180;
  const imgCx = ox + tgImgW * s / 2;
  const imgCy = oy + tgImgH * s / 2;
  ctx.translate(imgCx, imgCy);
  ctx.rotate(imgRad);
  ctx.translate(-imgCx, -imgCy);
  ctx.translate(tgImgTf.x * s, tgImgTf.y * s);

  // Checkerboard bg
  ctx.fillStyle = pat;
  ctx.fillRect(ox, oy, tgImgW * s, tgImgH * s);

  // Draw cutout
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'medium';
  ctx.drawImage(tgCutImg, ox, oy, tgImgW * s, tgImgH * s);

  // Draw borlas (composited from saved state)
  if (tgBorlaState && tgBorlaImg) {
    for (const b of tgBorlaState.borlas) {
      if (!b.visible) continue;
      const bw = b.scale * s;
      const bh = bw * (tgBorlaImg.height / tgBorlaImg.width);
      const bsx = ox + b.x * s;
      const bsy = oy + b.y * s;
      ctx.save();
      ctx.translate(bsx, bsy);
      ctx.rotate(b.rotation * Math.PI / 180);
      ctx.drawImage(tgBorlaImg, -bw / 2, 0, bw, bh);
      ctx.restore();
    }
  }

  // Selection rectangle for img-move tool (drawn in image group transform space, screen pixels)
  if (tgTool === 'img-move') {
    const pad = 6;
    ctx.save();
    ctx.strokeStyle = '#7dcfff'; ctx.lineWidth = 2; ctx.setLineDash([8, 5]);
    ctx.strokeRect(ox - pad, oy - pad, tgImgW * s + pad * 2, tgImgH * s + pad * 2);
    ctx.setLineDash([]);
    // Rotation handle: circle at top center with stem line
    const hx = ox + tgImgW * s / 2;
    const hy = oy - pad;
    const handleDist = 30;
    const handleR = 7;
    ctx.strokeStyle = '#7dcfff'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(hx, hy - handleDist); ctx.stroke();
    ctx.beginPath(); ctx.arc(hx, hy - handleDist, handleR, 0, Math.PI * 2);
    ctx.fillStyle = tgImgRotating ? '#e0af68' : '#7dcfff'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
    // Rotation icon
    ctx.fillStyle = '#1a1b26'; ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('\u21BB', hx, hy - handleDist);
    ctx.restore();
  }

  // Face markers (inside image group transform so they follow the photo)
  for (let i = 0; i < tgFaces.length; i++) {
    const isSeated = tgSeatedIdx.includes(i);
    const f = tgFaces[i];
    const fcx = ox + (f.x + f.w / 2) * tgImgW * s;
    const fcy = oy + (f.y + f.h / 2) * tgImgH * s;
    ctx.beginPath();
    ctx.arc(fcx, fcy, isSeated ? 6 : 4, 0, Math.PI * 2);
    if (isSeated) {
      ctx.fillStyle = 'rgba(224,175,104,0.7)'; ctx.fill();
      ctx.strokeStyle = '#e0af68'; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = '#e0af68'; ctx.font = 'bold 10px monospace';
      ctx.fillText('S', fcx + 8, fcy + 4);
    } else {
      ctx.fillStyle = 'rgba(122,162,247,0.3)'; ctx.fill();
      ctx.strokeStyle = 'rgba(122,162,247,0.5)'; ctx.lineWidth = 1; ctx.stroke();
    }
  }

  ctx.restore(); // END image group transform — togas are drawn independently below

  // --- Draw togas (independent of image group transform) ---
  ctx.save();
  const tgBaseOx = (cw - tgImgW * s) / 2 + pvX;
  const tgBaseOy = (ch - tgImgH * s) / 2 + pvY;
  const tgOx = tgBaseOx + tgGroupTf.x * s;
  const tgOy = tgBaseOy + tgGroupTf.y * s;
  const tgRad = tgGroupTf.rotation * Math.PI / 180;
  // Rotate around center of toga group
  if (Math.abs(tgRad) > 0.001) {
    const gcx = tgBaseOx + tgImgW * s / 2 + tgGroupTf.x * s;
    const gcy = tgBaseOy + tgImgH * s / 2 + tgGroupTf.y * s;
    ctx.translate(gcx, gcy);
    ctx.rotate(tgRad);
    ctx.translate(-gcx, -gcy);
  }

  for (let i = 0; i < tgTogas.length; i++) {
    const t = tgTogas[i];
    const togaImg = tgTogaImgs[t.variant];
    if (!togaImg) continue;

    const tw = t.scale * t.scaleX * s;
    const th = t.scale * tgRefAR * tgGroupTf.scaleY * s;
    const tx = tgOx + t.x * s;
    const ty = tgOy + t.y * s;

    ctx.save();
    ctx.translate(tx, ty);
    if (t.rotation) ctx.rotate(t.rotation * Math.PI / 180);
    if (t.flipH) ctx.scale(-1, 1);

    if (t.mask) {
      const mtw = Math.ceil(Math.abs(tw)), mth = Math.ceil(Math.abs(th));
      if (mtw > 0 && mth > 0) {
        const tmp = document.createElement('canvas');
        tmp.width = mtw; tmp.height = mth;
        const tc = tmp.getContext('2d');
        tc.drawImage(togaImg, 0, 0, mtw, mth);
        tc.globalCompositeOperation = 'destination-in';
        tc.drawImage(t.mask, 0, 0, mtw, mth);
        ctx.drawImage(tmp, -tw / 2, 0, tw, th);
      }
    } else {
      ctx.drawImage(togaImg, -tw / 2, 0, tw, th);
    }
    ctx.restore();

    // Selection rect + scaleX handles
    if (i === tgSelected && !tgDragging) {
      ctx.save();
      ctx.translate(tx, ty);
      if (t.rotation) ctx.rotate(t.rotation * Math.PI / 180);
      ctx.strokeStyle = '#7dcfff'; ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
      ctx.strokeRect(-tw / 2 - 2, -2, tw + 4, th + 4);
      ctx.setLineDash([]);
      // ScaleX drag handles (left + right midpoints)
      const hSize = 6;
      const hMidY = th / 2;
      ctx.fillStyle = tgScaleXDrag ? '#e0af68' : '#7dcfff';
      ctx.fillRect(-tw / 2 - 2 - hSize / 2, hMidY - hSize / 2, hSize, hSize);
      ctx.fillRect(tw / 2 + 2 - hSize / 2, hMidY - hSize / 2, hSize, hSize);
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
      ctx.strokeRect(-tw / 2 - 2 - hSize / 2, hMidY - hSize / 2, hSize, hSize);
      ctx.strokeRect(tw / 2 + 2 - hSize / 2, hMidY - hSize / 2, hSize, hSize);
      // Label
      ctx.fillStyle = '#7dcfff'; ctx.font = 'bold 11px monospace';
      ctx.fillText('#' + (i + 1), -tw / 2, -8);
      ctx.restore();
    }
  }
  ctx.restore(); // toga group transform

  // Guide lines (independent of image/toga group transforms)
  ctx.save();
  ctx.setLineDash([8, 4]);
  ctx.lineWidth = 1.5;
  const guideBaseOy = (ch - tgImgH * s) / 2 + pvY;
  for (let i = 0; i < tgGuides.length; i++) {
    const gy = guideBaseOy + tgGuides[i].y * s;
    ctx.strokeStyle = (tgTool === 'guide' || tgDraggingGuide === i) ? '#7dcfff' : 'rgba(125,207,255,0.4)';
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(cw, gy); ctx.stroke();
    ctx.fillStyle = '#7dcfff'; ctx.font = '10px monospace';
    ctx.fillText('Guia ' + (i + 1), 4, gy - 4);
  }
  ctx.setLineDash([]);
  ctx.restore();

  // Brush cursor
  if (tgTool === 'brush' && tgMousePos) {
    const r = tgBrushSize * pvZoom;
    ctx.save();
    ctx.strokeStyle = tgBrushAction === 'erase' ? '#f7768e' : '#9ece6a';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(tgMousePos.sx, tgMousePos.sy, r, 0, Math.PI * 2); ctx.stroke();
    // Feather ring
    if (tgBrushFeather > 0) {
      ctx.globalAlpha = 0.3;
      ctx.beginPath(); ctx.arc(tgMousePos.sx, tgMousePos.sy, r * (1 + tgBrushFeather * 0.5), 0, Math.PI * 2); ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.restore();
  }

  // Zoom label
  const lbl = document.getElementById('tgZoomLabel');
  if (lbl) lbl.textContent = Math.round(pvZoom * 100) + '%';
}

// --- Brush ---
function tgGetMask(t) {
  if (!t.mask) {
    const togaImg = tgTogaImgs[t.variant];
    if (!togaImg) return null;
    const c = document.createElement('canvas');
    c.width = Math.max(1, togaImg.width);
    c.height = Math.max(1, togaImg.height);
    const cx = c.getContext('2d');
    cx.fillStyle = '#fff'; cx.fillRect(0, 0, c.width, c.height);
    t.mask = c;
  }
  return t.mask;
}

function tgBrushOnMask(t, imgX, imgY) {
  const togaImg = tgTogaImgs[t.variant];
  if (!togaImg) return;
  const mask = tgGetMask(t);
  if (!mask) return;

  const togaW = t.scale * t.scaleX;
  const togaH = t.scale * tgRefAR * tgGroupTf.scaleY;

  // Account for group transform + individual position
  const dx = imgX - (t.x + tgGroupTf.x);
  const dy = imgY - (t.y + tgGroupTf.y);
  // Un-rotate for group + individual rotation
  const rad = -(tgGroupTf.rotation + (t.rotation || 0)) * Math.PI / 180;
  let rx = dx * Math.cos(rad) - dy * Math.sin(rad);
  let ry = dx * Math.sin(rad) + dy * Math.cos(rad);
  if (t.flipH) rx = -rx;

  // Convert to mask coords
  const localX = (rx + togaW / 2) / togaW * mask.width;
  const localY = ry / togaH * mask.height;
  const radius = tgBrushSize * (mask.width / togaW);

  const ctx = mask.getContext('2d');
  // Create gradient for feather
  if (tgBrushFeather > 0.01 && radius > 1) {
    const innerR = radius * (1 - tgBrushFeather);
    const grad = ctx.createRadialGradient(localX, localY, innerR, localX, localY, radius);
    if (tgBrushAction === 'erase') {
      ctx.globalCompositeOperation = 'destination-out';
      grad.addColorStop(0, `rgba(0,0,0,${tgBrushFlow})`);
      grad.addColorStop(1, 'rgba(0,0,0,0)');
    } else {
      ctx.globalCompositeOperation = 'source-over';
      grad.addColorStop(0, `rgba(255,255,255,${tgBrushFlow})`);
      grad.addColorStop(1, 'rgba(255,255,255,0)');
    }
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(localX, localY, radius, 0, Math.PI * 2); ctx.fill();
  } else {
    ctx.globalCompositeOperation = tgBrushAction === 'erase' ? 'destination-out' : 'source-over';
    ctx.fillStyle = tgBrushAction === 'erase'
      ? `rgba(0,0,0,${tgBrushFlow})`
      : `rgba(255,255,255,${tgBrushFlow})`;
    ctx.beginPath(); ctx.arc(localX, localY, radius, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalCompositeOperation = 'source-over';
}

// --- Undo/Redo ---
let tgUndoGroupId = 0;  // auto-increment group ID for batched undo

function tgSaveMaskSnapshot(idx, groupId) {
  const t = tgTogas[idx];
  const mask = tgGetMask(t);
  if (!mask) return;
  const snap = mask.getContext('2d').getImageData(0, 0, mask.width, mask.height);
  tgUndoStack.push({idx, data: snap, group: groupId});
  tgRedoStack = [];
  if (tgUndoStack.length > 200) tgUndoStack.shift();
}

function tgUndo() {
  if (!tgUndoStack.length) return;
  const topGroup = tgUndoStack[tgUndoStack.length - 1].group;
  // Pop ALL entries with the same group ID
  const redoBatch = [];
  while (tgUndoStack.length && tgUndoStack[tgUndoStack.length - 1].group === topGroup) {
    const snap = tgUndoStack.pop();
    const t = tgTogas[snap.idx];
    const mask = tgGetMask(t);
    if (!mask) continue;
    const cur = mask.getContext('2d').getImageData(0, 0, mask.width, mask.height);
    redoBatch.push({idx: snap.idx, data: cur, group: topGroup});
    mask.getContext('2d').putImageData(snap.data, 0, 0);
  }
  // Push redo batch (reversed so redo restores in original order)
  for (let i = redoBatch.length - 1; i >= 0; i--) tgRedoStack.push(redoBatch[i]);
  tgDraw();
}

function tgRedo() {
  if (!tgRedoStack.length) return;
  const topGroup = tgRedoStack[tgRedoStack.length - 1].group;
  const undoBatch = [];
  while (tgRedoStack.length && tgRedoStack[tgRedoStack.length - 1].group === topGroup) {
    const snap = tgRedoStack.pop();
    const t = tgTogas[snap.idx];
    const mask = tgGetMask(t);
    if (!mask) continue;
    const cur = mask.getContext('2d').getImageData(0, 0, mask.width, mask.height);
    undoBatch.push({idx: snap.idx, data: cur, group: topGroup});
    mask.getContext('2d').putImageData(snap.data, 0, 0);
  }
  for (let i = undoBatch.length - 1; i >= 0; i--) tgUndoStack.push(undoBatch[i]);
  tgDraw();
}

// --- Hit test ---
function tgHitToga(sx, sy) {
  // Togas are independent of image group transform (tgImgTf)
  for (let i = tgTogas.length - 1; i >= 0; i--) {
    const t = tgTogas[i];
    const togaImg = tgTogaImgs[t.variant];
    if (!togaImg) continue;
    const tw = t.scale * t.scaleX * pvZoom;
    const th = t.scale * tgRefAR * tgGroupTf.scaleY * pvZoom;
    const cw = tgCanvas.width, ch = tgCanvas.height;
    const baseOx = (cw - tgImgW * pvZoom) / 2 + pvX;
    const baseOy = (ch - tgImgH * pvZoom) / 2 + pvY;
    // Toga screen position (no image transform — togas are independent)
    const tx = baseOx + (t.x + tgGroupTf.x) * pvZoom;
    const ty = baseOy + (t.y + tgGroupTf.y) * pvZoom;
    // Simple AABB test
    if (sx >= tx - tw / 2 && sx <= tx + tw / 2 && sy >= ty && sy <= ty + th) return i;
  }
  return -1;
}

function tgHitScaleXHandle(sx, sy) {
  // Returns 'left' or 'right' if mouse is on a scaleX handle of the selected toga, else null
  if (tgSelected < 0 || tgSelected >= tgTogas.length) return null;
  const t = tgTogas[tgSelected];
  const s = pvZoom;
  const tw = t.scale * t.scaleX * s;
  const th = t.scale * tgRefAR * tgGroupTf.scaleY * s;
  const cw = tgCanvas.width, ch = tgCanvas.height;
  const baseOx = (cw - tgImgW * s) / 2 + pvX;
  const baseOy = (ch - tgImgH * s) / 2 + pvY;
  const tx = baseOx + (t.x + tgGroupTf.x) * s;
  const ty = baseOy + (t.y + tgGroupTf.y) * s;
  const hMidY = ty + th / 2;
  const hitR = 10; // generous hit radius
  // Left handle
  const lx = tx - tw / 2 - 2;
  if (Math.abs(sx - lx) < hitR && Math.abs(sy - hMidY) < hitR) return 'left';
  // Right handle
  const rx = tx + tw / 2 + 2;
  if (Math.abs(sx - rx) < hitR && Math.abs(sy - hMidY) < hitR) return 'right';
  return null;
}

function tgHitGuide(sy) {
  const ch = tgCanvas.height;
  const baseOy = (ch - tgImgH * pvZoom) / 2 + pvY;
  for (let i = 0; i < tgGuides.length; i++) {
    const gy = baseOy + tgGuides[i].y * pvZoom;
    if (Math.abs(sy - gy) < 8) return i;
  }
  return -1;
}

function _tgHitSeatedMarker(sx, sy) {
  // Returns index into tgSeatedIdx if click is near a seated face marker
  for (let si = 0; si < tgSeatedIdx.length; si++) {
    const fi = tgSeatedIdx[si];
    const f = tgFaces[fi];
    if (!f) continue;
    const scr = tgImgToScreen((f.x + f.w / 2) * tgImgW, (f.y + f.h / 2) * tgImgH);
    if (Math.hypot(sx - scr.x, sy - scr.y) < 12) return si;
  }
  return -1;
}

// --- Events ---
function tgBindEvents(wrap) {
  let panning = false, panStartX = 0, panStartY = 0, panOX = 0, panOY = 0;

  wrap.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = tgCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const oldZ = pvZoom;
    pvZoom = Math.max(0.05, Math.min(pvZoom * (e.deltaY < 0 ? 1.1 : 0.9), 20));
    const cx = tgCanvas.width / 2 + pvX, cy = tgCanvas.height / 2 + pvY;
    pvX += (mx - cx) * (1 - pvZoom / oldZ);
    pvY += (my - cy) * (1 - pvZoom / oldZ);
    tgDraw();
  }, {passive: false});

  wrap.addEventListener('mousedown', (e) => {
    const rect = tgCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;

    // Pan: middle click or space
    if (e.button === 1 || spaceHeld) {
      panning = true; panStartX = e.clientX; panStartY = e.clientY; panOX = pvX; panOY = pvY;
      wrap.style.cursor = 'grabbing'; e.preventDefault(); return;
    }
    if (e.button !== 0) return;

    // Add seated person
    if (tgTool === 'add-seat') {
      const img = tgScreenToImg(mx, my);
      const avgFaceW = tgFaces.length ? tgFaces.reduce((s,f)=>s+f.w,0)/tgFaces.length : 0.04;
      const avgFaceH = tgFaces.length ? tgFaces.reduce((s,f)=>s+f.h,0)/tgFaces.length : 0.05;
      const newFace = { x: img.x/tgImgW - avgFaceW/2, y: img.y/tgImgH - avgFaceH/2, w: avgFaceW, h: avgFaceH, conf: 1.0 };
      tgFaces.push(newFace);
      const newIdx = tgFaces.length - 1;
      tgSeatedIdx.push(newIdx);
      // Use same Y as existing togas, or calculate from this face
      const commonY = tgTogas.length > 0
        ? tgTogas[0].y
        : (newFace.y + newFace.h) * tgImgH + avgFaceH * tgImgH * 0.3;
      const avgScale = tgTogas.length > 0
        ? tgTogas.reduce((s,t)=>s+t.scale,0) / tgTogas.length
        : newFace.w * tgImgW * 1.8;
      const variant = tgTogaVariants[tgTogas.length % tgTogaVariants.length];
      tgTogas.push({
        id: tgTogas.length, variant,
        x: img.x, y: commonY,
        baseX: img.x, baseY: commonY,
        scale: avgScale, scaleX: 1.0,
        flipH: Math.random() > 0.5, rotation: 0, mask: null,
      });
      showToast(`Persona sentada #${tgSeatedIdx.length} agregada`, 'ok');
      tgRenderList(); tgDraw(); tgSaveState();
      return;
    }

    // Add guide
    if (tgTool === 'guide') {
      const base = tgScreenToBase(mx, my);
      tgGuides.push({y: base.y});
      tgDraw(); tgSaveState();
      return;
    }

    // Guide drag: works from ANY tool when mouse is near a guide line
    const gi = tgHitGuide(my);
    if (gi >= 0) {
      tgDraggingGuide = gi;
      wrap.style.cursor = 'ns-resize';
      return;
    }

    // Image move tool — check rotation handle first
    if (tgTool === 'img-move') {
      const cw2 = tgCanvas.width, ch2 = tgCanvas.height;
      const s2 = pvZoom;
      const ox2 = (cw2 - tgImgW * s2) / 2 + pvX;
      const oy2 = (ch2 - tgImgH * s2) / 2 + pvY;
      // Handle position in screen space (accounting for image transform)
      const hScrX = ox2 + tgImgW * s2 / 2 + tgImgTf.x * s2;
      const hScrY = oy2 - 6 - 30 + tgImgTf.y * s2;
      // Apply image rotation around image center for handle position
      const irad = tgImgTf.rotation * Math.PI / 180;
      const icx2 = ox2 + tgImgW * s2 / 2 + tgImgTf.x * s2;
      const icy2 = oy2 + tgImgH * s2 / 2 + tgImgTf.y * s2;
      const hdx = hScrX - icx2, hdy = hScrY - icy2;
      const rhx = hdx * Math.cos(irad) - hdy * Math.sin(irad) + icx2;
      const rhy = hdx * Math.sin(irad) + hdy * Math.cos(irad) + icy2;
      if (Math.hypot(mx - rhx, my - rhy) < 14) {
        // Start rotation
        tgImgRotating = true;
        tgImgRotStart = Math.atan2(my - icy2, mx - icx2) - irad;
        wrap.style.cursor = 'grab';
        return;
      }
      tgImgDragging = { startX: e.clientX, startY: e.clientY, ox: tgImgTf.x, oy: tgImgTf.y };
      wrap.style.cursor = 'move';
      return;
    }

    // Toga group move
    if (tgTool === 'toga-group') {
      tgGroupDragging = { startX: e.clientX, startY: e.clientY, ox: tgGroupTf.x, oy: tgGroupTf.y };
      wrap.style.cursor = 'move';
      return;
    }

    // Brush
    if (tgTool === 'brush' && (tgBrushAll ? tgTogas.length > 0 : tgSelected >= 0)) {
      const gid = ++tgUndoGroupId;
      if (tgBrushAll) {
        for (let bi = 0; bi < tgTogas.length; bi++) tgSaveMaskSnapshot(bi, gid);
      } else {
        tgSaveMaskSnapshot(tgSelected, gid);
      }
      tgBrushDrawing = true;
      const img = tgScreenToBase(mx, my);
      if (tgBrushAll) {
        for (const t of tgTogas) tgBrushOnMask(t, img.x, img.y);
      } else {
        tgBrushOnMask(tgTogas[tgSelected], img.x, img.y);
      }
      tgDraw(); return;
    }

    // Move tool: check scaleX handles first (if a toga is selected)
    const scaleHandle = tgHitScaleXHandle(mx, my);
    if (scaleHandle && tgSelected >= 0) {
      const t = tgTogas[tgSelected];
      tgScaleXDrag = { idx: tgSelected, side: scaleHandle, startMx: mx, startScaleX: t.scaleX };
      wrap.style.cursor = 'ew-resize';
      tgDraw(); return;
    }

    // Move tool: hit test togas
    const hit = tgHitToga(mx, my);
    if (hit >= 0) {
      tgSelected = hit;
      const t = tgTogas[hit];
      tgDragging = { idx: hit, startMx: mx, startX: t.x };
      wrap.style.cursor = 'ew-resize';
      tgRenderList(); tgDraw(); tgUpdateIndScaleXUI();
    } else {
      tgSelected = -1;
      tgRenderList(); tgDraw();
      // Pan on empty
      panning = true; panStartX = e.clientX; panStartY = e.clientY; panOX = pvX; panOY = pvY;
      wrap.style.cursor = 'grabbing';
    }
  });

  wrap.addEventListener('mousemove', (e) => {
    const rect = tgCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    tgMousePos = {sx: mx, sy: my};

    if (panning) {
      pvX = panOX + (e.clientX - panStartX); pvY = panOY + (e.clientY - panStartY);
      tgDraw(); return;
    }
    if (tgImgRotating) {
      const cw2 = tgCanvas.width, ch2 = tgCanvas.height;
      const s2 = pvZoom;
      const ox2 = (cw2 - tgImgW * s2) / 2 + pvX;
      const oy2 = (ch2 - tgImgH * s2) / 2 + pvY;
      const icx2 = ox2 + tgImgW * s2 / 2 + tgImgTf.x * s2;
      const icy2 = oy2 + tgImgH * s2 / 2 + tgImgTf.y * s2;
      const angle = Math.atan2(my - icy2, mx - icx2) - tgImgRotStart;
      tgImgTf.rotation = angle * 180 / Math.PI;
      // Clamp to ±15 degrees
      tgImgTf.rotation = Math.max(-15, Math.min(15, tgImgTf.rotation));
      const ri = document.getElementById('tgImgRotInput');
      if (ri) ri.value = tgImgTf.rotation.toFixed(1);
      tgDraw(); return;
    }
    if (tgImgDragging) {
      tgImgTf.x = tgImgDragging.ox + (e.clientX - tgImgDragging.startX) / pvZoom;
      tgImgTf.y = tgImgDragging.oy + (e.clientY - tgImgDragging.startY) / pvZoom;
      tgDraw(); return;
    }
    if (tgGroupDragging) {
      tgGroupTf.x = tgGroupDragging.ox + (e.clientX - tgGroupDragging.startX) / pvZoom;
      tgGroupTf.y = tgGroupDragging.oy + (e.clientY - tgGroupDragging.startY) / pvZoom;
      tgDraw(); return;
    }
    if (tgDraggingGuide >= 0) {
      const base = tgScreenToBase(mx, my);
      tgGuides[tgDraggingGuide].y = base.y;
      tgDraw(); return;
    }
    if (tgScaleXDrag) {
      const d = tgScaleXDrag;
      const deltaPx = mx - d.startMx;
      // Right handle: drag right = wider, left handle: drag left = wider
      const dir = d.side === 'right' ? 1 : -1;
      // Convert screen delta to scaleX delta: deltaScaleX = deltaPx / (t.scale * pvZoom)
      const t = tgTogas[d.idx];
      const dScaleX = (dir * deltaPx) / (t.scale * pvZoom);
      t.scaleX = Math.max(0.2, Math.min(3, d.startScaleX + dScaleX));
      tgUpdateIndScaleXUI();
      tgDraw(); return;
    }
    if (tgDragging) {
      // Togas only move horizontally individually
      const deltaScreen = mx - tgDragging.startMx;
      tgTogas[tgDragging.idx].x = tgDragging.startX + deltaScreen / pvZoom;
      tgDraw(); return;
    }
    if (tgBrushDrawing && tgTool === 'brush') {
      const img = tgScreenToBase(mx, my);
      if (tgBrushAll) {
        for (const t of tgTogas) tgBrushOnMask(t, img.x, img.y);
      } else if (tgSelected >= 0) {
        tgBrushOnMask(tgTogas[tgSelected], img.x, img.y);
      }
      tgDraw(); return;
    }
    if (tgTool === 'brush') tgDraw();
    // Cursor hint for scaleX handles
    if (tgTool === 'move' && tgSelected >= 0 && !tgDragging && !panning) {
      wrap.style.cursor = tgHitScaleXHandle(mx, my) ? 'ew-resize' : 'default';
    }
  });

  wrap.addEventListener('mouseup', () => {
    const hadAction = panning || tgDragging || tgDraggingGuide >= 0 || tgBrushDrawing || tgImgDragging || tgImgRotating || tgGroupDragging || tgScaleXDrag;
    panning = false; tgBrushDrawing = false;
    if (tgScaleXDrag) { tgScaleXDrag = null; tgRenderList(); tgDraw(); tgSaveState(); }
    if (tgDragging) { tgDragging = null; tgRenderList(); tgDraw(); }
    if (tgDraggingGuide >= 0) { tgDraggingGuide = -1; tgDraw(); }
    if (tgImgDragging) { tgImgDragging = null; }
    if (tgImgRotating) { tgImgRotating = false; tgDraw(); }
    if (tgGroupDragging) { tgGroupDragging = null; }
    const wrap = document.getElementById('tgWrap');
    if (wrap) {
      if (tgTool === 'add-seat' || tgTool === 'guide') wrap.style.cursor = 'crosshair';
      else if (tgTool === 'brush') wrap.style.cursor = 'none';
      else if (tgTool === 'img-move' || tgTool === 'toga-group') wrap.style.cursor = 'move';
      else wrap.style.cursor = 'default';
    }
    if (hadAction) tgSaveState();
  });

  wrap.addEventListener('mouseleave', () => {
    panning = false; tgDragging = null; tgDraggingGuide = -1; tgScaleXDrag = null;
    tgBrushDrawing = false; tgImgDragging = null; tgImgRotating = false; tgGroupDragging = null;
    tgMousePos = null; tgDraw();
  });

  wrap.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    const rect = tgCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;

    // Right-click on guide = delete guide
    const gi = tgHitGuide(my);
    if (gi >= 0) {
      tgGuides.splice(gi, 1);
      tgDraw(); tgSaveState();
      return;
    }

    // Right-click on seated marker = remove seated + its toga
    const hitSeat = _tgHitSeatedMarker(mx, my);
    if (hitSeat >= 0) {
      // Remove the toga associated with this seated index
      const seatedFaceIdx = tgSeatedIdx[hitSeat];
      // Find and remove the toga that was created for this face (by position match)
      const togaIdx = hitSeat < tgTogas.length ? hitSeat : -1;
      if (togaIdx >= 0) tgTogas.splice(togaIdx, 1);
      tgSeatedIdx.splice(hitSeat, 1);
      // Re-index toga ids
      tgTogas.forEach((t, i) => t.id = i);
      tgSelected = -1;
      showToast('Persona sentada eliminada', 'ok');
      tgRenderList(); tgDraw(); tgSaveState();
      return;
    }
  });
}

// --- Toga list ---
function tgRenderList() {
  const list = document.getElementById('tgList');
  if (!list) return;
  list.innerHTML = tgTogas.map((t, i) => {
    const sel = i === tgSelected ? 'selected' : '';
    const flipLabel = t.flipH ? 'H' : '';
    return `<div class="tg-item ${sel}" onclick="tgSelectToga(${i})">
      #${i+1} ${t.variant.replace('CAIDA_','')}${flipLabel}
      <button onclick="event.stopPropagation(); tgFlipToga(${i})" title="Voltear H">&#8644;</button>
      <button onclick="event.stopPropagation(); tgScaleXToga(${i}, 0.1)" title="+ Ancho">+</button>
      <button onclick="event.stopPropagation(); tgScaleXToga(${i}, -0.1)" title="- Ancho">-</button>
    </div>`;
  }).join('');
}

function tgSelectToga(i) { tgSelected = i; tgRenderList(); tgDraw(); tgUpdateIndScaleXUI(); }

function tgFlipToga(i) {
  tgTogas[i].flipH = !tgTogas[i].flipH;
  tgRenderList(); tgDraw(); tgSaveState();
}

function tgScaleXToga(i, delta) {
  tgTogas[i].scaleX = Math.max(0.2, Math.min(3, tgTogas[i].scaleX + delta));
  tgDraw(); tgSaveState();
}

// --- Save/Load ---
function tgSaveState() {
  if (!tgCbtis || !tgFilename) return;
  const state = {
    detectionDone: tgDetectionDone,
    faces: tgFaces, seatedIdx: tgSeatedIdx,
    groupTf: tgGroupTf, imgTf: tgImgTf,
    guides: tgGuides,
    brushFeather: tgBrushFeather, brushFlow: tgBrushFlow,
    togas: tgTogas.map(t => ({
      variant: t.variant, x: t.x, y: t.y, baseX: t.baseX, baseY: t.baseY,
      scale: t.scale, scaleX: t.scaleX, flipH: t.flipH, rotation: t.rotation,
      mask: t.mask ? t.mask.toDataURL('image/png') : null,
    }))
  };
  fetch(`/api/togas/state/${tgCbtis}/${encodeURIComponent(tgFilename)}`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(state)
  }).catch(() => {});
}

async function tgLoadState() {
  try {
    const res = await fetch(`/api/togas/state/${tgCbtis}/${encodeURIComponent(tgFilename)}`);
    const state = await res.json();
    if (!state || !state.togas) return false;
    tgDetectionDone = state.detectionDone || false;
    tgFaces = state.faces || [];
    tgSeatedIdx = state.seatedIdx || [];
    tgGroupTf = state.groupTf || {x:0, y:0, rotation:0, scaleY:1.0};
    tgImgTf = state.imgTf || {x:0, y:0, rotation:0};
    tgGuides = state.guides || [];
    tgBrushFeather = state.brushFeather ?? 0.5;
    tgBrushFlow = state.brushFlow ?? 1.0;
    tgTogas = state.togas.map((t, i) => ({
      id: i, variant: t.variant, x: t.x, y: t.y,
      baseX: t.baseX || t.x, baseY: t.baseY || t.y,
      scale: t.scale, scaleX: t.scaleX || 1.0,
      flipH: t.flipH || false, rotation: t.rotation || 0,
      mask: null, _maskData: t.mask,
    }));
    // Restore masks
    for (const t of tgTogas) {
      if (t._maskData) {
        const img = await loadImg(t._maskData);
        const c = document.createElement('canvas');
        c.width = img.width; c.height = img.height;
        c.getContext('2d').drawImage(img, 0, 0);
        t.mask = c; delete t._maskData;
      }
    }
    // Update UI controls
    const btn = document.getElementById('tgDetDoneBtn');
    if (btn) btn.innerHTML = tgDetectionDone ? '&#9745; Det. lista' : '&#9744; Det. lista';
    return true;
  } catch { return false; }
}

// --- Zoom/Fullscreen ---
function tgZoomFit() {
  if (!tgCutImg || !tgCanvas) return;
  pvZoom = Math.min(tgCanvas.width / tgImgW, tgCanvas.height / tgImgH) * 0.95;
  pvX = 0; pvY = 0; tgDraw();
}

function tgToggleFullscreen() {
  const panel = document.getElementById('previewPanel');
  if (!panel) return;
  _toggleRealFullscreen(panel, 'tgFullscreenBtn', () => {
    const wrap = document.getElementById('tgWrap');
    if (wrap && tgCanvas && tgMode) {
      const rect = wrap.getBoundingClientRect();
      tgCanvas.width = rect.width; tgCanvas.height = rect.height;
      tgZoomFit();
    }
  });
}

// --- TIFF Export ---
async function tgExportTiff() {
  if (tgTogas.length === 0) { showToast('No hay togas para exportar', 'warn'); return; }
  showToast('Generando TIFF con borlas + togas... espera', 'info');
  try {
    // --- Borlas: raw positions in cutout space (tgImgTf affects both cutout
    // and borlas equally on screen, so it cancels out in the TIFF) ---
    let borlas = [];
    if (tgBorlaState && tgBorlaState.borlas) {
      borlas = tgBorlaState.borlas.filter(b => b.visible).map(b => ({
        color: b.color, x: Math.round(b.x), y: Math.round(b.y),
        scale: Math.round(b.scale), rotation: b.rotation, visible: true,
        mask: b.mask || null,
      }));
    }
    // --- Togas: apply toga group transform only (independent of tgImgTf) ---
    const _imgCx = tgImgW / 2, _imgCy = tgImgH / 2;
    const _gRad = tgGroupTf.rotation * Math.PI / 180;
    const togas = tgTogas.map(t => {
      let px = t.x + tgGroupTf.x;
      let py = t.y + tgGroupTf.y;
      // Apply group orbital rotation around image center
      if (Math.abs(_gRad) > 0.001) {
        const dx = px - _imgCx, dy = py - _imgCy;
        px = _imgCx + dx * Math.cos(_gRad) - dy * Math.sin(_gRad);
        py = _imgCy + dx * Math.sin(_gRad) + dy * Math.cos(_gRad);
      }
      return {
        variant: t.variant, x: Math.round(px), y: Math.round(py),
        scale: Math.round(t.scale), scaleX: t.scaleX,
        flipH: t.flipH,
        rotation: (t.rotation || 0) + tgGroupTf.rotation,
        mask: t.mask ? t.mask.toDataURL('image/png') : null,
      };
    });
    const payload = {
      borlas, togas,
      togaGroupTransform: { scaleY: tgGroupTf.scaleY },
      imgTransform: { x: tgImgTf.x || 0, y: tgImgTf.y || 0, rotation: tgImgTf.rotation || 0 },
    };
    const res = await fetch(`/api/togas/export-tiff/${tgCbtis}/${encodeURIComponent(tgFilename)}`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    if (!res.ok) { showToast('Error TIFF: ' + res.status, 'error'); return; }
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = tgFilename.replace(/\.[^.]+$/, '') + '_togas.tif';
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(a.href);
    showToast('TIFF con togas descargado', 'ok');
  } catch (e) { showToast('Error: ' + e.message, 'error'); }
}

// ─── SOMBRAS (shadows) ──────────────────────────────────────────────────────
let shMode = false, shCbtis = '', shFilename = '';
let shCutImg = null;
let shBorlaState = null, shBorlaImg = null;  // borlas for compositing
let shTogaState = null, shTogaImgs = {};     // togas for compositing
let shTogaGroupTf = {x:0,y:0,rotation:0,scaleY:1.0};
let shImgTf = {x:0,y:0,rotation:0};  // cutout+borlas image transform from togas section
let shTogaRefAR = 1.5;
let shPolygons = [];  // [{id, points:[{x,y}], feather:{top,bottom,left,right}, opacity, layer:'below'|'above'}]
let shFeatherMax = 500;  // user-configurable feather max
let shSelected = -1;
let shTool = 'polygon';  // 'polygon'|'move'|'edit'
let shNewPoints = [];     // vertices being placed for new polygon
let shDrawing = false;    // true while placing polygon vertices
let shDraggingNode = -1;  // index of node being dragged
let shDraggingPoly = null; // {idx, offX, offY}
let shMousePos = null;
let shCanvas = null, shCtx = null;
let shImgW = 1, shImgH = 1;
let shUndoStack = [], shRedoStack = [];
// Feather defaults — persistent via localStorage
let shFeatherDefault = 10;
let shFeatherChained = true;
let shHardnessDefault = 30;  // 0=very soft, 100=sharp edges

try {
  const _shfd = localStorage.getItem('sh_featherDefault');
  const _shfc = localStorage.getItem('sh_featherChained');
  const _shfm = localStorage.getItem('sh_featherMax');
  const _shhd = localStorage.getItem('sh_hardnessDefault');
  if (_shfd !== null) shFeatherDefault = +_shfd;
  if (_shfc !== null) shFeatherChained = _shfc === 'true';
  if (_shfm !== null) shFeatherMax = +_shfm;
  if (_shhd !== null) shHardnessDefault = +_shhd;
} catch(e) {}

async function enterSombrasMode(cbtis, filename) {
  shCbtis = cbtis; shFilename = filename;
  shPolygons = []; shSelected = -1; shNewPoints = [];
  shDrawing = false; shDraggingNode = -1; shDraggingPoly = null;
  shUndoStack = []; shRedoStack = [];

  const panel = document.getElementById('previewPanel');
  panel.dataset.cbtis = cbtis;
  panel.dataset.cutout = filename;
  const _gd = groups.find(g => g.cbtis === cbtis && g.cutout === filename);
  if (_gd) { panel.dataset.group = _gd.group; panel.dataset.pano = _gd.output || ''; }
  panel.innerHTML = '<div class="rf-loading">Cargando modo Sombras...</div>';

  const encName = encodeURIComponent(filename);
  try {
    shCutImg = await loadImg(`/api/cutout/preview-scaled/${cbtis}/${encName}?maxw=99999&_=${Date.now()}`);
  } catch(e) {
    showToast('Error cargando imagen: ' + e.message, 'error');
    return;
  }
  shImgW = shCutImg.width;
  shImgH = shCutImg.height;

  // Load borlas state for compositing
  shBorlaState = null; shBorlaImg = null;
  try {
    const bsRes = await fetch(`/api/borlas/state/${cbtis}/${encName}`);
    const bs = await bsRes.json();
    if (bs && bs.borlas && bs.borlas.length > 0) {
      shBorlaState = bs;
      const color = bs.selectedColor || bs.borlas[0].color || 'DORADO';
      shBorlaImg = await loadImg(`/api/borlas/image/${color}?maxw=800`);
      // Restore hilo masks from data URLs
      for (const b of shBorlaState.borlas) {
        b._shHiloMask = null;
        if (b.hiloMask) {
          try {
            const img = await loadImg(b.hiloMask);
            const c = document.createElement('canvas');
            c.width = img.width; c.height = img.height;
            c.getContext('2d').drawImage(img, 0, 0);
            b._shHiloMask = c;
          } catch {}
        }
      }
    }
  } catch {}

  // Load togas state for compositing
  shTogaState = null; shTogaImgs = {};
  shTogaGroupTf = {x:0,y:0,rotation:0,scaleY:1.0};
  shImgTf = {x:0,y:0,rotation:0};
  try {
    const tsRes = await fetch(`/api/togas/state/${cbtis}/${encName}`);
    const ts = await tsRes.json();
    if (ts && ts.togas && ts.togas.length > 0) {
      shTogaState = ts;
      shTogaGroupTf = ts.groupTf || {x:0,y:0,rotation:0,scaleY:1.0};
      shImgTf = ts.imgTf || {x:0,y:0,rotation:0};
      // Load toga variant images
      const variants = [...new Set(ts.togas.map(t => t.variant))];
      await Promise.all(variants.map(async v => {
        try { shTogaImgs[v] = await loadImg(`/api/togas/image/${v}?maxw=600`); } catch {}
      }));
      // Compute average AR
      const ars = Object.values(shTogaImgs).map(img => img.height / img.width).filter(v => v > 0);
      shTogaRefAR = ars.length > 0 ? ars.reduce((a,b) => a+b, 0) / ars.length : 1.5;
      // Restore toga brush masks from data URLs
      for (const t of shTogaState.togas) {
        t._shMask = null;
        if (t.mask) {
          try {
            const img = await loadImg(t.mask);
            const c = document.createElement('canvas');
            c.width = img.width; c.height = img.height;
            c.getContext('2d').drawImage(img, 0, 0);
            t._shMask = c;
          } catch {}
        }
      }
    }
  } catch {}

  const wf = _gd && _gd.workflow ? _gd.workflow : {};
  const doneClass = wf.sombras_done ? 'is-done' : '';
  const doneText = wf.sombras_done ? '\u2714 Listo' : '\u2610 Listo';

  panel.innerHTML = `
    <div class="borlas-toolbar sombras-toolbar" style="display:flex;flex-wrap:wrap;gap:4px;padding:4px 8px;align-items:center;font-size:12px">
      <button class="rf-btn active" id="shToolPoly" onclick="shSetTool('polygon')" title="Dibujar polígono de sombra (P)">&#11039; Polígono</button>
      <button class="rf-btn" id="shToolMove" onclick="shSetTool('move')" title="Mover sombra (V)">&#9995; Mover</button>
      <button class="rf-btn" id="shToolEdit" onclick="shSetTool('edit')" title="Editar nodos (N)">&#9670; Nodos</button>
      <div class="bl-sep"></div>
      <button class="rf-btn" id="shChainBtn" onclick="shToggleChain()" title="Vincular/desvincular feather" style="opacity:${shFeatherChained ? 1 : 0.4}">&#128279;</button>
      <label>T:</label><input type="range" class="rf-size-slider" min="0" max="${shFeatherMax}" value="${shFeatherDefault}" style="width:40px" id="shFTop"
             oninput="shSetFeather('top',+this.value)"><span id="shFTopVal" class="rf-size-label">${shFeatherDefault}</span>
      <label>B:</label><input type="range" class="rf-size-slider" min="0" max="${shFeatherMax}" value="${shFeatherDefault}" style="width:40px" id="shFBot"
             oninput="shSetFeather('bottom',+this.value)"><span id="shFBotVal" class="rf-size-label">${shFeatherDefault}</span>
      <label>L:</label><input type="range" class="rf-size-slider" min="0" max="${shFeatherMax}" value="${shFeatherDefault}" style="width:40px" id="shFLeft"
             oninput="shSetFeather('left',+this.value)"><span id="shFLeftVal" class="rf-size-label">${shFeatherDefault}</span>
      <label>R:</label><input type="range" class="rf-size-slider" min="0" max="${shFeatherMax}" value="${shFeatherDefault}" style="width:40px" id="shFRight"
             oninput="shSetFeather('right',+this.value)"><span id="shFRightVal" class="rf-size-label">${shFeatherDefault}</span>
      <label title="Límite máximo de feather">Max:</label><input type="number" min="50" max="5000" value="${shFeatherMax}" style="width:44px;background:var(--bg2);color:var(--fg1);border:1px solid var(--bg3);border-radius:3px;font-size:11px" id="shFMax"
             onchange="shSetFeatherMax(+this.value)">
      <div class="bl-sep"></div>
      <label title="Dureza del borde (0=muy suave, 100=filoso)">Dureza:</label>
      <input type="range" class="rf-size-slider" min="0" max="100" value="${shHardnessDefault}" style="width:50px" id="shHardness"
             oninput="shSetHardness(+this.value)"><span id="shHardnessVal" class="rf-size-label">${shHardnessDefault}</span>
      <div class="bl-sep"></div>
      <label>Opacidad:</label>
      <input type="range" class="rf-size-slider" min="5" max="100" value="50" style="width:60px" id="shOpacity"
             oninput="shSetOpacity(+this.value)"><span id="shOpacityVal" class="rf-size-label">50</span>%
      <div class="bl-sep"></div>
      <label>Capa:</label>
      <select id="shLayerSel" onchange="shSetLayer(this.value)">
        <option value="below">Debajo ALUMNOS</option>
        <option value="above">Arriba ALUMNOS</option>
      </select>
      <div class="bl-sep"></div>
      <button class="rf-btn" onclick="shUndo()" title="Deshacer (Ctrl+Z)">&#8630;</button>
      <button class="rf-btn" onclick="shRedo()" title="Rehacer (Ctrl+Y)">&#8631;</button>
      <div class="bl-sep"></div>
      <button class="rf-btn" onclick="shExportTiff()">&#128190; TIFF Final</button>
      <button class="rf-btn ${doneClass}" id="shDoneBtn" onclick="toggleWorkflowDone('sombras',this)">${doneText}</button>
      <button class="rf-btn" onclick="exitSombrasMode()">&#10005; Salir</button>
      <div class="bl-sep"></div>
      <button class="rf-btn" onclick="shZoomFit()" title="Ajustar zoom">Ajustar</button>
      <span id="shZoomLabel" class="rf-size-label" style="min-width:36px;text-align:center">100%</span>
      <button class="rf-btn" id="shFullscreenBtn" onclick="shToggleFullscreen()">&#9974;</button>
    </div>
    <div style="display:flex;flex:1;overflow:hidden">
      <div id="shList" style="width:120px;overflow-y:auto;border-right:1px solid var(--bg3);padding:4px;font-size:11px"></div>
      <div id="shWrap" style="flex:1;position:relative;overflow:hidden;background:repeating-conic-gradient(var(--bg3) 0% 25%, var(--bg2) 0% 50%) 0 0/20px 20px">
        <canvas id="shCanvas" style="display:block;width:100%;height:100%"></canvas>
      </div>
    </div>
    <div style="padding:2px 8px;font-size:11px;color:var(--fg3);border-top:1px solid var(--bg3)">
      ${filename} (sombras) — Click: agregar vértice | Doble-click/Enter: cerrar polígono | Del: eliminar sombra | Right-click nodo: eliminar nodo
    </div>`;

  shMode = true;
  shCanvas = document.getElementById('shCanvas');
  shCtx = shCanvas.getContext('2d');

  setTimeout(() => {
    const wrap = document.getElementById('shWrap');
    if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    shCanvas.width = Math.max(rect.width, 200);
    shCanvas.height = Math.max(rect.height, 200);
    pvZoom = 1; pvX = 0; pvY = 0;
    shZoomFit();
    shBindEvents(wrap);
    shLoadState().then(loaded => {
      if (loaded && shPolygons.length > 0) {
        showToast(shPolygons.length + ' sombras restauradas', 'ok');
      }
      shRenderList();
      shDraw();
      // Auto-capture snapshot for Fixes section (after all images are loaded)
      shCaptureSnapshot();
    });
  }, 50);
}

function exitSombrasMode() {
  if (shDrawing) shFinishPolygon();
  shSaveState();
  shMode = false;
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  const panel = document.getElementById('previewPanel');
  if (panel) panel.classList.remove('faux-fullscreen');
  if (shCbtis && shFilename) {
    initPreviewPanel(
      `/api/cutout/preview/${shCbtis}/${encodeURIComponent(shFilename)}?t=${Date.now()}`,
      shFilename + ' (recortada)', true
    );
  }
  refreshGroups();
}

function shSaveState() {
  if (!shCbtis || !shFilename) return;
  const state = {
    polygons: shPolygons.map(p => ({
      points: p.points, feather: p.feather,
      opacity: p.opacity, layer: p.layer,
      hardness: p.hardness !== undefined ? p.hardness : shHardnessDefault
    }))
  };
  fetch(`/api/sombras/state/${shCbtis}/${encodeURIComponent(shFilename)}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(state)
  }).catch(() => {});
  // Also capture a snapshot for Fixes section
  shCaptureSnapshot();
}

async function shCaptureSnapshot() {
  if (!shCutImg || !shCbtis || !shFilename) return;
  // Compute canvas expansion to include all shadows (with feather padding)
  let minPx = 0, minPy = 0, maxPx = shImgW, maxPy = shImgH;
  for (const p of shPolygons) {
    if (p.points.length < 3) continue;
    const feather = p.feather || {};
    const mf = Math.max(feather.top || 0, feather.bottom || 0, feather.left || 0, feather.right || 0);
    const pad = mf * 2 + 20;
    for (const pt of p.points) {
      if (pt.x - pad < minPx) minPx = pt.x - pad;
      if (pt.y - pad < minPy) minPy = pt.y - pad;
      if (pt.x + pad > maxPx) maxPx = pt.x + pad;
      if (pt.y + pad > maxPy) maxPy = pt.y + pad;
    }
  }
  const offX = Math.floor(Math.min(0, minPx));
  const offY = Math.floor(Math.min(0, minPy));
  const snapW = Math.ceil(Math.max(shImgW, maxPx) - offX);
  const snapH = Math.ceil(Math.max(shImgH, maxPy) - offY);

  // Render at scale=1 on offscreen canvas (no checkerboard, no UI)
  // We want: ox = -offX, oy = -offY  (image top-left at canvas offset)
  // shDraw computes: ox = (cw - shImgW) / 2 + pvX
  // So: pvX = -offX - (snapW - shImgW) / 2
  const origCanvas = shCanvas, origCtx = shCtx;
  const savedZoom = pvZoom, savedX = pvX, savedY = pvY;
  const oc = document.createElement('canvas');
  oc.width = snapW; oc.height = snapH;
  shCanvas = oc; shCtx = oc.getContext('2d');
  pvZoom = 1;
  pvX = -offX - (snapW - shImgW) / 2;
  pvY = -offY - (snapH - shImgH) / 2;
  _shSnapshotMode = true;
  shDraw();
  _shSnapshotMode = false;
  // Restore original state
  shCanvas = origCanvas; shCtx = origCtx;
  pvZoom = savedZoom; pvX = savedX; pvY = savedY;
  // Upload snapshot
  try {
    const blob = await new Promise(r => oc.toBlob(r, 'image/webp', 0.92));
    await fetch(`/api/sombras/snapshot/${shCbtis}/${encodeURIComponent(shFilename)}`, {
      method: 'POST', body: blob
    });
  } catch(e) { console.warn('Snapshot save failed:', e); }
}

async function shLoadState() {
  try {
    const res = await fetch(`/api/sombras/state/${shCbtis}/${encodeURIComponent(shFilename)}`);
    const state = await res.json();
    if (!state || !state.polygons) return false;
    shPolygons = state.polygons.map((p, i) => ({
      id: i, points: p.points, feather: p.feather || {top:10,bottom:10,left:10,right:10},
      opacity: p.opacity ?? 0.5, layer: p.layer || 'below',
      hardness: p.hardness !== undefined ? p.hardness : shHardnessDefault
    }));
    return true;
  } catch { return false; }
}

function shSetTool(tool) {
  if (shDrawing && tool !== 'polygon') shFinishPolygon();
  shTool = tool;
  document.getElementById('shToolPoly').classList.toggle('active', tool === 'polygon');
  document.getElementById('shToolMove').classList.toggle('active', tool === 'move');
  document.getElementById('shToolEdit').classList.toggle('active', tool === 'edit');
  const wrap = document.getElementById('shWrap');
  if (wrap) wrap.style.cursor = tool === 'polygon' ? 'crosshair' : tool === 'edit' ? 'crosshair' : 'default';
  shDraw();
}

function shToggleChain() {
  shFeatherChained = !shFeatherChained;
  try { localStorage.setItem('sh_featherChained', shFeatherChained); } catch(e) {}
  const btn = document.getElementById('shChainBtn');
  if (btn) btn.style.opacity = shFeatherChained ? '1' : '0.4';
  showToast(shFeatherChained ? 'Feather vinculado' : 'Feather independiente', 'info');
}

function shSetFeather(dir, val) {
  if (shFeatherChained) {
    document.getElementById('shFTop').value = val;
    document.getElementById('shFBot').value = val;
    document.getElementById('shFLeft').value = val;
    document.getElementById('shFRight').value = val;
    document.getElementById('shFTopVal').textContent = val;
    document.getElementById('shFBotVal').textContent = val;
    document.getElementById('shFLeftVal').textContent = val;
    document.getElementById('shFRightVal').textContent = val;
    shFeatherDefault = val;
    try { localStorage.setItem('sh_featherDefault', val); } catch(e) {}
    if (shSelected >= 0) {
      shPolygons[shSelected].feather = {top:val, bottom:val, left:val, right:val};
    }
  } else {
    document.getElementById('shF' + {top:'Top',bottom:'Bot',left:'Left',right:'Right'}[dir]).value = val;
    document.getElementById('shF' + {top:'Top',bottom:'Bot',left:'Left',right:'Right'}[dir] + 'Val').textContent = val;
    if (shSelected >= 0) {
      shPolygons[shSelected].feather[dir] = val;
    }
  }
  shDraw();
}

function shSetFeatherMax(val) {
  shFeatherMax = Math.max(50, Math.min(5000, val));
  try { localStorage.setItem('sh_featherMax', shFeatherMax); } catch(e) {}
  // Update all slider max attributes
  for (const id of ['shFTop', 'shFBot', 'shFLeft', 'shFRight']) {
    const el = document.getElementById(id);
    if (el) el.max = shFeatherMax;
  }
  showToast('Límite feather: ' + shFeatherMax + 'px', 'ok');
}

function shSetOpacity(val) {
  document.getElementById('shOpacityVal').textContent = val;
  if (shSelected >= 0) {
    shPolygons[shSelected].opacity = val / 100;
    shDraw();
  }
}

function shSetHardness(val) {
  document.getElementById('shHardnessVal').textContent = val;
  shHardnessDefault = val;
  try { localStorage.setItem('sh_hardnessDefault', val); } catch(e) {}
  if (shSelected >= 0) {
    shPolygons[shSelected].hardness = val;
    shDraw();
  }
}

function shSetLayer(val) {
  if (shSelected >= 0) {
    shPolygons[shSelected].layer = val;
    shDraw();
  }
}

function shSelectPoly(idx) {
  shSelected = shSelected === idx ? -1 : idx;
  if (shSelected >= 0) {
    const p = shPolygons[shSelected];
    document.getElementById('shFTop').value = p.feather.top;
    document.getElementById('shFBot').value = p.feather.bottom;
    document.getElementById('shFLeft').value = p.feather.left;
    document.getElementById('shFRight').value = p.feather.right;
    document.getElementById('shFTopVal').textContent = p.feather.top;
    document.getElementById('shFBotVal').textContent = p.feather.bottom;
    document.getElementById('shFLeftVal').textContent = p.feather.left;
    document.getElementById('shFRightVal').textContent = p.feather.right;
    document.getElementById('shOpacity').value = Math.round(p.opacity * 100);
    document.getElementById('shOpacityVal').textContent = Math.round(p.opacity * 100);
    document.getElementById('shLayerSel').value = p.layer;
    const hv = p.hardness !== undefined ? p.hardness : shHardnessDefault;
    document.getElementById('shHardness').value = hv;
    document.getElementById('shHardnessVal').textContent = hv;
  }
  shRenderList();
  shDraw();
}

function shRenderList() {
  const list = document.getElementById('shList');
  if (!list) return;
  list.innerHTML = shPolygons.map((p, i) => {
    const sel = i === shSelected ? 'selected' : '';
    const layerIcon = p.layer === 'below' ? '\u2b07 Debajo' : '\u2b06 Arriba';
    return `<div class="bl-item ${sel}" onclick="shSelectPoly(${i})" style="display:flex;justify-content:space-between;align-items:center;padding:2px 4px">
      <span>${layerIcon}</span>
      <span style="flex:1;margin:0 3px">#${i+1} <span style="color:var(--fg3)">${Math.round(p.opacity*100)}%</span></span>
      <button onclick="event.stopPropagation();shDeletePoly(${i})" title="Eliminar" style="background:none;border:none;color:var(--red);cursor:pointer;padding:0">&times;</button>
    </div>`;
  }).join('');
}

// --- Undo/Redo ---
function shSaveSnapshot() {
  shUndoStack.push(JSON.parse(JSON.stringify(shPolygons)));
  shRedoStack = [];
  if (shUndoStack.length > 40) shUndoStack.shift();
}
function shUndo() {
  if (shUndoStack.length === 0) return;
  shRedoStack.push(JSON.parse(JSON.stringify(shPolygons)));
  shPolygons = shUndoStack.pop();
  if (shSelected >= shPolygons.length) shSelected = -1;
  shRenderList(); shDraw();
}
function shRedo() {
  if (shRedoStack.length === 0) return;
  shUndoStack.push(JSON.parse(JSON.stringify(shPolygons)));
  shPolygons = shRedoStack.pop();
  shRenderList(); shDraw();
}

// --- Polygon drawing ---
function shAddVertex(imgX, imgY) {
  if (!shDrawing) {
    shNewPoints = [{x: imgX, y: imgY}];
    shDrawing = true;
  } else {
    // Check if clicking near first point to close
    const first = shNewPoints[0];
    const dist = Math.hypot((imgX - first.x) * pvZoom, (imgY - first.y) * pvZoom);
    if (shNewPoints.length >= 3 && dist < 10) {
      shFinishPolygon();
      return;
    }
    shNewPoints.push({x: imgX, y: imgY});
  }
  shDraw();
}

function shFinishPolygon() {
  if (!shDrawing) return;
  shDrawing = false;
  if (shNewPoints.length >= 3) {
    shSaveSnapshot();
    const fv = shFeatherDefault;
    const f = shFeatherChained
      ? {top:fv, bottom:fv, left:fv, right:fv}
      : {
          top: +document.getElementById('shFTop').value,
          bottom: +document.getElementById('shFBot').value,
          left: +document.getElementById('shFLeft').value,
          right: +document.getElementById('shFRight').value
        };
    const opVal = +document.getElementById('shOpacity').value / 100;
    const layerVal = document.getElementById('shLayerSel').value;
    const hardVal = +document.getElementById('shHardness').value;
    shPolygons.push({
      id: shPolygons.length,
      points: shNewPoints.slice(),
      feather: f,
      opacity: opVal,
      layer: layerVal,
      hardness: hardVal
    });
    shSelected = shPolygons.length - 1;
    shRenderList();
    shSaveState();
    showToast('Sombra #' + shPolygons.length + ' creada', 'ok');
  }
  shNewPoints = [];
  shDraw();
}

function shDeletePoly(idx) {
  if (idx < 0 || idx >= shPolygons.length) return;
  shSaveSnapshot();
  shPolygons.splice(idx, 1);
  // Re-assign IDs
  shPolygons.forEach((p, i) => p.id = i);
  if (shSelected >= shPolygons.length) shSelected = shPolygons.length - 1;
  shRenderList(); shDraw(); shSaveState();
  showToast('Sombra eliminada', 'ok');
}

// --- Hit testing ---
function shPointInPoly(px, py, points) {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const xi = points[i].x, yi = points[i].y;
    const xj = points[j].x, yj = points[j].y;
    if (((yi > py) !== (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

function shHitTest(imgX, imgY) {
  for (let i = shPolygons.length - 1; i >= 0; i--) {
    if (shPointInPoly(imgX, imgY, shPolygons[i].points)) return i;
  }
  return -1;
}

function shHitNode(imgX, imgY) {
  if (shSelected < 0) return -1;
  const pts = shPolygons[shSelected].points;
  const r = 8 / pvZoom;
  for (let i = 0; i < pts.length; i++) {
    if (Math.hypot(imgX - pts[i].x, imgY - pts[i].y) <= r) return i;
  }
  return -1;
}

function shHitEdge(imgX, imgY) {
  // Find closest edge to insert a node
  if (shSelected < 0) return -1;
  const pts = shPolygons[shSelected].points;
  const threshold = 6 / pvZoom;
  let bestDist = Infinity, bestIdx = -1;
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length;
    const d = shDistToSegment(imgX, imgY, pts[i], pts[j]);
    if (d < threshold && d < bestDist) {
      bestDist = d; bestIdx = i;
    }
  }
  return bestIdx;
}

function shDistToSegment(px, py, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const len2 = dx*dx + dy*dy;
  if (len2 === 0) return Math.hypot(px - a.x, py - a.y);
  let t = ((px - a.x)*dx + (py - a.y)*dy) / len2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (a.x + t*dx), py - (a.y + t*dy));
}

// --- Drawing ---
function shImgToScreen(ix, iy) {
  const cw = shCanvas.width, ch = shCanvas.height;
  const ox = (cw - shImgW * pvZoom) / 2 + pvX;
  const oy = (ch - shImgH * pvZoom) / 2 + pvY;
  return {x: ox + ix * pvZoom, y: oy + iy * pvZoom};
}
function shScreenToImg(sx, sy) {
  const cw = shCanvas.width, ch = shCanvas.height;
  const ox = (cw - shImgW * pvZoom) / 2 + pvX;
  const oy = (ch - shImgH * pvZoom) / 2 + pvY;
  return {x: (sx - ox) / pvZoom, y: (sy - oy) / pvZoom};
}

let _shSnapshotMode = false;

function shDraw() {
  if (!shCanvas || !shCtx || !shCutImg) return;
  const ctx = shCtx;
  const cw = shCanvas.width, ch = shCanvas.height;
  ctx.clearRect(0, 0, cw, ch);
  const scale = pvZoom;
  const ox = (cw - shImgW * scale) / 2 + pvX;
  const oy = (ch - shImgH * scale) / 2 + pvY;

  // Draw shadows BELOW alumnos
  for (let i = 0; i < shPolygons.length; i++) {
    const p = shPolygons[i];
    if (p.layer !== 'below' || p.points.length < 3) continue;
    shDrawPolygon(ctx, p, i, ox, oy, scale);
  }

  // --- ALUMNOS group: cutout + borlas (with imgTf from togas section) ---
  ctx.save();
  if (shImgTf && (Math.abs(shImgTf.rotation || 0) > 0.01 || Math.abs(shImgTf.x || 0) > 0.1 || Math.abs(shImgTf.y || 0) > 0.1)) {
    const imgRad = (shImgTf.rotation || 0) * Math.PI / 180;
    const imgCx = ox + shImgW * scale / 2;
    const imgCy = oy + shImgH * scale / 2;
    ctx.translate(imgCx, imgCy);
    ctx.rotate(imgRad);
    ctx.translate(-imgCx, -imgCy);
    ctx.translate((shImgTf.x || 0) * scale, (shImgTf.y || 0) * scale);
  }
  ctx.drawImage(shCutImg, ox, oy, shImgW * scale, shImgH * scale);

  // Borlas (composited from saved state)
  if (shBorlaState && shBorlaImg) {
    const hiloColor = shBorlaState.hiloColor || '#d4a017';
    const hiloSize = shBorlaState.hiloSize || 3;
    const hiloFeather = shBorlaState.hiloFeather || 1;
    const hiloFlow = shBorlaState.hiloFlow || 100;
    for (const b of shBorlaState.borlas) {
      if (!b.visible) continue;
      const bw = b.scale * scale;
      const bh = bw * (shBorlaImg.height / shBorlaImg.width);
      const bsx = ox + b.x * scale;
      const bsy = oy + b.y * scale;
      ctx.save();
      ctx.translate(bsx, bsy);
      ctx.rotate(b.rotation * Math.PI / 180);
      ctx.drawImage(shBorlaImg, -bw / 2, 0, bw, bh);
      // Draw hilo (thread) for this borla
      if (b.hilo && b.hilo.length > 1) {
        const hs = scale;
        ctx.save();
        ctx.globalAlpha = hiloFlow / 100;
        ctx.strokeStyle = hiloColor;
        ctx.lineWidth = Math.max(1, hiloSize * hs);
        ctx.lineCap = 'round'; ctx.lineJoin = 'round';
        if (hiloFeather > 0.1) { ctx.shadowColor = hiloColor; ctx.shadowBlur = hiloFeather * hs; }
        if (b._shHiloMask) {
          // Render through hilo mask
          const pad = hiloSize * 2 + hiloFeather * 2 + 4;
          let hMinX = Infinity, hMinY = Infinity, hMaxX = -Infinity, hMaxY = -Infinity;
          for (const p of b.hilo) { if (p.x < hMinX) hMinX = p.x; if (p.y < hMinY) hMinY = p.y; if (p.x > hMaxX) hMaxX = p.x; if (p.y > hMaxY) hMaxY = p.y; }
          hMinX -= pad; hMinY -= pad; hMaxX += pad; hMaxY += pad;
          const tw = Math.max(1, Math.ceil((hMaxX - hMinX) * hs));
          const th = Math.max(1, Math.ceil((hMaxY - hMinY) * hs));
          if (tw > 0 && th > 0 && tw < 4000 && th < 4000) {
            const tmp = document.createElement('canvas');
            tmp.width = tw; tmp.height = th;
            const tc = tmp.getContext('2d');
            tc.strokeStyle = hiloColor;
            tc.lineWidth = Math.max(1, hiloSize * hs);
            tc.lineCap = 'round'; tc.lineJoin = 'round';
            tc.globalAlpha = hiloFlow / 100;
            if (hiloFeather > 0.1) { tc.shadowColor = hiloColor; tc.shadowBlur = hiloFeather * hs; }
            tc.beginPath();
            tc.moveTo((b.hilo[0].x - hMinX) * hs, (b.hilo[0].y - hMinY) * hs);
            for (let pi = 1; pi < b.hilo.length; pi++) {
              tc.lineTo((b.hilo[pi].x - hMinX) * hs, (b.hilo[pi].y - hMinY) * hs);
            }
            tc.stroke();
            tc.globalCompositeOperation = 'destination-in';
            tc.drawImage(b._shHiloMask, 0, 0, tw, th);
            ctx.globalAlpha = 1; ctx.shadowBlur = 0;
            ctx.drawImage(tmp, hMinX * hs, hMinY * hs);
          }
        } else {
          ctx.beginPath();
          ctx.moveTo(b.hilo[0].x * hs, b.hilo[0].y * hs);
          for (let pi = 1; pi < b.hilo.length; pi++) {
            ctx.lineTo(b.hilo[pi].x * hs, b.hilo[pi].y * hs);
          }
          ctx.stroke();
        }
        ctx.restore();
      }
      ctx.restore();
    }
  }

  ctx.restore(); // end ALUMNOS imgTf group

  // Togas (composited from saved state)
  if (shTogaState && shTogaState.togas) {
    ctx.save();
    const tgOxS = ox + shTogaGroupTf.x * scale;
    const tgOyS = oy + shTogaGroupTf.y * scale;
    const tgRad = shTogaGroupTf.rotation * Math.PI / 180;
    if (Math.abs(tgRad) > 0.001) {
      const gcx = ox + shImgW * scale / 2 + shTogaGroupTf.x * scale;
      const gcy = oy + shImgH * scale / 2 + shTogaGroupTf.y * scale;
      ctx.translate(gcx, gcy); ctx.rotate(tgRad); ctx.translate(-gcx, -gcy);
    }
    for (const t of shTogaState.togas) {
      const togaImg = shTogaImgs[t.variant];
      if (!togaImg) continue;
      const tw = t.scale * (t.scaleX || 1) * scale;
      const th = t.scale * shTogaRefAR * (shTogaGroupTf.scaleY || 1) * scale;
      const tx = tgOxS + t.x * scale;
      const ty = tgOyS + t.y * scale;
      ctx.save();
      ctx.translate(tx, ty);
      if (t.rotation) ctx.rotate(t.rotation * Math.PI / 180);
      if (t.flipH) ctx.scale(-1, 1);
      if (t._shMask) {
        // Composite through brush mask
        const mtw = Math.ceil(Math.abs(tw)), mth = Math.ceil(Math.abs(th));
        if (mtw > 0 && mth > 0) {
          const tmp = document.createElement('canvas');
          tmp.width = mtw; tmp.height = mth;
          const tc = tmp.getContext('2d');
          tc.drawImage(togaImg, 0, 0, mtw, mth);
          tc.globalCompositeOperation = 'destination-in';
          tc.drawImage(t._shMask, 0, 0, mtw, mth);
          ctx.drawImage(tmp, -tw / 2, 0, tw, th);
        }
      } else {
        ctx.drawImage(togaImg, -tw / 2, 0, tw, th);
      }
      ctx.restore();
    }
    ctx.restore();
  }
  // --- END ALUMNOS group ---

  // Draw shadows ABOVE alumnos
  for (let i = 0; i < shPolygons.length; i++) {
    const p = shPolygons[i];
    if (p.layer !== 'above' || p.points.length < 3) continue;
    shDrawPolygon(ctx, p, i, ox, oy, scale);
  }

  // Draw in-progress polygon (skip in snapshot mode)
  if (!_shSnapshotMode && shDrawing && shNewPoints.length > 0) {
    ctx.save();
    ctx.strokeStyle = '#7aa2f7';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.beginPath();
    const f = shNewPoints[0];
    ctx.moveTo(ox + f.x * scale, oy + f.y * scale);
    for (let i = 1; i < shNewPoints.length; i++) {
      ctx.lineTo(ox + shNewPoints[i].x * scale, oy + shNewPoints[i].y * scale);
    }
    // Rubber band to mouse
    if (shMousePos) {
      const mp = shScreenToImg(shMousePos.sx, shMousePos.sy);
      ctx.lineTo(ox + mp.x * scale, oy + mp.y * scale);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    // Vertices
    for (let i = 0; i < shNewPoints.length; i++) {
      const sx = ox + shNewPoints[i].x * scale;
      const sy = oy + shNewPoints[i].y * scale;
      ctx.beginPath();
      ctx.arc(sx, sy, i === 0 ? 6 : 4, 0, Math.PI * 2);
      ctx.fillStyle = i === 0 ? '#e0af68' : '#7aa2f7';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.restore();
  }

  // Selection highlight + nodes (skip in snapshot mode)
  if (!_shSnapshotMode && shSelected >= 0 && shSelected < shPolygons.length) {
    const p = shPolygons[shSelected];
    if (p.points.length >= 3) {
      ctx.save();
      ctx.strokeStyle = '#7aa2f7';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(ox + p.points[0].x * scale, oy + p.points[0].y * scale);
      for (let i = 1; i < p.points.length; i++) {
        ctx.lineTo(ox + p.points[i].x * scale, oy + p.points[i].y * scale);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
      // Draw nodes when in edit mode
      if (shTool === 'edit') {
        for (let i = 0; i < p.points.length; i++) {
          const sx = ox + p.points[i].x * scale;
          const sy = oy + p.points[i].y * scale;
          ctx.beginPath();
          ctx.arc(sx, sy, 5, 0, Math.PI * 2);
          ctx.fillStyle = '#7aa2f7';
          ctx.fill();
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
      ctx.restore();
    }
  }

  // Zoom label (skip in snapshot mode)
  if (!_shSnapshotMode) {
    const lbl = document.getElementById('shZoomLabel');
    if (lbl) lbl.textContent = Math.round(pvZoom * 100) + '%';
  }
}

// Helper: draw rounded polygon path with arcTo
function _shRoundedPath(c, arr, r) {
  c.beginPath();
  const n = arr.length;
  if (r < 0.5 || n < 3) {
    c.moveTo(arr[0].sx, arr[0].sy);
    for (let i = 1; i < n; i++) c.lineTo(arr[i].sx, arr[i].sy);
  } else {
    let minEdge = Infinity;
    for (let i = 0; i < n; i++) {
      const nx = arr[(i+1)%n];
      const d = Math.hypot(nx.sx - arr[i].sx, nx.sy - arr[i].sy);
      if (d < minEdge) minEdge = d;
    }
    const cr = Math.min(r, minEdge * 0.4);
    const last = arr[n-1];
    c.moveTo((last.sx + arr[0].sx)/2, (last.sy + arr[0].sy)/2);
    for (let i = 0; i < n; i++) {
      const nx = arr[(i+1)%n];
      c.arcTo(arr[i].sx, arr[i].sy, nx.sx, nx.sy, cr);
    }
  }
  c.closePath();
}

// Build or reuse cached blurred polygon canvas.
// Cache key: scale + hardness + feather + point coords.
// Cache is stored on the polygon object itself (p._shCache, p._shCacheKey).
// The cached canvas is in image-relative coordinates so it survives panning.
function _shBuildPolyCache(p, scale) {
  const pts = p.points;
  const ft = p.feather.top, fb = p.feather.bottom, fl = p.feather.left, fr = p.feather.right;
  const maxF = Math.max(ft, fb, fl, fr);
  const hardness = (p.hardness !== undefined ? p.hardness : 30) / 100;
  const cornerR = maxF > 0 ? maxF * scale * Math.max(0.15, (1 - hardness) * 0.6) : 0;

  // Cache key (does NOT include pan position)
  let ck = scale.toFixed(5) + '|' + (p.hardness||30) + '|' + ft+','+fb+','+fl+','+fr;
  for (const pt of pts) ck += '|' + pt.x.toFixed(1) + ',' + pt.y.toFixed(1);

  if (p._shCache && p._shCacheKey === ck) return; // already cached

  if (maxF < 0.5) { p._shCache = null; p._shCacheKey = ck; return; }

  // Image-space bounds
  let imgMinX=Infinity, imgMinY=Infinity, imgMaxX=-Infinity, imgMaxY=-Infinity;
  for (const pt of pts) {
    if (pt.x < imgMinX) imgMinX = pt.x; if (pt.y < imgMinY) imgMinY = pt.y;
    if (pt.x > imgMaxX) imgMaxX = pt.x; if (pt.y > imgMaxY) imgMaxY = pt.y;
  }
  const blurBoost = 1 + (1 - hardness) * 0.5;
  const hBlur = Math.max(fl, fr) * scale * blurBoost;
  const vBlur = Math.max(ft, fb) * scale * blurBoost;
  const padT = ft * scale * blurBoost * 3 + 10, padB = fb * scale * blurBoost * 3 + 10;
  const padL = fl * scale * blurBoost * 3 + 10, padR_px = fr * scale * blurBoost * 3 + 10;
  const tw = (imgMaxX - imgMinX) * scale + padL + padR_px;
  const th = (imgMaxY - imgMinY) * scale + padT + padB;
  if (tw <= 0 || th <= 0 || tw >= 16000 || th >= 16000) { p._shCache = null; p._shCacheKey = ck; return; }
  // Points relative to temp canvas origin
  const tPts = pts.map(pt => ({
    sx: (pt.x - imgMinX) * scale + padL,
    sy: (pt.y - imgMinY) * scale + padT
  }));
  // Pass 1: horizontal blur
  const tmp1 = document.createElement('canvas');
  tmp1.width = Math.ceil(tw); tmp1.height = Math.ceil(th);
  const tc1 = tmp1.getContext('2d');
  if (hBlur > 0.5) tc1.filter = 'blur(' + hBlur + 'px)';
  tc1.fillStyle = 'rgba(0,0,0,1)';
  _shRoundedPath(tc1, tPts, cornerR); tc1.fill();
  tc1.filter = 'none';
  // Pass 2: vertical blur
  const tmp2 = document.createElement('canvas');
  tmp2.width = Math.ceil(tw); tmp2.height = Math.ceil(th);
  const tc2 = tmp2.getContext('2d');
  if (vBlur > 0.5) tc2.filter = 'blur(' + vBlur + 'px)';
  tc2.drawImage(tmp1, 0, 0);
  tc2.filter = 'none';
  // Store cache
  p._shCache = tmp2;
  p._shCacheKey = ck;
  p._shCacheImgX = imgMinX;
  p._shCacheImgY = imgMinY;
  p._shCachePadL = padL;
  p._shCachePadT = padT;
}

function shDrawPolygon(ctx, p, idx, ox, oy, scale) {
  const pts = p.points;
  if (pts.length < 3) return;
  const maxF = Math.max(p.feather.top, p.feather.bottom, p.feather.left, p.feather.right);
  const hardness = (p.hardness !== undefined ? p.hardness : 30) / 100;

  if (maxF > 0.5) {
    // Build/reuse cached blurred canvas
    _shBuildPolyCache(p, scale);
    if (p._shCache) {
      ctx.save();
      ctx.globalAlpha = p.opacity;
      ctx.drawImage(p._shCache,
        ox + p._shCacheImgX * scale - p._shCachePadL,
        oy + p._shCacheImgY * scale - p._shCachePadT);
      ctx.restore();
      return;
    }
  }
  // No feather — draw directly with rounding
  const cornerR = maxF > 0 ? maxF * scale * Math.max(0.15, (1 - hardness) * 0.6) : 0;
  const sPts = pts.map(pt => ({sx: ox + pt.x * scale, sy: oy + pt.y * scale}));
  ctx.save();
  ctx.fillStyle = 'rgba(0,0,0,' + p.opacity + ')';
  _shRoundedPath(ctx, sPts, cornerR);
  ctx.fill();
  ctx.restore();
}

// --- Zoom/Pan ---
function shZoomFit() {
  if (!shCutImg || !shCanvas) return;
  pvZoom = Math.min(shCanvas.width / shImgW, shCanvas.height / shImgH) * 0.95;
  pvX = 0; pvY = 0;
  shDraw();
}
function shToggleFullscreen() {
  const panel = document.getElementById('previewPanel');
  if (!panel) return;
  _toggleRealFullscreen(panel, 'shFullscreenBtn', () => {
    const wrap = document.getElementById('shWrap');
    if (wrap && shCanvas && shMode) {
      const rect = wrap.getBoundingClientRect();
      shCanvas.width = rect.width; shCanvas.height = rect.height;
      shZoomFit();
    }
  });
}

// --- Events ---
function shBindEvents(wrap) {
  let panning = false, panStartX = 0, panStartY = 0, panOX = 0, panOY = 0;

  wrap.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = shCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const oldZ = pvZoom;
    pvZoom = Math.max(0.05, Math.min(pvZoom * (e.deltaY < 0 ? 1.1 : 0.9), 20));
    const cw = shCanvas.width, ch = shCanvas.height;
    pvX += (mx - cw/2 - pvX) * (1 - pvZoom / oldZ);
    pvY += (my - ch/2 - pvY) * (1 - pvZoom / oldZ);
    shDraw();
  }, {passive: false});

  wrap.addEventListener('mousedown', (e) => {
    const rect = shCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;

    // Middle click or space = pan
    if (e.button === 1 || spaceHeld) {
      panning = true; panStartX = e.clientX; panStartY = e.clientY;
      panOX = pvX; panOY = pvY;
      wrap.style.cursor = 'grabbing'; e.preventDefault(); return;
    }

    // Right-click on node in edit mode: remove node
    if (e.button === 2 && shTool === 'edit' && shSelected >= 0) {
      const img = shScreenToImg(mx, my);
      const ni = shHitNode(img.x, img.y);
      if (ni >= 0 && shPolygons[shSelected].points.length > 3) {
        shSaveSnapshot();
        shPolygons[shSelected].points.splice(ni, 1);
        shDraw(); shSaveState();
        showToast('Nodo eliminado', 'ok');
      }
      e.preventDefault(); return;
    }

    // Right-click finishes polygon
    if (e.button === 2 && shDrawing) {
      shFinishPolygon(); return;
    }

    if (e.button !== 0) return;
    const img = shScreenToImg(mx, my);

    // Polygon tool: add vertex
    if (shTool === 'polygon') {
      shAddVertex(img.x, img.y);
      return;
    }

    // Edit tool: drag node or add node on edge
    if (shTool === 'edit' && shSelected >= 0) {
      const ni = shHitNode(img.x, img.y);
      if (ni >= 0) {
        shSaveSnapshot();
        shDraggingNode = ni;
        wrap.style.cursor = 'grabbing';
        return;
      }
      const ei = shHitEdge(img.x, img.y);
      if (ei >= 0) {
        shSaveSnapshot();
        shPolygons[shSelected].points.splice(ei + 1, 0, {x: img.x, y: img.y});
        shDraggingNode = ei + 1;
        wrap.style.cursor = 'grabbing';
        shDraw();
        showToast('Nodo insertado', 'ok');
        return;
      }
    }

    // Move/select tool or general click: select polygon
    const hit = shHitTest(img.x, img.y);
    if (hit >= 0) {
      shSelectPoly(hit);
      if (shTool === 'move') {
        const p = shPolygons[hit];
        shDraggingPoly = {idx: hit, offX: img.x, offY: img.y};
        wrap.style.cursor = 'move';
      }
    } else {
      // If in edit tool and clicked outside, try selecting another poly
      if (shTool === 'edit') {
        shSelected = -1;
        shRenderList(); shDraw();
      } else if (shTool === 'move') {
        shSelected = -1;
        shRenderList(); shDraw();
        // Pan on empty space
        panning = true; panStartX = e.clientX; panStartY = e.clientY;
        panOX = pvX; panOY = pvY; wrap.style.cursor = 'grabbing';
      }
    }
  });

  wrap.addEventListener('mousemove', (e) => {
    const rect = shCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    shMousePos = {sx: mx, sy: my};

    if (panning) {
      pvX = panOX + (e.clientX - panStartX);
      pvY = panOY + (e.clientY - panStartY);
      shDraw(); return;
    }
    if (shDraggingNode >= 0 && shSelected >= 0) {
      const img = shScreenToImg(mx, my);
      shPolygons[shSelected].points[shDraggingNode] = {x: img.x, y: img.y};
      shDraw(); return;
    }
    if (shDraggingPoly) {
      const img = shScreenToImg(mx, my);
      const dx = img.x - shDraggingPoly.offX;
      const dy = img.y - shDraggingPoly.offY;
      const p = shPolygons[shDraggingPoly.idx];
      for (const pt of p.points) { pt.x += dx; pt.y += dy; }
      shDraggingPoly.offX = img.x; shDraggingPoly.offY = img.y;
      shDraw(); return;
    }
    // Redraw for rubber band or cursor
    if (shDrawing || shTool === 'edit') shDraw();
  });

  wrap.addEventListener('mouseup', () => {
    const wasDragging = !!shDraggingPoly;
    const wasDraggingNode = shDraggingNode >= 0;
    panning = false;
    shDraggingNode = -1;
    if (shDraggingPoly) { shDraggingPoly = null; }
    wrap.style.cursor = shTool === 'polygon' || shTool === 'edit' ? 'crosshair' : 'default';
    if (wasDragging || wasDraggingNode) { shSaveState(); shDraw(); }
  });

  wrap.addEventListener('dblclick', (e) => {
    if (shDrawing) { e.preventDefault(); shFinishPolygon(); }
  });

  wrap.addEventListener('mouseleave', () => {
    panning = false; shDraggingNode = -1; shDraggingPoly = null;
    shMousePos = null; shDraw();
  });

  wrap.addEventListener('contextmenu', (e) => e.preventDefault());
}

// --- TIFF Export ---
async function shExportTiff() {
  showToast('Generando TIFF final... espera', 'info');
  try {
    const payload = {
      sombras: shPolygons.map(p => ({
        points: p.points, feather: p.feather,
        opacity: p.opacity, layer: p.layer,
        hardness: p.hardness !== undefined ? p.hardness : 30
      })),
      // Send borlas/togas state + transforms so server can composite everything
      borlas: shBorlaState ? shBorlaState.borlas.filter(b => b.visible).map(b => ({
        color: b.color, x: Math.round(b.x), y: Math.round(b.y),
        scale: Math.round(b.scale), rotation: b.rotation, visible: true,
        mask: b.mask || null, hilo: b.hilo || null, hiloMask: b.hiloMask || null,
      })) : [],
      hiloColor: shBorlaState ? (shBorlaState.hiloColor || '#d4a017') : '#d4a017',
      hiloSize: shBorlaState ? (shBorlaState.hiloSize || 3) : 3,
      hiloFeather: shBorlaState ? (shBorlaState.hiloFeather || 1) : 1,
      hiloFlow: shBorlaState ? (shBorlaState.hiloFlow || 100) : 100,
      // Pre-transform togas: apply groupTf (x, y, rotation) to each toga position
      // Same logic as togas section export (tgExportTiff)
      togas: (() => {
        if (!shTogaState || !shTogaState.togas) return [];
        const gRad = (shTogaGroupTf.rotation || 0) * Math.PI / 180;
        const imgCx = shImgW / 2, imgCy = shImgH / 2;
        return shTogaState.togas.map(t => {
          let px = (t.x || 0) + (shTogaGroupTf.x || 0);
          let py = (t.y || 0) + (shTogaGroupTf.y || 0);
          if (Math.abs(gRad) > 0.001) {
            const dx = px - imgCx, dy = py - imgCy;
            px = imgCx + dx * Math.cos(gRad) - dy * Math.sin(gRad);
            py = imgCy + dx * Math.sin(gRad) + dy * Math.cos(gRad);
          }
          return {
            variant: t.variant, x: Math.round(px), y: Math.round(py),
            scale: Math.round(t.scale || 50), scaleX: t.scaleX || 1,
            flipH: t.flipH || false,
            rotation: (t.rotation || 0) + (shTogaGroupTf.rotation || 0),
            mask: t.mask || null,
          };
        });
      })(),
      togaGroupTransform: { scaleY: shTogaGroupTf.scaleY || 1.0 },
      imgTransform: shImgTf,
      togaRefAR: shTogaRefAR
    };
    const res = await fetch(`/api/sombras/export-tiff/${shCbtis}/${encodeURIComponent(shFilename)}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    if (!res.ok) { showToast('Error al generar TIFF: ' + res.status, 'error'); return; }
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = shFilename.replace(/\.[^.]+$/, '') + '_final.tif';
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(a.href);
    showToast('TIFF final descargado', 'ok');
  } catch(e) { showToast('Error: ' + e.message, 'error'); }
}


// ===================== FIXES SECTION =====================
let fxMode = false, fxCbtis = '', fxFilename = '';
let fxBaseImg = null, fxImgW = 1, fxImgH = 1;
let fxCanvas = null, fxCtx = null, fxCursorCanvas = null, fxCursorCtx = null;
let fxSourcePhotos = [], fxSourceGroup = '';
let fxFixes = [], fxSelected = -1, fxNextId = 0;
let fxTool = 'move', fxAction = 'erase', fxBrushSize = 15;
let fxDragging = null, fxRotating = null, fxScaling = null, fxDrawing = false;
let fxCurStroke = null, fxLassoPoints = [];
let fxUndoStack = [], fxRedoStack = [];
let fxBrushFeather = 0.5, fxBrushFlow = 1.0;
let fxLassoFreehand = false, fxLassoDownXY = null, fxDraggingNode = -1;
let fxDblClickPending = false;
// Sub-editor state
let fxSubMode = false, fxSubImg = null, fxSubImgW = 1, fxSubImgH = 1;
let fxSubStrokes = [], fxSubPhotoId = '', fxSubDrawing = false, fxSubCurStroke = null, fxSubLassoPoints = [];
let fxSubRotation = 0;  // Source photo rotation in degrees (0, 90, 180, 270)
let fxSubRotations = {};  // Per-photo rotation memory: { photoId: degrees }

function fxImgToScreen(ix, iy) {
  const cw = fxCanvas.width, ch = fxCanvas.height;
  const s = pvZoom;
  const ox = (cw - fxImgW * s) / 2 + pvX;
  const oy = (ch - fxImgH * s) / 2 + pvY;
  return { x: ox + ix * s, y: oy + iy * s };
}
function fxScreenToImg(sx, sy) {
  const cw = fxCanvas.width, ch = fxCanvas.height;
  const s = pvZoom;
  const ox = (cw - fxImgW * s) / 2 + pvX;
  const oy = (ch - fxImgH * s) / 2 + pvY;
  return { x: (sx - ox) / s, y: (sy - oy) / s };
}

// Convert screen coords to local coords of selected fix layer (0-1 normalized)
function fxScreenToFixLocal(sx, sy) {
  const s = pvZoom;
  const oxBase = (fxCanvas.width - fxImgW * s) / 2 + pvX;
  const oyBase = (fxCanvas.height - fxImgH * s) / 2 + pvY;
  const nx = (sx - oxBase) / (fxImgW * s);
  const ny = (sy - oyBase) / (fxImgH * s);
  if (fxSelected < 0 || fxSelected >= fxFixes.length) return null;
  const fix = fxFixes[fxSelected];
  const fxS = fix.fxScale || 1.0;
  const bw = fix.bboxW || (fix._workCanvas ? fix._workCanvas.width : 100);
  const bh = fix.bboxH || (fix._workCanvas ? fix._workCanvas.height : 100);
  const drawW = bw * fxS;
  const drawH = bh * fxS;
  const rot = (fix.rotation || 0) * Math.PI / 180;
  const dx = (nx - fix.x / fxImgW) * fxImgW;
  const dy = (ny - fix.y / fxImgH) * fxImgH;
  const rdx = dx * Math.cos(-rot) - dy * Math.sin(-rot);
  const rdy = dx * Math.sin(-rot) + dy * Math.cos(-rot);
  return { x: (rdx + drawW / 2) / drawW, y: (rdy + drawH / 2) / drawH };
}

// Convert fix-local coords (0-1) to screen coords
function fxLocalToScreen(lx, ly) {
  if (fxSelected < 0 || fxSelected >= fxFixes.length) return { x: 0, y: 0 };
  const fix = fxFixes[fxSelected];
  const s = pvZoom;
  const oxBase = (fxCanvas.width - fxImgW * s) / 2 + pvX;
  const oyBase = (fxCanvas.height - fxImgH * s) / 2 + pvY;
  const fxS = fix.fxScale || 1.0;
  const bw = fix.bboxW || (fix._workCanvas ? fix._workCanvas.width : 100);
  const bh = fix.bboxH || (fix._workCanvas ? fix._workCanvas.height : 100);
  const drawW = bw * fxS;
  const drawH = bh * fxS;
  const rot = (fix.rotation || 0) * Math.PI / 180;
  const rdx = (lx - 0.5) * drawW;
  const rdy = (ly - 0.5) * drawH;
  const dx = rdx * Math.cos(rot) - rdy * Math.sin(rot);
  const dy = rdx * Math.sin(rot) + rdy * Math.cos(rot);
  return { x: oxBase + (fix.x + dx) * s, y: oyBase + (fix.y + dy) * s };
}

async function enterFixesMode(cbtis, filename) {
  // Exit any other mode
  if (rfMode) await exitRefineMode();
  if (typeof blMode !== 'undefined' && blMode) exitBorlasMode();
  if (typeof tgMode !== 'undefined' && tgMode) exitTogasMode();
  if (shMode) exitSombrasMode();

  fxCbtis = cbtis; fxFilename = filename;
  fxMode = true; fxSubMode = false;
  fxFixes = []; fxSelected = -1; fxNextId = 0;
  fxTool = 'move'; fxAction = 'erase'; fxBrushSize = 15;
  fxUndoStack = []; fxRedoStack = [];
  fxDragging = null; fxRotating = null; fxScaling = null;

  const encName = encodeURIComponent(filename);
  const panel = document.getElementById('previewPanel');
  if (!panel) return;

  panel.innerHTML = `
    <div class="refine-toolbar" id="fxToolbar">
      <button class="rf-btn active" onclick="fxSetTool(this,'move')" title="Mover (V)">&#8596; Mover</button>
      <button class="rf-btn" onclick="fxSetTool(this,'brush')" title="Brocha (B)">&#9998; Brocha</button>
      <button class="rf-btn" onclick="fxSetTool(this,'lasso')" title="Lazo (L)">&#11044; Lazo</button>
      <span class="rf-sep"></span>
      <button class="rf-btn erase active" onclick="fxSetAction(this,'erase')" title="Borrar (E)">&#10007; Borrar</button>
      <button class="rf-btn restore" onclick="fxSetAction(this,'restore')" title="Restaurar (R)">&#10003; Restaurar</button>
      <span class="rf-sep"></span>
      <span style="font-size:11px">Tam:</span>
      <input type="range" class="rf-size-slider" min="2" max="80" value="${fxBrushSize}"
        oninput="fxBrushSize=+this.value;document.getElementById('fxSizeVal').textContent=this.value">
      <span class="rf-size-label" id="fxSizeVal">${fxBrushSize}</span>
      <label style="font-size:11px">Pluma:</label>
      <input type="range" class="rf-size-slider" min="0" max="100" value="${Math.round(fxBrushFeather*100)}" style="width:40px"
        oninput="fxBrushFeather=this.value/100;document.getElementById('fxFeatherVal').textContent=this.value+'%'">
      <span class="rf-size-label" id="fxFeatherVal">${Math.round(fxBrushFeather*100)}%</span>
      <label style="font-size:11px">Flujo:</label>
      <input type="range" class="rf-size-slider" min="5" max="100" value="${Math.round(fxBrushFlow*100)}" style="width:40px"
        oninput="fxBrushFlow=this.value/100;document.getElementById('fxFlowVal').textContent=this.value+'%'">
      <span class="rf-size-label" id="fxFlowVal">${Math.round(fxBrushFlow*100)}%</span>
      <span class="rf-sep"></span>
      <button class="rf-btn" onclick="fxUndo()" title="Deshacer (Ctrl+Z)">&#8617;</button>
      <button class="rf-btn" onclick="fxRedo()" title="Rehacer (Ctrl+Y)">&#8618;</button>
      <span class="rf-sep"></span>
      <span style="font-size:11px">Fotos:</span>
      <span id="fxThumbs"></span>
      <button class="rf-btn" id="fxNewSelBtn" onclick="fxStartNewSelection()" title="Nueva seleccion desde foto original" disabled>+ Seleccion</button>
      <span class="rf-sep"></span>
      <button class="rf-btn" onclick="toggleWorkflowDone('fixes',this)">${getWorkflowDoneState('fixes') ? '&#10003; Listo' : '&#9744; Listo'}</button>
      <button class="rf-btn" onclick="fxExportTiff()">&#128190; TIFF</button>
      <button class="rf-btn" onclick="exitFixesMode()">&#10005; Salir</button>
      <span class="rf-sep"></span>
      <button class="rf-btn" onclick="fxZoomFit()" title="Ajustar (0)">Ajustar</button>
      <button class="rf-btn" onclick="fxToggleFullscreen()" id="fxFullscreenBtn" title="Pantalla completa (F)">&#9974; Completa</button>
    </div>
    <div style="display:flex;flex:1;overflow:hidden;align-self:stretch;width:100%">
      <div class="rf-canvas-wrap" id="fxWrap" style="flex:1;position:relative">
        <canvas id="fxDisplay"></canvas>
        <canvas id="fxCursor" style="position:absolute;top:0;left:0;pointer-events:none"></canvas>
      </div>
      <div id="fxLayerPanel" style="width:160px;padding:4px;overflow-y:auto;border-left:1px solid var(--bg3);font-size:11px">
        <div style="font-weight:bold;padding:2px 4px;color:#bb9af7">CAPAS FIX</div>
        <div id="fxLayerList"></div>
      </div>
    </div>
  `;

  fxCanvas = document.getElementById('fxDisplay');
  fxCtx = fxCanvas.getContext('2d');
  fxCursorCanvas = document.getElementById('fxCursor');
  fxCursorCtx = fxCursorCanvas.getContext('2d');

  setTimeout(async () => {
    const wrap = document.getElementById('fxWrap');
    const rect = wrap.getBoundingClientRect();
    console.log('[FX] wrap rect:', rect.width, rect.height);
    fxCanvas.width = Math.max(rect.width, 200);
    fxCanvas.height = Math.max(rect.height, 200);
    fxCursorCanvas.width = fxCanvas.width;
    fxCursorCanvas.height = fxCanvas.height;

    // Load source photos list
    try {
      const spRes = await fetch(`/api/fixes/source-photos/${cbtis}/${encName}`);
      const spData = await spRes.json();
      if (spData.ok) {
        fxSourcePhotos = spData.photos.filter(p => p.exists);
        fxSourceGroup = spData.group;
        fxRenderThumbs();
      }
    } catch {}

    // Load composite base — use Sombras snapshot (pixel-perfect match)
    try {
      const snapUrl = `/api/sombras/snapshot/${cbtis}/${encName}?maxw=600`;
      let testRes = await fetch(snapUrl);
      let usedSnapshot = testRes.ok;
      if (!testRes.ok) {
        // Fallback to server-rendered composite
        console.warn('[FX] No snapshot, falling back to server composite');
        const fallbackUrl = `/api/fixes/composite-base/${cbtis}/${encName}?maxw=600`;
        testRes = await fetch(fallbackUrl);
        if (!testRes.ok) {
          showToast('Error: guarda el estado en Sombras primero', 'error');
          return;
        }
        showToast('Tip: ve a Sombras y guarda para mejor calidad', 'warn');
      }
      const blob = await testRes.blob();
      const blobUrl = URL.createObjectURL(blob);
      fxBaseImg = await loadImg(blobUrl);
      URL.revokeObjectURL(blobUrl);
      fxImgW = fxBaseImg.width; fxImgH = fxBaseImg.height;
      pvZoom = 1; pvX = 0; pvY = 0;
      fxZoomFit();
    } catch(e) { console.error('[FX] load error:', e); showToast('Error cargando base: ' + e.message, 'error'); }

    // Load saved state
    try {
      const stRes = await fetch(`/api/fixes/state/${cbtis}/${encName}`);
      const st = await stRes.json();
      if (st.fixes && st.fixes.length > 0) {
        fxNextId = 0;
        for (const fx of st.fixes) {
          fx._img = null; fx._workCanvas = null;
          fxNextId = Math.max(fxNextId, fx.id + 1);
          fxFixes.push(fx);
        }
        // Restore brush settings
        if (st.brushFeather != null) fxBrushFeather = st.brushFeather;
        if (st.brushFlow != null) fxBrushFlow = st.brushFlow;
        const fsl = document.getElementById('fxFeatherSlider');
        if (fsl) fsl.value = Math.round(fxBrushFeather * 100);
        const fvl = document.getElementById('fxFeatherVal');
        if (fvl) fvl.textContent = Math.round(fxBrushFeather * 100) + '%';
        const flsl = document.getElementById('fxFlowSlider');
        if (flsl) flsl.value = Math.round(fxBrushFlow * 100);
        const flvl = document.getElementById('fxFlowVal');
        if (flvl) flvl.textContent = Math.round(fxBrushFlow * 100) + '%';
        // Re-extract images for each fix
        await fxReloadFixImages();
        fxRenderLayerList();
        fxDraw();
        showToast(`${fxFixes.length} fixes restaurados`, 'ok');
      }
    } catch {}

    // Load hi-res base in background
    setTimeout(async () => {
      if (!fxMode) return;
      try {
        let hiUrl = `/api/sombras/snapshot/${fxCbtis}/${encodeURIComponent(fxFilename)}?maxw=99999`;
        let hiRes = await fetch(hiUrl);
        if (!hiRes.ok) {
          hiUrl = `/api/fixes/composite-base/${fxCbtis}/${encodeURIComponent(fxFilename)}?maxw=99999`;
          hiRes = await fetch(hiUrl);
          if (!hiRes.ok) return;
        }
        const hiBlob = await hiRes.blob();
        const hiBlobUrl = URL.createObjectURL(hiBlob);
        const hiImg = await loadImg(hiBlobUrl);
        URL.revokeObjectURL(hiBlobUrl);
        if (!fxMode) return;
        fxBaseImg = hiImg;
        fxImgW = hiImg.width; fxImgH = hiImg.height;
        // Re-extract fix images at hi-res too
        await fxReloadFixImages();
        fxDraw();
      } catch {}
    }, 100);

    fxBindEvents(wrap);
  }, 50);
}

async function fxReloadFixImages() {
  const encName = encodeURIComponent(fxFilename);
  for (const fx of fxFixes) {
    if (!fx.selectionStrokes || fx.selectionStrokes.length === 0) continue;
    try {
      const res = await fetch(`/api/fixes/extract/${fxCbtis}/${encName}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          photo_id: fx.sourceId, group: fx.group || fxSourceGroup,
          strokes: fx.selectionStrokes, fix_id: fx.id,
          rotation: fx.srcRotation || fx.rotation || 0
        })
      });
      if (!res.ok) { fx._img = null; continue; }
      const bboxHeader = res.headers.get('X-Fix-Bbox');
      const blob = await res.blob();
      fx._img = await loadImg(URL.createObjectURL(blob));
      if (bboxHeader) {
        fx._bbox = JSON.parse(bboxHeader);
        // Re-compute display dimensions using source-to-base scale
        const srcW = fx._bbox.src_w || 1;
        const srcToBase = Math.min(fxImgW / srcW, fxImgH / (fx._bbox.src_h || 1));
        fx.fxScale = fx.fxScale || 1.0;
        fx.bboxW = (fx._bbox.pw || fx._img.width) * srcToBase;
        fx.bboxH = (fx._bbox.ph || fx._img.height) * srcToBase;
      }
      fx._workCanvas = fxBuildWorkCanvas(fx);
    } catch { fx._img = null; }
  }
}

function fxBuildWorkCanvas(fx) {
  if (!fx._img) return null;
  const c = document.createElement('canvas');
  c.width = fx._img.width; c.height = fx._img.height;
  const ctx = c.getContext('2d');
  ctx.drawImage(fx._img, 0, 0);
  // Apply edit strokes
  if (fx.editStrokes) {
    for (const s of fx.editStrokes) {
      fxApplyEditStroke(ctx, s, c.width, c.height, fx._img);
    }
  }
  return c;
}

function fxApplyEditStroke(ctx, stroke, w, h, origImg) {
  const feather = stroke.feather != null ? stroke.feather : 0.5;
  const flow = stroke.flow != null ? stroke.flow : 1.0;

  if (stroke.type === 'lasso') {
    // Lasso: fill polygon with feather blur
    const tmpC = document.createElement('canvas');
    tmpC.width = w; tmpC.height = h;
    const tmpCtx = tmpC.getContext('2d');
    tmpCtx.fillStyle = '#000';
    fxDrawStrokePath(tmpCtx, stroke, w, h);
    // Apply feather blur (scale feather 0-1 to pixel radius)
    const blurR = Math.max(0.5, feather * 30);
    tmpCtx.filter = `blur(${blurR}px)`;
    tmpCtx.drawImage(tmpC, 0, 0);
    tmpCtx.filter = 'none';
    if (stroke.mode === 'erase') {
      ctx.save(); ctx.globalCompositeOperation = 'destination-out';
      ctx.drawImage(tmpC, 0, 0); ctx.restore();
    } else {
      // Restore: mask original pixels through feathered shape
      const panoC = document.createElement('canvas');
      panoC.width = w; panoC.height = h;
      const panoCtx = panoC.getContext('2d');
      panoCtx.drawImage(origImg, 0, 0);
      panoCtx.globalCompositeOperation = 'destination-in';
      panoCtx.drawImage(tmpC, 0, 0);
      ctx.save(); ctx.globalCompositeOperation = 'source-over';
      ctx.drawImage(panoC, 0, 0); ctx.restore();
    }
  } else {
    // Brush: Togas-style radial gradient (single accumulation canvas)
    const radius = stroke.radius * Math.max(w, h);
    const innerR = radius * (1 - feather);
    const tmpC = document.createElement('canvas');
    tmpC.width = w; tmpC.height = h;
    const tmpCtx = tmpC.getContext('2d');

    for (const p of stroke.points) {
      const px = p.x * w, py = p.y * h;
      if (feather > 0.01 && radius > 1) {
        const grad = tmpCtx.createRadialGradient(px, py, Math.max(0.5, innerR), px, py, radius);
        if (stroke.mode === 'erase') {
          grad.addColorStop(0, `rgba(0,0,0,${flow})`);
          grad.addColorStop(1, 'rgba(0,0,0,0)');
        } else {
          grad.addColorStop(0, `rgba(255,255,255,${flow})`);
          grad.addColorStop(1, 'rgba(255,255,255,0)');
        }
        tmpCtx.fillStyle = grad;
      } else {
        tmpCtx.fillStyle = stroke.mode === 'erase'
          ? `rgba(0,0,0,${flow})` : `rgba(255,255,255,${flow})`;
      }
      tmpCtx.beginPath(); tmpCtx.arc(px, py, radius, 0, Math.PI * 2); tmpCtx.fill();
    }
    // Connect points with lines for smooth strokes
    if (stroke.points.length > 1) {
      tmpCtx.lineWidth = radius * 2; tmpCtx.lineCap = 'round'; tmpCtx.lineJoin = 'round';
      tmpCtx.strokeStyle = stroke.mode === 'erase'
        ? `rgba(0,0,0,${flow})` : `rgba(255,255,255,${flow})`;
      tmpCtx.beginPath();
      tmpCtx.moveTo(stroke.points[0].x * w, stroke.points[0].y * h);
      for (let i = 1; i < stroke.points.length; i++)
        tmpCtx.lineTo(stroke.points[i].x * w, stroke.points[i].y * h);
      tmpCtx.stroke();
    }
    // Apply accumulated mask
    if (stroke.mode === 'erase') {
      ctx.save(); ctx.globalCompositeOperation = 'destination-out';
      ctx.drawImage(tmpC, 0, 0); ctx.restore();
    } else {
      const panoC = document.createElement('canvas');
      panoC.width = w; panoC.height = h;
      const panoCtx = panoC.getContext('2d');
      panoCtx.drawImage(origImg, 0, 0);
      panoCtx.globalCompositeOperation = 'destination-in';
      panoCtx.drawImage(tmpC, 0, 0);
      ctx.save(); ctx.globalCompositeOperation = 'source-over';
      ctx.drawImage(panoC, 0, 0); ctx.restore();
    }
  }
}

function fxDrawStrokePath(ctx, stroke, w, h) {
  const pts = stroke.points;
  if (!pts || pts.length === 0) return;
  if (stroke.type === 'lasso') {
    ctx.beginPath();
    ctx.moveTo(pts[0].x * w, pts[0].y * h);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x * w, pts[i].y * h);
    ctx.closePath(); ctx.fill();
  } else {
    const r = stroke.radius * Math.max(w, h);
    for (const p of pts) {
      ctx.beginPath(); ctx.arc(p.x * w, p.y * h, r, 0, Math.PI * 2); ctx.fill();
    }
    if (pts.length > 1) {
      ctx.lineWidth = r * 2; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.beginPath(); ctx.moveTo(pts[0].x * w, pts[0].y * h);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x * w, pts[i].y * h);
      ctx.stroke();
    }
  }
}

function exitFixesMode() {
  fxSaveState();
  fxMode = false; fxSubMode = false;
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  const panel = document.getElementById('previewPanel');
  if (panel) panel.classList.remove('faux-fullscreen');
  if (fxCbtis && fxFilename) {
    initPreviewPanel(
      `/api/cutout/preview/${fxCbtis}/${encodeURIComponent(fxFilename)}?t=${Date.now()}`,
      `${fxFilename} (recortada)`, true
    );
  }
  refreshGroups();
}

function fxSaveState() {
  if (!fxCbtis || !fxFilename) return;
  const data = {
    brushFeather: fxBrushFeather, brushFlow: fxBrushFlow,
    fixes: fxFixes.map(fx => ({
      id: fx.id, sourceId: fx.sourceId, group: fx.group || fxSourceGroup,
      selectionStrokes: fx.selectionStrokes,
      x: fx.x, y: fx.y,
      rotation: fx.rotation || 0,
      srcRotation: fx.srcRotation || 0,
      fxScale: fx.fxScale || 1.0,
      bboxW: fx.bboxW, bboxH: fx.bboxH,
      opacity: fx.opacity, visible: fx.visible,
      editStrokes: fx.editStrokes || [],
      bbox: fx._bbox
    }))
  };
  fetch(`/api/fixes/state/${fxCbtis}/${encodeURIComponent(fxFilename)}`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).catch(() => {});
}

function fxZoomFit() {
  if (!fxCanvas || (!fxBaseImg && !fxSubMode)) return;
  const cw = fxCanvas.width, ch = fxCanvas.height;
  let iw, ih;
  if (fxSubMode) {
    const swapped = (fxSubRotation === 90 || fxSubRotation === 270);
    iw = swapped ? fxSubImgH : fxSubImgW;
    ih = swapped ? fxSubImgW : fxSubImgH;
  } else {
    iw = fxImgW; ih = fxImgH;
  }
  const pad = 20;
  pvZoom = Math.min((cw - pad * 2) / iw, (ch - pad * 2) / ih);
  pvX = 0; pvY = 0;
  fxDraw();
}

function fxToggleFullscreen() {
  const panel = document.getElementById('previewPanel');
  if (!panel) return;
  _toggleRealFullscreen(panel, 'fxFullscreenBtn', () => {
    const wrap = document.getElementById('fxWrap');
    if (wrap && fxCanvas && fxMode) {
      const rect = wrap.getBoundingClientRect();
      fxCanvas.width = rect.width; fxCanvas.height = rect.height;
      fxCursorCanvas.width = rect.width; fxCursorCanvas.height = rect.height;
      fxZoomFit();
    }
  });
}

function fxSetTool(btn, tool) {
  fxTool = tool;
  btn.parentElement.querySelectorAll('.rf-btn').forEach(b => {
    if (b.getAttribute('onclick')?.includes('fxSetTool')) b.classList.remove('active');
  });
  btn.classList.add('active');
}

function fxSetAction(btn, action) {
  if (fxDrawing || fxSubDrawing) return;
  fxAction = action;
  btn.parentElement.querySelectorAll('.rf-btn.restore,.rf-btn.erase').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

// --- Thumbnails ---
function fxRenderThumbs() {
  const container = document.getElementById('fxThumbs');
  if (!container) return;
  container.innerHTML = '';
  for (const p of fxSourcePhotos) {
    const img = document.createElement('img');
    img.className = 'fx-source-thumb';
    img.src = `/api/fixes/source-image/${fxCbtis}/${fxSourceGroup}/${p.id}?maxw=120`;
    img.title = `IMG_${p.id}.jpg`;
    img.dataset.photoId = p.id;
    img.onclick = () => {
      container.querySelectorAll('.fx-source-thumb').forEach(t => t.classList.remove('active'));
      img.classList.add('active');
      document.getElementById('fxNewSelBtn').disabled = false;
    };
    container.appendChild(img);
  }
}

function fxGetSelectedPhotoId() {
  const active = document.querySelector('.fx-source-thumb.active');
  return active ? active.dataset.photoId : null;
}

// --- Layer list ---
function fxRenderLayerList() {
  const list = document.getElementById('fxLayerList');
  if (!list) return;
  list.innerHTML = '';
  for (let i = fxFixes.length - 1; i >= 0; i--) {
    const fx = fxFixes[i];
    const div = document.createElement('div');
    div.className = 'fx-layer-item' + (fxSelected === i ? ' selected' : '');
    div.innerHTML = `
      <span style="cursor:pointer" onclick="fxToggleVis(${i})">${fx.visible ? '&#128065;' : '&#128064;'}</span>
      <span style="flex:1;cursor:pointer" onclick="fxSelectLayer(${i})">Fix ${fx.id + 1} (${fx.sourceId})</span>
      <input type="range" min="0" max="100" value="${Math.round(fx.opacity * 100)}" style="width:40px"
        oninput="fxSetOpacity(${i}, +this.value/100)" title="Opacidad">
      <span style="cursor:pointer;color:#f77" onclick="fxDeleteLayer(${i})" title="Eliminar">&#10005;</span>
    `;
    list.appendChild(div);
  }
}

function fxToggleVis(idx) {
  if (idx >= 0 && idx < fxFixes.length) { fxFixes[idx].visible = !fxFixes[idx].visible; fxRenderLayerList(); fxDraw(); }
}

function fxSelectLayer(idx) {
  fxSelected = idx; fxRenderLayerList(); fxDraw();
}

function fxSetOpacity(idx, val) {
  if (idx >= 0 && idx < fxFixes.length) { fxFixes[idx].opacity = val; fxDraw(); }
}

function fxDeleteLayer(idx) {
  if (idx < 0 || idx >= fxFixes.length) return;
  fxFixes.splice(idx, 1);
  if (fxSelected >= fxFixes.length) fxSelected = fxFixes.length - 1;
  fxRenderLayerList(); fxDraw(); fxSaveState();
  showToast('Fix eliminado', 'ok');
}

// --- Main draw ---
function fxDraw() {
  if (!fxCtx || !fxCanvas) return;
  const ctx = fxCtx;
  const cw = fxCanvas.width, ch = fxCanvas.height;
  ctx.clearRect(0, 0, cw, ch);

  if (fxSubMode) {
    // Sub-editor: show source photo (with rotation)
    if (fxSubImg) {
      const s = pvZoom;
      const rot = fxSubRotation || 0;
      const swapped = (rot === 90 || rot === 270);
      const dw = swapped ? fxSubImgH : fxSubImgW;
      const dh = swapped ? fxSubImgW : fxSubImgH;
      const ox = (cw - dw * s) / 2 + pvX;
      const oy = (ch - dh * s) / 2 + pvY;
      ctx.save();
      ctx.translate(ox + dw * s / 2, oy + dh * s / 2);
      ctx.rotate(rot * Math.PI / 180);
      ctx.drawImage(fxSubImg, -fxSubImgW * s / 2, -fxSubImgH * s / 2, fxSubImgW * s, fxSubImgH * s);
      ctx.restore();
      // Draw strokes in rotated space
      for (const stroke of fxSubStrokes) {
        ctx.save();
        ctx.globalAlpha = 0.4;
        ctx.fillStyle = stroke.mode === 'erase' ? 'rgba(255,80,80,0.5)' : 'rgba(80,255,80,0.5)';
        ctx.strokeStyle = ctx.fillStyle;
        fxDrawStrokeOnCanvas(ctx, stroke, ox, oy, s, dw, dh);
        ctx.restore();
      }
      // In-progress stroke
      if (fxSubCurStroke) {
        ctx.save();
        ctx.globalAlpha = 0.4;
        ctx.fillStyle = fxAction === 'erase' ? 'rgba(255,80,80,0.5)' : 'rgba(80,255,80,0.5)';
        ctx.strokeStyle = ctx.fillStyle;
        fxDrawStrokeOnCanvas(ctx, fxSubCurStroke, ox, oy, s, dw, dh);
        ctx.restore();
      }
      // Lasso points
      if (fxSubLassoPoints.length > 0) {
        ctx.save();
        ctx.strokeStyle = fxAction === 'erase' ? '#ff5050' : '#50ff50';
        ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(ox + fxSubLassoPoints[0].x * dw * s, oy + fxSubLassoPoints[0].y * dh * s);
        for (let i = 1; i < fxSubLassoPoints.length; i++)
          ctx.lineTo(ox + fxSubLassoPoints[i].x * dw * s, oy + fxSubLassoPoints[i].y * dh * s);
        ctx.stroke(); ctx.setLineDash([]);
        ctx.restore();
      }
    }
    fxDrawSubBanner(ctx, cw, ch);
    return;
  }

  // Main mode: draw base + fix layers
  if (fxBaseImg) {
    const s = pvZoom;
    const ox = (cw - fxImgW * s) / 2 + pvX;
    const oy = (ch - fxImgH * s) / 2 + pvY;
    // Checkerboard background
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, fxImgW * s, fxImgH * s);
    ctx.clip();
    const pat = ctx.createPattern(fxCheckerboard(), 'repeat');
    ctx.fillStyle = pat;
    ctx.fillRect(ox, oy, fxImgW * s, fxImgH * s);
    ctx.restore();
    // Base composite
    ctx.drawImage(fxBaseImg, ox, oy, fxImgW * s, fxImgH * s);
    // Fix layers
    for (let i = 0; i < fxFixes.length; i++) {
      const fx = fxFixes[i];
      if (!fx.visible || !fx._workCanvas) continue;
      ctx.save();
      ctx.globalAlpha = fx.opacity;
      // bboxW/bboxH are already in base image pixel coords
      // fxScale is user's manual scale adjustment (1.0 = original)
      const fxS = fx.fxScale || 1.0;
      const bw = fx.bboxW || fx._workCanvas.width;
      const bh = fx.bboxH || fx._workCanvas.height;
      const drawW = bw * fxS * s;
      const drawH = bh * fxS * s;
      const fsx = ox + fx.x * s;
      const fsy = oy + fx.y * s;
      ctx.translate(fsx, fsy);
      if (fx.rotation) ctx.rotate(fx.rotation * Math.PI / 180);
      ctx.drawImage(fx._workCanvas, -drawW / 2, -drawH / 2, drawW, drawH);
      ctx.restore();
    }
    // Selection handles for selected fix
    if (fxSelected >= 0 && fxSelected < fxFixes.length) {
      const fx = fxFixes[fxSelected];
      if (fx._workCanvas) {
        fxDrawHandles(ctx, fx, ox, oy, s);
      }
    }
    // In-progress edit stroke on selected fix
    if (fxCurStroke && fxSelected >= 0) {
      const fix = fxFixes[fxSelected];
      ctx.save();
      ctx.globalAlpha = 0.4;
      ctx.fillStyle = fxAction === 'erase' ? 'rgba(255,80,80,0.5)' : 'rgba(80,255,80,0.5)';
      ctx.strokeStyle = ctx.fillStyle;
      if (fxCurStroke.local && fix) {
        // Draw in fix layer's transformed space
        const fxS = fix.fxScale || 1.0;
        const bw = fix.bboxW || (fix._workCanvas ? fix._workCanvas.width : 100);
        const bh = fix.bboxH || (fix._workCanvas ? fix._workCanvas.height : 100);
        const drawW = bw * fxS, drawH = bh * fxS;
        const fsx = ox + fix.x * s, fsy = oy + fix.y * s;
        ctx.translate(fsx, fsy);
        if (fix.rotation) ctx.rotate(fix.rotation * Math.PI / 180);
        // Draw stroke points in local coords (0-1)
        const pts = fxCurStroke.points;
        const r = fxCurStroke.radius * Math.max(drawW, drawH);
        if (fxCurStroke.type === 'lasso' && pts.length > 0) {
          ctx.beginPath();
          ctx.moveTo(-drawW/2 + pts[0].x * drawW, -drawH/2 + pts[0].y * drawH);
          for (let i = 1; i < pts.length; i++)
            ctx.lineTo(-drawW/2 + pts[i].x * drawW, -drawH/2 + pts[i].y * drawH);
          ctx.stroke();
        } else if (pts.length > 0) {
          for (const p of pts) {
            ctx.beginPath(); ctx.arc(-drawW/2 + p.x * drawW, -drawH/2 + p.y * drawH, r, 0, Math.PI * 2); ctx.fill();
          }
        }
      } else {
        fxDrawStrokeOnCanvas(ctx, fxCurStroke, ox, oy, s, fxImgW, fxImgH);
      }
      ctx.restore();
    }
    // In-progress lasso (main mode) — draw in fix layer space
    if (fxTool === 'lasso' && fxLassoPoints.length > 0 && fxSelected >= 0) {
      const fix = fxFixes[fxSelected];
      if (fix) {
        const fxS = fix.fxScale || 1.0;
        const bw = fix.bboxW || (fix._workCanvas ? fix._workCanvas.width : 100);
        const bh = fix.bboxH || (fix._workCanvas ? fix._workCanvas.height : 100);
        const drawW = bw * fxS, drawH = bh * fxS;
        const fsx = ox + fix.x * s, fsy = oy + fix.y * s;
        const lassoColor = fxAction === 'restore' ? 'rgba(158,206,106' : 'rgba(247,118,142';
        ctx.save();
        ctx.translate(fsx, fsy);
        if (fix.rotation) ctx.rotate(fix.rotation * Math.PI / 180);
        ctx.strokeStyle = lassoColor + ',0.8)';
        ctx.lineWidth = 2 / s; ctx.setLineDash([6/s, 4/s]);
        ctx.beginPath();
        ctx.moveTo(-drawW/2 + fxLassoPoints[0].x * drawW, -drawH/2 + fxLassoPoints[0].y * drawH);
        for (let i = 1; i < fxLassoPoints.length; i++)
          ctx.lineTo(-drawW/2 + fxLassoPoints[i].x * drawW, -drawH/2 + fxLassoPoints[i].y * drawH);
        ctx.stroke(); ctx.setLineDash([]);
        // Nodes
        const nodeR = 5 / s;
        for (let i = 0; i < fxLassoPoints.length; i++) {
          const px = -drawW/2 + fxLassoPoints[i].x * drawW;
          const py = -drawH/2 + fxLassoPoints[i].y * drawH;
          const isFirst = (i === 0 && fxLassoPoints.length >= 3);
          ctx.beginPath(); ctx.arc(px, py, isFirst ? nodeR * 1.6 : nodeR, 0, Math.PI * 2);
          ctx.fillStyle = isFirst ? 'rgba(255,255,100,0.95)' : lassoColor + ',0.9)';
          ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5/s; ctx.stroke();
        }
        ctx.restore();
      }
    }
  }
}

let _fxCheckerCache = null;
function fxCheckerboard() {
  if (_fxCheckerCache) return _fxCheckerCache;
  const c = document.createElement('canvas'); c.width = 16; c.height = 16;
  const x = c.getContext('2d');
  x.fillStyle = '#404040'; x.fillRect(0, 0, 16, 16);
  x.fillStyle = '#505050'; x.fillRect(0, 0, 8, 8); x.fillRect(8, 8, 8, 8);
  _fxCheckerCache = c;
  return c;
}

function fxDrawStrokeOnCanvas(ctx, stroke, ox, oy, scale, iw, ih) {
  const pts = stroke.points;
  if (!pts || pts.length === 0) return;
  if (stroke.type === 'lasso') {
    ctx.beginPath();
    ctx.moveTo(ox + pts[0].x * iw * scale, oy + pts[0].y * ih * scale);
    for (let i = 1; i < pts.length; i++)
      ctx.lineTo(ox + pts[i].x * iw * scale, oy + pts[i].y * ih * scale);
    ctx.closePath(); ctx.fill();
  } else {
    const r = stroke.radius * Math.max(iw, ih) * scale;
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(ox + p.x * iw * scale, oy + p.y * ih * scale, r, 0, Math.PI * 2);
      ctx.fill();
    }
    if (pts.length > 1) {
      ctx.lineWidth = r * 2; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(ox + pts[0].x * iw * scale, oy + pts[0].y * ih * scale);
      for (let i = 1; i < pts.length; i++)
        ctx.lineTo(ox + pts[i].x * iw * scale, oy + pts[i].y * ih * scale);
      ctx.stroke();
    }
  }
}

function fxDrawHandles(ctx, fx, ox, oy, scale) {
  if (!fx._workCanvas) return;
  const fxS = fx.fxScale || 1.0;
  const fw = (fx.bboxW || fx._workCanvas.width) * fxS * scale;
  const fh = (fx.bboxH || fx._workCanvas.height) * fxS * scale;
  const cx = ox + fx.x * scale;
  const cy = oy + fx.y * scale;
  ctx.save();
  ctx.translate(cx, cy);
  if (fx.rotation) ctx.rotate(fx.rotation * Math.PI / 180);
  // Bounding box
  ctx.strokeStyle = '#bb9af7'; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
  ctx.strokeRect(-fw / 2, -fh / 2, fw, fh);
  ctx.setLineDash([]);
  // Rotation handle (top center)
  ctx.beginPath(); ctx.moveTo(0, -fh / 2); ctx.lineTo(0, -fh / 2 - 25); ctx.strokeStyle = '#bb9af7'; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.beginPath(); ctx.arc(0, -fh / 2 - 25, 6, 0, Math.PI * 2);
  ctx.fillStyle = fxRotating ? '#e0af68' : '#bb9af7'; ctx.fill();
  // Scale handle (bottom-right corner)
  ctx.beginPath(); ctx.arc(fw / 2, fh / 2, 6, 0, Math.PI * 2);
  ctx.fillStyle = fxScaling ? '#e0af68' : '#bb9af7'; ctx.fill();
  ctx.restore();
}

function fxDrawSubBanner(ctx, cw, ch) {
  ctx.save();
  ctx.fillStyle = 'rgba(187,154,247,0.85)';
  ctx.fillRect(0, 0, cw, 24);
  ctx.fillStyle = '#fff'; ctx.font = '12px monospace'; ctx.textAlign = 'center';
  ctx.fillText(`Seleccionando de IMG_${fxSubPhotoId}.jpg — Usa Brocha/Lazo para seleccionar, luego click "Extraer"`, cw / 2, 16);
  ctx.restore();
}

// --- Sub-editor: select from source photo ---
function fxStartNewSelection() {
  const photoId = fxGetSelectedPhotoId();
  if (!photoId) { showToast('Selecciona una foto primero', 'warn'); return; }
  fxEnterSubEditor(photoId);
}

async function fxEnterSubEditor(photoId) {
  fxSubMode = true; fxSubPhotoId = photoId;
  fxSubStrokes = []; fxSubCurStroke = null; fxSubLassoPoints = []; fxSubDrawing = false;
  fxSubRotation = fxSubRotations[photoId] || 0;  // Recall saved rotation
  fxTool = 'brush'; fxAction = 'restore';  // Default to brush+restore in sub-editor
  // Update toolbar active buttons to match
  document.querySelectorAll('#fxToolbar .rf-btn.erase, #fxToolbar .rf-btn.restore').forEach(b => {
    b.classList.toggle('active', b.classList.contains('restore'));
  });

  // Update toolbar — show extract/cancel buttons
  const toolbar = document.getElementById('fxToolbar');
  if (toolbar) {
    const existing = document.getElementById('fxSubActions');
    if (existing) existing.remove();
    const span = document.createElement('span');
    span.id = 'fxSubActions';
    span.innerHTML = `<span class="rf-sep"></span>
      <span class="fx-sub-banner" style="display:inline-block">Modo Seleccion</span>
      <button class="rf-btn" onclick="fxSubRotateCW()" title="Rotar 90° CW">&#8635; Rotar</button>
      <button class="rf-btn" onclick="fxConfirmExtraction()" style="color:#50ff50">&#10003; Extraer</button>
      <button class="rf-btn" onclick="fxCancelSubEditor()" style="color:#ff5050">&#10005; Cancelar</button>`;
    toolbar.appendChild(span);
  }

  showToast('Cargando foto original...', 'info');
  try {
    fxSubImg = await loadImg(`/api/fixes/source-image/${fxCbtis}/${fxSourceGroup}/${photoId}?maxw=800`);
    fxSubImgW = fxSubImg.width; fxSubImgH = fxSubImg.height;
    pvZoom = 1; pvX = 0; pvY = 0;
    fxZoomFit();
    // Load hi-res in background
    setTimeout(async () => {
      if (!fxSubMode || fxSubPhotoId !== photoId) return;
      try {
        const hi = await loadImg(`/api/fixes/source-image/${fxCbtis}/${fxSourceGroup}/${photoId}?maxw=99999`);
        if (!fxSubMode || fxSubPhotoId !== photoId) return;
        fxSubImg = hi; fxSubImgW = hi.width; fxSubImgH = hi.height;
        fxDraw();
      } catch {}
    }, 100);
    showToast('Foto cargada. Selecciona la region.', 'ok');
  } catch(e) { showToast('Error: ' + e.message, 'error'); fxCancelSubEditor(); }
}

function fxCancelSubEditor() {
  fxSubMode = false; fxSubImg = null;
  const el = document.getElementById('fxSubActions');
  if (el) el.remove();
  fxZoomFit();
}

function fxSubRotateCW() {
  fxSubRotation = (fxSubRotation + 90) % 360;
  fxSubRotations[fxSubPhotoId] = fxSubRotation;  // Remember per photo
  // Clear strokes since they don't match new orientation
  if (fxSubStrokes.length > 0) {
    fxSubStrokes = [];
    showToast('Seleccion borrada por rotacion', 'warn');
  }
  fxDraw();
}

async function fxConfirmExtraction() {
  if (fxSubStrokes.length === 0) { showToast('Selecciona algo primero', 'warn'); return; }
  showToast('Extrayendo seleccion...', 'info');
  const encName = encodeURIComponent(fxFilename);
  const fixId = fxNextId++;
  try {
    const res = await fetch(`/api/fixes/extract/${fxCbtis}/${encName}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        photo_id: fxSubPhotoId, group: fxSourceGroup,
        strokes: fxSubStrokes, fix_id: fixId,
        rotation: fxSubRotation || 0
      })
    });
    if (!res.ok) {
      const errTxt = await res.text();
      console.error('[FX] extract error:', res.status, errTxt);
      showToast('Error extrayendo: ' + res.status, 'error');
      return;
    }
    const bboxHeader = res.headers.get('X-Fix-Bbox');
    const blob = await res.blob();
    const img = await loadImg(URL.createObjectURL(blob));
    const bbox = bboxHeader ? JSON.parse(bboxHeader) : { x: 0.3, y: 0.3, w: 0.4, h: 0.4 };

    // Create fix layer — position at center of base image for user to place
    // bbox.px/py/pw/ph are pixel coords on source photo
    // We map the extraction size to base image coordinates
    const srcW = bbox.src_w || 1;
    const srcH = bbox.src_h || 1;
    // Scale factor: map source photo pixels to base image pixels
    const srcToBase = Math.min(fxImgW / srcW, fxImgH / srcH);
    const fx = {
      id: fixId,
      sourceId: fxSubPhotoId,
      group: fxSourceGroup,
      selectionStrokes: [...fxSubStrokes],
      srcRotation: fxSubRotation || 0,  // Remember for re-extraction (server needs it)
      rotation: 0,  // Visual rotation on canvas starts at 0 (server already applied rotation)
      // Position: center of base image (user can move)
      x: fxImgW / 2,
      y: fxImgH / 2,
      // fxScale is user's manual scale (1.0 = natural mapped size)
      fxScale: 1.0,
      // Drawing size in base image pixel coords (already mapped)
      bboxW: (bbox.pw || img.width) * srcToBase,
      bboxH: (bbox.ph || img.height) * srcToBase,
      opacity: 1.0,
      visible: true,
      editStrokes: [],
      _img: img, _bbox: bbox
    };
    fx._workCanvas = fxBuildWorkCanvas(fx);
    fxFixes.push(fx);
    fxSelected = fxFixes.length - 1;

    // Exit sub-editor
    fxSubMode = false; fxSubImg = null;
    const el = document.getElementById('fxSubActions');
    if (el) el.remove();

    fxRenderLayerList();
    fxZoomFit();
    fxSaveState();
    showToast('Fix creado exitosamente', 'ok');
  } catch(e) { showToast('Error: ' + e.message, 'error'); }
}

// --- Undo/Redo ---
function fxSaveSnapshot() {
  const snap = { fixes: JSON.stringify(fxFixes.map(fx => ({
    id: fx.id, sourceId: fx.sourceId, group: fx.group,
    selectionStrokes: fx.selectionStrokes,
    x: fx.x, y: fx.y, rotation: fx.rotation, scale: fx.scale,
    opacity: fx.opacity, visible: fx.visible,
    editStrokes: fx.editStrokes || []
  }))), selected: fxSelected };
  fxUndoStack.push(snap);
  fxRedoStack = [];
  if (fxUndoStack.length > 30) fxUndoStack.shift();
}

async function fxRestoreSnapshot(snap) {
  const data = JSON.parse(snap.fixes);
  const oldFixes = fxFixes;
  fxFixes = [];
  for (const d of data) {
    // Try to reuse existing image
    const existing = oldFixes.find(f => f.id === d.id);
    d._img = existing ? existing._img : null;
    d._workCanvas = null;
    fxFixes.push(d);
  }
  fxSelected = snap.selected;
  // Rebuild work canvases
  for (const fx of fxFixes) {
    if (!fx._img && fx.selectionStrokes && fx.selectionStrokes.length > 0) {
      await fxReloadFixImages();
      break;
    }
    fx._workCanvas = fxBuildWorkCanvas(fx);
  }
  fxRenderLayerList(); fxDraw();
}

function fxUndo() {
  if (fxUndoStack.length === 0) return;
  const curSnap = { fixes: JSON.stringify(fxFixes.map(fx => ({
    id: fx.id, sourceId: fx.sourceId, group: fx.group,
    selectionStrokes: fx.selectionStrokes,
    x: fx.x, y: fx.y, rotation: fx.rotation, scale: fx.scale,
    opacity: fx.opacity, visible: fx.visible, editStrokes: fx.editStrokes || []
  }))), selected: fxSelected };
  fxRedoStack.push(curSnap);
  fxRestoreSnapshot(fxUndoStack.pop());
}

function fxRedo() {
  if (fxRedoStack.length === 0) return;
  const curSnap = { fixes: JSON.stringify(fxFixes.map(fx => ({
    id: fx.id, sourceId: fx.sourceId, group: fx.group,
    selectionStrokes: fx.selectionStrokes,
    x: fx.x, y: fx.y, rotation: fx.rotation, scale: fx.scale,
    opacity: fx.opacity, visible: fx.visible, editStrokes: fx.editStrokes || []
  }))), selected: fxSelected };
  fxUndoStack.push(curSnap);
  fxRestoreSnapshot(fxRedoStack.pop());
}

// --- Mouse events ---
function fxBindEvents(wrap) {
  wrap.addEventListener('mousedown', fxMouseDown);
  wrap.addEventListener('mousemove', fxMouseMove);
  wrap.addEventListener('mouseup', fxMouseUp);
  wrap.addEventListener('dblclick', fxDblClick);
  wrap.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = fxCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const oldZ = pvZoom;
    pvZoom = Math.max(0.05, Math.min(pvZoom * (e.deltaY < 0 ? 1.1 : 0.9), 20));
    const cw = fxCanvas.width, ch = fxCanvas.height;
    pvX += (mx - cw / 2 - pvX) * (1 - pvZoom / oldZ);
    pvY += (my - ch / 2 - pvY) * (1 - pvZoom / oldZ);
    fxDraw();
  }, { passive: false });
  wrap.addEventListener('contextmenu', e => e.preventDefault());
}

function fxMouseDown(e) {
  if (e.button === 1 || (e.button === 0 && (spaceHeld || fxTool === 'hand'))) {
    // Pan
    fxDragging = { pan: true, sx: e.clientX, sy: e.clientY, ox: pvX, oy: pvY };
    return;
  }
  if (e.button !== 0) return;
  const rect = fxCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;

  if (fxSubMode) {
    // Sub-editor: brush/lasso selection (with rotation)
    const rot = fxSubRotation || 0;
    const swapped = (rot === 90 || rot === 270);
    const dw = swapped ? fxSubImgH : fxSubImgW;
    const dh = swapped ? fxSubImgW : fxSubImgH;
    const s = pvZoom;
    const ox = (fxCanvas.width - dw * s) / 2 + pvX;
    const oy = (fxCanvas.height - dh * s) / 2 + pvY;
    const nx = (mx - ox) / (dw * s), ny = (my - oy) / (dh * s);
    if (nx < 0 || ny < 0 || nx > 1 || ny > 1) return;

    if (fxTool === 'lasso') {
      fxSubLassoPoints.push({ x: nx, y: ny });
      fxDraw();
      return;
    }
    // Brush
    fxSubDrawing = true;
    fxSubCurStroke = {
      type: 'brush', mode: fxAction,
      points: [{ x: nx, y: ny }],
      radius: fxBrushSize / Math.max(dw, dh)
    };
    fxDraw();
    return;
  }

  // Main mode
  if (fxTool === 'move') {
    // Check transform handles first
    if (fxSelected >= 0 && fxSelected < fxFixes.length) {
      const fx = fxFixes[fxSelected];
      if (fx._workCanvas) {
        const s = pvZoom;
        const oxBase = (fxCanvas.width - fxImgW * s) / 2 + pvX;
        const oyBase = (fxCanvas.height - fxImgH * s) / 2 + pvY;
        const fcx = oxBase + fx.x * s;
        const fcy = oyBase + fx.y * s;
        const fxS = fx.fxScale || 1.0;
        const fw = (fx.bboxW || fx._workCanvas.width) * fxS * s;
        const fh = (fx.bboxH || fx._workCanvas.height) * fxS * s;
        const rad = -(fx.rotation || 0) * Math.PI / 180;
        const dx = mx - fcx, dy = my - fcy;
        const rx = dx * Math.cos(rad) - dy * Math.sin(rad);
        const ry = dx * Math.sin(rad) + dy * Math.cos(rad);

        // Rotation handle
        if (Math.hypot(rx, ry + fh / 2 + 25) < 12) {
          fxSaveSnapshot();
          fxRotating = { idx: fxSelected, startAngle: Math.atan2(mx - fcx, -(my - fcy)), startRot: fx.rotation };
          return;
        }
        // Scale handle
        if (Math.hypot(rx - fw / 2, ry - fh / 2) < 12) {
          fxSaveSnapshot();
          fxScaling = { idx: fxSelected, startDist: Math.hypot(mx - fcx, my - fcy), startScale: fx.fxScale || 1.0 };
          return;
        }
        // Hit test for drag (inside bounding box)
        if (rx >= -fw / 2 && rx <= fw / 2 && ry >= -fh / 2 && ry <= fh / 2) {
          fxSaveSnapshot();
          fxDragging = { idx: fxSelected, offX: mx - fcx, offY: my - fcy };
          return;
        }
      }
    }
    // Hit test other layers
    for (let i = fxFixes.length - 1; i >= 0; i--) {
      const fx = fxFixes[i];
      if (!fx.visible || !fx._workCanvas) continue;
      const s = pvZoom;
      const oxBase = (fxCanvas.width - fxImgW * s) / 2 + pvX;
      const oyBase = (fxCanvas.height - fxImgH * s) / 2 + pvY;
      const fcx = oxBase + fx.x * s;
      const fcy = oyBase + fx.y * s;
      const fw = fx._workCanvas.width * fx.scale * s / fxImgW;
      const fh = fx._workCanvas.height * fx.scale * s / fxImgH;
      const rad = -(fx.rotation || 0) * Math.PI / 180;
      const dx = mx - fcx, dy = my - fcy;
      const rx = dx * Math.cos(rad) - dy * Math.sin(rad);
      const ry = dx * Math.sin(rad) + dy * Math.cos(rad);
      if (rx >= -fw / 2 && rx <= fw / 2 && ry >= -fh / 2 && ry <= fh / 2) {
        fxSelected = i;
        fxSaveSnapshot();
        fxDragging = { idx: i, offX: mx - fcx, offY: my - fcy };
        fxRenderLayerList(); fxDraw();
        return;
      }
    }
    // Clicked empty space — deselect
    fxSelected = -1; fxRenderLayerList(); fxDraw();
    return;
  }

  // Brush/Lasso editing on selected fix layer
  if (fxSelected < 0 || fxSelected >= fxFixes.length) {
    showToast('Selecciona un fix primero', 'warn');
    return;
  }
  const s = pvZoom;
  const oxBase = (fxCanvas.width - fxImgW * s) / 2 + pvX;
  const oyBase = (fxCanvas.height - fxImgH * s) / 2 + pvY;
  const nx = (mx - oxBase) / (fxImgW * s);
  const ny = (my - oyBase) / (fxImgH * s);

  // Convert base-image coords to fix-layer local coords
  const fix = fxFixes[fxSelected];
  const fxS = fix.fxScale || 1.0;
  const bw = fix.bboxW || (fix._workCanvas ? fix._workCanvas.width : 100);
  const bh = fix.bboxH || (fix._workCanvas ? fix._workCanvas.height : 100);
  const drawW = bw * fxS;
  const drawH = bh * fxS;
  const rot = (fix.rotation || 0) * Math.PI / 180;
  // Mouse in base image pixels relative to fix center
  const dx = (nx - fix.x / fxImgW) * fxImgW;
  const dy = (ny - fix.y / fxImgH) * fxImgH;
  // Un-rotate
  const rdx = dx * Math.cos(-rot) - dy * Math.sin(-rot);
  const rdy = dx * Math.sin(-rot) + dy * Math.cos(-rot);
  // Scale to work canvas
  const wkW = fix._workCanvas ? fix._workCanvas.width : bw;
  const wkH = fix._workCanvas ? fix._workCanvas.height : bh;
  const lx = (rdx + drawW / 2) / drawW;
  const ly = (rdy + drawH / 2) / drawH;
  if (lx < -0.1 || ly < -0.1 || lx > 1.1 || ly > 1.1) return;

  if (fxTool === 'lasso') {
    // --- LASSO (Refine-style) ---
    fxLassoFreehand = false;
    // Check close on first node
    if (fxLassoPoints.length >= 3) {
      const first = fxLassoPoints[0];
      // First node in screen coords for hit test
      const fsx = oxBase + fix.x * s + (first.x * drawW - drawW / 2) * Math.cos(rot) * s / drawW * fxImgW;
      const fsy = oyBase + fix.y * s;
      // Simple proximity test: reproject first node to screen
      const fDx = (first.x - 0.5) * drawW;
      const fDy = (first.y - 0.5) * drawH;
      const fSx = fDx * Math.cos(rot) - fDy * Math.sin(rot);
      const fSy = fDx * Math.sin(rot) + fDy * Math.cos(rot);
      const fsxS = oxBase + (fix.x + fSx / fxImgW) * s;
      const fsyS = oyBase + (fix.y + fSy / fxImgH) * s;
      if ((mx-fsxS)*(mx-fsxS) + (my-fsyS)*(my-fsyS) <= 144) {
        // Close lasso → apply as edit stroke
        fxSaveSnapshot();
        const stroke = { type: 'lasso', mode: fxAction, points: [...fxLassoPoints], radius: 0, feather: fxBrushFeather, flow: fxBrushFlow, local: true };
        if (!fix.editStrokes) fix.editStrokes = [];
        fix.editStrokes.push(stroke);
        fix._workCanvas = fxBuildWorkCanvas(fix);
        fxLassoPoints = [];
        fxDraw(); fxSaveState();
        return;
      }
    }
    // Check hit on existing node → drag
    const hitIdx = fxHitTestLassoNode(e);
    if (hitIdx >= 0) {
      fxDraggingNode = hitIdx;
    } else {
      fxLassoPoints.push({ x: lx, y: ly });
      fxLassoDownXY = { cx: e.clientX, cy: e.clientY, t: Date.now() };
    }
    fxDraw(); return;
  }
  // --- BRUSH (Togas-style with feather/flow) ---
  fxSaveSnapshot();
  fxDrawing = true;
  fxCurStroke = {
    type: 'brush', mode: fxAction,
    points: [{ x: lx, y: ly }],
    radius: fxBrushSize / Math.max(drawW, drawH),
    feather: fxBrushFeather, flow: fxBrushFlow, local: true
  };
  fxDraw();
}

function fxMouseMove(e) {
  const rect = fxCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;

  // Cursor
  if (fxCursorCtx && (fxTool === 'brush' || (fxSubMode && fxTool === 'brush'))) {
    fxCursorCtx.clearRect(0, 0, fxCursorCanvas.width, fxCursorCanvas.height);
    const r = fxBrushSize * pvZoom;
    fxCursorCtx.beginPath();
    fxCursorCtx.arc(mx, my, r, 0, Math.PI * 2);
    fxCursorCtx.strokeStyle = fxAction === 'erase' ? 'rgba(255,80,80,0.7)' : 'rgba(80,255,80,0.7)';
    fxCursorCtx.lineWidth = 1.5;
    fxCursorCtx.stroke();
  }

  // Pan
  if (fxDragging && fxDragging.pan) {
    pvX = fxDragging.ox + (e.clientX - fxDragging.sx);
    pvY = fxDragging.oy + (e.clientY - fxDragging.sy);
    fxDraw(); return;
  }

  // Sub-editor brush drag
  if (fxSubMode && fxSubDrawing && fxSubCurStroke) {
    const rot = fxSubRotation || 0;
    const swapped = (rot === 90 || rot === 270);
    const dw = swapped ? fxSubImgH : fxSubImgW;
    const dh = swapped ? fxSubImgW : fxSubImgH;
    const s = pvZoom;
    const ox = (fxCanvas.width - dw * s) / 2 + pvX;
    const oy = (fxCanvas.height - dh * s) / 2 + pvY;
    const nx = (mx - ox) / (dw * s), ny = (my - oy) / (dh * s);
    fxSubCurStroke.points.push({ x: nx, y: ny });
    fxDraw(); return;
  }

  // Main mode interactions
  if (fxRotating) {
    const fx = fxFixes[fxRotating.idx];
    const s = pvZoom;
    const oxBase = (fxCanvas.width - fxImgW * s) / 2 + pvX;
    const oyBase = (fxCanvas.height - fxImgH * s) / 2 + pvY;
    const fcx = oxBase + fx.x * s;
    const fcy = oyBase + fx.y * s;
    const curAngle = Math.atan2(mx - fcx, -(my - fcy));
    const delta = (curAngle - fxRotating.startAngle) * 180 / Math.PI;
    fx.rotation = Math.round((fxRotating.startRot + delta) * 10) / 10;
    fxDraw(); return;
  }
  if (fxScaling) {
    const fx = fxFixes[fxScaling.idx];
    const s = pvZoom;
    const oxBase = (fxCanvas.width - fxImgW * s) / 2 + pvX;
    const oyBase = (fxCanvas.height - fxImgH * s) / 2 + pvY;
    const fcx = oxBase + fx.x * s;
    const fcy = oyBase + fx.y * s;
    const curDist = Math.hypot(mx - fcx, my - fcy);
    const ratio = curDist / Math.max(1, fxScaling.startDist);
    fx.fxScale = Math.max(0.1, fxScaling.startScale * ratio);
    fxDraw(); return;
  }
  if (fxDragging && !fxDragging.pan) {
    const fx = fxFixes[fxDragging.idx];
    const s = pvZoom;
    const oxBase = (fxCanvas.width - fxImgW * s) / 2 + pvX;
    const oyBase = (fxCanvas.height - fxImgH * s) / 2 + pvY;
    fx.x = (mx - fxDragging.offX - oxBase) / s;
    fx.y = (my - fxDragging.offY - oyBase) / s;
    fxDraw(); return;
  }
  // Lasso node dragging
  if (fxDraggingNode >= 0 && fxTool === 'lasso') {
    const s = pvZoom;
    const oxBase = (fxCanvas.width - fxImgW * s) / 2 + pvX;
    const oyBase = (fxCanvas.height - fxImgH * s) / 2 + pvY;
    const local = fxScreenToFixLocal(mx, my);
    if (local) fxLassoPoints[fxDraggingNode] = { x: local.x, y: local.y };
    fxDraw(); return;
  }
  // Lasso freehand (20px + 200ms threshold)
  if (fxTool === 'lasso' && fxLassoDownXY && (e.buttons & 1) && fxSelected >= 0) {
    if (!fxLassoFreehand) {
      const ddx = e.clientX - fxLassoDownXY.cx;
      const ddy = e.clientY - fxLassoDownXY.cy;
      const elapsed = Date.now() - (fxLassoDownXY.t || 0);
      if (ddx*ddx + ddy*ddy < 400 || elapsed < 200) return;
      fxLassoFreehand = true;
    }
    const local = fxScreenToFixLocal(mx, my);
    if (!local) return;
    const last = fxLassoPoints[fxLassoPoints.length - 1];
    const dx = local.x - last.x, dy = local.y - last.y;
    if (dx*dx + dy*dy > 0.0003) { fxLassoPoints.push({ x: local.x, y: local.y }); fxDraw(); }
    return;
  }
  // Lasso hover cursor
  if (fxTool === 'lasso' && fxLassoPoints.length > 0 && fxSelected >= 0) {
    let cursor = 'crosshair';
    // Check proximity to first node
    if (fxLassoPoints.length >= 3) {
      const f = fxLassoPoints[0];
      const scr = fxLocalToScreen(f.x, f.y);
      if ((mx-scr.x)*(mx-scr.x) + (my-scr.y)*(my-scr.y) <= 144) cursor = 'pointer';
    }
    // Check node proximity
    if (cursor === 'crosshair') {
      for (const p of fxLassoPoints) {
        const sx = p.x * fxImgW * s + oxBase, sy = p.y * fxImgH * s + oyBase;
        if ((mx-sx)*(mx-sx) + (my-sy)*(my-sy) <= 64) { cursor = 'move'; break; }
      }
    }
    e.target.style.cursor = cursor;
  }
  // Brush edit drag
  if (fxDrawing && fxCurStroke) {
    const s = pvZoom;
    const local = fxScreenToFixLocal(mx, my);
    if (!local) return;
    fxCurStroke.points.push({ x: local.x, y: local.y });
    fxDraw();
  }
}

function fxMouseUp(e) {
  if (fxDragging && fxDragging.pan) { fxDragging = null; return; }
  if (fxRotating) { fxRotating = null; fxSaveState(); fxDraw(); return; }
  if (fxScaling) { fxScaling = null; fxSaveState(); fxDraw(); return; }
  if (fxDragging) { fxDragging = null; fxSaveState(); fxDraw(); return; }

  // Lasso node drag stop
  if (fxDraggingNode >= 0) { fxDraggingNode = -1; fxDraw(); return; }
  // Lasso freehand stop (keep points, don't close)
  if (fxTool === 'lasso' && fxLassoDownXY) {
    fxLassoDownXY = null; fxLassoFreehand = false;
    fxDraw(); return;
  }

  // Sub-editor brush up
  if (fxSubMode && fxSubDrawing && fxSubCurStroke) {
    fxSubStrokes.push(fxSubCurStroke);
    fxSubCurStroke = null; fxSubDrawing = false;
    fxDraw(); return;
  }

  // Main brush edit up
  if (fxDrawing && fxCurStroke && fxSelected >= 0) {
    const fx = fxFixes[fxSelected];
    if (!fx.editStrokes) fx.editStrokes = [];
    fx.editStrokes.push(fxCurStroke);
    fx._workCanvas = fxBuildWorkCanvas(fx);
    fxCurStroke = null; fxDrawing = false;
    fxDraw(); fxSaveState();
  }
}

function fxHitTestLassoNode(e) {
  if (fxLassoPoints.length === 0 || fxSelected < 0) return -1;
  const rect = fxCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  for (let i = 0; i < fxLassoPoints.length; i++) {
    const scr = fxLocalToScreen(fxLassoPoints[i].x, fxLassoPoints[i].y);
    if ((mx-scr.x)*(mx-scr.x) + (my-scr.y)*(my-scr.y) <= 64) return i;
  }
  return -1;
}

function fxDblClick(e) {
  // Close lasso in sub-editor
  if (fxSubMode && fxSubLassoPoints.length >= 3) {
    fxSubStrokes.push({ type: 'lasso', mode: fxAction, points: [...fxSubLassoPoints], radius: 0 });
    fxSubLassoPoints = [];
    fxDraw();
    return;
  }
  // Close lasso in main edit mode
  if (!fxSubMode && fxLassoPoints.length >= 3 && fxSelected >= 0) {
    fxSaveSnapshot();
    const stroke = { type: 'lasso', mode: fxAction, points: [...fxLassoPoints], radius: 0 };
    const fx = fxFixes[fxSelected];
    if (!fx.editStrokes) fx.editStrokes = [];
    fx.editStrokes.push(stroke);
    fx._workCanvas = fxBuildWorkCanvas(fx);
    fxLassoPoints = [];
    fxDraw(); fxSaveState();
  }
}

// --- TIFF Export ---
async function fxExportTiff() {
  if (fxFixes.length === 0) {
    showToast('No hay fixes para exportar. Usa el TIFF de Sombras.', 'warn');
    return;
  }
  fxSaveState();
  showToast('Generando TIFF con fixes...', 'info');
  // Use sombras export as base, fixes are applied on top
  // For now, trigger sombras TIFF (which already includes everything except fixes)
  // TODO: dedicated fixes TIFF endpoint
  try {
    const encName = encodeURIComponent(fxFilename);
    // Load sombras state to include in export
    let sombrasData = [];
    try {
      const sRes = await fetch(`/api/sombras/state/${fxCbtis}/${encName}`);
      const sd = await sRes.json();
      sombrasData = sd.polygons || [];
    } catch {}
    const res = await fetch(`/api/sombras/export-tiff/${fxCbtis}/${encName}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ sombras: sombrasData })
    });
    if (!res.ok) { showToast('Error exportando TIFF', 'error'); return; }
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = fxFilename.replace(/\.[^.]+$/, '') + '_final.tif';
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(a.href);
    showToast('TIFF descargado', 'ok');
  } catch(e) { showToast('Error: ' + e.message, 'error'); }
}


// --- Polling ---
async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    setRunning(data.running);
  } catch {}
}
setInterval(pollStatus, 3000);

// --- Keyboard shortcuts ---
document.addEventListener('keydown', (e) => {
  if (e.key === 'F5') { e.preventDefault(); refreshGroups(); }
  if (e.key === 'Escape') {
    if (document.fullscreenElement) return; // browser handles fullscreen exit
    if (fxMode) {
      if (fxSubMode && fxSubLassoPoints.length > 0) { fxSubLassoPoints = []; fxDraw(); }
      else if (fxSubMode) { fxCancelSubEditor(); }
      else if (fxLassoPoints.length > 0) { fxLassoPoints = []; fxDraw(); }
      else if (document.getElementById('previewPanel')?.classList.contains('faux-fullscreen')) {
        fxToggleFullscreen();
      } else { exitFixesMode(); }
    } else if (shMode) {
      if (shDrawing) { shFinishPolygon(); }
      else if (document.getElementById('previewPanel')?.classList.contains('faux-fullscreen')) {
        shToggleFullscreen();
      } else { exitSombrasMode(); }
    } else if (tgMode) {
      if (document.getElementById('previewPanel')?.classList.contains('faux-fullscreen')) {
        tgToggleFullscreen();
      } else {
        exitTogasMode();
      }
    } else if (blMode) {
      if (document.getElementById('previewPanel')?.classList.contains('faux-fullscreen')) {
        blToggleFullscreen();
      } else {
        exitBorlasMode();
      }
    } else {
      stopProcess();
    }
  }
  // Borlas keyboard shortcuts
  if (blMode && (e.ctrlKey || e.metaKey) && e.key === 'z') {
    e.preventDefault(); blUndo(); return;
  }
  if (blMode && (e.ctrlKey || e.metaKey) && e.key === 'y') {
    e.preventDefault(); blRedo(); return;
  }
  if (blMode && blSelected >= 0 && (e.key === 'Delete' || e.key === 'Backspace')) {
    if (blTool === 'hilo') {
      blHiloDelete();
    } else {
      blToggleBorla(blSelected);
    }
  }
  if (blMode && (e.key === 'Enter' || e.key === 'Escape') && blHiloDrawing) {
    e.preventDefault();
    blHiloFinish();
  }
  if (blMode && !e.ctrlKey && !e.metaKey) {
    if (e.key === 'v' || e.key === 'V') { if (blHiloDrawing) blHiloFinish(); blSetTool('move'); }
    if (e.key === '+' || e.key === 'a' || e.key === 'A') { if (blHiloDrawing) blHiloFinish(); blSetTool('add'); }
    if (e.key === 'b' || e.key === 'B') { if (blHiloDrawing) blHiloFinish(); blSetTool('brush'); }
    if (e.key === 'h' || e.key === 'H') { if (blHiloDrawing) blHiloFinish(); blSetTool('hilo'); }
    if (blTool === 'brush' && (e.key === 'x' || e.key === 'X')) {
      blSetBrushAction(blBrushAction === 'erase' ? 'restore' : 'erase');
    }
  }
  // Fixes keyboard shortcuts
  if (fxMode && (e.ctrlKey || e.metaKey) && e.key === 'z') {
    e.preventDefault(); fxUndo(); return;
  }
  if (fxMode && (e.ctrlKey || e.metaKey) && e.key === 'y') {
    e.preventDefault(); fxRedo(); return;
  }
  if (fxMode && (e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault(); fxSaveState(); showToast('Guardado', 'ok'); return;
  }
  if (fxMode && !e.ctrlKey && !e.metaKey) {
    if (e.key === 'v' || e.key === 'V') { const btn = document.querySelector('#fxToolbar .rf-btn[onclick*="move"]'); if (btn) fxSetTool(btn, 'move'); }
    if (e.key === 'b' || e.key === 'B') { const btn = document.querySelector('#fxToolbar .rf-btn[onclick*="brush"]'); if (btn) fxSetTool(btn, 'brush'); }
    if (e.key === 'l' || e.key === 'L') { const btn = document.querySelector('#fxToolbar .rf-btn[onclick*="lasso"]'); if (btn) fxSetTool(btn, 'lasso'); }
    if (e.key === 'r' || e.key === 'R') { fxAction = 'restore'; document.querySelectorAll('#fxToolbar .rf-btn.restore,#fxToolbar .rf-btn.erase').forEach(b => b.classList.remove('active')); const b = document.querySelector('#fxToolbar .rf-btn.restore'); if (b) b.classList.add('active'); }
    if (e.key === 'e' || e.key === 'E') { fxAction = 'erase'; document.querySelectorAll('#fxToolbar .rf-btn.restore,#fxToolbar .rf-btn.erase').forEach(b => b.classList.remove('active')); const b = document.querySelector('#fxToolbar .rf-btn.erase'); if (b) b.classList.add('active'); }
    if (e.key === 'f' || e.key === 'F') fxToggleFullscreen();
    if (e.key === '0') fxZoomFit();
    if (e.key === '[') { fxBrushSize = Math.max(2, fxBrushSize - 2); const sl = document.querySelector('#fxToolbar .rf-size-slider'); if (sl) sl.value = fxBrushSize; const v = document.getElementById('fxSizeVal'); if (v) v.textContent = fxBrushSize; }
    if (e.key === ']') { fxBrushSize = Math.min(80, fxBrushSize + 2); const sl = document.querySelector('#fxToolbar .rf-size-slider'); if (sl) sl.value = fxBrushSize; const v = document.getElementById('fxSizeVal'); if (v) v.textContent = fxBrushSize; }
    if (e.key === 'Delete' || e.key === 'Backspace') { if (fxSelected >= 0) fxDeleteLayer(fxSelected); }
  }
  // Togas keyboard shortcuts
  if (tgMode && (e.ctrlKey || e.metaKey) && e.key === 'z') {
    e.preventDefault(); tgUndo(); return;
  }
  if (tgMode && (e.ctrlKey || e.metaKey) && e.key === 'y') {
    e.preventDefault(); tgRedo(); return;
  }
  if (tgMode && !e.ctrlKey && !e.metaKey) {
    if (e.key === 'v' || e.key === 'V') tgSetTool('move');
    if (e.key === 'a' || e.key === 'A') tgSetTool('add-seat');
    if (e.key === 'b' || e.key === 'B') tgSetTool('brush');
    if (e.key === 'g' || e.key === 'G') tgSetTool('guide');
    if (e.key === 'i' || e.key === 'I') tgSetTool('img-move');
    if (e.key === 't' || e.key === 'T') tgSetTool('toga-group');
    if (tgTool === 'brush' && (e.key === 'x' || e.key === 'X')) {
      tgSetBrushAction(tgBrushAction === 'erase' ? 'restore' : 'erase');
    }
  }
  // Sombras keyboard shortcuts
  if (shMode && (e.ctrlKey || e.metaKey) && e.key === 'z') {
    e.preventDefault();
    if (shDrawing && shNewPoints.length > 1) {
      shNewPoints.pop(); shDraw();
    } else {
      shUndo();
    }
    return;
  }
  if (shMode && (e.ctrlKey || e.metaKey) && e.key === 'y') {
    e.preventDefault(); shRedo(); return;
  }
  if (shMode && shSelected >= 0 && (e.key === 'Delete' || e.key === 'Backspace')) {
    shDeletePoly(shSelected);
  }
  if (shMode && (e.key === 'Enter') && shDrawing) {
    e.preventDefault(); shFinishPolygon();
  }
  if (shMode && !e.ctrlKey && !e.metaKey) {
    if (e.key === 'p' || e.key === 'P') { if (shDrawing) shFinishPolygon(); shSetTool('polygon'); }
    if (e.key === 'v' || e.key === 'V') { if (shDrawing) shFinishPolygon(); shSetTool('move'); }
    if (e.key === 'n' || e.key === 'N') { if (shDrawing) shFinishPolygon(); shSetTool('edit'); }
  }
  // Space+drag = free pan (Photoshop-style) — works in preview AND refine
  if (e.code === 'Space' && !e.repeat && e.target === document.body) {
    e.preventDefault();
    spaceHeld = true;
    document.body.style.cursor = 'grab';
  }
});
document.addEventListener('keyup', (e) => {
  if (e.code === 'Space') {
    spaceHeld = false;
    document.body.style.cursor = '';
  }
});

// --- Sidebar resize ---
(function() {
  const sb = document.getElementById('sidebar');
  const handle = document.getElementById('sidebarResize');
  const saved = localStorage.getItem('arca_sidebar_w');
  if (saved) sb.style.width = saved + 'px';
  let dragging = false, startX = 0, startW = 0;
  handle.addEventListener('pointerdown', e => {
    e.preventDefault();
    dragging = true; startX = e.clientX; startW = sb.offsetWidth;
    handle.classList.add('active');
    handle.setPointerCapture(e.pointerId);
  });
  handle.addEventListener('pointermove', e => {
    if (!dragging) return;
    const w = Math.max(200, Math.min(window.innerWidth * 0.6, startW + e.clientX - startX));
    sb.style.width = w + 'px';
  });
  handle.addEventListener('pointerup', e => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('active');
    localStorage.setItem('arca_sidebar_w', sb.offsetWidth);
  });
})();

// --- Init ---
loadConfig();
refreshGroups();

// Reconnect to log stream if a process is running (or has logs from current session)
(async () => {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.running || data.log_count > 0) {
      startListening();
    }
  } catch {}
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import argparse
    import webbrowser

    parser = argparse.ArgumentParser(description="ARCA Panorama GUI")
    parser.add_argument("-p", "--port", type=int, default=5800)
    parser.add_argument("-w", "--workspace", type=str, help="Override workspace path")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    # Silence Werkzeug request logging (the /api/status polling spam)
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    load_config()
    if args.workspace:
        config["workspace"] = win_to_wsl(args.workspace)
        save_config()

    PORT = args.port
    url = f"http://localhost:{PORT}"

    # Detect LAN IP for network access
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = "N/A"
    lan_url = f"http://{lan_ip}:{PORT}"

    print(f"""
  ╔══════════════════════════════════════════╗
  ║         ARCA Panorama GUI                ║
  ╠══════════════════════════════════════════╣
  ║  Local:     {url:<28s}║
  ║  Red:       {lan_url:<28s}║
  ║  Workspace: {config['workspace'][:28]:<28s}║
  ╠══════════════════════════════════════════╣
  ║  F5 = Actualizar | ESC = Detener         ║
  ║  Ctrl+C para cerrar el servidor          ║
  ╚══════════════════════════════════════════╝
""")

    if not args.no_browser:
        def try_open_browser():
            try:
                webbrowser.open(url)
            except Exception:
                print(f"  No se pudo abrir el navegador automáticamente.")
                print(f"  Abre manualmente: {url}\n")
        threading.Timer(1.0, try_open_browser).start()

    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
