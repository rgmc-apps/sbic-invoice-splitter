from __future__ import annotations
import base64, datetime, io, json, os, re, secrets, statistics, tempfile, threading, time, traceback, zipfile
from pathlib import Path
from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for

import fitz  # PyMuPDF
import google.auth
import google.auth.transport.requests as _google_requests
from google.cloud import storage as gcs
from PIL import Image

# pyzbar needs the native libzbar0 shared library.
# Import it optionally so a missing system library doesn't crash startup;
# barcode detection is simply skipped when unavailable.
try:
    from pyzbar import pyzbar as _pyzbar
    _ZBAR_OK = True
except Exception:
    _pyzbar = None  # type: ignore[assignment]
    _ZBAR_OK = False

# pymupdf4llm provides layout-aware, markdown-formatted text extraction that
# preserves table structure and reading order — significantly better than the
# raw get_text("text") approach for structured invoice documents.
# Import optionally so the service starts even if the package is absent.
try:
    import pymupdf4llm as _pymupdf4llm
    _ML4LLM_OK = True
except Exception:
    _pymupdf4llm = None  # type: ignore[assignment]
    _ML4LLM_OK = False

# pytesseract wraps the Tesseract OCR engine.  OCR reads from rendered pixels,
# completely bypassing the font-encoding layer — the only reliable approach
# when invoice PDFs use custom glyph-to-Unicode mappings that cause get_text()
# to return garbled characters (e.g. "U:rrr&?8s" instead of "Nº 51285").
# Requires the tesseract-ocr system package (installed in Dockerfile).
try:
    import pytesseract as _pytesseract
    _TESS_OK = True
except Exception:
    _pytesseract = None  # type: ignore[assignment]
    _TESS_OK = False

UPLOAD_BUCKET     = os.environ.get("UPLOAD_BUCKET", "sbic-splitter-uploads")
OPTIMIZE_LIMIT_MB = int(os.environ.get("OPTIMIZE_LIMIT_MB", 2048))  # Flask hard cap for /optimize-only
TMP_DIR           = Path(tempfile.gettempdir()) / "inv_sep"
TMP_DIR.mkdir(exist_ok=True)
RESULT_TTL = 3600  # seconds — local temp files expire after 1 hour

# ── Crop-region config ─────────────────────────────────────────────────────────
# Persisted alongside main.py so it survives container restarts.
CROP_CONFIG_PATH  = Path(__file__).parent / "crop_config.json"
CODES_CONFIG_PATH = Path(__file__).parent / "customer_codes.json"

_crop_region_cache: dict | None = None
_crop_region_mtime: float = 0.0


def _load_crop_region() -> dict | None:
    """Return saved crop region or None. Caches in memory; re-reads only on file change."""
    global _crop_region_cache, _crop_region_mtime
    try:
        if CROP_CONFIG_PATH.exists():
            mtime = CROP_CONFIG_PATH.stat().st_mtime
            if mtime != _crop_region_mtime:
                cfg = json.loads(CROP_CONFIG_PATH.read_text())
                _crop_region_mtime = mtime
                _crop_region_cache = (
                    cfg
                    if cfg.get("enabled") and all(k in cfg for k in ("x1", "y1", "x2", "y2"))
                    else None
                )
            return _crop_region_cache
        else:
            _crop_region_cache = None
            return None
    except Exception:
        return None

# ── Customer-code mapping ──────────────────────────────────────────────────────
# Premade defaults — shown on a fresh install before the user adds any codes.
# These are seeded into customer_codes.json the first time any edit is saved.
_PREMADE_CODES: dict[str, str] = {
    "MTC":  "240762",
    "SBIC": "190275",
}

_codes_cache: dict[str, str] | None = None
_codes_mtime: float = 0.0


def _load_customer_codes() -> dict[str, str]:
    """Return keyword→code mapping.  Reads customer_codes.json; falls back to
    premade defaults when the file does not yet exist."""
    global _codes_cache, _codes_mtime
    try:
        if CODES_CONFIG_PATH.exists():
            mtime = CODES_CONFIG_PATH.stat().st_mtime
            if mtime != _codes_mtime or _codes_cache is None:
                _codes_cache = json.loads(CODES_CONFIG_PATH.read_text())
                _codes_mtime = mtime
            return _codes_cache
    except Exception:
        pass
    return dict(_PREMADE_CODES)  # file absent or unreadable — use defaults


def _save_customer_codes(codes: dict[str, str]) -> None:
    """Persist codes to disk and invalidate the in-memory cache."""
    global _codes_cache, _codes_mtime
    CODES_CONFIG_PATH.write_text(json.dumps(codes, indent=2))
    _codes_cache = None  # force reload on next access


def _resolve_customer_code(filename: str) -> str:
    """Return the SI customer code whose key appears in *filename* (stem only).
    Falls back to 'UNKNOWN' if no mapping key is found."""
    codes = _load_customer_codes()
    stem = Path(filename).stem.upper()
    for key, code in codes.items():
        if key.upper() in stem:
            return code
    return "UNKNOWN"


# ── SI-number extraction patterns ─────────────────────────────────────────────
# Tried in order — first match wins.
_SI_RE = [
    # 0a. "SI #: 051285" — the label printed on MNLtaste / MTC invoices.
    #     The hash-colon variant is what Tesseract reads from these PDFs.
    re.compile(r'\bSI\s*#\s*:?\s*([0-9]{4,})', re.I),
    # 0b. "Nº 51285" / "No 51285" — the large bold display number on these invoices.
    #     Tesseract renders the numero sign (Nº) as "No"; we also accept "N°".
    #     Constrained to 5–6 digits so it doesn't accidentally match 10-digit PO numbers.
    re.compile(r'\bN[º°o]\s*\.?\s*([0-9]{5,6})\b'),
    # 1. Explicit "SI No:" / "S.I. No:" / "S.I. NO:" label (with optional dot/space variants)
    re.compile(r'\bS\.?\s*I\.?\s*[Nn][Oo]\.?\s*:?\s*([A-Z0-9]{4,}(?:[-/]\d+)*)', re.I),
    # 2. Bare "NO:" at the start of a line or after whitespace — common shorthand on
    #    invoice bodies where the SI context is already implied by the surrounding text.
    #    Using MULTILINE + start-of-line anchor prevents false matches on "PO NO:", etc.
    re.compile(r'(?:^|(?<=\s))NO\s*:\s*([A-Z0-9]{4,}(?:[-/]\d+)*)', re.I | re.MULTILINE),
    # 3. "Sales Invoice No." / "Invoice No."
    re.compile(r'(?:Sales\s+Invoice|Invoice)\s*[Nn]o\.?\s*[:\-]?\s*([A-Z0-9]{5,20})', re.I),
    # 4. Generic S.I. with optional number keyword (legacy broad match)
    re.compile(r'\bS\.?I\.?\s*(?:Num(?:ber)?|#)\s*[:\-]?\s*([A-Z0-9]{4,}(?:[-]\d+)*)', re.I),
    # 5. Bare "SI-NNNN" token
    re.compile(r'\b(SI[-]\d{4,}(?:[-]\d+)*)\b', re.I),
    # 6. Long bare numeric — last resort
    re.compile(r'\b(\d{7,13})\b'),
]


def _sanitize(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|\r\n\t]', '_', name).strip().strip('._')[:80] or "unknown"


def _find_si(text: str) -> str | None:
    # Collapse newlines so label/value pairs split across lines still match
    # (e.g. "SI No:\n123456" → "SI No: 123456").
    normalized = re.sub(r'[ \t]*[\r\n]+[ \t]*', ' ', text).strip()
    for pat in _SI_RE:
        m = pat.search(normalized)
        if m:
            v = m.group(1).strip()
            return _sanitize(v) if v else None
    return None


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting so the SI regex patterns work on clean text.

    pymupdf4llm returns markdown that includes bold markers (**), heading
    markers (#), table-cell separators (|), alignment rows (|:---|), and
    code fences (```).  Stripping these gives plain prose that the existing
    _SI_RE patterns can match without modification.
    """
    # Fenced code blocks — remove the whole block, not just the backticks
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'~~~[\s\S]*?~~~', ' ', text)
    # Inline code
    text = re.sub(r'`[^`\n]*`', ' ', text)
    # Bold / italic / bold-italic (order matters: longest first)
    text = re.sub(r'\*{1,3}', '', text)
    text = re.sub(r'_{1,2}', '', text)
    # ATX headings (# Heading)
    text = re.sub(r'^#{1,6}[ \t]*', '', text, flags=re.MULTILINE)
    # Table alignment rows  (|:---|:---| …)
    text = re.sub(r'^\|[\s:|\\-]+\|\s*$', '', text, flags=re.MULTILINE)
    # Table cell separators → space so "| SI No | 12345 |" → " SI No  12345 "
    text = re.sub(r'\|', ' ', text)
    # Blockquote markers
    text = re.sub(r'^>[ \t]*', '', text, flags=re.MULTILINE)
    # Setext / thematic break lines (--- *** ___)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Collapse runs of horizontal whitespace to a single space per line
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()


def _page_text(doc: fitz.Document, idx: int, clip: fitz.Rect | None = None) -> str:
    """Extract text from a page region.

    Tries pymupdf4llm first (layout-aware, table-preserving markdown output,
    then stripped clean) and falls back to PyMuPDF's native get_text() if
    pymupdf4llm is unavailable or returns nothing for that region.
    """
    if _ML4LLM_OK:
        try:
            kwargs: dict = {"pages": [idx]}
            if clip is not None:
                kwargs["clip"] = clip
            md = _pymupdf4llm.to_markdown(doc, **kwargs)
            stripped = _strip_markdown(md)
            if stripped:
                return stripped
        except Exception:
            pass  # fall through to get_text()

    page = doc[idx]
    if clip is not None:
        return page.get_text("text", clip=clip)
    return page.get_text("text")


def _ocr_text(page: fitz.Page, clip: fitz.Rect | None = None) -> str:
    """Render a page region to pixels at 3× zoom and run Tesseract OCR on it.

    This completely bypasses the PDF font-encoding layer, making it the only
    reliable approach when invoices use custom glyph-to-Unicode mappings that
    cause get_text() to return garbled output (e.g. "U:rrr&?8s" instead of
    "Nº 51285").  Returns an empty string if Tesseract is unavailable or fails.
    """
    if not _TESS_OK:
        return ""
    try:
        mat = fitz.Matrix(3, 3)  # 3× zoom → ~216 DPI for A4/Letter; enough for Tesseract
        px_kwargs: dict = {"matrix": mat, "colorspace": fitz.csGRAY}
        if clip is not None:
            px_kwargs["clip"] = clip
        pix = page.get_pixmap(**px_kwargs)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        return _pytesseract.image_to_string(img, config="--oem 1 --psm 11")
    except Exception:
        return ""


def _extract_si(doc: fitz.Document, idx: int) -> tuple[str, str]:
    page = doc[idx]
    h, w = page.rect.height, page.rect.width

    # ── Priority 1: user-configured crop region (OCR first, then text layer) ──
    crop = _load_crop_region()
    if crop:
        clip = fitz.Rect(w * crop["x1"], h * crop["y1"], w * crop["x2"], h * crop["y2"])
        if _TESS_OK:
            si = _find_si(_ocr_text(page, clip))
            if si:
                return si, "ocr-crop"
        si = _find_si(_page_text(doc, idx, clip))
        if si:
            return si, "text-crop"

    # ── Priority 2: bottom 40 % of page ───────────────────────────────────────
    si = _find_si(_page_text(doc, idx, fitz.Rect(0, h * 0.60, w, h)))
    if si:
        return si, "text-bottom"
    if _TESS_OK:
        si = _find_si(_ocr_text(page, fitz.Rect(0, h * 0.60, w, h)))
        if si:
            return si, "ocr-bottom"

    # ── Priority 3: right half of page ────────────────────────────────────────
    si = _find_si(_page_text(doc, idx, fitz.Rect(w * 0.50, 0, w, h)))
    if si:
        return si, "text-right"
    if _TESS_OK:
        si = _find_si(_ocr_text(page, fitz.Rect(w * 0.50, 0, w, h)))
        if si:
            return si, "ocr-right"

    # ── Priority 4: full page ─────────────────────────────────────────────────
    si = _find_si(_page_text(doc, idx))
    if si:
        return si, "text-full"
    if _TESS_OK:
        si = _find_si(_ocr_text(page))
        if si:
            return si, "ocr-full"

    # No SI found by any method — use fallback page name.
    return f"page_{idx + 1:04d}", "fallback"


def _infer_from_sequence(raw: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Detect a consecutive numeric pattern across successfully-extracted pages and
    fill in any 'fallback' pages by inferring the expected SI number.

    Algorithm:
      1. Collect all pages whose SI is a pure integer string and method != 'fallback'.
      2. For every consecutive pair of known pages compute (value_diff / page_gap).
      3. Take the median step — robust against a few OCR mis-reads.
      4. If the step is a clean integer (±0.15 tolerance), anchor on the median
         known page and project forward/backward to fill fallback pages.
      5. Zero-padding width is matched to the widest known SI string.

    Returns the same-length list with inferred values substituted where applicable.
    """
    # ── collect known numeric pages ────────────────────────────────────────────
    known: dict[int, int] = {}   # page_idx → integer SI value
    pad_width: int = 0           # widest known SI string (for zero-padding)
    for i, (si, method) in enumerate(raw):
        if method != "fallback" and re.fullmatch(r"\d+", si):
            known[i] = int(si)
            pad_width = max(pad_width, len(si))

    if len(known) < 2:
        return raw  # not enough data to detect a pattern

    # ── detect step ───────────────────────────────────────────────────────────
    sorted_known = sorted(known.items())
    steps: list[float] = []
    for j in range(1, len(sorted_known)):
        i1, v1 = sorted_known[j]
        i0, v0 = sorted_known[j - 1]
        gap = i1 - i0
        if gap > 0:
            steps.append((v1 - v0) / gap)

    if not steps:
        return raw

    step_f = statistics.median(steps)
    if abs(step_f - round(step_f)) > 0.15:
        return raw  # non-integer step — not a reliable sequence
    step = int(round(step_f))
    if step == 0:
        return raw  # all same value — nothing useful to infer

    # ── anchor on the median known page (minimises propagation error) ─────────
    mid = len(sorted_known) // 2
    anchor_idx, anchor_val = sorted_known[mid]

    def _project(page_idx: int) -> str:
        val = anchor_val + (page_idx - anchor_idx) * step
        if val < 0:
            return str(val)
        return str(val).zfill(pad_width)

    # ── fill fallback pages ────────────────────────────────────────────────────
    result = list(raw)
    for i, (si, method) in enumerate(raw):
        if method == "fallback":
            inferred = _sanitize(_project(i))
            result[i] = (inferred, "sequence")

    return result


def _optimize_pdf(pdf_bytes: bytes) -> bytes:
    """Lossless rewrite: strip unreferenced objects, recompress streams/images/fonts."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True, deflate_images=True, deflate_fonts=True, clean=True)
    doc.close()
    buf.seek(0)
    optimized = buf.read()
    return optimized if len(optimized) < len(pdf_bytes) else pdf_bytes


def process_pdf(pdf_bytes: bytes, customer_code: str) -> tuple[list[dict], bytes]:
    """Split every page into its own PDF named SI_{customer_code}_{si_number}.pdf."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # ── Pass 1: extract SI from every page individually ───────────────────────
    raw: list[tuple[str, str]] = [_extract_si(doc, i) for i in range(doc.page_count)]

    # ── Pass 2: fill fallback pages via consecutive-sequence inference ─────────
    raw = _infer_from_sequence(raw)

    # ── Pass 3: build ZIP with final filenames ────────────────────────────────
    results: list[dict] = []
    used: dict[str, int] = {}
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (si, method) in enumerate(raw):
            base = si
            if base in used:
                used[base] += 1
                si = f"{base}_{used[base]}"
            else:
                used[base] = 0

            single = fitz.open()
            single.insert_pdf(doc, from_page=i, to_page=i)
            page_bytes = single.tobytes(deflate=True)
            single.close()

            filename = f"SI_{customer_code}_{si}.pdf"
            zf.writestr(filename, page_bytes)
            results.append({"page": i + 1, "si": si, "filename": filename, "method": method})

    doc.close()
    zip_buf.seek(0)
    return results, zip_buf.read()


def _cleanup_old_files() -> None:
    now = time.time()
    for p in TMP_DIR.glob("*"):
        try:
            if now - p.stat().st_mtime > RESULT_TTL:
                p.unlink()
        except OSError:
            pass


# ── Google Cloud Storage ───────────────────────────────────────────────────────

_gcs_client: gcs.Client | None = None


def _storage() -> gcs.Client:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = gcs.Client()
    return _gcs_client


def _configure_bucket() -> None:
    """Set 1-day lifecycle rule and CORS for browser PUT uploads. Non-fatal."""
    try:
        bucket = _storage().bucket(UPLOAD_BUCKET)
        bucket.reload()

        # 1-day auto-delete lifecycle
        needs_lifecycle = not any(
            r.get("action", {}).get("type") == "Delete"
            for r in bucket.lifecycle_rules
        )
        if needs_lifecycle:
            bucket.lifecycle_rules = [{"action": {"type": "Delete"}, "condition": {"age": 1}}]

        # CORS — allow browsers to PUT directly to the bucket via signed URLs
        needs_cors = not any("PUT" in c.get("method", []) for c in (bucket.cors or []))
        if needs_cors:
            bucket.cors = [{
                "origin": ["*"],
                "method": ["PUT"],
                "responseHeader": ["Content-Type"],
                "maxAgeSeconds": 3600,
            }]

        if needs_lifecycle or needs_cors:
            bucket.patch()
    except Exception:
        pass


def _generate_upload_url(blob_name: str) -> str:
    """
    Generate a V4 signed PUT URL for direct browser-to-GCS upload.

    Uses the Cloud Run service account's access token to call IAM signBlob,
    so no private-key file is required.  The service account must have
    roles/iam.serviceAccountTokenCreator (or iam.serviceAccounts.signBlob).
    """
    credentials, _ = google.auth.default()
    credentials.refresh(_google_requests.Request())
    blob = _storage().bucket(UPLOAD_BUCKET).blob(blob_name)
    return blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=15),
        method="PUT",
        content_type="application/pdf",
        service_account_email=credentials.service_account_email,
        access_token=credentials.token,
    )


def _save_result(results: list[dict], zip_bytes: bytes,
                 original_size: int, optimized_size: int,
                 original_filename: str = "upload") -> str:
    token = secrets.token_urlsafe(24)

    # Upload ZIP to GCS so any Cloud Run instance can serve the download.
    zip_blob_name = f"results/{token}.zip"
    _storage().bucket(UPLOAD_BUCKET).blob(zip_blob_name).upload_from_string(
        zip_bytes, content_type="application/zip"
    )

    # Derive the download filename: strip extension, sanitize, append suffix.
    stem = re.sub(r'[\\/*?:"<>|\r\n\t]', '_', Path(original_filename).stem).strip().strip('._') or "invoices"
    download_name = f"{stem}_invoices.zip"

    # Keep a local JSON sidecar for the result page (tiny, fast).
    (TMP_DIR / f"{token}.json").write_text(json.dumps({
        "results": results,
        "original_size": original_size,
        "optimized_size": optimized_size,
        "zip_blob": zip_blob_name,
        "download_name": download_name,
    }))
    return token


# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
# Hard cap only applies to /optimize-only; all other routes are JSON or tiny.
app.config["MAX_CONTENT_LENGTH"] = OPTIMIZE_LIMIT_MB * 1024 * 1024

# Configure bucket (lifecycle + CORS) once in background — never blocks port bind.
threading.Thread(target=_configure_bucket, daemon=True).start()


@app.errorhandler(413)
def too_large(e):
    return render_template("error_413.html", optimize_limit_mb=OPTIMIZE_LIMIT_MB), 413


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok", "zbar": _ZBAR_OK, "ml4llm": _ML4LLM_OK, "tesseract": _TESS_OK})


@app.get("/")
def index():
    return render_template("index.html")


# ── New primary upload flow (bypasses Cloud Run 32 MB ingress limit) ───────────

@app.get("/request-upload-url")
def request_upload_url():
    """
    Step 1 — return a short-lived signed GCS URL the browser can PUT to directly.
    The file bytes never pass through Cloud Run, so there is no size limit.
    """
    filename = request.args.get("filename", "upload.pdf")
    safe = re.sub(r'[^A-Za-z0-9._-]', '_', Path(filename).name)[:80]
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    blob_name = f"uploads/{ts}-{secrets.token_hex(4)}-{safe}"
    try:
        url = _generate_upload_url(blob_name)
    except Exception as exc:
        return jsonify({"error": f"Could not generate upload URL: {exc}"}), 500
    return jsonify({"url": url, "blob_name": blob_name})


@app.post("/process")
def process():
    """
    Step 2 — read the already-uploaded blob from GCS, optimise, split, return token.
    Request body is tiny JSON; only the GCS→Cloud Run download happens here.
    """
    _cleanup_old_files()
    data = request.get_json(silent=True) or {}
    blob_name = (data.get("blob_name") or "").strip()
    filename  = (data.get("filename")  or "upload.pdf").strip()

    if not blob_name:
        return jsonify({"error": "blob_name is required"}), 400

    # ── Step A: download from GCS ──────────────────────────────────────────────
    try:
        pdf_bytes     = _storage().bucket(UPLOAD_BUCKET).blob(blob_name).download_as_bytes()
        original_size = len(pdf_bytes)
    except Exception as exc:
        return jsonify({
            "error":  f"Failed to download file from cloud storage: {exc}",
            "cause":  "gcs_download",
            "hint":   "The upload may have failed or the signed URL expired. Try uploading again.",
            "detail": traceback.format_exc(),
        }), 502

    # ── Step B: lossless optimisation ─────────────────────────────────────────
    try:
        customer_code  = _resolve_customer_code(filename)
        pdf_bytes      = _optimize_pdf(pdf_bytes)
        optimized_size = len(pdf_bytes)
    except MemoryError:
        return jsonify({
            "error":  "Out of memory while optimising the PDF.",
            "cause":  "oom_optimize",
            "hint":   "The file is too large for the current Cloud Run memory limit. "
                      "Increase the instance memory to 2 GB or higher in the Cloud Run service settings.",
            "detail": traceback.format_exc(),
        }), 500
    except Exception as exc:
        return jsonify({
            "error":  f"PDF optimisation failed: {exc}",
            "cause":  "optimization",
            "hint":   "The file may be corrupt or password-protected.",
            "detail": traceback.format_exc(),
        }), 500

    # ── Step C: split pages ────────────────────────────────────────────────────
    try:
        results, zip_bytes = process_pdf(pdf_bytes, customer_code)
    except MemoryError:
        return jsonify({
            "error":  "Out of memory while splitting the PDF.",
            "cause":  "oom_split",
            "hint":   "The file is too large for the current Cloud Run memory limit. "
                      "Increase the instance memory to 2 GB or higher in the Cloud Run service settings.",
            "detail": traceback.format_exc(),
        }), 500
    except Exception as exc:
        return jsonify({
            "error":  f"PDF splitting failed: {exc}",
            "cause":  "split",
            "hint":   "The PDF may be corrupt, encrypted, or contain unsupported content.",
            "detail": traceback.format_exc(),
        }), 500

    token = _save_result(results, zip_bytes, original_size, optimized_size, filename)
    return jsonify({"token": token})


# ── Optimize-only (compress and return — no splitting) ────────────────────────

@app.post("/optimize-only")
def optimize_only():
    f = request.files.get("file")
    if not f or not f.filename:
        return render_template("error_413.html", optimize_limit_mb=OPTIMIZE_LIMIT_MB,
                               error="No file provided."), 400
    if not f.filename.lower().endswith(".pdf"):
        return render_template("error_413.html", optimize_limit_mb=OPTIMIZE_LIMIT_MB,
                               error="Only PDF files are accepted."), 400

    raw = f.read()
    optimized = _optimize_pdf(raw)
    resp = send_file(io.BytesIO(optimized), as_attachment=True,
                     download_name=f"{Path(f.filename).stem}_optimized.pdf",
                     mimetype="application/pdf")
    resp.headers["X-Original-Size"]  = str(len(raw))
    resp.headers["X-Optimized-Size"] = str(len(optimized))
    resp.headers["X-Saved-Pct"]      = str(round((len(raw) - len(optimized)) / len(raw) * 100, 1))
    return resp


# ── Result & download ─────────────────────────────────────────────────────────

@app.get("/result/<token>")
def result(token: str):
    if not re.fullmatch(r'[A-Za-z0-9_\-]{24,40}', token):
        abort(404)
    meta_path = TMP_DIR / f"{token}.json"
    if not meta_path.exists():
        abort(404)
    meta = json.loads(meta_path.read_text())
    return render_template("result.html", results=meta["results"],
                           original_size=meta["original_size"],
                           optimized_size=meta["optimized_size"], token=token)


@app.get("/download/<token>")
def download(token: str):
    if not re.fullmatch(r'[A-Za-z0-9_\-]{24,40}', token):
        abort(404)

    # Resolve the GCS blob name from the JSON sidecar.
    meta_path = TMP_DIR / f"{token}.json"
    if not meta_path.exists():
        abort(404)
    meta = json.loads(meta_path.read_text())
    zip_blob_name = meta.get("zip_blob")
    if not zip_blob_name:
        abort(404)
    download_name = meta.get("download_name", "invoices.zip")

    # Generate a short-lived signed URL and redirect the browser to it.
    try:
        credentials, _ = google.auth.default()
        credentials.refresh(_google_requests.Request())
        blob = _storage().bucket(UPLOAD_BUCKET).blob(zip_blob_name)
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=15),
            method="GET",
            response_disposition=f'attachment; filename="{download_name}"',
            service_account_email=credentials.service_account_email,
            access_token=credentials.token,
        )
    except Exception as exc:
        return jsonify({"error": f"Could not generate download URL: {exc}"}), 500

    return redirect(signed_url, code=302)


# ── Crop-region setup ─────────────────────────────────────────────────────────

@app.get("/crop-setup")
def crop_setup():
    crop = _load_crop_region()
    return render_template("crop_setup.html", current_crop=json.dumps(crop) if crop else "null")


@app.post("/render-sample-page")
def render_sample_page():
    """Accept a PDF upload, render its first page as a base64 PNG for the crop UI."""
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file provided"}), 400
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted"}), 400
    try:
        pdf_bytes = f.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]
        mat = fitz.Matrix(2, 2)  # render at 2× zoom for clarity
        pix = page.get_pixmap(matrix=mat)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
        page_w, page_h = page.rect.width, page.rect.height
        doc.close()
        return jsonify({
            "image":      f"data:image/png;base64,{img_b64}",
            "img_width":  pix.width,
            "img_height": pix.height,
            "page_width": page_w,
            "page_height": page_h,
        })
    except Exception as exc:
        return jsonify({"error": f"Failed to render page: {exc}"}), 500


@app.post("/save-crop-region")
def save_crop_region():
    """Persist crop region (as page fractions 0–1) to crop_config.json."""
    data = request.get_json(silent=True) or {}
    try:
        x1 = float(data["x1"])
        y1 = float(data["y1"])
        x2 = float(data["x2"])
        y2 = float(data["y2"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "x1, y1, x2, y2 are required floats"}), 400

    x1, y1, x2, y2 = (max(0.0, min(1.0, v)) for v in (x1, y1, x2, y2))
    if x2 <= x1 or y2 <= y1:
        return jsonify({"error": "Invalid region: x2 must be > x1 and y2 must be > y1"}), 400

    cfg = {"x1": round(x1, 6), "y1": round(y1, 6),
           "x2": round(x2, 6), "y2": round(y2, 6), "enabled": True}
    CROP_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    return jsonify({"ok": True, "region": cfg})


@app.post("/disable-crop-region")
def disable_crop_region():
    """Disable (but preserve) the saved crop region."""
    try:
        if CROP_CONFIG_PATH.exists():
            cfg = json.loads(CROP_CONFIG_PATH.read_text())
            cfg["enabled"] = False
            CROP_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass
    return jsonify({"ok": True})


@app.get("/get-crop-region")
def get_crop_region():
    """Return current crop region config (or empty object if none)."""
    crop = _load_crop_region()
    return jsonify(crop or {})


# ── Customer-code configuration ───────────────────────────────────────────────

@app.get("/customer-codes")
def customer_codes_page():
    codes = _load_customer_codes()
    return render_template("customer_codes.html",
                           codes=codes,
                           premade=_PREMADE_CODES)


@app.post("/customer-codes/upsert")
def customer_codes_upsert():
    """Add or update a keyword→code mapping."""
    data = request.get_json(silent=True) or {}
    keyword = str(data.get("keyword", "")).strip().upper()
    code    = str(data.get("code", "")).strip()
    if not keyword or not code:
        return jsonify({"error": "keyword and code are required"}), 400
    if not re.fullmatch(r'[A-Z0-9 _\-]{1,40}', keyword):
        return jsonify({"error": "keyword must be 1–40 alphanumeric characters"}), 400
    if not re.fullmatch(r'[A-Za-z0-9\-]{1,40}', code):
        return jsonify({"error": "code must be 1–40 alphanumeric characters"}), 400
    codes = _load_customer_codes()
    codes[keyword] = code
    _save_customer_codes(codes)
    return jsonify({"ok": True, "keyword": keyword, "code": code})


@app.post("/customer-codes/delete")
def customer_codes_delete():
    """Remove a keyword→code mapping."""
    data = request.get_json(silent=True) or {}
    keyword = str(data.get("keyword", "")).strip().upper()
    if not keyword:
        return jsonify({"error": "keyword is required"}), 400
    codes = _load_customer_codes()
    if keyword not in codes:
        return jsonify({"error": "keyword not found"}), 404
    del codes[keyword]
    _save_customer_codes(codes)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
