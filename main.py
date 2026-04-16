from __future__ import annotations
import datetime, io, json, os, re, secrets, tempfile, threading, time, zipfile
from pathlib import Path
from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for

import fitz  # PyMuPDF
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

UPLOAD_LIMIT_MB  = int(os.environ.get("UPLOAD_LIMIT_MB",  500))   # soft cap for normal uploads
OPTIMIZE_LIMIT_MB = int(os.environ.get("OPTIMIZE_LIMIT_MB", 2048))  # hard cap for the optimize-only endpoint
MAX_MB = UPLOAD_LIMIT_MB   # kept for template compatibility
UPLOAD_BUCKET = os.environ.get("UPLOAD_BUCKET", "sbic-splitter-uploads")
TMP_DIR = Path(tempfile.gettempdir()) / "inv_sep"
TMP_DIR.mkdir(exist_ok=True)
RESULT_TTL = 3600  # seconds — local temp files expire after 1 hour

# ── Customer-code mapping ──────────────────────────────────────────────────────
# Maps a keyword found in the uploaded filename (case-insensitive) to the
# customer's SI code used in the output filename prefix.
# To add a new customer: insert a new entry below — no other code changes needed.
_CUSTOMER_CODES: dict[str, str] = {
    "MTC":  "240762",
    "SBIC": "190275",
}


def _resolve_customer_code(filename: str) -> str:
    """Return the SI customer code whose key appears in *filename* (stem only).
    Falls back to 'UNKNOWN' if no mapping key is found."""
    stem = Path(filename).stem.upper()
    for key, code in _CUSTOMER_CODES.items():
        if key in stem:
            return code
    return "UNKNOWN"


# ── SI-number extraction patterns ─────────────────────────────────────────────
# Philippine sales invoices typically show "SI No.: XXXXXXXX" or a bare number
# printed under a barcode near the lower-right corner of the page.
_SI_RE = [
    re.compile(r'\bS\.?I\.?\s*(?:No\.?|Num(?:ber)?|#|:)?\s*[:\-]?\s*([A-Z0-9]{4,}(?:[-]\d+)*)', re.I),
    re.compile(r'(?:Sales\s+Invoice|Invoice)\s*(?:No\.?|#|:)?\s*([A-Z0-9]{5,20})', re.I),
    re.compile(r'\b(SI[-]\d{4,}(?:[-]\d+)*)\b', re.I),
    re.compile(r'\b(\d{7,13})\b'),  # long bare numeric — last resort
]


def _sanitize(name: str) -> str:
    """Strip characters illegal in filenames and cap length."""
    return re.sub(r'[\\/*?:"<>|\r\n\t]', '_', name).strip().strip('._')[:80] or "unknown"


def _find_si(text: str) -> str | None:
    for pat in _SI_RE:
        m = pat.search(text)
        if m:
            v = m.group(1).strip()
            return _sanitize(v) if v else None
    return None


def _extract_si(doc: fitz.Document, idx: int) -> tuple[str, str]:
    """Return (si_number, method_used) for the given page index."""
    page = doc[idx]
    h = page.rect.height
    w = page.rect.width

    # 1. Text — bottom 40 % of page (SI number is usually here)
    si = _find_si(page.get_text("text", clip=fitz.Rect(0, h * 0.60, w, h)))
    if si:
        return si, "text-bottom"

    # 2. Text — right half (lower-right corner fallback)
    si = _find_si(page.get_text("text", clip=fitz.Rect(w * 0.50, 0, w, h)))
    if si:
        return si, "text-right"

    # 3. Full-page text
    si = _find_si(page.get_text("text"))
    if si:
        return si, "text-full"

    # 4. Barcode decode — render bottom half at 2× scale (only when libzbar0 is present)
    if _ZBAR_OK:
        try:
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(0, h * 0.50, w, h), colorspace=fitz.csGRAY)
            img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
            codes = _pyzbar.decode(img)
            if not codes:
                # Retry on full page at 1.5×
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), colorspace=fitz.csGRAY)
                img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
                codes = _pyzbar.decode(img)
            for code in codes:
                data = code.data.decode("utf-8", errors="ignore").strip()
                if data:
                    return _sanitize(data), "barcode"
        except Exception:
            pass

    return f"page_{idx + 1:04d}", "fallback"


def _optimize_pdf(pdf_bytes: bytes) -> bytes:
    """
    Rewrite the PDF with PyMuPDF's maximum lossless compression:
      garbage=4        — remove all unreferenced objects and deduplicate streams
      deflate=True     — zlib-compress all compressible streams
      deflate_images   — compress embedded image streams
      deflate_fonts    — compress embedded font streams
      clean=True       — normalise and sanitise content streams

    This is lossless — visual quality is unchanged.
    Typical reduction: 20–60 % for scanned PDFs, 10–30 % for text PDFs.
    Returns the original bytes unchanged if optimisation produces a larger file.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    buf = io.BytesIO()
    doc.save(
        buf,
        garbage=4,
        deflate=True,
        deflate_images=True,
        deflate_fonts=True,
        clean=True,
    )
    doc.close()
    buf.seek(0)
    optimized = buf.read()
    return optimized if len(optimized) < len(pdf_bytes) else pdf_bytes


def process_pdf(pdf_bytes: bytes, customer_code: str) -> tuple[list[dict], bytes]:
    """Split every page into its own PDF named SI_{customer_code}_{si_number}.pdf,
    return (results_list, zip_bytes)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results: list[dict] = []
    used: dict[str, int] = {}
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(doc.page_count):
            si, method = _extract_si(doc, i)

            # Deduplicate: append _2, _3 … for repeated SI numbers on different pages
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

            # Output format: SI_{customer_code}_{si_number}.pdf
            filename = f"SI_{customer_code}_{si}.pdf"
            zf.writestr(filename, page_bytes)
            results.append({"page": i + 1, "si": si, "filename": filename, "method": method})

    doc.close()
    zip_buf.seek(0)
    return results, zip_buf.read()


def _cleanup_old_files() -> None:
    """Remove local temp files older than RESULT_TTL seconds."""
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


def _ensure_lifecycle() -> None:
    """
    Apply a 1-day auto-delete lifecycle rule to UPLOAD_BUCKET if one is not
    already present. 1 day is the minimum age GCS Object Lifecycle supports.
    Called once at startup; failures are non-fatal.
    """
    try:
        bucket = _storage().bucket(UPLOAD_BUCKET)
        bucket.reload()
        for rule in bucket.lifecycle_rules:
            if rule.get("action", {}).get("type") == "Delete":
                return  # a delete rule already exists — leave it alone
        bucket.lifecycle_rules = [
            {"action": {"type": "Delete"}, "condition": {"age": 1}}
        ]
        bucket.patch()
    except Exception:
        pass  # non-fatal: bucket may not exist yet, or ADC is unavailable locally


def _upload_to_gcs(pdf_bytes: bytes, original_filename: str) -> str:
    """Upload raw PDF bytes to GCS and return the blob path."""
    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r'[^A-Za-z0-9._-]', '_', Path(original_filename).name)[:80]
    blob_name = f"uploads/{ts}-{secrets.token_hex(4)}-{safe}"
    _storage().bucket(UPLOAD_BUCKET).blob(blob_name).upload_from_string(
        pdf_bytes, content_type="application/pdf"
    )
    return blob_name


# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
# Set the global hard limit to OPTIMIZE_LIMIT_MB so the /optimize-only
# endpoint can accept files larger than the normal upload cap.
# The /upload route enforces UPLOAD_LIMIT_MB itself.
app.config["MAX_CONTENT_LENGTH"] = OPTIMIZE_LIMIT_MB * 1024 * 1024

# Run lifecycle setup in a daemon thread so it never delays gunicorn's port bind.
# Cloud Run kills the container if the port isn't ready within the startup timeout;
# moving this off the critical path ensures the server is always reachable first.
threading.Thread(target=_ensure_lifecycle, daemon=True).start()


@app.errorhandler(413)
def too_large(e):
    return render_template("error_413.html", upload_limit_mb=UPLOAD_LIMIT_MB,
                           optimize_limit_mb=OPTIMIZE_LIMIT_MB), 413


@app.get("/healthz")
def healthz():
    """Lightweight health check used by Cloud Run startup/liveness probes."""
    return jsonify({"status": "ok", "zbar": _ZBAR_OK})


@app.get("/")
def index():
    return render_template("index.html", max_mb=MAX_MB, upload_limit_mb=UPLOAD_LIMIT_MB)


@app.post("/optimize-only")
def optimize_only():
    """Accept a PDF (up to OPTIMIZE_LIMIT_MB), compress it, return the optimized file."""
    f = request.files.get("file")
    if not f or not f.filename:
        return render_template("error_413.html", upload_limit_mb=UPLOAD_LIMIT_MB,
                               optimize_limit_mb=OPTIMIZE_LIMIT_MB,
                               error="No file provided."), 400
    if not f.filename.lower().endswith(".pdf"):
        return render_template("error_413.html", upload_limit_mb=UPLOAD_LIMIT_MB,
                               optimize_limit_mb=OPTIMIZE_LIMIT_MB,
                               error="Only PDF files are accepted."), 400

    raw = f.read()
    optimized = _optimize_pdf(raw)
    saved_pct = round((len(raw) - len(optimized)) / len(raw) * 100, 1)

    stem = Path(f.filename).stem
    dl_name = f"{stem}_optimized.pdf"

    resp = send_file(
        io.BytesIO(optimized),
        as_attachment=True,
        download_name=dl_name,
        mimetype="application/pdf",
    )
    # Surface compression stats in response headers for the JS layer
    resp.headers["X-Original-Size"]  = str(len(raw))
    resp.headers["X-Optimized-Size"] = str(len(optimized))
    resp.headers["X-Saved-Pct"]      = str(saved_pct)
    return resp


@app.post("/upload")
def upload():
    _cleanup_old_files()
    f = request.files.get("file")
    if not f or not f.filename:
        return render_template("index.html", error="No file selected.",
                               max_mb=MAX_MB, upload_limit_mb=UPLOAD_LIMIT_MB)
    if not f.filename.lower().endswith(".pdf"):
        return render_template("index.html", error="Only PDF files are accepted.",
                               max_mb=MAX_MB, upload_limit_mb=UPLOAD_LIMIT_MB)

    # Manual soft cap — files between UPLOAD_LIMIT_MB and OPTIMIZE_LIMIT_MB reach
    # this route because the global Flask limit is OPTIMIZE_LIMIT_MB.
    content_len = request.content_length or 0
    if content_len > UPLOAD_LIMIT_MB * 1024 * 1024:
        return render_template("error_413.html", upload_limit_mb=UPLOAD_LIMIT_MB,
                               optimize_limit_mb=OPTIMIZE_LIMIT_MB), 413

    try:
        raw_bytes = f.read()
        original_size = len(raw_bytes)
        customer_code = _resolve_customer_code(f.filename)

        # Optimise before storing and processing — reduces GCS cost and speeds up splitting.
        pdf_bytes = _optimize_pdf(raw_bytes)
        optimized_size = len(pdf_bytes)

        # Persist the optimised PDF to GCS (auto-deleted after 1 day).
        _upload_to_gcs(pdf_bytes, f.filename)

        # Process from the same in-memory bytes — no second GCS round-trip needed.
        results, zip_bytes = process_pdf(pdf_bytes, customer_code)
    except Exception as exc:
        return render_template("index.html", error=f"Processing failed: {exc}", max_mb=MAX_MB)

    token = secrets.token_urlsafe(24)
    (TMP_DIR / f"{token}.zip").write_bytes(zip_bytes)
    meta = {
        "results": results,
        "original_size": original_size,
        "optimized_size": optimized_size,
    }
    (TMP_DIR / f"{token}.json").write_text(json.dumps(meta))

    return redirect(url_for("result", token=token))


@app.get("/result/<token>")
def result(token: str):
    if not re.fullmatch(r'[A-Za-z0-9_\-]{24,40}', token):
        abort(404)
    meta_path = TMP_DIR / f"{token}.json"
    if not meta_path.exists():
        abort(404)
    meta = json.loads(meta_path.read_text())
    return render_template(
        "result.html",
        results=meta["results"],
        original_size=meta["original_size"],
        optimized_size=meta["optimized_size"],
        token=token,
    )


@app.get("/download/<token>")
def download(token: str):
    if not re.fullmatch(r'[A-Za-z0-9_\-]{24,40}', token):
        abort(404)
    zip_path = TMP_DIR / f"{token}.zip"
    if not zip_path.exists():
        abort(404)
    return send_file(
        zip_path,
        as_attachment=True,
        download_name="invoices.zip",
        mimetype="application/zip",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
