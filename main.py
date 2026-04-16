from __future__ import annotations
import datetime, io, json, os, re, secrets, tempfile, threading, time, zipfile
from pathlib import Path
from flask import Flask, abort, redirect, render_template, request, send_file, url_for

import fitz  # PyMuPDF
from google.cloud import storage as gcs
from PIL import Image
from pyzbar import pyzbar as zbar

MAX_MB = 500
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

    # 4. Barcode decode — render bottom half at 2× scale using pdfium
    try:
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(0, h * 0.50, w, h), colorspace=fitz.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        codes = zbar.decode(img)
        if not codes:
            # Retry on full page at 1.5×
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), colorspace=fitz.csGRAY)
            img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
            codes = zbar.decode(img)
        for code in codes:
            data = code.data.decode("utf-8", errors="ignore").strip()
            if data:
                return _sanitize(data), "barcode"
    except Exception:
        pass

    return f"page_{idx + 1:04d}", "fallback"


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
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

# Run lifecycle setup in a daemon thread so it never delays gunicorn's port bind.
# Cloud Run kills the container if the port isn't ready within the startup timeout;
# moving this off the critical path ensures the server is always reachable first.
threading.Thread(target=_ensure_lifecycle, daemon=True).start()


@app.get("/")
def index():
    return render_template("index.html", max_mb=MAX_MB)


@app.post("/upload")
def upload():
    _cleanup_old_files()
    f = request.files.get("file")
    if not f or not f.filename:
        return render_template("index.html", error="No file selected.", max_mb=MAX_MB)
    if not f.filename.lower().endswith(".pdf"):
        return render_template("index.html", error="Only PDF files are accepted.", max_mb=MAX_MB)

    try:
        pdf_bytes = f.read()
        customer_code = _resolve_customer_code(f.filename)

        # Persist the original upload to GCS (auto-deleted after 1 day).
        _upload_to_gcs(pdf_bytes, f.filename)

        # Process from the same in-memory bytes — no second GCS round-trip needed.
        results, zip_bytes = process_pdf(pdf_bytes, customer_code)
    except Exception as exc:
        return render_template("index.html", error=f"Processing failed: {exc}", max_mb=MAX_MB)

    token = secrets.token_urlsafe(24)
    (TMP_DIR / f"{token}.zip").write_bytes(zip_bytes)
    (TMP_DIR / f"{token}.json").write_text(json.dumps(results))

    return redirect(url_for("result", token=token))


@app.get("/result/<token>")
def result(token: str):
    if not re.fullmatch(r'[A-Za-z0-9_\-]{24,40}', token):
        abort(404)
    meta = TMP_DIR / f"{token}.json"
    if not meta.exists():
        abort(404)
    results = json.loads(meta.read_text())
    return render_template("result.html", results=results, token=token)


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
