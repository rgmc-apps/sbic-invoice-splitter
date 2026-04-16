from __future__ import annotations
import datetime, io, json, os, re, secrets, tempfile, threading, time, zipfile
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

UPLOAD_BUCKET     = os.environ.get("UPLOAD_BUCKET", "sbic-splitter-uploads")
OPTIMIZE_LIMIT_MB = int(os.environ.get("OPTIMIZE_LIMIT_MB", 2048))  # Flask hard cap for /optimize-only
TMP_DIR           = Path(tempfile.gettempdir()) / "inv_sep"
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
_SI_RE = [
    re.compile(r'\bS\.?I\.?\s*(?:No\.?|Num(?:ber)?|#|:)?\s*[:\-]?\s*([A-Z0-9]{4,}(?:[-]\d+)*)', re.I),
    re.compile(r'(?:Sales\s+Invoice|Invoice)\s*(?:No\.?|#|:)?\s*([A-Z0-9]{5,20})', re.I),
    re.compile(r'\b(SI[-]\d{4,}(?:[-]\d+)*)\b', re.I),
    re.compile(r'\b(\d{7,13})\b'),  # long bare numeric — last resort
]


def _sanitize(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|\r\n\t]', '_', name).strip().strip('._')[:80] or "unknown"


def _find_si(text: str) -> str | None:
    for pat in _SI_RE:
        m = pat.search(text)
        if m:
            v = m.group(1).strip()
            return _sanitize(v) if v else None
    return None


def _extract_si(doc: fitz.Document, idx: int) -> tuple[str, str]:
    page = doc[idx]
    h, w = page.rect.height, page.rect.width

    si = _find_si(page.get_text("text", clip=fitz.Rect(0, h * 0.60, w, h)))
    if si:
        return si, "text-bottom"

    si = _find_si(page.get_text("text", clip=fitz.Rect(w * 0.50, 0, w, h)))
    if si:
        return si, "text-right"

    si = _find_si(page.get_text("text"))
    if si:
        return si, "text-full"

    if _ZBAR_OK:
        try:
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(0, h * 0.50, w, h), colorspace=fitz.csGRAY)
            img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
            codes = _pyzbar.decode(img)
            if not codes:
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
    results: list[dict] = []
    used: dict[str, int] = {}
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(doc.page_count):
            si, method = _extract_si(doc, i)
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
                 original_size: int, optimized_size: int) -> str:
    token = secrets.token_urlsafe(24)
    (TMP_DIR / f"{token}.zip").write_bytes(zip_bytes)
    (TMP_DIR / f"{token}.json").write_text(json.dumps({
        "results": results,
        "original_size": original_size,
        "optimized_size": optimized_size,
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
    return jsonify({"status": "ok", "zbar": _ZBAR_OK})


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

    try:
        pdf_bytes     = _storage().bucket(UPLOAD_BUCKET).blob(blob_name).download_as_bytes()
        original_size = len(pdf_bytes)
        customer_code = _resolve_customer_code(filename)
        pdf_bytes     = _optimize_pdf(pdf_bytes)
        optimized_size = len(pdf_bytes)
        results, zip_bytes = process_pdf(pdf_bytes, customer_code)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    token = _save_result(results, zip_bytes, original_size, optimized_size)
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
    zip_path = TMP_DIR / f"{token}.zip"
    if not zip_path.exists():
        abort(404)
    return send_file(zip_path, as_attachment=True,
                     download_name="invoices.zip", mimetype="application/zip")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
