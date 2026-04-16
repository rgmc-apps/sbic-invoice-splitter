from __future__ import annotations
import io, json, os, re, secrets, tempfile, time, zipfile
from pathlib import Path
from flask import Flask, abort, redirect, render_template, request, send_file, url_for

import fitz  # PyMuPDF
from PIL import Image
from pyzbar import pyzbar as zbar

MAX_MB = 500
TMP_DIR = Path(tempfile.gettempdir()) / "inv_sep"
TMP_DIR.mkdir(exist_ok=True)
RESULT_TTL = 3600  # seconds — temp files expire after 1 hour

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


def process_pdf(pdf_bytes: bytes) -> tuple[list[dict], bytes]:
    """Split every page into its own PDF, name each by SI number, return ZIP bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results: list[dict] = []
    used: dict[str, int] = {}
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(doc.page_count):
            si, method = _extract_si(doc, i)

            # Deduplicate: append _2, _3 … for repeated SI values
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

            filename = f"{si}.pdf"
            zf.writestr(filename, page_bytes)
            results.append({"page": i + 1, "si": si, "filename": filename, "method": method})

    doc.close()
    zip_buf.seek(0)
    return results, zip_buf.read()


def _cleanup_old_files() -> None:
    """Remove temp files older than RESULT_TTL seconds."""
    now = time.time()
    for p in TMP_DIR.glob("*"):
        try:
            if now - p.stat().st_mtime > RESULT_TTL:
                p.unlink()
        except OSError:
            pass


# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


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
        results, zip_bytes = process_pdf(pdf_bytes)
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
