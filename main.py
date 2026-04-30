from __future__ import annotations
import base64, concurrent.futures, datetime, io, json, os, re, secrets, statistics, tempfile, threading, time, traceback, zipfile
from pathlib import Path
from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for

import fitz  # PyMuPDF
import google.auth
import google.auth.transport.requests as _google_requests
from google.cloud import storage as gcs
from PIL import Image, ImageEnhance, ImageFilter

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
JOB_TTL    = 3600  # seconds — in-memory job entries expire after 1 hour

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_MAX_WORKERS = int(os.environ.get("MAX_WORKERS", min(os.cpu_count() or 4, 8)))

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
    "SVI":  "190275"
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


# ── OCR digit-normalization ────────────────────────────────────────────────────
# Tesseract often mistakes digits for visually similar letters (e.g. "5" → "S",
# "0" → "O", "8" → "B").  We correct these only inside sequences that are
# already 4+ characters of digits/digit-like letters, so short label words
# ("SI", "No") are never touched.
_OCR_DIGIT_MAP = str.maketrans('SOsoQIlBGZ', '5005001862')


def _ocr_clean(text: str) -> str:
    """Fix letter-for-digit OCR mistakes in numeric-looking sequences."""
    def _fix(m: re.Match) -> str:
        return m.group().translate(_OCR_DIGIT_MAP)
    return re.sub(r'[0-9SOsoQIlBGZ]{4,}', _fix, text)


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


# ocr-full retry schedule: (psm, zoom, bin_threshold)
# Attempt 0 (primary)  : PSM 11 sparse-text, 3× zoom, threshold 140 — standard full-page
# Retry 1              : PSM 6  uniform-block, 4× zoom, threshold 128 — higher-res, softer binarize
# Retry 2              : PSM 3  auto-segment,  3× zoom, threshold 150 — alternative segmentation
_OCR_FULL_ATTEMPTS: list[tuple[int, int, int]] = [
    (11, 3, 140),
    (6,  4, 128),
    (3,  3, 150),
]


def _ocr_text(page: fitz.Page, clip: fitz.Rect | None = None, psm: int = 11,
              zoom: int | None = None, bin_threshold: int = 140,
              _cached_pix=None) -> str:
    """Render a page region to pixels and run Tesseract OCR on it.

    Bypasses the PDF font-encoding layer entirely — the only reliable approach
    when invoices use custom glyph-to-Unicode mappings that cause get_text()
    to return garbled output (e.g. "U:rrr&?8s" instead of "Nº 51285").

    psm controls Tesseract's page-segmentation mode:
      6 = single uniform text block (best for small crop regions)
     11 = sparse text, find as much as possible (best for large/full-page regions)

    zoom overrides the default (4× for clip regions, 3× for full-page).
    bin_threshold controls binarization: pixels above it become white (background).
    _cached_pix accepts a pre-rendered fitz.Pixmap to skip the render step entirely.

    Image pipeline (applied before Tesseract):
      1. 4× zoom for clip regions, 3× for full-page — more pixels let Tesseract
         distinguish digit shapes that look similar at low resolution (9 vs 6,
         5 vs 8, etc.).
      2. Unsharp mask — sharpens edges with controlled radius; better than the
         basic SHARPEN filter at preserving stroke details on small numerals.
      3. Contrast ×2 — darkens ink, lightens background.
      4. Binarization — converts the gray-gradient result to pure black/white.
         Mid-gray pixels are what cause 9↔6 and 5↔8 digit confusion; a hard
         threshold eliminates them.
    """
    if not _TESS_OK:
        return ""
    try:
        effective_zoom = zoom if zoom is not None else (4 if clip is not None else 3)
        if _cached_pix is not None:
            pix = _cached_pix
        else:
            # Higher zoom for focused clip regions — descenders ("9") and ascenders
            # ("6") become clearly distinct at 4× vs ambiguous at 3×.
            mat = fitz.Matrix(effective_zoom, effective_zoom)
            px_kwargs: dict = {"matrix": mat, "colorspace": fitz.csGRAY}
            if clip is not None:
                px_kwargs["clip"] = clip
            pix = page.get_pixmap(**px_kwargs)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)

        # Unsharp mask: radius=2 sharpens thin digit strokes without
        # over-amplifying noise; percent=150 is a moderate gain.
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        img = ImageEnhance.Contrast(img).enhance(2.0)

        # Binarize at bin_threshold: pixels above → white (background), below → black (ink).
        # Default 140 (slightly above 128) handles light-gray invoice backgrounds
        # without swallowing thin stroke tails on digits like "9".
        thr = bin_threshold
        img = img.point(lambda p: 255 if p > thr else 0)

        # load_system_dawg=0 / load_freq_dawg=0: disable word-frequency and
        # dictionary lookups so Tesseract relies purely on character shape
        # rather than word-completion heuristics that can corrupt lone numerals.
        config = f"--oem 1 --psm {psm} -c load_system_dawg=0 -c load_freq_dawg=0"
        raw = _pytesseract.image_to_string(img, config=config)

        # Correct common letter-for-digit OCR mistakes (e.g. "S"→"5", "O"→"0")
        return _ocr_clean(raw)
    except Exception:
        return ""


def _infer_from_sequence(raw: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Detect a consecutive numeric pattern across successfully-extracted pages,
    fill in 'fallback' pages, and correct OCR digit misreads on extracted pages.

    Algorithm:
      1. Collect all pages whose SI is a pure integer string and method != 'fallback'.
      2. For every consecutive pair of known pages compute (value_diff / page_gap).
      3. Take the median step — robust against a few OCR mis-reads.
      4. For each known page compute its "virtual page-0" value (v − step × idx).
         Take the mode across all known pages — this is the true sequence base
         even when some pages have digit-confusion OCR errors (e.g. "5"→"8"),
         because those erroneous pages project to a different v0 than the majority.
      5. Fill 'fallback' pages by projecting from the anchor  → method "sequence".
      6. Validate every OCR-extracted value against the sequence prediction.
         If the value has the same digit count and differs in ≤ 2 positions
         (digit-level Hamming distance), it is a digit-confusion OCR error —
         replace with the predicted value  → method "corrected".

    Returns the same-length list with inferred/corrected values where applicable.
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

    # ── anchor via mode of virtual-page-0 values ──────────────────────────────
    # Each correctly-read page gives: v0 = value − step × page_idx.
    # Pages with OCR digit errors produce a different v0 (minority).
    # The most common v0 is the true base of the sequence.
    v0_counts: dict[int, int] = {}
    for idx, val in known.items():
        v0 = val - step * idx
        v0_counts[v0] = v0_counts.get(v0, 0) + 1
    best_v0 = max(v0_counts, key=lambda k: v0_counts[k])

    def _project(page_idx: int) -> str:
        val = best_v0 + step * page_idx
        if val < 0:
            return str(val)
        return str(val).zfill(pad_width)

    # ── fill fallback pages ────────────────────────────────────────────────────
    result = list(raw)
    for i, (si, method) in enumerate(raw):
        if method == "fallback":
            result[i] = (_sanitize(_project(i)), "sequence")

    # ── correct OCR digit misreads against the sequence ───────────────────────
    # If an extracted value has the same digit count as the prediction and
    # differs in at most 2 digit positions, treat it as a digit-confusion error
    # (e.g. "5" misread as "8") and replace with the predicted value.
    for i, (si, method) in enumerate(result):
        if method in ("sequence", "corrected"):
            continue  # already filled or corrected
        predicted = _project(i)
        if len(si) != len(predicted):
            continue  # different digit count — leave untouched
        mismatches = sum(a != b for a, b in zip(si, predicted))
        if 0 < mismatches <= 2:
            result[i] = (predicted, "corrected")

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

    # ── Phase 1: single-pass pre-extraction (one full-PDF open) ───────────────
    # Opens the full document exactly once to pre-extract per-page bytes and
    # all native text clips. Workers in Phase 2 then open only tiny 1-page PDFs
    # (or skip PDF opens entirely when text methods succeed), eliminating the
    # N × full-PDF-parse overhead that was the primary bottleneck for large files.
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    n    = doc.page_count
    crop = _load_crop_region()

    page_bytes_list: list[bytes]               = []
    page_dims:       list[tuple[float, float]] = []
    page_fast_text:  list[str]                 = []
    page_bot_text:   list[str]                 = []
    page_right_text: list[str]                 = []
    page_crop_text:  list[str]                 = []

    for i in range(n):
        pg = doc[i]
        h, w = pg.rect.height, pg.rect.width
        page_dims.append((h, w))
        page_fast_text.append(pg.get_text("text"))
        page_bot_text.append(pg.get_text("text", clip=fitz.Rect(0, h * 0.60, w, h)))
        page_right_text.append(pg.get_text("text", clip=fitz.Rect(w * 0.50, 0, w, h)))
        page_crop_text.append(
            pg.get_text("text", clip=fitz.Rect(
                w * crop["x1"], h * crop["y1"], w * crop["x2"], h * crop["y2"]
            )) if crop else ""
        )
        single = fitz.open()
        single.insert_pdf(doc, from_page=i, to_page=i)
        page_bytes_list.append(single.tobytes(deflate=True))
        single.close()

    doc.close()

    # ── Phase 2: parallel SI extraction using pre-extracted data ──────────────
    # Text methods use pre-extracted strings — zero PDF opens.
    # OCR and pymupdf4llm open a single-page PDF (much cheaper than the full doc).
    # All OCR fallbacks share one single-page PDF open per worker invocation.
    raw: list[tuple[str, str]] = [("", "")] * n

    def _worker(i: int) -> tuple[int, tuple[str, str]]:
        h, w = page_dims[i]

        # Open the single-page PDF once; reused by all OCR priorities in this worker.
        # Skipped entirely when Tesseract is unavailable.
        ocr_doc = fitz.open(stream=page_bytes_list[i], filetype="pdf") if _TESS_OK else None
        try:
            p = ocr_doc[0] if ocr_doc is not None else None

            # Priority 1: ocr-full — preferred result; 1 try + 2 retries with varied params.
            # OCR is first because custom glyph-to-Unicode mappings in these invoices
            # corrupt the native text layer; reading pixels bypasses that entirely.
            # Pixmaps are cached per zoom level to avoid re-rendering identical bitmaps.
            if _TESS_OK:
                _pix_cache: dict[int, object] = {}
                for psm, zm, thr in _OCR_FULL_ATTEMPTS:
                    if zm not in _pix_cache:
                        _pix_cache[zm] = p.get_pixmap(
                            matrix=fitz.Matrix(zm, zm), colorspace=fitz.csGRAY
                        )
                    si = _find_si(_ocr_text(p, psm=psm, zoom=zm, bin_threshold=thr,
                                            _cached_pix=_pix_cache[zm]))
                    if si:
                        return i, (si, "ocr-full")

            # Priority 2: crop region (ocr-crop / text-crop)
            if crop:
                if _TESS_OK:
                    clip_r = fitz.Rect(w * crop["x1"], h * crop["y1"], w * crop["x2"], h * crop["y2"])
                    si = _find_si(_ocr_text(p, clip_r, psm=6))
                    if si:
                        return i, (si, "ocr-crop")
                si = _find_si(page_crop_text[i])
                if si:
                    return i, (si, "text-crop")

            # Priorities 3–5: native text (pre-extracted, no PDF open needed)
            si = _find_si(page_bot_text[i])
            if si:
                return i, (si, "text-bottom")

            si = _find_si(page_right_text[i])
            if si:
                return i, (si, "text-right")

            si = _find_si(page_fast_text[i])
            if si:
                return i, (si, "text-full")

            # Priority 6: pymupdf4llm layout pass (needs its own doc open — may mutate state)
            if _ML4LLM_OK:
                d = fitz.open(stream=page_bytes_list[i], filetype="pdf")
                try:
                    md = _pymupdf4llm.to_markdown(d, pages=[0])
                    si = _find_si(_strip_markdown(md))
                except Exception:
                    si = None
                finally:
                    d.close()
                if si:
                    return i, (si, "text-full")

            # Priorities 7–8: targeted OCR sub-regions (reuses ocr_doc)
            if _TESS_OK:
                si = _find_si(_ocr_text(p, fitz.Rect(0, h * 0.60, w, h)))
                if si:
                    return i, (si, "ocr-bottom")
                si = _find_si(_ocr_text(p, fitz.Rect(w * 0.50, 0, w, h)))
                if si:
                    return i, (si, "ocr-right")

            return i, (f"page_{i + 1:04d}", "fallback")
        finally:
            if ocr_doc is not None:
                ocr_doc.close()

    workers = min(_MAX_WORKERS, n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_worker, i): i for i in range(n)}
        for fut in concurrent.futures.as_completed(futs):
            idx, result = fut.result()
            raw[idx] = result

    # ── Pass 2: fill fallback pages via consecutive-sequence inference ─────────
    raw = _infer_from_sequence(raw)

    # ── Pass 3: assemble ZIP from already-built page bytes ────────────────────
    results: list[dict] = []
    used:    dict[str, int] = {}
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_STORED) as zf:
        for i, (si, method) in enumerate(raw):
            base = si
            if base in used:
                used[base] += 1
                si = f"{base}_{used[base]}"
            else:
                used[base] = 0
            filename = f"SI_{customer_code}_{si}.pdf"
            zf.writestr(filename, page_bytes_list[i])
            results.append({"page": i + 1, "si": si, "filename": filename, "method": method,
                            "customer_code": customer_code})

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
    with _jobs_lock:
        stale = [jid for jid, j in _jobs.items() if now - j.get("created", now) > JOB_TTL]
        for jid in stale:
            del _jobs[jid]


def _process_job(job_id: str, blob_name: str, filename: str) -> None:
    """Background worker: download → optimise → split → save.  Updates _jobs[job_id]."""
    def _fail(error: str, cause: str, hint: str) -> None:
        with _jobs_lock:
            _jobs[job_id].update({
                "status": "error", "error": error, "cause": cause,
                "hint": hint, "detail": traceback.format_exc(),
            })

    try:
        try:
            pdf_bytes     = _storage().bucket(UPLOAD_BUCKET).blob(blob_name).download_as_bytes()
            original_size = len(pdf_bytes)
        except Exception as exc:
            _fail(
                f"Failed to download file from cloud storage: {exc}", "gcs_download",
                "The upload may have failed or the signed URL expired. Try uploading again.",
            )
            return

        try:
            customer_code  = _resolve_customer_code(filename)
            pdf_bytes      = _optimize_pdf(pdf_bytes)
            optimized_size = len(pdf_bytes)
        except MemoryError:
            _fail(
                "Out of memory while optimising the PDF.", "oom_optimize",
                "Increase the Cloud Run instance memory to 2 GB or higher.",
            )
            return
        except Exception as exc:
            _fail(
                f"PDF optimisation failed: {exc}", "optimization",
                "The file may be corrupt or password-protected.",
            )
            return

        try:
            results, zip_bytes = process_pdf(pdf_bytes, customer_code)
        except MemoryError:
            _fail(
                "Out of memory while splitting the PDF.", "oom_split",
                "Increase the Cloud Run instance memory to 2 GB or higher.",
            )
            return
        except Exception as exc:
            _fail(
                f"PDF splitting failed: {exc}", "split",
                "The PDF may be corrupt, encrypted, or contain unsupported content.",
            )
            return

        token = _save_result(results, zip_bytes, original_size, optimized_size, filename)
        with _jobs_lock:
            _jobs[job_id].update({"status": "done", "token": token})

    except Exception as exc:
        _fail(f"Unexpected error: {exc}", "unexpected", "Please try again.")


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

    zip_blob_name = f"results/{token}.zip"
    stem = re.sub(r'[\\/*?:"<>|\r\n\t]', '_', Path(original_filename).stem).strip().strip('._') or "invoices"
    download_name = f"{stem}_invoices.zip"

    meta = {
        "results": results,
        "original_size": original_size,
        "optimized_size": optimized_size,
        "zip_blob": zip_blob_name,
        "download_name": download_name,
    }
    meta_json = json.dumps(meta)

    # Upload ZIP and JSON sidecar to GCS in parallel — both are independent.
    bucket = _storage().bucket(UPLOAD_BUCKET)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        zip_fut  = pool.submit(bucket.blob(zip_blob_name).upload_from_string,
                               zip_bytes, "application/zip")
        json_fut = pool.submit(bucket.blob(f"results/{token}.json").upload_from_string,
                               meta_json, "application/json")
        zip_fut.result()
        json_fut.result()

    # Cache locally so the /result redirect on the same instance skips GCS.
    (TMP_DIR / f"{token}.json").write_text(meta_json)
    return token


def _load_result_meta(token: str) -> dict | None:
    """Return result metadata for *token*, or None if not found.

    Tries the local /tmp cache first (zero-latency on the same instance),
    then falls back to GCS so cross-instance redirects always work.
    If the GCS copy is found it is written to the local cache so subsequent
    calls on this instance skip the network round-trip.
    """
    local = TMP_DIR / f"{token}.json"
    if local.exists():
        try:
            return json.loads(local.read_text())
        except Exception:
            pass  # corrupt local file — fall through to GCS

    try:
        data = _storage().bucket(UPLOAD_BUCKET).blob(
            f"results/{token}.json"
        ).download_as_text()
        local.write_text(data)   # populate local cache for this instance
        return json.loads(data)
    except Exception:
        return None


def _rebuild_zip_renamed(zip_bytes: bytes, old_name: str, new_name: str) -> bytes:
    """Return a new ZIP identical to *zip_bytes* except one entry is renamed."""
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as src:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as dst:
            for info in src.infolist():
                data = src.read(info.filename)
                dst.writestr(new_name if info.filename == old_name else info.filename, data,
                             compress_type=zipfile.ZIP_STORED)
    return buf.getvalue()


def _update_result_entry(token: str, page_idx: int, new_si: str, new_method: str,
                         zip_bytes: bytes, meta: dict) -> None:
    """Rename one page entry in both the ZIP and the metadata JSON, then re-upload both."""
    entry        = meta["results"][page_idx]
    old_filename = entry["filename"]
    cc           = entry.get("customer_code", "UNKNOWN")
    new_filename = f"SI_{cc}_{new_si}.pdf"

    new_zip = _rebuild_zip_renamed(zip_bytes, old_filename, new_filename)
    entry.update({"si": new_si, "filename": new_filename, "method": new_method})
    meta_json = json.dumps(meta)

    bucket = _storage().bucket(UPLOAD_BUCKET)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(bucket.blob(meta["zip_blob"]).upload_from_string,
                         new_zip, "application/zip")
        f2 = pool.submit(bucket.blob(f"results/{token}.json").upload_from_string,
                         meta_json, "application/json")
        f1.result(); f2.result()

    (TMP_DIR / f"{token}.json").write_text(meta_json)


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
    Step 2 — enqueue a background job and return a job_id immediately.
    The client polls GET /status/<job_id> until status is 'done' or 'error'.
    This avoids 504 timeouts on large PDFs that take several minutes to process.
    """
    _cleanup_old_files()
    data = request.get_json(silent=True) or {}
    blob_name = (data.get("blob_name") or "").strip()
    filename  = (data.get("filename")  or "upload.pdf").strip()

    if not blob_name:
        return jsonify({"error": "blob_name is required"}), 400

    job_id = secrets.token_urlsafe(16)
    with _jobs_lock:
        _jobs[job_id] = {"status": "processing", "token": None, "created": time.time()}

    threading.Thread(target=_process_job, args=(job_id, blob_name, filename), daemon=True).start()
    return jsonify({"job_id": job_id, "status": "processing"})


@app.get("/status/<job_id>")
def job_status(job_id):
    """Poll the status of an async processing job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({k: v for k, v in job.items() if k != "created"})


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
    meta = _load_result_meta(token)
    if meta is None:
        abort(404)
    return render_template("result.html", results=meta["results"],
                           original_size=meta["original_size"],
                           optimized_size=meta["optimized_size"], token=token)


@app.get("/download/<token>")
def download(token: str):
    if not re.fullmatch(r'[A-Za-z0-9_\-]{24,40}', token):
        abort(404)

    # Resolve the GCS blob name from the JSON sidecar.
    meta = _load_result_meta(token)
    if meta is None:
        abort(404)
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


# ── Per-page re-evaluate & rename ────────────────────────────────────────────

def _validate_token_and_page(token: str, page_idx) -> tuple[dict, int] | tuple[None, None]:
    """Return (meta, page_idx) or (None, None) on any validation failure."""
    if not re.fullmatch(r'[A-Za-z0-9_\-]{24,40}', token):
        return None, None
    meta = _load_result_meta(token)
    if not meta:
        return None, None
    try:
        page_idx = int(page_idx)
    except (TypeError, ValueError):
        return None, None
    if page_idx < 0 or page_idx >= len(meta["results"]):
        return None, None
    return meta, page_idx


@app.post("/re-evaluate")
def re_evaluate():
    """Force OCR re-extraction for one page and rename it in the ZIP if the SI changes."""
    data     = request.get_json(silent=True) or {}
    token    = str(data.get("token", "")).strip()
    page_idx = data.get("page_index")

    if not token or page_idx is None:
        return jsonify({"error": "token and page_index are required"}), 400

    meta, page_idx = _validate_token_and_page(token, page_idx)
    if meta is None:
        return jsonify({"error": "Invalid token or page_index"}), 400

    entry        = meta["results"][page_idx]
    old_filename = entry["filename"]
    cc           = entry.get("customer_code", "UNKNOWN")

    # Pull the single-page PDF from the existing ZIP — no need to re-download the
    # original upload; the ZIP already contains stand-alone page PDFs.
    try:
        zip_bytes = _storage().bucket(UPLOAD_BUCKET).blob(
            meta["zip_blob"]).download_as_bytes()
    except Exception as exc:
        return jsonify({"error": f"Failed to download result: {exc}"}), 500

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            page_pdf_bytes = zf.read(old_filename)
    except Exception as exc:
        return jsonify({"error": f"Could not extract page: {exc}"}), 500

    doc  = fitz.open(stream=page_pdf_bytes, filetype="pdf")
    page = doc[0]
    h, w = page.rect.height, page.rect.width

    new_si, new_method = None, None

    crop = _load_crop_region()

    # Priority 1: ocr-full with retries — consistent with main processor priority order.
    if _TESS_OK:
        _pix_cache: dict[int, object] = {}
        for psm, zm, thr in _OCR_FULL_ATTEMPTS:
            if zm not in _pix_cache:
                _pix_cache[zm] = page.get_pixmap(
                    matrix=fitz.Matrix(zm, zm), colorspace=fitz.csGRAY
                )
            new_si = _find_si(_ocr_text(page, psm=psm, zoom=zm, bin_threshold=thr,
                                        _cached_pix=_pix_cache[zm]))
            if new_si:
                new_method = "ocr-full"
                break

    # Priority 2: crop region
    if new_si is None and crop and _TESS_OK:
        clip_r = fitz.Rect(w * crop["x1"], h * crop["y1"], w * crop["x2"], h * crop["y2"])
        new_si = _find_si(_ocr_text(page, clip_r, psm=6))
        if new_si:
            new_method = "ocr-crop"

    # Priorities 3–4: sub-region OCR
    if new_si is None and _TESS_OK:
        new_si = _find_si(_ocr_text(page, fitz.Rect(0, h * 0.60, w, h)))
        if new_si:
            new_method = "ocr-bottom"

    if new_si is None and _TESS_OK:
        new_si = _find_si(_ocr_text(page, fitz.Rect(w * 0.50, 0, w, h)))
        if new_si:
            new_method = "ocr-right"

    doc.close()

    if new_si is None:
        return jsonify({"ok": False, "message": "No SI number found via OCR"})

    new_si       = _sanitize(new_si)
    new_filename = f"SI_{cc}_{new_si}.pdf"

    # Check for SI conflict with another page.
    for i, r in enumerate(meta["results"]):
        if i != page_idx and r["filename"] == new_filename:
            return jsonify({"ok": False,
                            "message": f"SI {new_si} already used by page {r['page']}"})

    changed = new_filename != old_filename or new_method != entry["method"]
    if changed:
        _update_result_entry(token, page_idx, new_si, new_method, zip_bytes, meta)
    else:
        # Method unchanged and filename unchanged — nothing to rewrite.
        pass

    return jsonify({"ok": True, "changed": changed, "si": new_si,
                    "filename": new_filename, "method": new_method})


@app.post("/rename-page")
def rename_page():
    """Manually rename one page's SI number, rebuilding the ZIP entry."""
    data     = request.get_json(silent=True) or {}
    token    = str(data.get("token", "")).strip()
    page_idx = data.get("page_index")
    new_si   = str(data.get("new_si", "")).strip()

    if not token or page_idx is None or not new_si:
        return jsonify({"error": "token, page_index, and new_si are required"}), 400

    meta, page_idx = _validate_token_and_page(token, page_idx)
    if meta is None:
        return jsonify({"error": "Invalid token or page_index"}), 400

    new_si = _sanitize(new_si)
    if not new_si:
        return jsonify({"error": "Invalid SI number"}), 400

    entry        = meta["results"][page_idx]
    old_filename = entry["filename"]
    cc           = entry.get("customer_code", "UNKNOWN")
    new_filename = f"SI_{cc}_{new_si}.pdf"

    # Reject if another page already uses this filename.
    for i, r in enumerate(meta["results"]):
        if i != page_idx and r["filename"] == new_filename:
            return jsonify({"error": f"'{new_filename}' already used by page {r['page']}"}), 409

    if new_filename == old_filename:
        return jsonify({"ok": True, "si": entry["si"], "filename": old_filename,
                        "method": entry["method"]})

    try:
        zip_bytes = _storage().bucket(UPLOAD_BUCKET).blob(
            meta["zip_blob"]).download_as_bytes()
    except Exception as exc:
        return jsonify({"error": f"Failed to download result: {exc}"}), 500

    _update_result_entry(token, page_idx, new_si, "manual", zip_bytes, meta)
    return jsonify({"ok": True, "si": new_si, "filename": new_filename, "method": "manual"})


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
