# rgmc-invoice-separator

Web service that accepts a multi-page sales invoice PDF, splits it page-by-page, extracts the SI (Sales Invoice) number from each page, and packages everything into a downloadable ZIP — each PDF named after its SI number.

Built with **Flask + PyMuPDF + pyzbar**. Ready for deployment on **Google Cloud Run**.

---

## How SI numbers are extracted (in order)

| Method | How |
|---|---|
| `text-bottom` | Text extraction from the bottom 40 % of the page (fastest) |
| `text-right` | Text extraction from the right half of the page |
| `text-full` | Full-page text extraction |
| `barcode` | Renders the page as an image and decodes any barcode (Code 128, QR, etc.) |
| `fallback` | Page named `page_NNNN.pdf` if nothing is found |

---

## Local development

```bash
pip install -r requirements.txt
python main.py
# Open http://localhost:8080
```

> **Linux/macOS** — `pyzbar` requires `libzbar0`:
> ```bash
> sudo apt-get install libzbar0   # Debian/Ubuntu
> brew install zbar               # macOS
> ```

---

## Docker

```bash
docker build -t invoice-separator .
docker run -p 8080:8080 invoice-separator
```

---

## Deploy to Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT/invoice-separator

# Deploy
gcloud run deploy invoice-separator \
  --image gcr.io/YOUR_PROJECT/invoice-separator \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --concurrency 10
```

Replace `YOUR_PROJECT` with your GCP project ID.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8080` | Port the server listens on |

---

## Project structure

```
rgmc-invoice-separator/
├── main.py              # Flask app — upload, process, download routes
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── templates/
│   ├── index.html       # Upload form
│   └── result.html      # Results table + download button
└── static/
    └── logo.png
```
