# VisionAI — Intelligent Furniture Detection, Condition Assessment & Smart Marketplace Search

<p align="center">
  <strong>Final Year Project — BS Computer Science</strong><br>
  AI-powered system that detects household furniture, assesses its condition, and finds exact or alternative matches from local marketplaces using computer vision and large language models.
</p>

---

## The Problem

When people move homes, renovate, or buy/sell used furniture, they face a common set of challenges:

- **No easy way to catalog furniture** — Manually listing every item in a room is tedious and error-prone.
- **Condition assessment is subjective** — One person's "good condition" is another's "worn out." There's no standardized, objective way to evaluate furniture condition.
- **Finding replacements or matches is painful** — Searching for "the same sofa I saw in that photo" across OLX, Daraz, and local markets requires hours of manual browsing.
- **No price benchmarking** — Without knowing the market rate, buyers overpay and sellers underprice.

**VisionAI solves all of this in one workflow**: upload a photo → AI detects every furniture piece → assesses each item's condition → searches local marketplaces for exact matches or alternatives → generates a professional report with pricing.

---

## Why This Project?

We chose this as our Final Year Project because it sits at the intersection of three rapidly evolving AI domains:

1. **Computer Vision (YOLOv11)** — Real-time object detection is now accurate enough to identify specific furniture types and their condition from a single image.
2. **Large Language Models (Google Gemini)** — Multimodal LLMs can analyze a cropped furniture image and extract attributes (material, style, color, dimensions) that no traditional CV model can.
3. **AI-Powered Search** — Combining vision analysis with intelligent search creates an experience that didn't exist before: "find me this exact chair in Lahore under Rs. 50,000."

This project demonstrates a full production pipeline — from model training and MLOps to a deployed web application with real-time AI inference.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER WORKFLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. Upload Image ──► 2. YOLO Detection ──► 3. Crop Objects    │
│                                                                 │
│   4. Select City ──► 5a. Find Exact Match                      │
│                  ──► 5b. Find Alternative                      │
│                                                                 │
│   6. View Results ──► 7. Generate Report ──► 8. Download PDF   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Flow

| Step | What Happens | Technology |
|------|-------------|------------|
| **1. Upload** | User uploads a room photo or furniture image | Next.js Frontend |
| **2. Detect** | YOLOv11 model detects all furniture objects with bounding boxes | Ultralytics YOLO, FastAPI |
| **3. Crop** | Each detected object is individually cropped from the image | OpenCV, NumPy |
| **4. Analyze** | Gemini Vision AI analyzes each crop — extracts type, material, color, style, dimensions, condition, brand | Google Gemini 2.5 Flash |
| **5a. Exact Match** | AI searches local marketplaces in the selected city for the same item | Gemini Search / SerpAPI |
| **5b. Alternative** | AI searches for similar furniture (e.g., "5-seater L-shaped sofa") at various price points | Gemini Search / SerpAPI |
| **6. Results** | Product listings with prices in PKR, store names, and locations are displayed | React UI |
| **7. Report** | Gemini generates a professional assessment report with pricing tables and recommendations | Gemini 2.5 Flash |
| **8. Download** | Report can be printed or saved as PDF directly from the browser | Browser Print API |

---

## System Architecture

```
┌──────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   Next.js 16     │────►│   FastAPI Backend     │────►│  YOLOv11 Model  │
│   React 19 UI    │◄────│   REST API            │◄────│  (PyTorch)      │
│   Tailwind CSS 4 │     │                      │     └─────────────────┘
└──────────────────┘     │   /detect/full       │
                         │   /analyze           │────►┌─────────────────┐
                         │   /search            │────►│  Google Gemini   │
                         │   /report            │◄────│  2.5 Flash       │
                         └──────────────────────┘     │  (Vision + Text) │
                                                      └─────────────────┘
                                                             │
                                                      ┌──────┴──────┐
                                                      │  SerpAPI    │
                                                      │  (Optional) │
                                                      └─────────────┘
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check — model status and loaded classes |
| `/detect/full` | POST | Upload image → returns detections, annotated image, and cropped images (all base64) |
| `/analyze` | POST | Send a cropped image → Gemini Vision returns structured furniture analysis |
| `/search` | POST | Send analysis + city + mode → returns local marketplace listings |
| `/report` | POST | Send all items data → Gemini generates a comprehensive markdown report |

---

## Project Structure

```
VisionAI/
├── backend/                        # FastAPI REST API
│   ├── main.py                     # All endpoints, YOLO integration, image processing
│   └── services/
│       ├── gemini_service.py       # Gemini Vision analysis + report generation
│       └── search_service.py       # SerpAPI + Gemini marketplace search
│
├── frontend/                       # Next.js web application
│   └── app/
│       ├── layout.tsx              # Root layout, fonts, metadata
│       ├── page.tsx                # Main UI — upload, results, search, report views
│       └── globals.css             # Futuristic dark theme, glassmorphism, animations
│
├── computer-vision/                # ML pipeline
│   ├── src/
│   │   ├── train.py                # YOLO training with MLflow tracking
│   │   ├── inference.py            # Image/video/webcam detection with visualization
│   │   ├── evaluate.py             # Model validation and DVC-trackable metrics
│   │   ├── preprocess.py           # Dataset preparation (video→frames, splits)
│   │   └── utils.py                # Shared helpers, logging, class definitions
│   ├── models/                     # Trained model weights (.pt files)
│   ├── data/                       # Training data (DVC-tracked)
│   └── requirements.txt            # CV-specific Python dependencies
│
├── notebooks/
│   └── Train_VisionAI_Colab.ipynb  # Google Colab training notebook
│
├── scripts/                        # Automation helpers
│   ├── setup_mlops.py              # One-time MLOps configuration
│   ├── push_data.py                # DVC data push/pull wrapper
│   ├── compare_experiments.py      # MLflow experiment comparison
│   ├── run_training.bat            # Windows training launcher
│   └── run_webcam.bat              # Windows webcam inference launcher
│
├── .github/workflows/
│   └── mlops-pipeline.yml          # CI/CD: lint, build, DVC pipeline
│
├── params.yaml                     # Global training hyperparameters
├── dvc.yaml                        # DVC pipeline (prepare → train → evaluate → export)
├── requirements.txt                # Root Python dependencies
└── .env.example                    # Environment variables template
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Object Detection** | YOLOv11 (Ultralytics) | Real-time furniture detection and condition classification |
| **Deep Learning** | PyTorch | Model training and inference engine |
| **Vision AI** | Google Gemini 2.5 Flash | Multimodal furniture analysis (image → structured attributes) |
| **Product Search** | Gemini + SerpAPI | Location-based marketplace search with real listings |
| **Backend API** | FastAPI + Uvicorn | High-performance async REST API |
| **Frontend** | Next.js 16, React 19, TypeScript | Modern web UI with server-side rendering |
| **Styling** | Tailwind CSS 4 | Utility-first CSS with custom dark theme |
| **Image Processing** | OpenCV, NumPy, Pillow | Crop, annotate, and encode detection results |
| **MLOps** | DVC + MLflow + DagsHub | Data versioning, experiment tracking, model registry |
| **CI/CD** | GitHub Actions | Automated linting, builds, and pipeline validation |
| **Version Control** | Git + DVC | Code versioning + large file (model/data) versioning |

---

## Getting Started

### Prerequisites

- **Python 3.12+**
- **Node.js 18+**
- **Git**
- A free [Google Gemini API key](https://aistudio.google.com) (required for analysis and search)

### 1. Clone and Install

```bash
git clone https://github.com/arsalbinmanan1/FYP-2.git
cd FYP-2

# Python environment
python -m venv venv
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install
cd ..
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here    # Required — free at aistudio.google.com
SERPAPI_KEY=your_serpapi_key_here           # Optional — free tier at serpapi.com
```

### 3. Add a Trained Model

Place your trained YOLO model in `computer-vision/models/`:
- **`best.pt`** (highest priority) — your trained model from Colab
- The system falls back to `my_model.pt` → `last.pt` → default `yolo11n.pt`

### 4. Run the Application

**Terminal 1 — Backend:**
```bash
cd backend
uvicorn main:app --reload
# API running at http://localhost:8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
# UI running at http://localhost:3000
```

Open **http://localhost:3000** in your browser.

---

## Training the Model

### Option A: Google Colab (Recommended)

1. Open `notebooks/Train_VisionAI_Colab.ipynb` in Google Colab
2. Upload your labeled dataset
3. Run all cells — training uses a free GPU
4. Download the resulting `best.pt`
5. Place it in `computer-vision/models/best.pt`

### Option B: Local Training

```bash
python computer-vision/src/train.py --epochs 50 --batch 16
```

### Option C: DVC Pipeline

```bash
dvc repro   # Runs: prepare_data → train → evaluate → export_model
```

---

## Training Classes

The model is trained to detect household furniture and assess condition:

| Class | Description |
|-------|-------------|
| `chair_broken` | Chair in broken/unusable condition |
| `chair_wornout` | Chair showing wear and aging |
| `sofa_broken` | Sofa with structural damage |
| `sofa_wornout` | Sofa with fabric/cushion wear |
| `table_broken` | Table with structural damage |
| `table_wornout` | Table showing surface wear |

The current model also detects 30+ general household objects (bed, cabinet, carpet, lamp, mirror, refrigerator, etc.) depending on the training dataset.

---

## Webcam Real-Time Detection

```bash
# Activate venv first
python computer-vision/src/inference.py --source webcam
```

Controls:
- **Q** — Quit
- **S** — Save screenshot
- **P** — Pause/resume

---

## MLOps Pipeline

The project includes a full MLOps stack for reproducible experiments:

```
params.yaml ──► dvc.yaml ──► DagsHub (data + models)
                    │              │
                    ▼              ▼
              DVC Pipeline    MLflow Tracking
              (4 stages)      (metrics, artifacts)
```

**Setup:**
```bash
python scripts/setup_mlops.py   # Interactive configuration
dvc repro                       # Run full pipeline
python scripts/compare_experiments.py  # Compare runs
```

---

## Use Cases

1. **Real Estate** — Photograph furnished properties, auto-generate furniture inventory with condition reports and replacement costs.
2. **Insurance Claims** — Document furniture condition before/after incidents with AI-verified assessments.
3. **Used Furniture Marketplace** — Sellers upload photos; AI suggests fair pricing based on condition and local market rates.
4. **Interior Design** — Designers analyze client's existing furniture and find matching or complementary pieces.
5. **Moving Companies** — Inventory household items for quotes and insurance purposes.

---

## Team

| Name | Role |
|------|------|
| Arsal Bin Manan | Full-Stack Development, MLOps, System Architecture |

---

## License

This project was developed as a Final Year Project for academic purposes.

---

<p align="center">
  Built with YOLOv11, Google Gemini, FastAPI, and Next.js<br>
  <strong>VisionAI</strong> — See your furniture differently.
</p>
