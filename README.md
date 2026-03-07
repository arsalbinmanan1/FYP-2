# VisionAI - Household Furniture Condition Detection

A Final Year Project that detects household furniture (chairs, sofas, tables) and classifies their condition (broken / worn out) using **YOLOv11**, served through a **FastAPI** backend and a **Next.js** frontend, with a full **MLOps pipeline** powered by **DVC**, **MLflow**, and **DagsHub**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Code File Reference](#code-file-reference)
5. [MLOps Pipeline](#mlops-pipeline)
6. [Detection Classes](#detection-classes)
7. [Prerequisites](#prerequisites)
8. [Getting Started](#getting-started)
9. [Running the Application](#running-the-application)
10. [Training on Google Colab](#training-on-google-colab)
11. [DVC Data Versioning](#dvc-data-versioning)
12. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
13. [CI/CD with GitHub Actions](#cicd-with-github-actions)
14. [Developer Workflow](#developer-workflow)
15. [Troubleshooting](#troubleshooting)
16. [Tech Stack](#tech-stack)

---

## Project Overview

VisionAI is a computer vision system that:

- Detects household furniture in images and video streams.
- Classifies the condition of each detected object as **broken** or **worn out**.
- Exposes predictions through a REST API.
- Displays results in a web interface with annotated bounding boxes.
- Versions all data and models with DVC, tracks every training experiment with MLflow, and stores everything on DagsHub.

The system is split into three layers:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Computer Vision** | YOLOv11 (Ultralytics) + PyTorch | Detection and condition classification |
| **Backend API** | FastAPI + Uvicorn | Serve model predictions over HTTP |
| **Frontend** | Next.js 16 + React 19 + Tailwind CSS 4 | Upload images, display results |
| **MLOps** | DVC + MLflow + DagsHub + GitHub Actions | Data versioning, experiment tracking, CI/CD |

---

## Architecture

```
                          ┌─────────────────┐
                          │   Google Colab   │
                          │   GPU Training   │
                          └────────┬────────┘
                                   │ train + log
                    ┌──────────────┼──────────────┐
                    │              │              │
               ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐
               │  DagsHub  │  │  MLflow  │  │   DVC    │
               │   (Git)   │  │ Tracking │  │ Storage  │
               └────┬──────┘  └──────────┘  └────┬─────┘
                    │                             │
                    │   git push / dvc pull        │
                    ▼                             ▼
              ┌───────────────────────────────────────┐
              │           Local Machine               │
              │                                       │
              │  ┌──────────┐      ┌──────────────┐   │
              │  │ FastAPI  │◄────►│  YOLO Model  │   │
              │  │ Backend  │      │  (best.pt)   │   │
              │  │ :8000    │      └──────────────┘   │
              │  └────┬─────┘                         │
              │       │  REST API                     │
              │  ┌────▼─────┐                         │
              │  │ Next.js  │                         │
              │  │ Frontend │                         │
              │  │ :3000    │                         │
              │  └──────────┘                         │
              └───────────────────────────────────────┘
```

---

## Project Structure

```
FYP-2/
├── computer-vision/            # AI/ML core
│   ├── src/
│   │   ├── __init__.py         # Package init, exports key symbols
│   │   ├── train.py            # Model training with MLflow logging
│   │   ├── inference.py        # Image/video/webcam detection
│   │   ├── evaluate.py         # Model validation, writes metrics.json
│   │   ├── preprocess.py       # Frame extraction, dataset creation
│   │   └── utils.py            # Shared helpers, class maps, config
│   ├── data/
│   │   ├── raw/                # Raw videos/images (DVC-tracked)
│   │   └── processed/          # YOLO-format dataset (DVC-tracked)
│   │       └── data.yaml.template
│   ├── models/                 # Trained weights
│   │   ├── my_model.pt         # Custom household object detector
│   │   ├── yolov8n.pt          # Base COCO model
│   │   └── best.pt             # Best condition-detection model (after training)
│   ├── notebooks/
│   │   ├── Train_With_MLOps.ipynb
│   │   └── Train_YOLO_Models.ipynb
│   └── requirements.txt
│
├── backend/
│   └── main.py                 # FastAPI REST API for detection
│
├── frontend/                   # Next.js web application
│   ├── app/
│   │   ├── page.tsx            # Main detection UI
│   │   ├── layout.tsx          # Root layout and metadata
│   │   └── globals.css         # Theme variables and styles
│   ├── next.config.ts          # API proxy config
│   ├── package.json            # Dependencies and scripts
│   └── .env.local              # API URL config (not committed)
│
├── scripts/
│   ├── setup_mlops.py          # Interactive MLOps setup wizard
│   ├── push_data.py            # DVC push/pull helper
│   ├── compare_experiments.py  # Query MLflow runs
│   ├── setup_dvc.py            # DVC initialization helper
│   ├── run_training.bat        # Windows training shortcut
│   └── run_webcam.bat          # Windows webcam shortcut
│
├── notebooks/
│   └── Train_VisionAI_Colab.ipynb  # Full Colab training notebook
│
├── .github/workflows/
│   └── mlops-pipeline.yml      # CI/CD pipeline
│
├── params.yaml                 # Global hyperparameters
├── dvc.yaml                    # DVC pipeline stages
├── .dvc/config                 # DVC remote config (DagsHub S3)
├── .dvcignore                  # DVC ignore rules
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Root Python dependencies
└── README.md                   # This file
```

---

## Code File Reference

### Computer Vision Module (`computer-vision/src/`)

#### `utils.py` — Shared Utilities

The foundation module imported by every other script.

- **`load_params(path)`** — Loads `params.yaml` from multiple search locations, falls back to defaults.
- **`FURNITURE_CLASSES`** — Dictionary mapping class IDs (0-5) to names like `chair_broken`, `sofa_wornout`.
- **`CONDITION_COLORS`** — BGR color map for bounding box drawing (`broken` = red, `wornout` = orange).
- **`get_condition_from_class(class_id)`** / **`get_furniture_type(class_id)`** — Extract condition or furniture type from a class ID.
- **`get_latest_model(models_dir)`** — Finds the best available model file with priority: `best.pt` > `last.pt` > most recent `.pt`.
- **`validate_dataset(data_dir)`** — Checks that a YOLO dataset has the required `train/images`, `train/labels`, `val/images`, `val/labels` folders.
- **`create_data_yaml(data_dir, output_path)`** — Generates a YOLO `data.yaml` configuration file.
- **`setup_dagshub(owner, repo)`** / **`setup_mlflow(uri, name)`** — Initialize DagsHub and MLflow for experiment tracking.
- **`print_banner(title)`** / **`print_detection_results(detections)`** — Formatted console output.

#### `train.py` — Model Training

Trains a YOLOv11 model with automatic MLflow experiment tracking.

- **`setup_experiment_tracking(owner, repo, name)`** — Initializes DagsHub + MLflow. Sets `MLFLOW_TRACKING_URI` as an environment variable so Ultralytics auto-logs metrics, parameters, and artifacts without manual `mlflow.start_run()` calls.
- **`train_yolo(data_yaml, model, epochs, ...)`** — Core training function. Loads a YOLO model, trains it, and saves a `metrics.json` alongside the run. If MLflow is active, also logs the best model and training plots as artifacts.
- **`train_from_params(params_path, **overrides)`** — Reads all config from `params.yaml` and delegates to `train_yolo()`.
- **CLI**: `python train.py --data data.yaml --epochs 50` or `python train.py --from-params`.

#### `inference.py` — Detection Engine

Runs object detection on images, videos, or webcam feeds.

- **`VisionAIDetector`** — Main class wrapping YOLO inference.
  - `detect(image)` — Returns a list of detection dictionaries (`bbox`, `class_name`, `confidence`, `furniture_type`, `condition`).
  - `draw_detections(image, detections)` — Draws color-coded bounding boxes and labels on the image.
  - `crop_detections(image, detections, output_dir)` — Saves cropped images of each detected object.
- **`process_image(detector, path)`** — Single image pipeline: detect, annotate, save, display.
- **`process_video(detector, path)`** — Video pipeline with FPS overlay and progress indicator. Press `q` to quit, `p` to pause.
- **`process_webcam(detector)`** — Real-time webcam detection. Press `s` for screenshot, `q` to quit.
- **CLI**: `python inference.py --source image.jpg`, `--source video.mp4`, or `--source webcam`.

#### `evaluate.py` — Model Evaluation

Validates a trained model and writes DVC-trackable metrics.

- **`evaluate_model(model_path, data_yaml, output_dir)`** — Runs `model.val()`, extracts mAP50, mAP50-95, precision, recall, fitness, and per-class metrics. Writes everything to `evaluation_results/metrics.json`.
- This file is called by the `evaluate` stage in `dvc.yaml`. DVC tracks `metrics.json` and can diff it between commits.
- **CLI**: `python evaluate.py --model best.pt --data data.yaml`.

#### `preprocess.py` — Data Preparation

Converts raw data into a YOLO-format training dataset.

- **`extract_frames(video_path, output_dir, frame_skip)`** — Extracts every Nth frame from a video file and saves as images.
- **`create_yolo_dataset(source_dir, output_dir, train_split)`** — Takes organized image folders (`chair/broken/`, `sofa/wornout/`, etc.) and creates a YOLO dataset with `train/` and `val/` splits, labels, and `data.yaml`.
- **`prepare_annotations_from_roboflow(export_dir, output_dir)`** — Copies a Roboflow export into the project's dataset folder.
- **CLI**: `python preprocess.py extract --video test.mp4 --skip 5` or `python preprocess.py create-dataset --source raw/ --output processed/`.

---

### Backend (`backend/`)

#### `main.py` — FastAPI REST API

Serves YOLO predictions over HTTP. The frontend sends images here for detection.

- **Model loading**: `get_model()` loads the best available `.pt` file with priority: `best.pt` > `my_model.pt` > `last.pt` > fallback to `yolo11n.pt`.
- **Model-agnostic**: Works with any YOLO model regardless of class count. Dynamically reads class names from the loaded model.
- **`parse_class_name(name)`** — Splits class names like `chair_broken` into furniture type and condition.
- **Color coding**: Condition-based colors (red = broken, orange = wornout) for annotated images, or a Tableau-10 palette for general models.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET /` | Health check | Returns model status, path, and class list |
| `POST /detect` | JSON response | Upload image, get detections as JSON |
| `POST /detect/annotated` | Image response | Upload image, get back annotated JPEG |
| `POST /detect/full` | JSON + image | Upload image, get detections JSON plus annotated image as base64 |

All `POST` endpoints accept a `confidence` query parameter (0.1–1.0, default 0.5).

---

### Frontend (`frontend/`)

#### `page.tsx` — Main UI

A single-page application built with React 19 and Tailwind CSS.

- **Drag-and-drop upload**: Drop an image or click to browse. Accepts JPEG, PNG, WebP.
- **Confidence slider**: Adjustable threshold (10%–100%) that controls minimum detection confidence.
- **Detection display**: Shows the original image overlaid with annotated bounding boxes from the backend.
- **Results panel**: Detection count, inference time, image dimensions, unique classes found.
- **Detections list**: Grouped by class name, each detection shows condition badge and confidence percentage.
- **Model classes**: Shows all classes the loaded model can detect.
- **Backend status**: Green/red dot indicating whether the API server is reachable.

#### `layout.tsx` — Root Layout

Sets page metadata (`<title>`, `<meta description>`), loads the Inter font, and wraps children with the global CSS.

#### `globals.css` — Theme System

CSS custom properties for both light and dark mode. Defines colors for background, foreground, accent (indigo), danger (red), warning (amber), success (green), and card/border/muted tones.

#### `next.config.ts` — API Proxy

Rewrites `/api/*` requests to `http://localhost:8000/*` so the frontend can call the backend without CORS issues in production.

---

### Scripts (`scripts/`)

#### `setup_mlops.py` — MLOps Setup Wizard

Interactive script that configures everything in one run:
1. Sets DVC remote to DagsHub S3.
2. Stores DagsHub token as DVC credential.
3. Verifies the DVC pipeline DAG.
4. Creates a `.env` file with all tokens and URIs.

Run: `python scripts/setup_mlops.py`

#### `push_data.py` — DVC Data Push/Pull

Convenience wrapper around DVC commands.
- `python scripts/push_data.py push` — Tracks data with DVC and pushes to DagsHub.
- `python scripts/push_data.py pull` — Pulls data from DagsHub.
- `python scripts/push_data.py status` — Shows DVC remote and tracking status.

#### `compare_experiments.py` — Experiment Comparison

Queries MLflow on DagsHub and prints a table of the top N training runs sorted by any metric. Run: `python scripts/compare_experiments.py --top 5 --metric mAP50`.

#### `setup_dvc.py` — DVC Initialization

An older helper that initializes DVC and configures the DagsHub remote. Superseded by `setup_mlops.py` but still functional.

#### `run_training.bat` / `run_webcam.bat`

Windows batch files for quick local training or webcam detection. They activate the venv and call the Python scripts with sensible defaults.

---

### Configuration Files

#### `params.yaml` — Hyperparameters

Central configuration file read by both DVC and the training scripts. Contains:

| Section | Key Parameters |
|---------|---------------|
| `model` | `architecture` (yolo11s.pt), `input_size` (640), `num_classes` (6) |
| `train` | `epochs` (50), `batch_size` (16), `learning_rate` (0.01), `patience` (10) |
| `data` | `config_path`, `train_split` (0.8), `augmentation` (true) |
| `augmentation` | HSV shifts, rotation, translation, scale, flip, mosaic, mixup |
| `experiment` | `dagshub_owner`, `dagshub_repo`, `mlflow_tracking` |
| `output` | `runs_dir` (VisionAI_Runs), `models_dir` |
| `classes` | List of 6 class names |

To change training settings, edit this file and run `dvc repro` — DVC detects parameter changes and re-runs affected stages.

#### `dvc.yaml` — Pipeline Definition

Defines four stages that run in sequence:

1. **`prepare_data`** — Runs `preprocess.py create-dataset` to convert raw data into YOLO format. Depends on `computer-vision/data/raw` and `params.yaml` split ratios.
2. **`train`** — Runs `train.py` with all hyperparameters from `params.yaml`. Logs to MLflow via DagsHub. Outputs model weights.
3. **`evaluate`** — Runs `evaluate.py` on `best.pt` against the validation set. Writes `evaluation_results/metrics.json` as a DVC metric.
4. **`export_model`** — Copies `best.pt` to `computer-vision/models/best.pt` for the backend to pick up.

#### `.env.example` — Environment Template

Copy to `.env` and fill in your DagsHub token. Used for:
- `DAGSHUB_USER_TOKEN` — Authentication for DagsHub, MLflow, and DVC.
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — DVC S3 remote auth (same DagsHub token).
- `MLFLOW_TRACKING_URI` / `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`.

#### `.github/workflows/mlops-pipeline.yml` — CI/CD

Three jobs triggered on push to `main`:

| Job | Trigger | What it does |
|-----|---------|-------------|
| `lint-and-test` | Every push | Validates DVC pipeline, compiles Python sources, checks params.yaml |
| `dvc-pipeline` | Manual dispatch only | Pulls data, runs `dvc repro`, pushes results, commits `dvc.lock` |
| `frontend-build` | Every push | Runs `npm ci`, lint, and `next build` |

#### `.dvc/config` — DVC Remote

```ini
[core]
    remote = origin
    autostage = true
['remote "origin"']
    url = s3://dvc
    endpointurl = https://dagshub.com/arsal6010/FYP_visionAI.s3
```

Points DVC to DagsHub's S3-compatible storage. Credentials are stored locally (not committed) via `dvc remote modify --local`.

---

## MLOps Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ prepare_data │────>│    train     │────>│   evaluate   │     │ export_model │
│              │     │              │     │              │     │              │
│ preprocess.py│     │ train.py     │     │ evaluate.py  │     │ copy best.pt │
│ raw -> YOLO  │     │ YOLO + MLflow│     │ -> metrics   │     │ -> models/   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                     │
       │               ┌────┴────┐               │                    │
       ▼               ▼         ▼               ▼                    ▼
   data.yaml     DagsHub    MLflow        metrics.json           best.pt
                 DVC S3    Dashboard    (DVC-tracked)       (backend loads)
```

**Run the full pipeline:**
```bash
dvc repro
```

**View metrics:**
```bash
dvc metrics show
```

**Compare across commits:**
```bash
dvc metrics diff
dvc plots diff
```

---

## Detection Classes

| ID | Class Name | Description |
|----|-----------|-------------|
| 0 | `chair_broken` | Chair with structural damage |
| 1 | `chair_wornout` | Chair with wear and tear |
| 2 | `sofa_broken` | Sofa with structural damage |
| 3 | `sofa_wornout` | Sofa with wear and tear |
| 4 | `table_broken` | Table with structural damage |
| 5 | `table_wornout` | Table with wear and tear |

---

## Prerequisites

- **Python 3.12** (3.14 has PyTorch DLL issues on Windows)
- **Node.js 18+** and npm
- **Git**
- **DagsHub account** with an API token ([get one here](https://dagshub.com/user/settings/tokens))
- (Optional) **NVIDIA GPU** with CUDA for local training

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://dagshub.com/arsal6010/FYP_visionAI.git FYP-2
cd FYP-2
```

### 2. Create Python virtual environment

```bash
# Windows
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3.12 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set up MLOps credentials

```bash
python scripts/setup_mlops.py
```

This will ask for your DagsHub username and token, then configure DVC remote auth and create a `.env` file.

### 5. Pull data and models from DVC

```bash
dvc pull -r origin
```

### 6. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Running the Application

You need two terminal windows:

### Terminal 1 — Backend (port 8000)

```powershell
.\venv\Scripts\Activate.ps1
cd backend
python main.py
```

The backend starts at `http://localhost:8000`. It auto-loads the best available model from `computer-vision/models/`.

### Terminal 2 — Frontend (port 3000)

```powershell
cd frontend
npm run dev
```

Open **http://localhost:3000** in your browser. Upload an image and click "Run Detection".

### Quick CLI Detection (no frontend needed)

```bash
# Image
python computer-vision/src/inference.py --source path/to/image.jpg

# Video
python computer-vision/src/inference.py --source path/to/video.mp4

# Webcam
python computer-vision/src/inference.py --source webcam
```

---

## Training on Google Colab

Since training requires a GPU, use the provided Colab notebook.

1. Open `notebooks/Train_VisionAI_Colab.ipynb` in Google Colab.
2. Set your DagsHub credentials in cell 4.
3. The notebook will:
   - Clone the repo from DagsHub.
   - Pull training data via DVC.
   - Train YOLOv11 with MLflow logging.
   - Push the trained model back to DVC.
   - Commit the DVC tracking file.
4. After training, pull the model locally:

```bash
dvc pull -r origin
```

The backend automatically picks up `best.pt` on next restart.

---

## DVC Data Versioning

DVC tracks large files (datasets, model weights) that don't belong in Git.

### Adding new data

```bash
# Place data in computer-vision/data/processed/ (YOLO format)
dvc add computer-vision/data/processed
git add computer-vision/data/processed.dvc .gitignore
git commit -m "update training dataset"
dvc push -r origin
git push
```

### Pulling data on a new machine

```bash
dvc pull -r origin
```

### Checking what's tracked

```bash
dvc data status
python scripts/push_data.py status
```

---

## MLflow Experiment Tracking

Every training run automatically logs to MLflow on DagsHub.

**What gets logged:**
- Hyperparameters (model, epochs, batch size, learning rate, etc.)
- Metrics per epoch (mAP50, mAP50-95, precision, recall, box loss, class loss)
- Artifacts (best.pt model, training plots, confusion matrix)

**View your experiments:**
- Dashboard: https://dagshub.com/arsal6010/FYP_visionAI.mlflow
- CLI: `python scripts/compare_experiments.py --top 5`

**How it works:**
1. `train.py` calls `dagshub.init()` which sets `MLFLOW_TRACKING_URI`.
2. Ultralytics detects this env var and auto-logs all metrics.
3. After training, `train.py` additionally logs the model file and plots as artifacts.

---

## CI/CD with GitHub Actions

The workflow at `.github/workflows/mlops-pipeline.yml` runs on every push.

### Automatic (every push to main):
- Validates `params.yaml` and `dvc.yaml`.
- Compiles all Python source files.
- Builds the Next.js frontend.

### Manual (workflow dispatch):
- Pulls data from DVC.
- Runs `dvc repro` (full pipeline).
- Pushes results back to DVC.
- Commits updated `dvc.lock` and `metrics.json`.

### Required GitHub Secrets:

| Secret | Value |
|--------|-------|
| `DAGSHUB_OWNER` | `arsal6010` |
| `DAGSHUB_REPO` | `FYP_visionAI` |
| `DAGSHUB_USER_TOKEN` | Your DagsHub API token |

---

## Developer Workflow

This is how a developer works on this project day to day.

### Making a model improvement

```bash
# 1. Edit hyperparameters
#    Open params.yaml, change epochs, learning_rate, etc.

# 2. Train on Colab
#    Open notebooks/Train_VisionAI_Colab.ipynb in Colab
#    The notebook logs everything to MLflow

# 3. Pull the new model
dvc pull -r origin

# 4. Compare with previous runs
python scripts/compare_experiments.py --top 5

# 5. Test locally
.\venv\Scripts\Activate.ps1
cd backend && python main.py          # Terminal 1
cd frontend && npm run dev            # Terminal 2
# Upload images at http://localhost:3000

# 6. Commit if satisfied
git add params.yaml dvc.lock
git commit -m "improve: increase epochs to 100, mAP50 0.82 -> 0.87"
git push dagshub main
```

### Adding new training data

```bash
# 1. Place new annotated images in computer-vision/data/processed/
#    Follow YOLO format: train/images/, train/labels/, val/images/, val/labels/

# 2. Track with DVC
dvc add computer-vision/data/processed

# 3. Push data to DagsHub
dvc push -r origin

# 4. Commit tracking files
git add computer-vision/data/processed.dvc .gitignore
git commit -m "data: add 200 new sofa images"
git push dagshub main

# 5. Retrain on Colab (data will be pulled via dvc pull)
```

### Modifying the frontend

```bash
cd frontend
npm run dev           # hot-reload on http://localhost:3000
# Edit app/page.tsx
npm run lint          # check for issues
npm run build         # verify production build
git add -A && git commit -m "ui: add detection history panel"
```

### Modifying the backend

```bash
.\venv\Scripts\Activate.ps1
cd backend
python main.py        # hot-reload enabled, edit main.py and save
# Test: curl -X POST http://localhost:8000/detect/full -F "file=@image.jpg"
```

### Running the full DVC pipeline locally

```bash
# Only works if you have training data in computer-vision/data/raw/
dvc repro             # runs all stages
dvc metrics show      # print evaluation_results/metrics.json
dvc metrics diff      # compare with last commit
dvc plots diff        # visual comparison of training curves
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `OSError: [WinError 1114] DLL initialization failed` | Install Visual C++ Redistributable: `winget install Microsoft.VCRedist.2015+.x64` |
| PyTorch fails on Python 3.14 | Use Python 3.12 instead: `py -3.12 -m venv venv` |
| `dvc pull` asks for credentials | Run `python scripts/setup_mlops.py` or set `AWS_ACCESS_KEY_ID` env var to your DagsHub token |
| Backend returns "Model not found" | Place a `.pt` model in `computer-vision/models/` or run `dvc pull -r origin` |
| Frontend shows "API Offline" | Start the backend first: `cd backend && python main.py` |
| `dvc.lock is git-ignored` | Remove `/dvc.lock` from `.gitignore` (already fixed in this repo) |
| CUDA not available | PyTorch on Python 3.14 only has CPU wheels. Use Python 3.12, or train on Colab for GPU |

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Object Detection | YOLOv11 (Ultralytics) | 8.4+ |
| Deep Learning | PyTorch | 2.10+ |
| Computer Vision | OpenCV | 4.13+ |
| Backend | FastAPI + Uvicorn | 0.135+ |
| Frontend | Next.js + React | 16.0 / 19.2 |
| Styling | Tailwind CSS | 4.0 |
| Language (FE) | TypeScript | 5.x |
| Data Versioning | DVC | 3.66+ |
| Experiment Tracking | MLflow | 3.10+ |
| MLOps Platform | DagsHub | 0.6+ |
| CI/CD | GitHub Actions | - |
| Python | 3.12 recommended | 3.12.10 |
| Node.js | 20+ | 24.12 |

---

## Links

- **DagsHub Repository**: https://dagshub.com/arsal6010/FYP_visionAI
- **MLflow Dashboard**: https://dagshub.com/arsal6010/FYP_visionAI.mlflow
- **DVC Remote**: S3-compatible storage via DagsHub

---

*VisionAI — Final Year Project 2025-2026*
