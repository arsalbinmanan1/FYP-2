# 🏠 VisionAI - Household Object Condition Detection

A Final Year Project implementing real-time furniture condition assessment using YOLOv11, with a complete MLOps stack for professional-grade experiment tracking and data versioning.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)
![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-purple.svg)

---

## 📋 Project Overview

This project consists of:

1. **Computer Vision (YOLOv11)** - Object detection model for identifying furniture and assessing condition (broken/worn out)
2. **Frontend (Next.js)** - Modern web interface for real-time detection display
3. **MLOps Stack** - Professional experiment tracking and data versioning
   - **DagsHub** - Central hub for code, data, and experiments
   - **MLflow** - Experiment tracking and model registry
   - **DVC** - Data version control

---

## 📁 Project Structure

```
FYP-2/
│
├── 📄 README.md                    # This file
├── 📄 params.yaml                  # Global training parameters
├── 📄 dvc.yaml                     # DVC pipeline definition
├── 📄 .dvcignore                   # DVC ignore patterns
│
├── 📂 computer-vision/             # AI/ML Core
│   ├── 📂 data/
│   │   ├── 📂 raw/                 # Raw images/videos
│   │   └── 📂 processed/           # YOLO format dataset
│   │
│   ├── 📂 models/                  # Trained model weights
│   │   ├── 🤖 yolov8n.pt          # Base model
│   │   ├── 🤖 my_model.pt         # Custom trained model
│   │   └── 🤖 best.pt             # Best fine-tuned model
│   │
│   ├── 📂 notebooks/               # Jupyter Notebooks
│   │   ├── 📓 Train_With_MLOps.ipynb    # MLflow integrated training
│   │   └── 📓 Train_YOLO_Models.ipynb   # Original training notebook
│   │
│   ├── 📂 src/                     # Python modules
│   │   ├── 📄 train.py            # Training with MLflow
│   │   ├── 📄 inference.py        # Detection/inference
│   │   ├── 📄 preprocess.py       # Data preprocessing
│   │   └── 📄 utils.py            # Helper functions
│   │
│   └── 📄 requirements.txt         # Python dependencies
│
└── 📂 frontend/                    # Next.js Web Application
    ├── 📂 app/                     # Next.js App Router
    ├── 📂 public/                  # Static assets
    ├── 📄 package.json
    └── 📄 tsconfig.json
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Git
- (Optional) NVIDIA GPU with CUDA for faster training

### 1. Clone & Setup

```bash
# Clone repository
git clone https://github.com/yourusername/FYP-2.git
cd FYP-2

# Setup Python environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install Python dependencies
pip install -r computer-vision/requirements.txt
```

### 2. Setup DagsHub & DVC

```bash
# Install MLOps tools
pip install dagshub mlflow dvc dvc-s3

# Login to DagsHub
dagshub login

# Initialize DVC (if not already)
dvc init

# Add DagsHub as remote
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/arsal6010/FYP_visionAI.s3
```

### 3. Pull Data with DVC

```bash
# Pull versioned data from DagsHub
dvc pull -r origin
```

### 4. Train Model

```bash
# Option 1: Use CLI
cd computer-vision
python src/train.py --data data/processed/data.yaml --epochs 50

# Option 2: Use Google Colab
# Open notebooks/Train_With_MLOps.ipynb in Colab
```

### 5. Run Inference

```bash
# Detect in image
python src/inference.py --source image.jpg

# Detect in video
python src/inference.py --source video.mp4

# Webcam detection
python src/inference.py --source webcam
```

### 6. Start Frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

---

## 🔧 Configuration

### Training Parameters (`params.yaml`)

```yaml
model:
  architecture: "yolo11s.pt"   # yolo11n.pt, yolo11s.pt, yolo11m.pt
  input_size: 640

train:
  epochs: 50
  batch_size: 16
  learning_rate: 0.01
  patience: 10

experiment:
  dagshub_owner: "arsal6010"
  dagshub_repo: "VisionAI"
```

### DVC Pipeline (`dvc.yaml`)

The pipeline has 4 stages:

1. **prepare_data** - Convert raw data to YOLO format
2. **train** - Train model with MLflow logging
3. **evaluate** - Run validation
4. **export_model** - Save best model

Run the entire pipeline:

```bash
dvc repro
```

---

## 📊 MLOps Stack

### DagsHub (Central Hub)

- **Repository**: https://dagshub.com/arsal6010/VisionAI
- **MLflow Dashboard**: https://dagshub.com/arsal6010/VisionAI.mlflow
- **DVC Storage**: S3-compatible storage for data/models

### MLflow (Experiment Tracking)

Every training run logs:
- Hyperparameters (epochs, batch_size, learning_rate)
- Metrics (mAP50, mAP50-95, precision, recall)
- Artifacts (model weights, training plots)

```python
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/arsal6010/VisionAI.mlflow")
mlflow.set_experiment("YOLOv11-Household-Objects")
```

### DVC (Data Version Control)

Version your data and models:

```bash
# Add data to DVC tracking
dvc add computer-vision/data/processed

# Push to remote storage
dvc push -r origin

# Pull on another machine
dvc pull -r origin
```

---

## 🏗️ Project Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    VisionAI Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Stage 1    │───▶│   Stage 2    │───▶│   Stage 3    │      │
│  │ Data Ingest  │    │   Training   │    │   Registry   │      │
│  │    + DVC     │    │   + MLflow   │    │   + Deploy   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│   ┌───────────┐      ┌───────────┐      ┌───────────┐          │
│   │Raw Images │      │ Metrics   │      │ best.pt   │          │
│   │Annotations│      │ mAP, Loss │      │  Model    │          │
│   │  DVC      │      │  MLflow   │      │ Registry  │          │
│   └───────────┘      └───────────┘      └───────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 1: Data Ingestion & Versioning
- Raw videos/images of household objects
- Annotated using Roboflow/LabelImg
- Versioned with DVC
- Stored in DagsHub S3

### Stage 2: Model Training
- Fine-tune YOLOv11 on custom dataset
- Automatic MLflow logging
- Hyperparameter tracking
- Training curves visualization

### Stage 3: Model Registry
- Best model selected by mAP
- Versioned with DVC
- Ready for deployment

### Stage 4: Deployment
- Python inference engine
- Next.js web interface
- Real-time object detection

---

## 🎯 Detection Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | chair_broken | Damaged chair |
| 1 | chair_wornout | Worn out chair |
| 2 | sofa_broken | Damaged sofa |
| 3 | sofa_wornout | Worn out sofa |
| 4 | table_broken | Damaged table |
| 5 | table_wornout | Worn out table |

---

## 📱 Frontend Features

- Real-time video feed with bounding boxes
- Object condition status display
- Detection history logging
- Mobile-responsive design

---

## 🧪 Running Tests

```bash
# Test inference on sample video
python computer-vision/src/inference.py \
    --source computer-vision/data/raw/test-1.mp4 \
    --output test_results

# Validate model
python -c "from ultralytics import YOLO; YOLO('computer-vision/models/best.pt').val()"
```

---

## 📈 Results

View experiment results at:
- **MLflow Dashboard**: https://dagshub.com/arsal6010/VisionAI.mlflow

Example metrics:
- mAP@50: 0.85+
- mAP@50-95: 0.65+
- Inference Speed: 30+ FPS (GPU)

---

## 🛠️ Technologies

### Computer Vision
- **YOLOv11** (Ultralytics) - State-of-the-art object detection
- **OpenCV** - Image/video processing
- **PyTorch** - Deep learning framework

### MLOps
- **DagsHub** - MLOps platform
- **MLflow** - Experiment tracking
- **DVC** - Data version control

### Frontend
- **Next.js 15** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling

---

## 👥 Contributors

- FYP Team - Final Year Project 2025

---

## 📝 License

This project is part of a Final Year Project (FYP).

---

## 🔗 Links

- **DagsHub Repository**: https://dagshub.com/arsal6010/FYP_visionAI
- **MLflow Dashboard**: https://dagshub.com/arsal6010/FYP_visionAI.mlflow
- **Documentation**: [Project Wiki]

---

*Last updated: December 2025*
