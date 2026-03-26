"""
VisionAI - FastAPI Backend
Serves YOLO model predictions via REST API
"""

import io
import sys
import base64
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "computer-vision" / "src"))
sys.path.insert(0, str(BACKEND_DIR))
from utils import get_latest_model

app = FastAPI(
    title="VisionAI API",
    description="Household Object Condition Detection API powered by YOLOv11",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_model_path = None

CONDITION_KEYWORDS = {"broken", "wornout", "damaged", "worn"}
CONDITION_COLORS_BGR = {
    "broken": (0, 0, 220),
    "wornout": (0, 140, 255),
    "damaged": (0, 0, 220),
    "good": (0, 200, 0),
}
DEFAULT_COLOR = (200, 120, 50)

# Tableau-10 palette (BGR) for general class coloring
PALETTE = [
    (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
    (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241),
    (98, 118, 150), (172, 176, 184),
]


class Detection(BaseModel):
    bbox: list[int]
    class_id: int
    class_name: str
    confidence: float
    furniture_type: Optional[str] = None
    condition: Optional[str] = None


class DetectionResponse(BaseModel):
    detections: list[Detection]
    count: int
    inference_time_ms: float
    image_width: int
    image_height: int
    model_path: Optional[str] = None
    model_classes: list[str] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str]
    classes: list[str]


def get_model():
    global _model, _model_path
    if _model is None:
        from ultralytics import YOLO

        models_dir = PROJECT_ROOT / "computer-vision" / "models"
        # Priority: best.pt > my_model.pt > last.pt > any .pt
        for name in ["best.pt", "my_model.pt", "last.pt"]:
            candidate = models_dir / name
            if candidate.exists():
                _model_path = str(candidate)
                break
        else:
            _model_path = get_latest_model(str(models_dir)) or "yolo11n.pt"

        _model = YOLO(_model_path)
    return _model


def parse_class_name(class_name: str):
    """Extract furniture type and condition from a class name like 'chair_broken'."""
    parts = class_name.lower().replace("-", "_").split("_")
    condition = None
    furniture = class_name

    for kw in CONDITION_KEYWORDS:
        if kw in parts:
            condition = kw
            furniture = class_name.lower().replace(f"_{kw}", "").replace(f"{kw}_", "")
            break

    return furniture, condition


def run_detection(image: np.ndarray, conf: float = 0.5) -> list[dict]:
    model = get_model()
    results = model(image, verbose=False, conf=conf)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            xyxy = boxes[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            class_id = int(boxes[i].cls.item())
            confidence = float(boxes[i].conf.item())
            class_name = model.names.get(class_id, f"class_{class_id}")
            furniture_type, condition = parse_class_name(class_name)

            detections.append({
                "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "furniture_type": furniture_type,
                "condition": condition,
            })

    return detections


def color_for_detection(det: dict) -> tuple:
    condition = det.get("condition")
    if condition and condition in CONDITION_COLORS_BGR:
        return CONDITION_COLORS_BGR[condition]
    return PALETTE[det["class_id"] % len(PALETTE)]


def draw_boxes(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    annotated = image.copy()

    for det in detections:
        xmin, ymin, xmax, ymax = det["bbox"]
        color = color_for_detection(det)

        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)

        cond_str = f": {det['condition'].upper()} " if det.get("condition") else " "
        label = f"{det['class_name'].upper()}{cond_str}{int(det['confidence'] * 100)}%"

        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(ymin, label_size[1] + 10)
        cv2.rectangle(
            annotated,
            (xmin, label_ymin - label_size[1] - 10),
            (xmin + label_size[0] + 4, label_ymin + baseline - 10),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            annotated, label,
            (xmin + 2, label_ymin - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    return annotated


@app.get("/", response_model=HealthResponse)
async def health_check():
    model = get_model()
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_path=_model_path,
        classes=list(model.names.values()),
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.1, le=1.0),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG, PNG, etc.)")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image")

    t0 = time.perf_counter()
    detections = run_detection(image, conf=confidence)
    inference_ms = (time.perf_counter() - t0) * 1000

    model = get_model()
    h, w = image.shape[:2]
    return DetectionResponse(
        detections=detections,
        count=len(detections),
        inference_time_ms=round(inference_ms, 2),
        image_width=w,
        image_height=h,
        model_path=_model_path,
        model_classes=list(model.names.values()),
    )


@app.post("/detect/annotated")
async def detect_annotated(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.1, le=1.0),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image")

    detections = run_detection(image, conf=confidence)
    annotated = draw_boxes(image, detections)

    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"X-Detection-Count": str(len(detections))},
    )


def crop_detections(image: np.ndarray, detections: list[dict]) -> list[str]:
    """Crop each detected object and return as base64 JPEG strings."""
    crops = []
    for det in detections:
        xmin, ymin, xmax, ymax = det["bbox"]
        h, w = image.shape[:2]
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        crop = image[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            crops.append("")
            continue
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        crops.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
    return crops


@app.post("/detect/full")
async def detect_full(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.1, le=1.0),
):
    """Returns detections JSON, annotated image, and cropped images as base64."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image")

    t0 = time.perf_counter()
    detections = run_detection(image, conf=confidence)
    inference_ms = (time.perf_counter() - t0) * 1000

    annotated = draw_boxes(image, detections)
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")

    cropped_images = crop_detections(image, detections)

    model = get_model()
    h, w = image.shape[:2]
    return {
        "detections": detections,
        "count": len(detections),
        "inference_time_ms": round(inference_ms, 2),
        "image_width": w,
        "image_height": h,
        "annotated_image": f"data:image/jpeg;base64,{b64_image}",
        "cropped_images": [f"data:image/jpeg;base64,{c}" if c else "" for c in cropped_images],
        "model_classes": list(model.names.values()),
    }


# ── LLM Analysis & Search Endpoints ──


class AnalyzeRequest(BaseModel):
    crop_image: str  # base64 data URI or raw base64


class SearchRequest(BaseModel):
    analysis: dict
    city: str
    mode: str = "exact"  # "exact" or "alternative"


class ReportRequest(BaseModel):
    items: list[dict]


def _strip_data_uri(b64_string: str) -> str:
    """Remove data:image/...;base64, prefix if present."""
    if "," in b64_string and b64_string.startswith("data:"):
        return b64_string.split(",", 1)[1]
    return b64_string


@app.post("/analyze")
async def analyze_furniture_endpoint(req: AnalyzeRequest):
    """Send a cropped furniture image to Gemini Vision for analysis."""
    from services.gemini_service import analyze_furniture

    raw_b64 = _strip_data_uri(req.crop_image)
    try:
        analysis = analyze_furniture(raw_b64)
        return {"status": "ok", "analysis": analysis}
    except Exception as e:
        err = str(e)
        if "429" in err or "Too Many Requests" in err or "RESOURCE_EXHAUSTED" in err:
            raise HTTPException(429, "Gemini rate limit reached. Please wait a moment and try again.")
        if "API_KEY" in err or "api_key" in err.lower():
            raise HTTPException(401, "Invalid or missing Gemini API key. Check your .env file.")
        raise HTTPException(500, f"Gemini analysis failed: {err}")


@app.post("/search")
async def search_furniture_endpoint(req: SearchRequest):
    """Search for furniture products based on Gemini analysis."""
    from services.search_service import search_exact_match, search_alternative

    try:
        if req.mode == "alternative":
            results = search_alternative(req.analysis, req.city)
        else:
            results = search_exact_match(req.analysis, req.city)
        return {"status": "ok", "results": results, "mode": req.mode, "city": req.city}
    except Exception as e:
        raise HTTPException(500, f"Search failed: {e}")


@app.post("/report")
async def generate_report_endpoint(req: ReportRequest):
    """Generate a comprehensive furniture report from all analyzed items."""
    from services.gemini_service import generate_report

    try:
        report_md = generate_report(req.items)
        return {"status": "ok", "report": report_md}
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
