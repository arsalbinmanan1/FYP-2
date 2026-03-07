"""
Evaluation Module for VisionAI
Runs model validation and writes DVC-trackable metrics.json
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import logger, print_banner


def evaluate_model(
    model_path: str,
    data_yaml: str,
    output_dir: str = "evaluation_results",
    img_size: int = 640,
    conf: float = 0.5,
    device: str = "auto",
) -> dict:
    from ultralytics import YOLO

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print_banner("MODEL EVALUATION")
    print(f"  Model: {model_path}")
    print(f"  Data:  {data_yaml}")

    model = YOLO(str(model_path))
    results = model.val(
        data=str(data_path),
        imgsz=img_size,
        conf=conf,
        device=device if device != "auto" else None,
        project=str(out),
        name="val",
        exist_ok=True,
    )

    rd = results.results_dict
    metrics = {
        "mAP50": round(float(rd.get("metrics/mAP50(B)", 0)), 5),
        "mAP50-95": round(float(rd.get("metrics/mAP50-95(B)", 0)), 5),
        "precision": round(float(rd.get("metrics/precision(B)", 0)), 5),
        "recall": round(float(rd.get("metrics/recall(B)", 0)), 5),
        "fitness": round(float(rd.get("fitness", 0)), 5),
    }

    per_class = {}
    if hasattr(results, "ap_class_index") and hasattr(results, "names"):
        for i, cls_idx in enumerate(results.ap_class_index):
            cls_name = results.names[int(cls_idx)]
            per_class[cls_name] = {
                "precision": round(float(results.box.p[i]), 5),
                "recall": round(float(results.box.r[i]), 5),
                "ap50": round(float(results.box.ap50[i]), 5),
                "ap": round(float(results.box.ap[i]), 5),
            }

    output = {
        "model": str(model_path),
        "dataset": str(data_yaml),
        **metrics,
        "per_class": per_class,
    }

    metrics_path = out / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Metrics written to {metrics_path}")

    print_banner("EVALUATION RESULTS")
    print(f"  mAP@50:    {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    if per_class:
        print()
        for cls, vals in per_class.items():
            print(f"  {cls:20s}  P={vals['precision']:.3f}  R={vals['recall']:.3f}  AP50={vals['ap50']:.3f}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate VisionAI model")
    parser.add_argument("--model", "-m", required=True, help="Path to .pt model")
    parser.add_argument("--data", "-d", required=True, help="Path to data.yaml")
    parser.add_argument("--output", "-o", default="evaluation_results")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        output_dir=args.output,
        img_size=args.img_size,
        conf=args.conf,
        device=args.device,
    )


if __name__ == "__main__":
    main()
