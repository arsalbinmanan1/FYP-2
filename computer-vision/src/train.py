"""
Training Module for VisionAI with MLOps Integration
Supports MLflow experiment tracking via DagsHub
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent))
from utils import logger, load_params, print_banner, ensure_dir


def setup_experiment_tracking(
    dagshub_owner: str = None,
    dagshub_repo: str = None,
    experiment_name: str = "YOLOv11-Household-Objects",
) -> bool:
    if not dagshub_owner or not dagshub_repo:
        logger.info("No DagsHub credentials — MLflow tracking disabled")
        return False

    try:
        import dagshub
        import mlflow

        dagshub.init(
            repo_owner=dagshub_owner,
            repo_name=dagshub_repo,
            mlflow=True,
        )
        logger.info(f"DagsHub initialized: {dagshub_owner}/{dagshub_repo}")

        tracking_uri = f"https://dagshub.com/{dagshub_owner}/{dagshub_repo}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow tracking: {tracking_uri}")

        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

        return True
    except Exception as e:
        logger.warning(f"Experiment tracking setup failed: {e}")
        return False


def train_yolo(
    data_yaml: str,
    model: str = "yolo11s.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    learning_rate: float = 0.01,
    patience: int = 10,
    device: str = "auto",
    project: str = "VisionAI_Runs",
    name: str = None,
    resume: bool = False,
    pretrained: bool = True,
    dagshub_owner: str = None,
    dagshub_repo: str = None,
    save_metrics: bool = True,
) -> Dict[str, Any]:
    from ultralytics import YOLO

    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")

    if name is None:
        name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print_banner(f"YOLO TRAINING: {name}")
    print(f"  Model:         {model}")
    print(f"  Data:          {data_yaml}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch Size:    {batch_size}")
    print(f"  Image Size:    {img_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Patience:      {patience}")
    print(f"  Device:        {device}")

    mlflow_enabled = setup_experiment_tracking(
        dagshub_owner=dagshub_owner,
        dagshub_repo=dagshub_repo,
    )

    # Ultralytics detects MLFLOW_TRACKING_URI env var and auto-logs
    # metrics, params, and model artifacts — no manual logging needed.
    yolo_model = YOLO(model)

    results = yolo_model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=learning_rate,
        patience=patience,
        device=device if device != "auto" else None,
        project=project,
        name=name,
        resume=resume,
        pretrained=pretrained,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )

    run_dir = Path(project) / name
    best_model = run_dir / "weights" / "best.pt"
    last_model = run_dir / "weights" / "last.pt"

    metrics = {
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
    }

    if save_metrics:
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

    # Log extra artifacts to MLflow (model file + plots)
    if mlflow_enabled:
        try:
            import mlflow

            active = mlflow.active_run()
            if active:
                if best_model.exists():
                    mlflow.log_artifact(str(best_model), "models")
                for plot in run_dir.glob("*.png"):
                    mlflow.log_artifact(str(plot), "plots")
                mlflow.log_metrics(metrics)
                logger.info("Logged artifacts to MLflow")
        except Exception as e:
            logger.warning(f"MLflow artifact logging failed: {e}")

    output = {
        "run_dir": str(run_dir),
        "best_model": str(best_model) if best_model.exists() else None,
        "last_model": str(last_model) if last_model.exists() else None,
        "metrics": metrics,
    }

    print_banner("TRAINING COMPLETE")
    print(f"  Run Directory: {output['run_dir']}")
    print(f"  Best Model:    {output['best_model']}")
    print(f"  mAP@50:        {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95:     {metrics['mAP50-95']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")

    return output


def train_from_params(params_path: str = "params.yaml", **overrides) -> Dict[str, Any]:
    params = load_params(params_path)

    train_config = {
        "model": params.get("model", {}).get("architecture", "yolo11s.pt"),
        "epochs": params.get("train", {}).get("epochs", 50),
        "batch_size": params.get("train", {}).get("batch_size", 16),
        "img_size": params.get("model", {}).get("input_size", 640),
        "learning_rate": params.get("train", {}).get("learning_rate", 0.01),
        "patience": params.get("train", {}).get("patience", 10),
        "data_yaml": params.get("data", {}).get("config_path", "data.yaml"),
        "project": params.get("output", {}).get("runs_dir", "VisionAI_Runs"),
        "dagshub_owner": params.get("experiment", {}).get("dagshub_owner"),
        "dagshub_repo": params.get("experiment", {}).get("dagshub_repo"),
    }
    train_config.update(overrides)
    return train_yolo(**train_config)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO model for VisionAI with MLOps integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --from-params
  python train.py --data data.yaml --epochs 100 --batch 32
  python train.py --data data.yaml --dagshub-owner arsal6010 --dagshub-repo FYP_visionAI
        """,
    )
    parser.add_argument("--data", "-d", type=str, help="Path to data.yaml")
    parser.add_argument("--from-params", action="store_true", help="Use params.yaml")
    parser.add_argument("--model", "-m", type=str, default="yolo11s.pt")
    parser.add_argument("--epochs", "-e", type=int, default=50)
    parser.add_argument("--batch", "-b", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--project", type=str, default="VisionAI_Runs")
    parser.add_argument("--name", "-n", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dagshub-owner", type=str)
    parser.add_argument("--dagshub-repo", type=str)
    parser.add_argument("--no-mlflow", action="store_true")

    args = parser.parse_args()

    if args.from_params:
        overrides = {}
        if args.data:
            overrides["data_yaml"] = args.data
        if args.epochs != 50:
            overrides["epochs"] = args.epochs
        if args.batch != 16:
            overrides["batch_size"] = args.batch
        results = train_from_params(**overrides)
    else:
        if not args.data:
            parser.error("--data is required (or use --from-params)")

        results = train_yolo(
            data_yaml=args.data,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            learning_rate=args.lr,
            patience=args.patience,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume,
            dagshub_owner=None if args.no_mlflow else args.dagshub_owner,
            dagshub_repo=None if args.no_mlflow else args.dagshub_repo,
        )

    print(f"\nTraining completed! Best model: {results['best_model']}")


if __name__ == "__main__":
    main()
