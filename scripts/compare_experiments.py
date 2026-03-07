"""
Compare MLflow experiment runs from DagsHub.
Usage:
  python scripts/compare_experiments.py
  python scripts/compare_experiments.py --top 5
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "computer-vision" / "src"))
from utils import load_params


def main():
    parser = argparse.ArgumentParser(description="Compare MLflow experiments")
    parser.add_argument("--top", type=int, default=10, help="Number of runs to show")
    parser.add_argument("--metric", type=str, default="mAP50", help="Sort metric")
    args = parser.parse_args()

    params = load_params()
    owner = os.getenv("DAGSHUB_OWNER", params.get("experiment", {}).get("dagshub_owner", "arsal6010"))
    repo = os.getenv("DAGSHUB_REPO", params.get("experiment", {}).get("dagshub_repo", "FYP_visionAI"))

    tracking_uri = f"https://dagshub.com/{owner}/{repo}.mlflow"

    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)

        experiment_name = params.get("experiment", {}).get("experiment_name", "YOLOv11-Household-Objects")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"No experiment found: {experiment_name}")
            print(f"Check your MLflow dashboard: {tracking_uri}")
            return

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=args.top,
            order_by=[f"metrics.{args.metric} DESC"],
        )

        if runs.empty:
            print("No runs found.")
            return

        cols = ["run_id", "start_time", "status"]
        metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
        param_cols = [c for c in runs.columns if c.startswith("params.") and c in [
            "params.epochs", "params.batch_size", "params.lr0", "params.model"
        ]]
        display_cols = cols + param_cols + metric_cols

        available = [c for c in display_cols if c in runs.columns]
        print(f"\nTop {args.top} runs sorted by {args.metric}:\n")
        print(runs[available].to_string(index=False))
        print(f"\nMLflow dashboard: {tracking_uri}")

    except ImportError:
        print("mlflow not installed. Run: pip install mlflow")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Make sure you can access: {tracking_uri}")


if __name__ == "__main__":
    main()
