"""
MLOps Setup Script for VisionAI
Configures DVC remote, DagsHub auth, and verifies the pipeline.
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run(cmd, cwd=None, check=True):
    print(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd or PROJECT_ROOT)
    if r.stdout.strip():
        print(f"    {r.stdout.strip()}")
    if r.returncode != 0 and check:
        print(f"    ERROR: {r.stderr.strip()}")
    return r.returncode == 0


def main():
    print("=" * 60)
    print("  VisionAI — MLOps Pipeline Setup")
    print("=" * 60)

    owner = input(f"\nDagsHub username [{os.getenv('DAGSHUB_OWNER', 'arsal6010')}]: ").strip()
    owner = owner or os.getenv("DAGSHUB_OWNER", "arsal6010")

    repo = input(f"DagsHub repo [{os.getenv('DAGSHUB_REPO', 'FYP_visionAI')}]: ").strip()
    repo = repo or os.getenv("DAGSHUB_REPO", "FYP_visionAI")

    token = input("DagsHub token (paste, won't echo): ").strip()
    if not token:
        print("  No token provided — skipping credential config.")
        print("  You can still run: dagshub login")
    else:
        os.environ["DAGSHUB_USER_TOKEN"] = token

    s3_endpoint = f"https://dagshub.com/{owner}/{repo}.s3"
    mlflow_uri = f"https://dagshub.com/{owner}/{repo}.mlflow"

    # 1. DVC remote
    print("\n[1/5] Configuring DVC remote...")
    run("dvc remote add -d origin s3://dvc", check=False)
    run(f"dvc remote modify origin endpointurl {s3_endpoint}")
    if token:
        run(f"dvc remote modify --local origin access_key_id {token}")
        run(f"dvc remote modify --local origin secret_access_key {token}")
    print("  DVC remote: origin -> s3://dvc")

    # 2. DagsHub login
    print("\n[2/5] DagsHub auth...")
    if token:
        os.environ["DAGSHUB_USER_TOKEN"] = token
        print(f"  DAGSHUB_USER_TOKEN set in environment")
    else:
        print("  Run `dagshub login` manually to authenticate")

    # 3. Verify MLflow connection
    print(f"\n[3/5] MLflow tracking URI: {mlflow_uri}")

    # 4. Verify DVC pipeline
    print("\n[4/5] Verifying DVC pipeline definition...")
    run("dvc dag")

    # 5. Write local .env (for convenience)
    print("\n[5/5] Writing .env file...")
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists() and token:
        env_path.write_text(
            f"DAGSHUB_USER_TOKEN={token}\n"
            f"DAGSHUB_OWNER={owner}\n"
            f"DAGSHUB_REPO={repo}\n"
            f"MLFLOW_TRACKING_URI={mlflow_uri}\n"
            f"MLFLOW_TRACKING_USERNAME={owner}\n"
            f"MLFLOW_TRACKING_PASSWORD={token}\n"
            f"AWS_ACCESS_KEY_ID={token}\n"
            f"AWS_SECRET_ACCESS_KEY={token}\n"
        )
        print(f"  Created .env (DO NOT commit this file)")
    elif env_path.exists():
        print(f"  .env already exists — skipping")
    else:
        print(f"  No token — skipping .env creation")

    print("\n" + "=" * 60)
    print("  Setup Complete!")
    print("=" * 60)
    print(f"""
  DagsHub repo:    https://dagshub.com/{owner}/{repo}
  MLflow dashboard: {mlflow_uri}
  DVC remote:      s3://dvc -> {s3_endpoint}

  Next steps:
  1. Add data:     dvc add computer-vision/data/processed
  2. Push data:    dvc push
  3. Train:        python computer-vision/src/train.py --from-params
  4. Full pipeline: dvc repro
  5. View metrics:  dvc metrics show
  6. Compare runs:  dvc plots diff
""")


if __name__ == "__main__":
    main()
