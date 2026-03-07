"""
Push/pull data to DagsHub DVC remote.
Usage:
  python scripts/push_data.py push
  python scripts/push_data.py pull
  python scripts/push_data.py status
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run(cmd):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"

    if action == "push":
        print("Tracking data with DVC...")
        run("dvc add computer-vision/data/processed")
        run("dvc add computer-vision/data/raw")
        print("\nPushing to DagsHub remote...")
        run("dvc push -r origin")
        print("\nDon't forget to commit the .dvc files:")
        print("  git add computer-vision/data/*.dvc .gitignore")
        print('  git commit -m "chore: update DVC-tracked data"')

    elif action == "pull":
        print("Pulling data from DagsHub remote...")
        run("dvc pull -r origin")

    elif action == "status":
        print("DVC status:")
        run("dvc status")
        print("\nDVC remote list:")
        run("dvc remote list")
        print("\nDVC data status:")
        run("dvc data status")

    else:
        print(f"Unknown action: {action}")
        print("Usage: python scripts/push_data.py [push|pull|status]")


if __name__ == "__main__":
    main()
