#!/usr/bin/env python3
"""
DVC Setup Script for VisionAI
Run this script to initialize DVC and configure DagsHub remote storage
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the output."""
    print(f"  Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return False
    if result.stdout:
        print(f"  Output: {result.stdout.strip()}")
    return True


def main():
    print("=" * 60)
    print("  VisionAI - DVC Setup Script")
    print("=" * 60)
    print()
    
    # Configuration
    DAGSHUB_OWNER = input("Enter your DagsHub username [arsal6010]: ").strip() or "arsal6010"
    DAGSHUB_REPO = input("Enter your DagsHub repo name [FYP_visionAI]: ").strip() or "FYP_visionAI"
    
    print()
    print(f"  DagsHub Owner: {DAGSHUB_OWNER}")
    print(f"  DagsHub Repo: {DAGSHUB_REPO}")
    print()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"  Working directory: {project_root}")
    print()
    
    # Step 1: Check if DVC is installed
    print("Step 1: Checking DVC installation...")
    if not run_command("dvc version", check=False):
        print("  DVC not found. Installing...")
        run_command("pip install dvc dvc-s3")
    else:
        print("  DVC is installed!")
    print()
    
    # Step 2: Initialize DVC (if not already)
    print("Step 2: Initializing DVC...")
    dvc_dir = project_root / ".dvc"
    if dvc_dir.exists():
        print("  DVC already initialized!")
    else:
        run_command("dvc init")
    print()
    
    # Step 3: Configure DagsHub remote
    print("Step 3: Configuring DagsHub remote storage...")
    s3_endpoint = f"https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.s3"
    
    run_command("dvc remote add -d origin s3://dvc", check=False)
    run_command(f"dvc remote modify origin endpointurl {s3_endpoint}")
    print()
    
    # Step 4: Setup authentication
    print("Step 4: Setting up authentication...")
    print("  Please ensure you have logged into DagsHub:")
    print("  Run: dagshub login")
    print()
    
    # Step 5: Create placeholder files for data directories
    print("Step 5: Creating data directory structure...")
    data_dirs = [
        "computer-vision/data/raw",
        "computer-vision/data/processed",
        "computer-vision/models"
    ]
    
    for dir_path in data_dirs:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        gitkeep = full_path / ".gitkeep"
        gitkeep.touch()
        print(f"  Created: {dir_path}")
    print()
    
    # Step 6: Add data to DVC (if exists)
    print("Step 6: Ready to track data with DVC")
    print()
    print("  To add data to DVC tracking, run:")
    print("  dvc add computer-vision/data/processed")
    print("  dvc add computer-vision/models/best.pt")
    print()
    print("  To push data to DagsHub storage:")
    print("  dvc push -r origin")
    print()
    
    print("=" * 60)
    print("  DVC Setup Complete!")
    print("=" * 60)
    print()
    print("  Next steps:")
    print("  1. Login to DagsHub: dagshub login")
    print(f"  2. View your repo: https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}")
    print("  3. Add your dataset and run: dvc push -r origin")
    print()


if __name__ == "__main__":
    main()

