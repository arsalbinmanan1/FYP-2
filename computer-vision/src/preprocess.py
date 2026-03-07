"""
Data Preprocessing Module for VisionAI
Handles video to frames conversion and dataset preparation
"""

import os
import sys
import argparse
import cv2
import shutil
import random
import yaml
from pathlib import Path
from typing import Optional, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import logger, ensure_dir, print_banner, get_class_names, load_params


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_skip: int = 1,
    frame_format: str = 'jpg',
    prefix: str = 'frame',
    max_frames: Optional[int] = None
) -> int:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        frame_skip: Extract every Nth frame (1 = all frames)
        frame_format: Output image format (jpg, png, bmp)
        prefix: Prefix for output filenames
        max_frames: Maximum number of frames to extract (None = all)
        
    Returns:
        Number of frames extracted
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return 0
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print_banner("VIDEO INFORMATION")
    print(f"  File: {os.path.basename(video_path)}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Frame skip: Every {frame_skip} frame(s)")
    
    expected = total_frames // frame_skip
    if max_frames:
        expected = min(expected, max_frames)
    print(f"  Expected output: ~{expected} frames")
    
    # Create output directory
    ensure_dir(output_dir)
    
    frame_count = 0
    saved_count = 0
    
    logger.info("Extracting frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames and saved_count >= max_frames:
            break
        
        # Save frame if it matches the skip pattern
        if frame_count % frame_skip == 0:
            filename = f"{prefix}_{saved_count:06d}.{frame_format}"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                logger.info(f"  Extracted {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    
    print_banner("EXTRACTION COMPLETE")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Frames saved: {saved_count}")
    print(f"  Output directory: {os.path.abspath(output_dir)}")
    
    return saved_count


def create_yolo_dataset(
    source_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    furniture_types: List[str] = None,
    conditions: List[str] = None
) -> Optional[str]:
    """
    Create YOLO format dataset from organized image folders.
    
    Expected source structure:
        source_dir/
            chair/
                broken/
                wornout/
            sofa/
                broken/
                wornout/
            table/
                broken/
                wornout/
    
    Args:
        source_dir: Source directory with organized images
        output_dir: Output directory for YOLO dataset
        train_split: Fraction of data for training (0.8 = 80%)
        furniture_types: List of furniture types
        conditions: List of conditions
        
    Returns:
        Path to created data.yaml file, or None if failed
    """
    if furniture_types is None:
        furniture_types = ["chair", "sofa", "table"]
    if conditions is None:
        conditions = ["broken", "wornout"]
    
    print_banner("CREATING YOLO DATASET")
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create directory structure
    (output_path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (output_path / "val" / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "val" / "labels").mkdir(parents=True, exist_ok=True)
    
    # Create class mapping
    class_mapping = {}
    class_id = 0
    for furniture in furniture_types:
        for condition in conditions:
            class_mapping[f"{furniture}_{condition}"] = class_id
            class_id += 1
    
    logger.info(f"Class mapping: {class_mapping}")
    
    # Collect all images
    all_images = []
    
    for furniture in furniture_types:
        for condition in conditions:
            folder_path = source_path / furniture / condition
            if folder_path.exists():
                images = (
                    list(folder_path.glob("*.jpg")) + 
                    list(folder_path.glob("*.jpeg")) + 
                    list(folder_path.glob("*.png"))
                )
                logger.info(f"Found {len(images)} {furniture} {condition} images")
                
                for img_path in images:
                    all_images.append({
                        'path': img_path,
                        'furniture': furniture,
                        'condition': condition,
                        'class_id': class_mapping[f"{furniture}_{condition}"]
                    })
    
    if len(all_images) == 0:
        logger.error(f"No images found in {source_dir}")
        return None
    
    logger.info(f"Total images found: {len(all_images)}")
    
    # Shuffle and split
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_split)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    logger.info(f"Training images: {len(train_images)}")
    logger.info(f"Validation images: {len(val_images)}")
    
    # Process images
    for split, images in [("train", train_images), ("val", val_images)]:
        for i, img_data in enumerate(images):
            # Read image to get dimensions
            img = cv2.imread(str(img_data['path']))
            if img is None:
                logger.warning(f"Could not read: {img_data['path']}")
                continue
            
            h, w = img.shape[:2]
            
            # Copy image with unique name
            img_name = f"{img_data['furniture']}_{img_data['condition']}_{i:04d}.jpg"
            dest_img = output_path / split / "images" / img_name
            cv2.imwrite(str(dest_img), img)
            
            # Create YOLO label (full image bounding box)
            # Format: class_id x_center y_center width height (normalized)
            x_center = 0.5
            y_center = 0.5
            bbox_width = 0.8   # 80% of image
            bbox_height = 0.8
            
            label_name = img_name.replace('.jpg', '.txt')
            label_path = output_path / split / "labels" / label_name
            
            with open(label_path, 'w') as f:
                f.write(f"{img_data['class_id']} {x_center} {y_center} {bbox_width} {bbox_height}\n")
    
    # Create data.yaml
    class_names = [f"{f}_{c}" for f in furniture_types for c in conditions]
    
    config = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print_banner("DATASET CREATED")
    print(f"  Location: {output_path.absolute()}")
    print(f"  Training images: {len(train_images)}")
    print(f"  Validation images: {len(val_images)}")
    print(f"  Classes: {class_names}")
    print(f"  Config: {yaml_path}")
    
    return str(yaml_path)


def prepare_annotations_from_roboflow(
    export_dir: str,
    output_dir: str
) -> str:
    """
    Prepare dataset exported from Roboflow.
    Roboflow exports are already in YOLO format, this just validates and copies.
    
    Args:
        export_dir: Directory containing Roboflow export
        output_dir: Where to copy the prepared dataset
        
    Returns:
        Path to data.yaml
    """
    export_path = Path(export_dir)
    output_path = Path(output_dir)
    
    # Check for data.yaml in export
    yaml_files = list(export_path.glob("*.yaml"))
    
    if not yaml_files:
        logger.error("No YAML config found in Roboflow export")
        return None
    
    # Copy entire export
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.copytree(export_path, output_path)
    
    # Find and return data.yaml path
    yaml_path = list(output_path.glob("*.yaml"))[0]
    logger.info(f"Dataset prepared at: {output_path}")
    
    return str(yaml_path)


def main():
    parser = argparse.ArgumentParser(
        description='Data preprocessing for VisionAI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract frames command
    extract_parser = subparsers.add_parser('extract', help='Extract frames from video')
    extract_parser.add_argument('--video', '-v', required=True, help='Path to video file')
    extract_parser.add_argument('--output', '-o', default=None, help='Output directory')
    extract_parser.add_argument('--skip', '-s', type=int, default=5, help='Frame skip interval')
    extract_parser.add_argument('--format', '-f', default='jpg', choices=['jpg', 'png'])
    extract_parser.add_argument('--max', '-m', type=int, default=None, help='Max frames')
    
    # Create dataset command
    dataset_parser = subparsers.add_parser('create-dataset', help='Create YOLO dataset')
    dataset_parser.add_argument('--source', '-s', required=True, help='Source directory')
    dataset_parser.add_argument('--output', '-o', required=True, help='Output directory')
    dataset_parser.add_argument('--split', type=float, default=0.8, help='Train split ratio')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        output_dir = args.output or f"{Path(args.video).stem}_frames"
        extract_frames(
            video_path=args.video,
            output_dir=output_dir,
            frame_skip=args.skip,
            frame_format=args.format,
            max_frames=args.max
        )
        
    elif args.command == 'create-dataset':
        create_yolo_dataset(
            source_dir=args.source,
            output_dir=args.output,
            train_split=args.split
        )
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

