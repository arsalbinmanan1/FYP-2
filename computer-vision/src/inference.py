"""
Inference Module for VisionAI
Real-time object detection and condition assessment
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    logger, print_banner, ensure_dir, get_latest_model,
    FURNITURE_CLASSES, CONDITION_COLORS,
    get_condition_from_class, get_furniture_type, print_detection_results
)


class VisionAIDetector:
    """
    Object detector for furniture condition assessment.
    Wraps YOLO model with custom visualization and output handling.
    """
    
    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.5,
        device: str = "auto"
    ):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLO model weights (.pt file)
            confidence_threshold: Minimum confidence for detections
            device: Inference device (auto, cpu, cuda:0, etc.)
        """
        from ultralytics import YOLO
        
        # Find model if not specified
        if model_path is None:
            model_path = get_latest_model()
        
        if model_path is None or not Path(model_path).exists():
            logger.warning(f"Model not found at {model_path}, using default yolo11n.pt")
            model_path = "yolo11n.pt"
        
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.conf_threshold = confidence_threshold
        self.device = device
        self.labels = self.model.names
        
        # Visualization colors (Tableau 10 color scheme)
        self.bbox_colors = [
            (164, 120, 87), (68, 148, 228), (93, 97, 209),
            (178, 182, 133), (88, 159, 106), (96, 202, 231),
            (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)
        ]
        
        logger.info(f"Loaded model: {model_path}")
        logger.info(f"Classes: {list(self.labels.values())}")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run detection on a single image.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        results = self.model(image, verbose=False, conf=self.conf_threshold)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for i in range(len(boxes)):
                # Get bounding box
                xyxy = boxes[i].xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                
                # Get class and confidence
                class_id = int(boxes[i].cls.item())
                confidence = float(boxes[i].conf.item())
                class_name = self.labels.get(class_id, f"class_{class_id}")
                
                # Get furniture type and condition
                furniture_type = get_furniture_type(class_id)
                condition = get_condition_from_class(class_id)
                
                detections.append({
                    'bbox': [xmin, ymin, xmax, ymax],
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'furniture_type': furniture_type,
                    'condition': condition
                })
        
        return detections
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        show_confidence: bool = True,
        show_condition: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: BGR image as numpy array
            detections: List of detection dictionaries
            show_confidence: Show confidence percentage
            show_condition: Show condition label
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for det in detections:
            xmin, ymin, xmax, ymax = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence']
            condition = det.get('condition', 'unknown')
            furniture = det.get('furniture_type', det['class_name'])
            
            # Get color based on condition
            if condition in CONDITION_COLORS:
                color = CONDITION_COLORS[condition]
            else:
                color = self.bbox_colors[class_id % len(self.bbox_colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Create label
            if show_condition and condition != 'unknown':
                label = f"{furniture.upper()}: {condition.upper()}"
            else:
                label = furniture.upper()
            
            if show_confidence:
                label += f" {int(confidence * 100)}%"
            
            # Draw label background
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(
                annotated,
                (xmin, label_ymin - label_size[1] - 10),
                (xmin + label_size[0], label_ymin + baseline - 10),
                color, cv2.FILLED
            )
            
            # Draw label text
            cv2.putText(
                annotated, label,
                (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        
        return annotated
    
    def crop_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        output_dir: str,
        prefix: str = ""
    ) -> List[str]:
        """
        Crop and save detected objects.
        
        Args:
            image: Original image
            detections: List of detections
            output_dir: Directory to save crops
            prefix: Prefix for filenames
            
        Returns:
            List of saved file paths
        """
        ensure_dir(output_dir)
        saved_paths = []
        
        for i, det in enumerate(detections):
            xmin, ymin, xmax, ymax = det['bbox']
            crop = image[ymin:ymax, xmin:xmax]
            
            if crop.size > 0:
                furniture = det.get('furniture_type', 'object')
                condition = det.get('condition', 'unknown')
                confidence = int(det['confidence'] * 100)
                
                filename = f"{prefix}{furniture}_{condition}_{confidence}_{i}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, crop)
                saved_paths.append(filepath)
        
        return saved_paths


def process_image(
    detector: VisionAIDetector,
    image_path: str,
    output_dir: str = "output",
    save_crops: bool = True,
    show_result: bool = True
) -> List[Dict]:
    """Process a single image."""
    print_banner(f"Processing: {os.path.basename(image_path)}")
    
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return []
    
    # Run detection
    detections = detector.detect(image)
    
    # Draw results
    annotated = detector.draw_detections(image, detections)
    
    # Save annotated image
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, annotated)
    logger.info(f"Saved annotated image: {output_path}")
    
    # Save crops
    if save_crops and detections:
        crops_dir = os.path.join(output_dir, "crops")
        crop_paths = detector.crop_detections(image, detections, crops_dir)
        logger.info(f"Saved {len(crop_paths)} crops to {crops_dir}")
    
    # Print results
    print_detection_results(detections)
    
    # Show result
    if show_result:
        cv2.imshow("VisionAI Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return detections


def process_video(
    detector: VisionAIDetector,
    video_path: str,
    output_dir: str = "output",
    display: bool = True,
    record: bool = True,
    frame_skip: int = 1
) -> Dict:
    """
    Process video with real-time detection.
    
    Args:
        detector: VisionAIDetector instance
        video_path: Path to video file
        output_dir: Output directory
        display: Show live preview
        record: Save annotated video
        frame_skip: Process every N frames
        
    Returns:
        Statistics dictionary
    """
    print_banner(f"Processing Video: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return {}
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    
    # Setup video writer
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, f"annotated_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) if record else None
    
    # Processing loop
    frame_count = 0
    total_detections = 0
    fps_buffer = []
    
    while True:
        t_start = time.perf_counter()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection on selected frames
        if frame_count % frame_skip == 0:
            detections = detector.detect(frame)
            annotated = detector.draw_detections(frame, detections)
            total_detections += len(detections)
        else:
            annotated = frame
        
        # Add FPS overlay
        if fps_buffer:
            avg_fps = np.mean(fps_buffer[-30:])
            cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Progress overlay
        progress = frame_count / total_frames * 100
        cv2.putText(annotated, f"Progress: {progress:.1f}%", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Write and display
        if writer:
            writer.write(annotated)
        
        if display:
            cv2.imshow("VisionAI Detection", annotated)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)
        
        # Calculate FPS
        t_end = time.perf_counter()
        fps_buffer.append(1 / (t_end - t_start))
        
        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count}/{total_frames} frames...")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    stats = {
        "frames_processed": frame_count,
        "total_detections": total_detections,
        "avg_fps": np.mean(fps_buffer) if fps_buffer else 0,
        "output_path": output_path if record else None
    }
    
    print_banner("VIDEO PROCESSING COMPLETE")
    print(f"  Frames Processed: {stats['frames_processed']}")
    print(f"  Total Detections: {stats['total_detections']}")
    print(f"  Average FPS: {stats['avg_fps']:.2f}")
    if stats['output_path']:
        print(f"  Output: {stats['output_path']}")
    
    return stats


def process_webcam(
    detector: VisionAIDetector,
    camera_index: int = 0,
    resolution: Tuple[int, int] = None,
    record: bool = False,
    output_path: str = "webcam_output.mp4"
):
    """
    Run real-time detection on webcam feed.
    
    Args:
        detector: VisionAIDetector instance
        camera_index: Camera device index
        resolution: Optional (width, height) tuple
        record: Whether to record the output
        output_path: Path for recorded video
    """
    print_banner("WEBCAM DETECTION")
    print("  Press 'q' to quit")
    print("  Press 's' to save screenshot")
    print("  Press 'p' to pause")
    
    cap = cv2.VideoCapture(camera_index)
    
    if resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    writer = None
    if record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    fps_buffer = []
    screenshot_count = 0
    
    while True:
        t_start = time.perf_counter()
        
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame from webcam")
            break
        
        # Detect and annotate
        detections = detector.detect(frame)
        annotated = detector.draw_detections(frame, detections)
        
        # FPS overlay
        if fps_buffer:
            avg_fps = np.mean(fps_buffer[-30:])
            cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Object count overlay
        cv2.putText(annotated, f"Objects: {len(detections)}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if writer:
            writer.write(annotated)
        
        cv2.imshow("VisionAI - Webcam", annotated)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{screenshot_count}.jpg"
            cv2.imwrite(filename, annotated)
            logger.info(f"Saved screenshot: {filename}")
            screenshot_count += 1
        elif key == ord('p'):
            cv2.waitKey(0)
        
        t_end = time.perf_counter()
        fps_buffer.append(1 / (t_end - t_start))
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    logger.info(f"Average FPS: {np.mean(fps_buffer):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='VisionAI Inference - Furniture Condition Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect in image
  python inference.py --source image.jpg
  
  # Detect in video
  python inference.py --source video.mp4
  
  # Webcam detection
  python inference.py --source webcam
  
  # Use custom model
  python inference.py --model best.pt --source image.jpg
        """
    )
    
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to YOLO model (.pt file)')
    parser.add_argument('--source', '-s', required=True,
                        help='Input source: image, video, folder, webcam, or usb0')
    parser.add_argument('--conf', '-c', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable live display')
    parser.add_argument('--no-record', action='store_true',
                        help='Disable video recording')
    parser.add_argument('--resolution', type=str, default=None,
                        help='Resolution WxH (e.g., 640x480)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = VisionAIDetector(
        model_path=args.model,
        confidence_threshold=args.conf
    )
    
    source = args.source.lower()
    
    # Parse resolution
    resolution = None
    if args.resolution:
        w, h = args.resolution.split('x')
        resolution = (int(w), int(h))
    
    # Determine source type and process
    if source in ['webcam', 'camera', 'usb0']:
        cam_idx = 0
        if source.startswith('usb'):
            cam_idx = int(source[3:])
        process_webcam(
            detector,
            camera_index=cam_idx,
            resolution=resolution,
            record=not args.no_record
        )
    
    elif os.path.isfile(source):
        # Check if video or image
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        if Path(source).suffix.lower() in video_exts:
            process_video(
                detector,
                video_path=source,
                output_dir=args.output,
                display=not args.no_display,
                record=not args.no_record
            )
        else:
            process_image(
                detector,
                image_path=source,
                output_dir=args.output,
                show_result=not args.no_display
            )
    
    elif os.path.isdir(source):
        # Process all images in folder
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        images = [f for f in Path(source).iterdir() 
                 if f.suffix.lower() in image_exts]
        
        logger.info(f"Found {len(images)} images in folder")
        for img_path in images:
            process_image(
                detector,
                image_path=str(img_path),
                output_dir=args.output,
                show_result=False
            )
    
    else:
        logger.error(f"Invalid source: {source}")
        sys.exit(1)


if __name__ == "__main__":
    main()

