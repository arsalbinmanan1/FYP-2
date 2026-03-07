"""
Utility functions for VisionAI project
Shared helper functions used across training, inference, and preprocessing
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== Configuration Management ==============

def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """
    Load parameters from params.yaml file.
    
    Args:
        params_path: Path to params.yaml file
        
    Returns:
        Dictionary containing all parameters
    """
    # Try multiple locations for params.yaml
    search_paths = [
        params_path,
        Path(__file__).parent.parent.parent / "params.yaml",  # Root directory
        Path.cwd() / "params.yaml",
    ]
    
    for path in search_paths:
        if Path(path).exists():
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            logger.info(f"Loaded parameters from: {path}")
            return params
    
    logger.warning("params.yaml not found, using default parameters")
    return get_default_params()


def get_default_params() -> Dict[str, Any]:
    """Return default training parameters."""
    return {
        'model': {
            'architecture': 'yolo11s.pt',
            'input_size': 640,
            'num_classes': 6
        },
        'train': {
            'epochs': 50,
            'batch_size': 16,
            'learning_rate': 0.01,
            'optimizer': 'auto',
            'patience': 10,
            'workers': 4
        },
        'data': {
            'train_split': 0.8,
            'val_split': 0.2,
            'augmentation': True
        },
        'experiment': {
            'project_name': 'VisionAI',
            'experiment_name': 'YOLOv11-Household-Objects'
        }
    }


def save_params(params: Dict[str, Any], params_path: str = "params.yaml") -> None:
    """Save parameters to YAML file."""
    with open(params_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved parameters to: {params_path}")


# ============== Class Mappings ==============

# Furniture condition classes
FURNITURE_CLASSES = {
    0: "chair_broken",
    1: "chair_wornout", 
    2: "sofa_broken",
    3: "sofa_wornout",
    4: "table_broken",
    5: "table_wornout"
}

# Reverse mapping
CLASS_NAME_TO_ID = {v: k for k, v in FURNITURE_CLASSES.items()}

# Condition colors for visualization (BGR format)
CONDITION_COLORS = {
    "broken": (0, 0, 255),      # Red
    "wornout": (0, 165, 255),   # Orange
    "good": (0, 255, 0),        # Green
    "unknown": (128, 128, 128)  # Gray
}


def get_condition_from_class(class_id: int) -> str:
    """Extract condition (broken/wornout) from class ID."""
    class_name = FURNITURE_CLASSES.get(class_id, "unknown")
    if "broken" in class_name:
        return "broken"
    elif "wornout" in class_name:
        return "wornout"
    return "unknown"


def get_furniture_type(class_id: int) -> str:
    """Extract furniture type (chair/sofa/table) from class ID."""
    class_name = FURNITURE_CLASSES.get(class_id, "unknown")
    for furniture in ["chair", "sofa", "table"]:
        if furniture in class_name:
            return furniture
    return "unknown"


def get_class_names() -> List[str]:
    """Get ordered list of class names."""
    return [FURNITURE_CLASSES[i] for i in range(len(FURNITURE_CLASSES))]


# ============== Path Utilities ==============

def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "params.yaml").exists() or (current / "dvc.yaml").exists():
            return current
        if (current / "computer-vision").exists():
            return current
        current = current.parent
    return Path.cwd()


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_latest_model(models_dir: str = "computer-vision/models") -> Optional[str]:
    """Find the latest/best model in the models directory."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return None
    
    # Priority: best.pt > last.pt > any .pt file
    for model_name in ["best.pt", "last.pt"]:
        model_path = models_path / model_name
        if model_path.exists():
            return str(model_path)
    
    # Find any .pt file
    pt_files = list(models_path.glob("*.pt"))
    if pt_files:
        # Return most recently modified
        return str(max(pt_files, key=lambda p: p.stat().st_mtime))
    
    return None


# ============== Dataset Utilities ==============

def create_data_yaml(
    data_dir: str,
    output_path: str,
    class_names: Optional[List[str]] = None
) -> str:
    """
    Create YOLO format data.yaml file.
    
    Args:
        data_dir: Root directory containing train/val folders
        output_path: Where to save the data.yaml file
        class_names: List of class names (uses default if None)
        
    Returns:
        Path to created data.yaml file
    """
    if class_names is None:
        class_names = get_class_names()
    
    data_dir = Path(data_dir).absolute()
    
    config = {
        'path': str(data_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created data.yaml at: {output_path}")
    return output_path


def validate_dataset(data_dir: str) -> Tuple[bool, str]:
    """
    Validate YOLO dataset structure.
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        Tuple of (is_valid, message)
    """
    data_path = Path(data_dir)
    
    required_dirs = [
        "train/images",
        "train/labels", 
        "val/images",
        "val/labels"
    ]
    
    for dir_path in required_dirs:
        full_path = data_path / dir_path
        if not full_path.exists():
            return False, f"Missing directory: {dir_path}"
        
        if "images" in dir_path:
            images = list(full_path.glob("*.jpg")) + list(full_path.glob("*.png"))
            if len(images) == 0:
                return False, f"No images found in: {dir_path}"
    
    return True, "Dataset structure is valid"


# ============== Experiment Tracking Utilities ==============

def setup_mlflow(
    tracking_uri: str = None,
    experiment_name: str = None
) -> None:
    """
    Setup MLflow for experiment tracking.
    
    Args:
        tracking_uri: MLflow tracking server URI (uses DagsHub if None)
        experiment_name: Name of the experiment
    """
    try:
        import mlflow
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        
        if experiment_name:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment set to: {experiment_name}")
            
    except ImportError:
        logger.warning("MLflow not installed. Experiment tracking disabled.")


def setup_dagshub(repo_owner: str, repo_name: str, mlflow: bool = True) -> None:
    """
    Initialize DagsHub integration.
    
    Args:
        repo_owner: DagsHub username
        repo_name: Repository name
        mlflow: Whether to initialize MLflow tracking
    """
    try:
        import dagshub
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=mlflow)
        logger.info(f"DagsHub initialized: {repo_owner}/{repo_name}")
    except ImportError:
        logger.warning("DagsHub not installed. Please run: pip install dagshub")
    except Exception as e:
        logger.error(f"Failed to initialize DagsHub: {e}")


# ============== Display Utilities ==============

def print_banner(title: str, width: int = 60) -> None:
    """Print a formatted banner."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_detection_results(detections: List[Dict]) -> None:
    """Pretty print detection results."""
    if not detections:
        print("No detections found.")
        return
    
    print(f"\nFound {len(detections)} object(s):")
    print("-" * 50)
    
    for i, det in enumerate(detections, 1):
        furniture = det.get('furniture_type', 'unknown')
        condition = det.get('condition', 'unknown')
        confidence = det.get('confidence', 0)
        print(f"  {i}. {furniture.upper()}: {condition} (confidence: {confidence:.2%})")
    
    print("-" * 50)


if __name__ == "__main__":
    # Test utilities
    print_banner("VisionAI Utilities Test")
    
    params = load_params()
    print(f"Loaded params: {params}")
    
    print(f"\nClass names: {get_class_names()}")
    print(f"Project root: {get_project_root()}")

