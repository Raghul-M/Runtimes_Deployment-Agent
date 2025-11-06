"""Configuration loader and parser for model-car YAML files.

Handles:
1. YAML parsing and validation
2. Model requirement extraction
3. Accelerator configuration management
"""

import json
import subprocess
import yaml
from typing import Dict
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_llm_model_config(config_path: str) -> dict:
    """Load and parse the model-car YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        dict: Parsed configuration as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def _skopeo_inspect(image_name: str) -> dict | None:
    image_ref = image_name.split("://", 1)[1] if "://" in image_name else image_name
    image_ref = f"docker://{image_ref}"
    try:
        result = subprocess.run(
            ["skopeo", "inspect", image_ref],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        logger.error("skopeo inspect failed for %s: %s", image_name, exc)
        return None

def _estimate_model_size(image_name: str) -> int:
    """Estimate the model size based on the image name using podman inspect.

    Args:
        image_name (str): The name of the container image.
    Returns:
        int: Size of the image in bytes.
    """
    metadata = _skopeo_inspect(image_name)

    if not metadata:
        return 0
    layers = metadata.get("LayersData", [])
    return sum(layer.get("Size", 0) for layer in layers if isinstance(layer, dict))/1024**3
    
def _supported_arch(image_name: str) -> str:
    """Estimate the supported architecture based on the image name using podman inspect.

    Args:
        image_name (str): The name of the container image.
    Returns:
        str: Supported architecture of the image.
    """
    metadata = _skopeo_inspect(image_name)

    if not metadata:
        return "unknown"
    arch = metadata.get("Architecture") or metadata.get("Labels", {}).get("architecture")
    return arch or "unknown"

def estimate_model_size(image_name: str) -> int:
    """Public wrapper to estimate model size bytes for a container image."""
    return _estimate_model_size(image_name)
    
def get_model_requirements(config: Dict) -> Dict:
    """Extract model requirements from the configuration.

    Args:
        config (Dict): Parsed configuration dictionary.
    Returns:
        Dict: Model requirements including model name and accelerator info.
    """
    model_info = config.get('model-car', [])
    if isinstance(model_info, dict):
        models = [model_info]
    else:
        models = [m for m in model_info if isinstance(m, dict)]
    
    requirements = {}
    for model_info in models:
        name = model_info.get('name', 'unknown')
        requirements[name] = {
            'model_name': model_info.get('name', 'unknown'),
            'image': model_info.get('image', 'default-image'),
            'gpu_count': model_info.get('serving_arguments', {}).get('gpu_count', 0),
            'arguments': model_info.get('serving_arguments', {}).get('args', []),
            'model_size_gb': _estimate_model_size(model_info.get('image', 'default-image')),
            'supported_arch': _supported_arch(model_info.get('image', 'default-image'))
        }

    return requirements
    
