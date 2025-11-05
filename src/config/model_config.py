"""Configuration loader and parser for model-car YAML files.

Handles:
1. YAML parsing and validation
2. Model requirement extraction
3. Accelerator configuration management
"""

import subprocess
import yaml
from typing import Dict


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

def _estimate_model_size(image_name: str) -> int:
    """Estimate the model size based on the image name using podman inspect.

    Args:
        image_name (str): The name of the container image.
    Returns:
        int: Size of the image in bytes.
    """
    try:
        result = subprocess.run(
            ['podman', 'inspect', image_name, '--format', '{{.Size}}'],
            capture_output=True, text=True, check=True
        )
        size_bytes = int(result.stdout.strip())
        return size_bytes
    except Exception:
        return 0  # Default to 0 if inspection fails
    
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
            'model_size_bytes': _estimate_model_size(model_info.get('image', 'default-image'))
        }

    return requirements
    
