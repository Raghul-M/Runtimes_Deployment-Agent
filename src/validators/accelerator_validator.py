"""Validators for accelerator compatibility.

Provides validation for:
1. CUDA compatibility
2. ROCm compatibility
3. vLLM-Spyre-x86 compatibility
4. GPU memory and capacity requirements
"""

import subprocess
import re


def check_gpu_availability():
    """
    Check if the GPU is available by running OpenShift command to check for GPU nodes.
    
    Returns:
        tuple: (gpu_status: bool, gpu_provider: str)
        - gpu_status: True if GPU is available, False otherwise
        - gpu_provider: "NVIDIA", "AMD", or "NONE"
    """
    try:
        # Run the oc command to get GPU information from nodes
        result = subprocess.run(
            ["oc", "get", "nodes", "-o", "custom-columns=NAME:.metadata.name,GPUs:.status.allocatable", "--no-headers"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            # Check the output for GPU providers
            output = result.stdout
            
            # Check for NVIDIA GPU
            if "nvidia.com/gpu" in output:
                return True, "NVIDIA"
            # Check for AMD GPU
            elif "amd.com/gpu" in output:
                return True, "AMD"
            else:
                return False, "NONE"
        else:
            return False, "NONE"
            
    except subprocess.TimeoutExpired:
        return False, "NONE"
    except FileNotFoundError:
        return False, "NONE"
    except Exception:
        return False, "NONE"


def get_gpu_info():
    """
    Get detailed GPU information based on the GPU provider and save to gpu_info.txt file.
    
    Returns:
        str: Path to the created gpu_info.txt file
    """
    try:
        # Get GPU availability status
        gpu_status, gpu_provider = check_gpu_availability()
        
        if not gpu_status:
            info_content = "No GPU available in the cluster\n"
        else:
            # Get detailed GPU information based on provider
            if gpu_provider == "NVIDIA":
                info_content = get_nvidia_gpu_details()
            elif gpu_provider == "AMD":
                info_content = get_amd_gpu_details()
            else:
                info_content = f"Unknown GPU provider: {gpu_provider}\n"
        
        # Save to file
        with open("gpu_info.txt", "w") as f:
            f.write(info_content)
        
        return "gpu_info.txt"
        
    except Exception as e:
        error_content = f"Error getting GPU info: {str(e)}\n"
        with open("gpu_info.txt", "w") as f:
            f.write(error_content)
        return "gpu_info.txt"


def _get_cloud_provider():
    """
    Get cloud provider from oc cluster-info command.
    
    Returns:
        str: Cloud provider name (AWS, Azure, GCP, IBM, etc.) or "Unknown"
    """
    try:
        result = subprocess.run(
            ["oc", "cluster-info"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            output = result.stdout.lower()
            
            # Check for  cloud providers 
            if "amazonaws.com" in output or "aws" in output:
                return "AWS"
            elif "azure.com" in output or "azure" in output:
                return "Azure"
            elif "googleapis.com" in output or "gcp" in output or "google" in output:
                return "GCP"
            elif "ibm.com" in output or "ibm" in output:
                return "IBM"
            else:
                return "Unknown"
        else:
            return "Unknown"
            
    except Exception:
        return "Unknown"


def _convert_to_gb(value):
    """
    Convert Kubernetes memory/storage values to GB.
    
    Args:
        value: Memory or storage value (e.g., "16Gi", "1000Mi", "5000000000")
        
    Returns:
        str: Value in GB format
    """
    if not value or value == "Unknown":
        return "Unknown"
    
    try:
        # Handle different units
        if value.endswith('Gi'):
            # Already in GiB, convert to GB (1 GiB = 1.074 GB)
            num = float(value[:-2])
            return f"{num * 1.074:.1f}"
        elif value.endswith('Mi'):
            # Convert MiB to GB
            num = float(value[:-2])
            return f"{num / 1024 * 1.074:.1f}"
        elif value.endswith('Ki'):
            # Convert KiB to GB
            num = float(value[:-2])
            return f"{num / (1024 * 1024) * 1.074:.1f}"
        elif value.endswith('G'):
            # Already in GB
            return value[:-1]
        elif value.endswith('M'):
            # Convert MB to GB
            num = float(value[:-1])
            return f"{num / 1000:.1f}"
        elif value.endswith('K'):
            # Convert KB to GB
            num = float(value[:-1])
            return f"{num / (1000 * 1000):.1f}"
        else:
            # Assume it's in bytes, convert to GB
            num = float(value)
            return f"{num / (1000 * 1000 * 1000):.1f}"
    except (ValueError, TypeError):
        return "Unknown"


def get_nvidia_gpu_details():
    """Get detailed NVIDIA GPU information."""
    try:
        # Get nodes with NVIDIA GPU
        result = subprocess.run(
            ["oc", "get", "nodes", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return "Failed to get node information\n"
        
        import json
        nodes_data = json.loads(result.stdout)
        
        gpu_info = []
        for node in nodes_data.get("items", []):
            allocatable = node.get("status", {}).get("allocatable", {})
            if "nvidia.com/gpu" in allocatable:
                node_name = node.get("metadata", {}).get("name", "Unknown")
                gpu_count = allocatable.get("nvidia.com/gpu", "0")
                memory = allocatable.get("memory", "Unknown")
                storage = allocatable.get("ephemeral-storage", "Unknown")
                
                # Get instance type from labels
                labels = node.get("metadata", {}).get("labels", {})
                instance_type = labels.get("node.kubernetes.io/instance-type", "Unknown")
                
                # Get cloud provider from cluster info
                cloud_provider = _get_cloud_provider()
                
                # Convert memory to GB
                memory_gb = _convert_to_gb(memory)
                storage_gb = _convert_to_gb(storage)
                
                gpu_info.append(f"• Cloud Provider: {cloud_provider}")
                gpu_info.append(f"• Instance Type: {instance_type}")
                gpu_info.append(f"• GPU Provider: NVIDIA")
                gpu_info.append(f"• Allocatable GPUs: {gpu_count}")
                gpu_info.append(f"• Memory: {memory_gb} GB")
                gpu_info.append(f"• Storage: {storage_gb} GB")
                gpu_info.append(f"• Node Name: {node_name}")
                gpu_info.append("")  # Empty line between nodes
        
        return "\n".join(gpu_info) if gpu_info else "No NVIDIA GPU nodes found\n"
        
    except Exception as e:
        return f"Error getting NVIDIA GPU details: {str(e)}\n"


def get_amd_gpu_details():
    """Get detailed AMD GPU information."""
    try:
        # Get nodes with AMD GPU
        result = subprocess.run(
            ["oc", "get", "nodes", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return "Failed to get node information\n"
        
        import json
        nodes_data = json.loads(result.stdout)
        
        gpu_info = []
        for node in nodes_data.get("items", []):
            allocatable = node.get("status", {}).get("allocatable", {})
            if "amd.com/gpu" in allocatable:
                node_name = node.get("metadata", {}).get("name", "Unknown")
                gpu_count = allocatable.get("amd.com/gpu", "0")
                memory = allocatable.get("memory", "Unknown")
                storage = allocatable.get("ephemeral-storage", "Unknown")
                
                # Get instance type from labels
                labels = node.get("metadata", {}).get("labels", {})
                instance_type = labels.get("node.kubernetes.io/instance-type", "Unknown")
                
                # Get cloud provider from cluster info
                cloud_provider = _get_cloud_provider()
                
                # Try to determine AMD GPU model from instance type
                gpu_model = "Unknown"
                if "MI300X" in instance_type:
                    gpu_model = "MI300X"
                elif "MI250" in instance_type:
                    gpu_model = "MI250"
                elif "MI100" in instance_type:
                    gpu_model = "MI100"
                
                # Convert memory to GB
                memory_gb = _convert_to_gb(memory)
                storage_gb = _convert_to_gb(storage)
                
                gpu_info.append(f"• Cloud Provider: {cloud_provider}")
                gpu_info.append(f"• Instance Type: {instance_type}")
                gpu_info.append(f"• GPU Provider: AMD ({gpu_model})")
                gpu_info.append(f"• Allocatable GPUs: {gpu_count}")
                gpu_info.append(f"• Memory: {memory_gb} GB")
                gpu_info.append(f"• Storage: {storage_gb} GB")
                gpu_info.append(f"• Node Name: {node_name}")
                gpu_info.append("")  # Empty line between nodes
        
        return "\n".join(gpu_info) if gpu_info else "No AMD GPU nodes found\n"
        
    except Exception as e:
        return f"Error getting AMD GPU details: {str(e)}\n"


# Test the functions
if __name__ == "__main__":
    gpu_status, gpu_provider = check_gpu_availability()
    print(f"GPU Status: {gpu_status}")
    print(f"GPU Provider: {gpu_provider}")
    
    if gpu_status:
        file_path = get_gpu_info()
        print(f"GPU information saved to: {file_path}")
    