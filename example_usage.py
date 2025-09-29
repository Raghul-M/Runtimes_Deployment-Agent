#!/usr/bin/env python3
"""Example usage of the cluster utility functions."""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.cluster_utils import check_cluster_auth, get_cluster_info


def main():
    """Example of how to use the cluster utility functions."""
    
    print("🔍 Checking cluster authentication...")
    print("=" * 40)
    
    # Method 1: Simple authentication check
    auth_status = check_cluster_auth()
    
    if auth_status.is_authenticated:
        print(f"✅ Authenticated to {auth_status.cluster_type.value} cluster")
        print(f"   URL: {auth_status.cluster_url}")
        print(f"   User: {auth_status.username}")
        print(f"   Namespace: {auth_status.namespace}")
    else:
        print(f"❌ Not authenticated")
        print(f"   Error: {auth_status.error_message}")
        return
    
    print("\n📊 Getting detailed cluster information...")
    print("=" * 40)
    
    # Method 2: Get detailed cluster information
    cluster_info = get_cluster_info()
    
    for key, value in cluster_info.items():
        if key == "authenticated":
            continue
        elif key in ["available_projects", "available_namespaces"]:
            print(f"{key}: {len(value)} items")
            for item in value[:5]:  # Show first 5
                print(f"  - {item}")
            if len(value) > 5:
                print(f"  ... and {len(value) - 5} more")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
