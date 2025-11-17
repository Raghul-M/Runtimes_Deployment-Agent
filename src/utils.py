import os
import sys
import subprocess


def check_cluster_login():
    """
    Check if the cluster is logged in by running 'oc cluster-info' command.
    
    Returns:
        str: "true cluster logged in" if successful, "failed" if not
    """
    try:
        result = subprocess.run(
            ["oc", "cluster-info"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return "Cluster is logged in"
        else:
            return "Login to your cluster to continue"

    except subprocess.TimeoutExpired:
        return "failed"
    except FileNotFoundError:
        return "failed"
    except Exception:
        return "failed"
