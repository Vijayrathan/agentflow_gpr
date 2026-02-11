"""
Workspace utilities for GPR simulation.

This module provides workspace directory management for generated files.
"""

from pathlib import Path
import os


def get_workspace_directory() -> Path:
    """
    Get the workspace directory outside Flask's working directory.
    This prevents file changes from triggering Flask's hot reload.
    
    Returns:
        Path: Absolute path to the workspace directory
    """
    # Use environment variable if set, otherwise use /tmp or a directory outside project
    workspace_base = os.getenv("GPR_WORKSPACE_DIR", None)
    
    if workspace_base:
        workspace_dir = Path(workspace_base)
    else:
        # Default to /tmp/intelligent_gpr_workspace (outside Flask working directory)
        workspace_dir = Path("/tmp/intelligent_gpr_workspace")
    
    # Create directory if it doesn't exist
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir
