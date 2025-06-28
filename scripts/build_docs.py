#!/usr/bin/env python3
"""
Cross-platform documentation build script for Godot Stat Math.

This script automates the two-step documentation build process:
1. Generate RST files from GDScript sources
2. Build HTML documentation using Sphinx

Usage:
    python scripts/build_docs.py

Requirements:
    - Python 3.x
    - Sphinx (pip install sphinx)
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, cwd=None, description=None):
    """Run a command and handle errors appropriately."""
    if description:
        print(f"\n{'='*60}")
        print(f"STEP: {description}")
        print(f"{'='*60}")
        print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        print(f"Working directory: {cwd or os.getcwd()}")
        print()
    
    try:
        if isinstance(cmd, str):
            # Use shell=True for string commands
            result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, 
                                  capture_output=False, text=True)
        else:
            # Use list format for better cross-platform compatibility
            result = subprocess.run(cmd, cwd=cwd, check=True, 
                                  capture_output=False, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"ERROR: Command not found: {e}")
        print("Make sure all required tools are installed and in your PATH")
        sys.exit(1)


def get_project_root():
    """Get the project root directory (parent of scripts folder)."""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    return project_root


def check_requirements():
    """Check that required files and tools exist."""
    project_root = get_project_root()
    
    # Check required files exist
    required_files = [
        project_root / "docs" / "generate_api_rst.py",
        project_root / "addons" / "godot-stat-math" / "core",
        project_root / "docs" / "modules"
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            print(f"ERROR: Required file/directory not found: {file_path}")
            sys.exit(1)
    
    # Check Python is available
    try:
        subprocess.run([sys.executable, "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: Python not found or not working properly")
        sys.exit(1)
    
    # Check Sphinx is available
    try:
        subprocess.run([sys.executable, "-c", "import sphinx"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("ERROR: Sphinx not found. Install with: pip install sphinx")
        sys.exit(1)
    
    print("✓ All requirements checked successfully")


def generate_api_rst():
    """Generate RST files from GDScript sources."""
    project_root = get_project_root()
    
    cmd = [
        sys.executable,
        "docs/generate_api_rst.py", 
        "addons/godot-stat-math/core",
        "docs/modules"
    ]
    
    run_command(cmd, cwd=project_root, 
                description="Generating RST files from GDScript sources")


def clean_build_directory():
    """Clean the build directory and Sphinx cache for a fresh build."""
    project_root = get_project_root()
    build_dir = project_root / "docs" / "_build"
    doctrees_dir = project_root / "docs" / ".doctrees"
    
    import shutil
    
    # Remove build directory if it exists
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"✓ Removed build directory: {build_dir}")
    
    # Remove doctrees cache if it exists
    if doctrees_dir.exists():
        shutil.rmtree(doctrees_dir)
        print(f"✓ Removed Sphinx cache: {doctrees_dir}")


def build_html_docs():
    """Build HTML documentation using Sphinx."""
    project_root = get_project_root()
    scripts_dir = project_root / "scripts"
    
    # Always clean build directory first for consistency across platforms
    print(f"\n{'='*60}")
    print(f"STEP: Cleaning build directory and Sphinx cache")
    print(f"{'='*60}")
    clean_build_directory()
    
    # Determine the appropriate build command based on platform
    if platform.system() == "Windows":
        # Use make.bat on Windows (but it will skip cleaning since we already did it)
        cmd = [str(scripts_dir / "make.bat"), "html"]
        run_command(cmd, cwd=project_root,
                    description="Building HTML documentation (Windows)")
    else:
        # Use make on Unix-like systems (Linux, macOS, etc.)
        cmd = ["make", "-f", str(scripts_dir / "Makefile"), "html"]
        run_command(cmd, cwd=project_root,
                    description="Building HTML documentation (Unix)")


def main():
    """Main entry point for the documentation build process."""
    print("Godot Stat Math Documentation Builder")
    print("=" * 40)
    
    # Get project root and validate environment
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    
    # Check all requirements
    check_requirements()
    
    try:
        # Step 1: Generate API RST files
        generate_api_rst()
        
        # Step 2: Build HTML documentation
        build_html_docs()
        
        # Success message
        print(f"\n{'='*60}")
        print("SUCCESS: Documentation build completed!")
        print(f"{'='*60}")
        print(f"📁 Output location: {project_root}/docs/_build/html/")
        print(f"🌐 Open in browser: {project_root}/docs/_build/html/index.html")
        print()
        
    except KeyboardInterrupt:
        print("\n\nBuild cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Unexpected error during build: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 