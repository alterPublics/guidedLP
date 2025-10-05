#!/usr/bin/env python3
"""
Debug script to help diagnose GitHub pip installation issues.

Run this on machines where installation shows "UNKNOWN":
    python debug_installation.py
"""

import subprocess
import sys
import tempfile
import os
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def main():
    print("=== GitHub Pip Installation Debug ===")
    print()
    
    # Check Python and pip versions
    print("1. Environment Check:")
    code, stdout, stderr = run_command("python3 --version")
    print(f"   Python: {stdout.strip()}")
    
    code, stdout, stderr = run_command("pip3 --version")
    print(f"   Pip: {stdout.strip()}")
    
    code, stdout, stderr = run_command("pip3 show setuptools")
    if code == 0:
        for line in stdout.split('\n'):
            if line.startswith('Version:'):
                print(f"   Setuptools: {line}")
                break
    
    print()
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"2. Testing GitHub clone in: {temp_dir}")
        
        # Clone the repository
        clone_cmd = "git clone https://github.com/alterPublics/guidedLP.git"
        code, stdout, stderr = run_command(clone_cmd, cwd=temp_dir)
        
        if code != 0:
            print(f"   ❌ Git clone failed: {stderr}")
            return
        
        repo_path = Path(temp_dir) / "guidedLP"
        print(f"   ✅ Repository cloned successfully")
        
        # Check repository structure
        print("\n3. Repository Structure Check:")
        pyproject_path = repo_path / "pyproject.toml"
        if pyproject_path.exists():
            print("   ✅ pyproject.toml found at root")
            
            # Read and check pyproject.toml content
            content = pyproject_path.read_text()
            if 'name = "guidedLP"' in content:
                print("   ✅ Package name 'guidedLP' found")
            else:
                print("   ❌ Package name 'guidedLP' NOT found")
                
            if 'version = "0.1.0"' in content:
                print("   ✅ Version '0.1.0' found")
            else:
                print("   ❌ Version '0.1.0' NOT found")
                
            if 'dynamic = []' in content:
                print("   ✅ 'dynamic = []' found")
            else:
                print("   ❌ 'dynamic = []' NOT found")
                
        else:
            print("   ❌ pyproject.toml NOT found at root")
            
        # Check if guidedLP subdirectory exists
        guidedlp_dir = repo_path / "guidedLP"
        if guidedlp_dir.exists():
            print("   ✅ guidedLP/ subdirectory found")
            
            src_dir = guidedlp_dir / "src" / "guidedLP"
            if src_dir.exists():
                print("   ✅ guidedLP/src/guidedLP/ source directory found")
            else:
                print("   ❌ guidedLP/src/guidedLP/ source directory NOT found")
        else:
            print("   ❌ guidedLP/ subdirectory NOT found")
        
        print("\n4. Test Local Installation:")
        
        # Try to install locally
        install_cmd = "pip3 install -e ."
        code, stdout, stderr = run_command(install_cmd, cwd=repo_path)
        
        if code == 0:
            print("   ✅ Local installation successful")
            
            # Check what package was installed
            if "guidedLP" in stdout and "UNKNOWN" not in stdout:
                print("   ✅ Package installed as 'guidedLP'")
            else:
                print("   ❌ Package installed as 'UNKNOWN' or other")
                print(f"   Installation output: {stdout}")
                
        else:
            print(f"   ❌ Local installation failed: {stderr}")
            
        print("\n5. Test Direct GitHub Installation:")
        
        # Uninstall first
        run_command("pip3 uninstall guidedLP -y")
        
        # Try GitHub installation
        github_cmd = "pip3 install git+https://github.com/alterPublics/guidedLP.git"
        code, stdout, stderr = run_command(github_cmd)
        
        if code == 0:
            if "guidedLP" in stdout and "UNKNOWN" not in stdout:
                print("   ✅ GitHub installation successful as 'guidedLP'")
            else:
                print("   ❌ GitHub installation shows 'UNKNOWN'")
                print(f"   Installation output: {stdout}")
        else:
            print(f"   ❌ GitHub installation failed: {stderr}")
            
        # Check installed package
        print("\n6. Installed Package Check:")
        code, stdout, stderr = run_command("pip3 show guidedLP")
        if code == 0:
            print("   ✅ Package 'guidedLP' found in pip list")
            for line in stdout.split('\n'):
                if line.startswith(('Name:', 'Version:')):
                    print(f"   {line}")
        else:
            print("   ❌ Package 'guidedLP' NOT found in pip list")
            
            # Check for UNKNOWN
            code, stdout, stderr = run_command("pip3 show UNKNOWN")
            if code == 0:
                print("   ❌ Found package installed as 'UNKNOWN'")
                for line in stdout.split('\n'):
                    if line.startswith(('Name:', 'Version:')):
                        print(f"   {line}")

if __name__ == "__main__":
    main()