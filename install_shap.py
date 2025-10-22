#!/usr/bin/env python3
"""
Simple script to install SHAP for feature importance calculations.
Run this script to install SHAP if it's not already available.
"""

import subprocess
import sys

def install_shap():
    """Install SHAP package for feature importance calculations."""
    try:
        import shap
        print("✅ SHAP is already installed!")
        return True
    except ImportError:
        print("📦 Installing SHAP for feature importance calculations...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
            print("✅ SHAP installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install SHAP: {e}")
            print("💡 You can install it manually with: pip install shap")
            return False

if __name__ == "__main__":
    install_shap()
