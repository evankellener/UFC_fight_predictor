#!/usr/bin/env python3
"""
Simple script to install tqdm for progress bars in the greedy forward search algorithm.
Run this script to install tqdm if it's not already available.
"""

import subprocess
import sys

def install_tqdm():
    """Install tqdm package for progress bars."""
    try:
        import tqdm
        print("✅ tqdm is already installed!")
        return True
    except ImportError:
        print("📦 Installing tqdm for progress bars...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
            print("✅ tqdm installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install tqdm: {e}")
            print("💡 You can install it manually with: pip install tqdm")
            return False

if __name__ == "__main__":
    install_tqdm()
