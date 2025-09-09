#!/usr/bin/env python3
"""
Debug script for Heroku deployment
Run this locally to test the same environment as Heroku
"""

import os
import sys

def debug_environment():
    """Debug the current environment"""
    print("=== HEROKU DEBUG SCRIPT ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check for data file
    print("\n=== CHECKING FOR DATA FILE ===")
    possible_paths = [
        '../data/tmp/final.csv',
        'data/tmp/final.csv',
        os.path.join(os.getcwd(), 'data', 'tmp', 'final.csv'),
        os.path.join(os.getcwd(), '..', 'data', 'tmp', 'final.csv')
    ]
    
    for path in possible_paths:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"  {path}: {'EXISTS' if exists else 'NOT FOUND'} ({size:,} bytes)")
    
    # Check directory structure
    print("\n=== DIRECTORY STRUCTURE ===")
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
    
    # Test imports
    print("\n=== TESTING IMPORTS ===")
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
    
    try:
        import sklearn
        print("✅ sklearn imported successfully")
    except ImportError as e:
        print(f"❌ sklearn import failed: {e}")
    
    try:
        import flask
        print("✅ flask imported successfully")
    except ImportError as e:
        print(f"❌ flask import failed: {e}")

if __name__ == "__main__":
    debug_environment()
