#!/usr/bin/env python3
"""
Simple test script to verify deployment works
"""

import os
import sys

def test_imports():
    """Test if all required imports work"""
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ sklearn imported successfully")
    except ImportError as e:
        print(f"❌ sklearn import failed: {e}")
        return False
    
    try:
        import flask
        print("✅ flask imported successfully")
    except ImportError as e:
        print(f"❌ flask import failed: {e}")
        return False
    
    return True

def test_data_file():
    """Test if data file can be found"""
    possible_paths = [
        '../data/tmp/final.csv',
        'data/tmp/final.csv',
        '../data/final.csv',
        'data/final.csv',
        os.path.join(os.getcwd(), 'data', 'tmp', 'final.csv'),
        os.path.join(os.getcwd(), '..', 'data', 'tmp', 'final.csv'),
        os.path.join(os.getcwd(), 'data', 'final.csv'),
        os.path.join(os.getcwd(), '..', 'data', 'final.csv')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✅ Found data file at: {path} ({size:,} bytes)")
            return True
    
    print("❌ Data file not found in any expected location")
    return False

def test_model_import():
    """Test if the model can be imported"""
    try:
        sys.path.append('src')
        from app.model import UFCFightPredictor
        print("✅ Model import successful")
        return True
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False

if __name__ == "__main__":
    print("=== DEPLOYMENT TEST ===")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    print("\n=== TESTING IMPORTS ===")
    imports_ok = test_imports()
    
    print("\n=== TESTING DATA FILE ===")
    data_ok = test_data_file()
    
    print("\n=== TESTING MODEL IMPORT ===")
    model_ok = test_model_import()
    
    print("\n=== SUMMARY ===")
    if imports_ok and data_ok and model_ok:
        print("✅ All tests passed! Deployment should work.")
    else:
        print("❌ Some tests failed. Check the errors above.")
