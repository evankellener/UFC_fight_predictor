#!/usr/bin/env python3
"""
Deployment script for UFC Fight Predictor
This script helps prepare the project for deployment to various platforms.
"""

import os
import shutil
import subprocess
import sys

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'app/app.py',
        'app/model.py',
        'app/templates/index.html',
        'app/static/css/style.css',
        'app/static/js/app.js',
        'requirements.txt',
        'data/tmp/final.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files present")
    return True

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
ufc_env/
ufc_env_old/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log

# Database
*.db
*.sqlite

# Temporary files
*.tmp
*.temp

# Model files (too large for git)
saved_models/
*.h5
*.joblib
*.pkl

# Data files (too large for git)
data/tmp/*.csv
data/sqlite_db/
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore file")

def main():
    """Main deployment preparation function"""
    print("🚀 Preparing UFC Fight Predictor for deployment...")
    
    # Check requirements
    if not check_requirements():
        print("❌ Deployment preparation failed due to missing files")
        sys.exit(1)
    
    # Create .gitignore
    create_gitignore()
    
    print("\n✅ Project is ready for deployment!")
    print("\n📋 Next steps:")
    print("1. Initialize git repository: git init")
    print("2. Add files: git add .")
    print("3. Commit: git commit -m 'Initial commit'")
    print("4. Create GitHub repository and push")
    print("5. Deploy to Railway, Render, or Heroku")
    
    print("\n🌐 Recommended deployment platforms:")
    print("• Railway: https://railway.app (easiest)")
    print("• Render: https://render.com (great for Flask)")
    print("• Heroku: https://heroku.com (most popular)")

if __name__ == "__main__":
    main()
