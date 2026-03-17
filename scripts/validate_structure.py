#!/usr/bin/env python3
"""Simple structure test for predictive maintenance system."""

import os
import sys
from pathlib import Path

def test_project_structure():
    """Test that all required files and directories exist."""
    print("🧪 Testing Project Structure")
    print("=" * 50)
    
    required_files = [
        "README.md",
        "requirements.txt", 
        "pyproject.toml",
        ".gitignore",
        "configs/config.yaml",
        ".pre-commit-config.yaml",
        ".github/workflows/ci.yml"
    ]
    
    required_dirs = [
        "src",
        "src/models",
        "src/pipelines", 
        "src/export",
        "src/utils",
        "data",
        "data/raw",
        "data/processed",
        "configs",
        "scripts",
        "tests",
        "assets",
        "demo",
        ".github/workflows"
    ]
    
    passed = 0
    total = len(required_files) + len(required_dirs)
    
    print("📁 Checking directories...")
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
            passed += 1
        else:
            print(f"❌ {dir_path}")
    
    print("\n📄 Checking files...")
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
            passed += 1
        else:
            print(f"❌ {file_path}")
    
    print("\n" + "=" * 50)
    print(f"📊 Structure Test Results: {passed}/{total} items found")
    
    if passed == total:
        print("🎉 All required files and directories exist!")
        return True
    else:
        print("⚠️ Some files or directories are missing.")
        return False

def test_file_contents():
    """Test that key files have expected content."""
    print("\n🔍 Testing File Contents")
    print("=" * 50)
    
    tests = [
        ("README.md", "Predictive Maintenance Edge AI System"),
        ("requirements.txt", "tensorflow"),
        ("pyproject.toml", "predictive-maintenance-edge"),
        ("configs/config.yaml", "device_configs"),
        (".gitignore", "__pycache__"),
    ]
    
    passed = 0
    total = len(tests)
    
    for file_path, expected_content in tests:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if expected_content in content:
                        print(f"✅ {file_path} contains '{expected_content}'")
                        passed += 1
                    else:
                        print(f"❌ {file_path} missing '{expected_content}'")
            except Exception as e:
                print(f"❌ {file_path} read error: {e}")
        else:
            print(f"❌ {file_path} not found")
    
    print("\n" + "=" * 50)
    print(f"📊 Content Test Results: {passed}/{total} files have expected content")
    
    if passed == total:
        print("🎉 All files have expected content!")
        return True
    else:
        print("⚠️ Some files are missing expected content.")
        return False

def test_python_syntax():
    """Test Python syntax of key files."""
    print("\n🐍 Testing Python Syntax")
    print("=" * 50)
    
    python_files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/pipelines/__init__.py",
        "src/export/__init__.py",
        "src/utils/__init__.py",
        "tests/test_models.py",
    ]
    
    passed = 0
    total = len(python_files)
    
    for file_path in python_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                compile(content, file_path, 'exec')
                print(f"✅ {file_path} syntax OK")
                passed += 1
            except SyntaxError as e:
                print(f"❌ {file_path} syntax error: {e}")
            except Exception as e:
                print(f"❌ {file_path} error: {e}")
        else:
            print(f"❌ {file_path} not found")
    
    print("\n" + "=" * 50)
    print(f"📊 Syntax Test Results: {passed}/{total} files have valid syntax")
    
    if passed == total:
        print("🎉 All Python files have valid syntax!")
        return True
    else:
        print("⚠️ Some Python files have syntax errors.")
        return False

def main():
    """Run all tests."""
    print("🚀 Predictive Maintenance System - Structure Validation")
    print("=" * 60)
    
    tests = [
        test_project_structure,
        test_file_contents,
        test_python_syntax,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Overall Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed! Project structure is valid.")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run training: python scripts/train.py --n-samples 100 --epochs 5")
        print("3. Launch demo: streamlit run demo/streamlit_app.py")
        return True
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
