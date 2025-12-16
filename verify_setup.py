#!/usr/bin/env python3
"""
Verify Matt AI Setup

Checks that all files are in place and basic structure is correct.
"""

import json
import os
import sys
from pathlib import Path


def check_file(path, description):
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} MISSING: {path}")
        return False


def verify_ethical_standards():
    """Verify ethical standards file."""
    path = "ethical_standards.json"
    
    if not Path(path).exists():
        print(f"✗ Ethical standards file missing")
        return False
    
    try:
        with open(path, 'r') as f:
            standards = json.load(f)
        
        # Check structure
        required_keys = ["version", "ethical_standards"]
        for key in required_keys:
            if key not in standards:
                print(f"✗ Missing key in ethical standards: {key}")
                return False
        
        ethics = standards["ethical_standards"]
        required_sections = [
            "core_principles",
            "prohibited_actions",
            "content_filters",
            "operational_constraints",
            "learning_guidelines",
            "safety_measures"
        ]
        
        for section in required_sections:
            if section not in ethics:
                print(f"✗ Missing section in ethical standards: {section}")
                return False
        
        print(f"✓ Ethical standards file valid")
        print(f"  Version: {standards['version']}")
        print(f"  Core principles: {len(ethics['core_principles'])}")
        print(f"  Prohibited actions: {len(ethics['prohibited_actions'])}")
        
        # Check file permissions (read-only check)
        mode = os.stat(path).st_mode
        print(f"  File permissions: {oct(mode)[-3:]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading ethical standards: {e}")
        return False


def verify_python_modules():
    """Verify Python module structure."""
    print("\n--- Python Modules ---")
    
    modules = [
        ("src/matt_ai/__init__.py", "Main module init"),
        ("src/matt_ai/ethical_controller.py", "Ethical Controller"),
        ("src/matt_ai/model.py", "Self-Training LLM"),
        ("src/matt_ai/trainer.py", "Iterative Trainer"),
        ("src/matt_ai/data_utils.py", "Data Utilities"),
    ]
    
    all_good = True
    for path, description in modules:
        if not check_file(path, description):
            all_good = False
    
    return all_good


def verify_config_files():
    """Verify configuration files."""
    print("\n--- Configuration Files ---")
    
    files = [
        ("requirements.txt", "Python requirements"),
        ("setup.py", "Setup configuration"),
        (".gitignore", "Git ignore rules"),
    ]
    
    all_good = True
    for path, description in files:
        if not check_file(path, description):
            all_good = False
    
    return all_good


def verify_scripts():
    """Verify executable scripts."""
    print("\n--- Scripts ---")
    
    scripts = [
        ("main.py", "Main entry point"),
        ("example_usage.py", "Example usage script"),
    ]
    
    all_good = True
    for path, description in scripts:
        if not check_file(path, description):
            all_good = False
    
    return all_good


def verify_documentation():
    """Verify documentation."""
    print("\n--- Documentation ---")
    
    docs = [
        ("README.md", "Main README"),
    ]
    
    all_good = True
    for path, description in docs:
        if not check_file(path, description):
            all_good = False
    
    return all_good


def count_lines():
    """Count lines of code."""
    print("\n--- Code Statistics ---")
    
    python_files = [
        "src/matt_ai/__init__.py",
        "src/matt_ai/ethical_controller.py",
        "src/matt_ai/model.py",
        "src/matt_ai/trainer.py",
        "src/matt_ai/data_utils.py",
        "main.py",
        "example_usage.py",
        "setup.py",
    ]
    
    total_lines = 0
    for path in python_files:
        if Path(path).exists():
            with open(path, 'r') as f:
                lines = len(f.readlines())
                total_lines += lines
    
    print(f"Total Python code lines: {total_lines}")


def main():
    """Run all verifications."""
    print("=" * 60)
    print("Matt AI - Setup Verification")
    print("=" * 60)
    
    results = []
    
    print("\n--- Ethical Standards ---")
    results.append(verify_ethical_standards())
    
    results.append(verify_python_modules())
    results.append(verify_config_files())
    results.append(verify_scripts())
    results.append(verify_documentation())
    
    count_lines()
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All verifications passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run examples: python example_usage.py")
        print("3. View system info: python main.py info")
        print("4. See README.md for full documentation")
        return 0
    else:
        print("✗ Some verifications failed")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
