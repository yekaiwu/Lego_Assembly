"""
Installation validation script.
Checks dependencies, configuration, and basic functionality.
"""

import sys
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version >= (3, 8):
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_dependencies():
    """Check required Python packages."""
    print("\nChecking dependencies...")
    
    required = [
        "pdf2image",
        "fitz",  # PyMuPDF
        "PIL",   # Pillow
        "numpy",
        "scipy",
        "requests",
        "dotenv",
        "pydantic",
        "jsonschema",
        "diskcache",
        "loguru",
        "cv2"    # opencv-python
    ]
    
    missing = []
    for package in required:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def check_poppler():
    """Check if Poppler is installed."""
    print("\nChecking Poppler (for PDF processing)...")
    try:
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFInfoNotInstalledError
        
        # Try to access pdfinfo
        import subprocess
        result = subprocess.run(["pdfinfo", "-v"], capture_output=True)
        if result.returncode == 0:
            print("✓ Poppler installed")
            return True
        else:
            raise FileNotFoundError
    except (FileNotFoundError, PDFInfoNotInstalledError):
        print("✗ Poppler not found")
        print("  Install with:")
        print("    macOS: brew install poppler")
        print("    Ubuntu: sudo apt-get install poppler-utils")
        return False

def check_configuration():
    """Check configuration file."""
    print("\nChecking configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("✗ .env file not found")
        print("  Create with: cp ENV_TEMPLATE.txt .env")
        return False
    
    print("✓ .env file exists")
    
    # Check for API keys
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    api_keys = {
        "DASHSCOPE_API_KEY": "Qwen-VL (Primary)",
        "DEEPSEEK_API_KEY": "DeepSeek (Secondary)",
        "MOONSHOT_API_KEY": "Kimi (Fallback)"
    }
    
    configured = 0
    for key, name in api_keys.items():
        if os.getenv(key):
            print(f"✓ {name} configured")
            configured += 1
        else:
            print(f"○ {name} not configured (optional)")
    
    if configured == 0:
        print("\n⚠ No API keys configured. At least one is required.")
        return False
    
    return True

def check_module_imports():
    """Check if source modules can be imported."""
    print("\nChecking module imports...")
    
    modules = [
        "src.utils.config",
        "src.utils.cache",
        "src.api.qwen_vlm",
        "src.api.deepseek_api",
        "src.api.kimi_api",
        "src.vision_processing.manual_input_handler",
        "src.vision_processing.vlm_step_extractor",
        "src.vision_processing.dependency_graph",
        "src.plan_generation.part_database",
        "src.plan_generation.spatial_reasoning",
        "src.plan_generation.plan_structure"
    ]
    
    all_ok = True
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module}: {e}")
            all_ok = False
    
    return all_ok

def check_directories():
    """Check required directories."""
    print("\nChecking directories...")
    
    dirs = ["src", "data", "examples"]
    all_ok = True
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ (missing)")
            all_ok = False
    
    return all_ok

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        # Test config loading
        from src.utils import get_config
        config = get_config()
        print("✓ Configuration loading")
        
        # Test cache initialization
        from src.utils import get_cache
        cache = get_cache()
        print("✓ Cache initialization")
        
        # Test part database initialization
        from src.plan_generation import PartDatabase
        db = PartDatabase()
        print("✓ Part database initialization")
        
        # Test spatial reasoning
        from src.plan_generation import SpatialReasoning
        spatial = SpatialReasoning()
        print("✓ Spatial reasoning engine")
        
        return True
    
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def main():
    """Run all validation checks."""
    print("=" * 80)
    print("LEGO Assembly System - Installation Validation")
    print("=" * 80)
    
    results = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Poppler": check_poppler(),
        "Configuration": check_configuration(),
        "Directories": check_directories(),
        "Module Imports": check_module_imports(),
        "Basic Functionality": test_basic_functionality()
    }
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {check}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All checks passed! System ready to use.")
        print("\nNext steps:")
        print("1. Read QUICK_START.md for usage examples")
        print("2. Run: python main.py /path/to/lego_manual.pdf")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nRefer to README.md for detailed setup instructions.")
    print("=" * 80)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())

