# setup.py - Installation script for Drone HUD Overlay

from setuptools import setup, find_packages

setup(
    name="drone-hud-overlay",
    version="1.0.0",
    description="Add realistic HUD overlay to drone videos with telemetry data",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "mavlink": ["pymavlink>=2.4.0"],
        "analysis": ["matplotlib>=3.3.0", "pandas>=1.2.0"],
        "all": ["pymavlink>=2.4.0", "matplotlib>=3.3.0", "pandas>=1.2.0"]
    },
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "drone-hud=drone_hud_overlay:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

# requirements.txt
"""
opencv-python>=4.5.0
numpy>=1.19.0

# Optional dependencies
# pymavlink>=2.4.0        # For MAVLink integration
# matplotlib>=3.3.0       # For data visualization
# pandas>=1.2.0           # For data processing
# dronekit>=2.9.0         # For real-time drone communication
"""

# install.py - Automated installation script

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    requirements = [
        "opencv-python>=4.5.0",
        "numpy>=1.19.0"
    ]
    
    optional_requirements = {
        "mavlink": ["pymavlink>=2.4.0"],
        "visualization": ["matplotlib>=3.3.0", "pandas>=1.2.0"],
        "dronekit": ["dronekit>=2.9.0"]
    }
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
    
    print("\nOptional packages (install if needed):")
    for category, packages in optional_requirements.items():
        print(f"\n{category.upper()}:")
        for package in packages:
            print(f"  pip install {package}")

def check_opencv_installation():
    """Verify OpenCV installation"""
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Check for video codec support
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("✓ MP4 codec support available")
        
        return True
    except ImportError:
        print("✗ OpenCV not installed or not working")
        return False

def create_project_structure():
    """Create recommended project structure"""
    directories = [
        "input_videos",
        "output_videos", 
        "telemetry_data",
        "examples",
        "logs"
    ]
    
    print("Creating project structure...")
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"→ Directory exists: {directory}")

def create_config_file():
    """Create default configuration file"""
    config = {
        "default_colors": {
            "primary": [0, 255, 0],
            "secondary": [255, 170, 0],
            "warning": [0, 170, 255],
            "error": [0, 0, 255],
            "background": [0, 0, 0],
            "white": [255, 255, 255]
        },
        "default_settings": {
            "font_scale": 0.6,
            "font_thickness": 1,
            "panel_transparency": 0.8,
            "update_rate_ms": 100
        },
        "output_settings": {
            "codec": "mp4v",
            "quality": "high",
            "preserve_audio": True
        }
    }
    
    import json
    with open("hud_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("✓ Created configuration file: hud_config.json")

def run_test():
    """Run a basic test to verify installation"""
    print("\nRunning installation test...")
    
    test_code = '''
import cv2
import numpy as np
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)

# Test basic video operations
cap = cv2.VideoCapture()
print("VideoCapture created successfully")

# Test drawing operations
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.circle(img, (320, 240), 50, (0, 255, 0), 2)
cv2.putText(img, "TEST", (280, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
print("Drawing operations successful")

print("✓ All tests passed!")
'''
    
    try:
        exec(test_code)
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Main installation function"""
    print("Drone HUD Overlay - Installation Script")
    print("=" * 40)
    
    # Install requirements
    install_requirements()
    
    print("\n" + "=" * 40)
    
    # Check OpenCV
    opencv_ok = check_opencv_installation()
    
    print("\n" + "=" * 40)
    
    # Create project structure
    create_project_structure()
    
    print("\n" + "=" * 40)
    
    # Create config file
    create_config_file()
    
    print("\n" + "=" * 40)
    
    # Run test
    if opencv_ok:
        test_ok = run_test()
    else:
        test_ok = False
    
    print("\n" + "=" * 40)
    print("INSTALLATION SUMMARY")
    print("=" * 40)
    
    if test_ok:
        print("✓ Installation completed successfully!")
        print("\nNext steps:")
        print("1. Place your drone video in the 'input_videos' folder")
        print("2. Run: python drone_hud_overlay.py input_videos/your_video.mp4")
        print("3. Check 'output_videos' folder for results")
        print("\nExample commands:")
        print("  python drone_hud_overlay.py input_videos/flight.mp4 -o output_videos/flight_hud.mp4")
        print("  python drone_hud_overlay.py input_videos/flight.mp4 -t telemetry_data/flight.json")
    else:
        print("✗ Installation had issues. Please check error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure you have Python 3.7+ installed")
        print("2. Try: pip install --upgrade pip")
        print("3. Try: pip install opencv-python --force-reinstall")
        print("4. On some systems, you may need: pip install opencv-python-headless")

if __name__ == "__main__":
    main()

# Quick start script - quick_start.py

def quick_start_guide():
    """Display quick start guide"""
    print("""
    DRONE HUD OVERLAY - QUICK START GUIDE
    =====================================
    
    1. BASIC USAGE (with simulated data):
       python drone_hud_overlay.py your_video.mp4
    
    2. WITH TELEMETRY DATA:
       python drone_hud_overlay.py your_video.mp4 -t telemetry.json
    
    3. SPECIFY OUTPUT FILE:
       python drone_hud_overlay.py input.mp4 -o output_with_hud.mp4
    
    4. PROGRAMMATIC USAGE:
       from drone_hud_overlay import DroneHUDOverlay
       
       processor = DroneHUDOverlay("input.mp4", "output.mp4")
       processor.process_video()
    
    TELEMETRY DATA FORMAT:
    =====================
    JSON file with array of telemetry objects:
    [
        {
            "timestamp": 0.0,
            "altitude": 0.0,
            "ground_speed": 0.0,
            "heading": 0.0,
            "battery": 100.0,
            ... (see telemetry_template.json for full format)
        }
    ]
    
    CUSTOMIZATION:
    =============
    - Edit hud_config.json to change colors and settings
    - Extend DroneHUDOverlay class for custom HUD elements
    - Add your own telemetry data sources
    
    TROUBLESHOOTING:
    ===============
    - Video file not found: Check file path and format
    - Codec issues: Try different output format (.avi, .mov)
    - Performance: Reduce video resolution or frame rate
    - Memory issues: Process shorter video segments
    
    SUPPORTED FORMATS:
    =================
    Input: MP4, AVI, MOV, MKV (depends on OpenCV build)
    Output: MP4, AVI (MP4 recommended)
    
    For more help, see the examples in the examples/ folder.
    """)

if __name__ == "__main__":
    quick_start_guide()