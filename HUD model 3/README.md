# Drone HUD intergration 

## Main Script (vid_overlay.py)

Full-featured Python script that recreates your HTML HUD using OpenCV
Real-time telemetry simulation or external data loading
Professional HUD elements: crosshair, attitude indicator, compass, speed/altitude tapes, information panels
Configurable colors, fonts, and layouts

## Examples & Documentation (telemetry_data.py)

Multiple usage examples (basic, with telemetry, batch processing)
MAVLink integration guide for real drone data
Performance optimization tips
Testing and validation guidelines
Custom HUD element examples

## Setup & Installation (setup_req.py)

Automated installation script
Requirements management
Project structure creation
Configuration file generation
Installation verification tests

### basic usage:
python video_overlay.py input1.mp4

### with tel_Data:
python video_overlay.py input1.mp4 -t telemetry.json