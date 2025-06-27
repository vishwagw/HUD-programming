# Example usage and telemetry data format

import json
import math

def create_sample_telemetry(duration_seconds=60, fps=30):
    """
    Create sample telemetry data for testing
    This simulates a realistic drone flight mission
    """
    total_frames = duration_seconds * fps
    telemetry_data = []
    
    for frame in range(total_frames):
        time_elapsed = frame / fps
        
        # Simulate a mission with takeoff, cruise, and landing phases
        if time_elapsed < 10:  # Takeoff phase
            altitude = time_elapsed * 25  # Rising to 250m
            speed = time_elapsed * 2  # Accelerating
            mode_phase = 0
        elif time_elapsed < 50:  # Cruise phase
            altitude = 250 + math.sin(time_elapsed / 10) * 20
            speed = 15 + math.sin(time_elapsed / 5) * 3
            mode_phase = 1
        else:  # Landing phase
            altitude = max(0, 250 - (time_elapsed - 50) * 25)
            speed = max(0, 15 - (time_elapsed - 50) * 1.5)
            mode_phase = 2
        
        telemetry = {
            "timestamp": time_elapsed,
            "frame_number": frame,
            "altitude": altitude,
            "ground_speed": speed,
            "vertical_speed": math.sin(time_elapsed / 4) * 2 if mode_phase == 1 else (5 if mode_phase == 0 else -5),
            "heading": (time_elapsed * 5) % 360,
            "roll": math.sin(time_elapsed / 6) * 8,
            "pitch": math.sin(time_elapsed / 7) * 5,
            "battery": max(15, 100 - (time_elapsed / duration_seconds * 85)),
            "gps_satellites": min(12, 4 + int(time_elapsed / 5)),
            "distance_home": abs(time_elapsed - duration_seconds/2) * 0.2,
            "wind_speed": 8 + math.sin(time_elapsed / 15) * 3,
            "wind_direction": (270 + math.sin(time_elapsed / 20) * 30) % 360,
            "temperature": 23 + math.sin(time_elapsed / 30) * 5,
            "pressure": 1013 + math.sin(time_elapsed / 25) * 10,
            "mission_time": time_elapsed,
            "current_waypoint": min(8, int(time_elapsed / 7) + 1),
            "total_waypoints": 8,
            "flight_mode": ["TAKEOFF", "AUTO", "RTL"][mode_phase]
        }
        
        telemetry_data.append(telemetry)
    
    return telemetry_data

def save_telemetry_to_file(telemetry_data, filename):
    """Save telemetry data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(telemetry_data, f, indent=2)
    print(f"Telemetry data saved to {filename}")

# Example usage scenarios

def example_basic_usage():
    """Basic usage with simulated data"""
    from video_overlay import DroneHUDOverlay
    
    print("=== Basic Usage Example ===")
    
    try:
        hud_processor = DroneHUDOverlay(
            video_path="input_drone_video.mp4",
            output_path="output_with_hud.mp4"
        )
        hud_processor.process_video()
        print("Video processing completed successfully!")
    except FileNotFoundError:
        print("Input video file not found. Please check the file path.")
    except Exception as e:
        print(f"Error: {e}")

def example_with_telemetry_file():
    """Usage with external telemetry data"""
    from video_overlay import DroneHUDOverlay
    
    print("=== Usage with Telemetry File Example ===")
    
    # Create sample telemetry data
    telemetry_data = create_sample_telemetry(duration_seconds=120, fps=30)
    save_telemetry_to_file(telemetry_data, "sample_telemetry.json")
    
    try:
        # Load telemetry and process video
        with open("sample_telemetry.json", 'r') as f:
            telemetry_data = json.load(f)
        
        hud_processor = DroneHUDOverlay(
            video_path="input_drone_video.mp4",
            output_path="output_with_real_telemetry.mp4",
            telemetry_data=telemetry_data
        )
        hud_processor.process_video()
        print("Video processing with telemetry completed successfully!")
    except FileNotFoundError:
        print("Input video or telemetry file not found.")
    except Exception as e:
        print(f"Error: {e}")

def example_batch_processing():
    """Process multiple videos with different settings"""
    from video_overlay import DroneHUDOverlay
    import os
    
    print("=== Batch Processing Example ===")
    
    video_files = [
        "flight1.mp4",
        "flight2.mp4", 
        "flight3.mp4"
    ]
    
    for i, video_file in enumerate(video_files):
        if os.path.exists(video_file):
            try:
                output_file = f"flight_{i+1}_with_hud.mp4"
                hud_processor = DroneHUDOverlay(
                    video_path=video_file,
                    output_path=output_file
                )
                hud_processor.process_video()
                print(f"Processed {video_file} -> {output_file}")
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
        else:
            print(f"File not found: {video_file}")

def create_telemetry_template():
    """Create a template file for manual telemetry data entry"""
    template = {
        "description": "Telemetry data template for drone HUD overlay",
        "format": "Array of telemetry objects, one per frame",
        "required_fields": {
            "timestamp": "float - time in seconds",
            "frame_number": "int - frame index",
            "altitude": "float - altitude in meters",
            "ground_speed": "float - speed in m/s",
            "vertical_speed": "float - vertical speed in m/s",
            "heading": "float - heading in degrees (0-360)",
            "roll": "float - roll angle in degrees",
            "pitch": "float - pitch angle in degrees",
            "battery": "float - battery percentage (0-100)",
            "gps_satellites": "int - number of GPS satellites",
            "distance_home": "float - distance to home in km",
            "wind_speed": "float - wind speed in m/s",
            "wind_direction": "float - wind direction in degrees",
            "temperature": "float - temperature in Celsius",
            "pressure": "float - pressure in hPa",
            "mission_time": "float - mission elapsed time in seconds",
            "current_waypoint": "int - current waypoint number",
            "total_waypoints": "int - total waypoints in mission",
            "flight_mode": "string - current flight mode"
        },
        "sample_data": [
            {
                "timestamp": 0.0,
                "frame_number": 0,
                "altitude": 0.0,
                "ground_speed": 0.0,
                "vertical_speed": 0.0,
                "heading": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "battery": 100.0,
                "gps_satellites": 8,
                "distance_home": 0.0,
                "wind_speed": 5.0,
                "wind_direction": 270.0,
                "temperature": 23.0,
                "pressure": 1013.2,
                "mission_time": 0.0,
                "current_waypoint": 1,
                "total_waypoints": 5,
                "flight_mode": "TAKEOFF"
            }
        ]
    }
    
    with open("telemetry_template.json", 'w') as f:
        json.dump(template, f, indent=2)
    print("Telemetry template saved to telemetry_template.json")

def integrate_with_mavlink():
    """
    Example of how to integrate with MAVLink telemetry data
    This would require pymavlink library: pip install pymavlink
    """
    print("=== MAVLink Integration Example ===")
    print("""
    To integrate with real MAVLink telemetry data:
    
    1. Install pymavlink: pip install pymavlink
    2. Connect to your flight controller or log file
    3. Extract telemetry data and convert to the required format
    
    Example code structure:
    
    from pymavlink import mavutil
    
    def extract_telemetry_from_mavlink(log_file):
        mavlink_connection = mavutil.mavlink_connection(log_file)
        telemetry_data = []
        
        while True:
            msg = mavlink_connection.recv_match(blocking=False)
            if msg is None:
                break
                
            if msg.get_type() == 'ATTITUDE':
                # Extract attitude data
                roll = math.degrees(msg.roll)
                pitch = math.degrees(msg.pitch)
                yaw = math.degrees(msg.yaw)
                
            elif msg.get_type() == 'GLOBAL_POSITION_INT':
                # Extract position data
                altitude = msg.relative_alt / 1000.0  # Convert mm to m
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7
                
            elif msg.get_type() == 'VFR_HUD':
                # Extract speed and heading data
                ground_speed = msg.groundspeed
                heading = msg.heading
                altitude = msg.alt
                
            elif msg.get_type() == 'SYS_STATUS':
                # Extract system status
                battery_voltage = msg.voltage_battery / 1000.0
                battery_remaining = msg.battery_remaining
                
            # Combine all data into telemetry format
            telemetry_entry = {
                "timestamp": msg.time_boot_ms / 1000.0,
                "frame_number": len(telemetry_data),
                "altitude": altitude,
                "ground_speed": ground_speed,
                "heading": heading,
                "roll": roll,
                "pitch": pitch,
                "battery": battery_remaining,
                # ... add other fields
            }
            telemetry_data.append(telemetry_entry)
            
        return telemetry_data
    """)

def requirements_and_installation():
    """Display installation requirements and setup instructions"""
    print("=== Installation Requirements ===")
    print("""
    Required Python packages:
    
    pip install opencv-python numpy
    
    Optional packages for advanced features:
    
    pip install pymavlink          # For MAVLink telemetry integration
    pip install dronekit          # For real-time drone communication
    pip install matplotlib        # For telemetry data visualization
    pip install pandas           # For telemetry data processing
    
    System Requirements:
    - Python 3.7 or higher
    - Sufficient disk space for output videos
    - Good CPU for video processing (GPU acceleration possible with OpenCV build)
    
    Usage Examples:
    
    # Command line usage
    python drone_hud_overlay.py input_video.mp4 -o output_with_hud.mp4
    python drone_hud_overlay.py input_video.mp4 -t telemetry.json -o output.mp4
    
    # Python script usage
    from drone_hud_overlay import DroneHUDOverlay
    
    processor = DroneHUDOverlay("input.mp4", "output.mp4")
    processor.process_video()
    """)

def advanced_customization_example():
    """Show how to customize HUD elements"""
    print("=== Advanced Customization Example ===")
    
    customization_code = '''
    # Create a custom HUD processor class
    class CustomDroneHUD(DroneHUDOverlay):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Custom colors (BGR format)
            self.colors = {
                'primary': (0, 255, 255),    # Yellow
                'secondary': (255, 0, 255),   # Magenta  
                'warning': (0, 165, 255),     # Orange
                'error': (0, 0, 255),         # Red
                'background': (50, 50, 50),   # Dark gray
                'white': (255, 255, 255)      # White
            }
            
        def draw_custom_logo(self, frame):
            """Add custom logo or branding"""
            logo_text = "MY DRONE"
            text_size = cv2.getTextSize(logo_text, self.font, 1.0, 2)[0]
            x = self.width - text_size[0] - 20
            y = self.height - 30
            
            cv2.putText(frame, logo_text, (x, y), 
                       self.font, 1.0, self.colors['primary'], 2)
        
        def draw_custom_telemetry_graph(self, frame, telemetry_history):
            """Draw a small altitude graph"""
            if len(telemetry_history) > 1:
                graph_x = self.width - 200
                graph_y = self.height - 150
                graph_w = 150
                graph_h = 80
                
                # Background
                cv2.rectangle(frame, (graph_x, graph_y), 
                             (graph_x + graph_w, graph_y + graph_h), 
                             self.colors['background'], -1)
                cv2.rectangle(frame, (graph_x, graph_y), 
                             (graph_x + graph_w, graph_y + graph_h), 
                             self.colors['primary'], 1)
                
                # Plot altitude line
                altitudes = [t['altitude'] for t in telemetry_history[-50:]]
                if altitudes:
                    min_alt = min(altitudes)
                    max_alt = max(altitudes)
                    alt_range = max_alt - min_alt if max_alt > min_alt else 1
                    
                    points = []
                    for i, alt in enumerate(altitudes):
                        x = graph_x + int(i * graph_w / len(altitudes))
                        y = graph_y + graph_h - int((alt - min_alt) / alt_range * graph_h)
                        points.append((x, y))
                    
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i-1], points[i], 
                                self.colors['primary'], 2)
        
        def process_frame(self, frame, telemetry):
            """Override to add custom elements"""
            # Call parent method first
            frame = super().process_frame(frame, telemetry)
            
            # Add custom elements
            self.draw_custom_logo(frame)
            
            return frame
    
    # Usage
    custom_hud = CustomDroneHUD("input.mp4", "custom_output.mp4")
    custom_hud.process_video()
    '''
    
    print(customization_code)

def performance_optimization_tips():
    """Provide performance optimization suggestions"""
    print("=== Performance Optimization Tips ===")
    print("""
    1. Video Resolution:
       - Lower resolution = faster processing
       - Consider downscaling input if HUD elements are too small
    
    2. Frame Rate:
       - Process every Nth frame for preview
       - Use lower FPS for faster processing during development
    
    3. Memory Management:
       - Process videos in chunks for very large files
       - Clear telemetry data not needed for current frame
    
    4. OpenCV Optimization:
       - Use compiled OpenCV with optimizations
       - Consider GPU acceleration if available
    
    5. Code Optimization:
       - Pre-calculate static elements
       - Use efficient drawing operations
       - Minimize redundant calculations
    
    Example optimized processing:
    
    class OptimizedDroneHUD(DroneHUDOverlay):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.static_elements = self.precompute_static_elements()
        
        def precompute_static_elements(self):
            # Pre-calculate positions, sizes, etc.
            return {
                'crosshair_center': (self.width // 2, self.height // 2),
                'panel_positions': {
                    'top_left': (20, 20),
                    'top_right': (self.width - 270, 20),
                    # ... etc
                }
            }
        
        def process_frame_optimized(self, frame, telemetry):
            # Use pre-calculated values
            # Minimize drawing operations
            # Use vectorized operations where possible
            pass
    """)

def testing_and_validation():
    """Provide testing guidelines"""
    print("=== Testing and Validation ===")
    print("""
    1. Test with Different Video Formats:
       - MP4, AVI, MOV
       - Different codecs and bitrates
       - Various resolutions and frame rates
    
    2. Validate Telemetry Synchronization:
       - Ensure telemetry matches video timeline
       - Check for frame drops or timing issues
       - Verify data accuracy
    
    3. Quality Assurance:
       - Check HUD element positioning
       - Verify text readability
       - Test color visibility in different lighting
    
    4. Performance Testing:
       - Measure processing time
       - Monitor memory usage
       - Test with large files
    
    Test Script Example:
    
    def run_tests():
        test_cases = [
            ("test_video_1080p.mp4", "telemetry_1080p.json"),
            ("test_video_720p.mp4", "telemetry_720p.json"),
            ("test_video_4k.mp4", None)  # Use simulated data
        ]
        
        for video_file, telem_file in test_cases:
            print(f"Testing {video_file}...")
            start_time = time.time()
            
            try:
                processor = DroneHUDOverlay(video_file, f"test_output_{video_file}")
                processor.process_video()
                
                processing_time = time.time() - start_time
                print(f"✓ Success: {processing_time:.2f}s")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
    """)

if __name__ == "__main__":
    print("Drone HUD Overlay - Examples and Documentation")
    print("=" * 50)
    
    # Show all examples
    requirements_and_installation()
    print("\n" + "="*50 + "\n")
    
    example_basic_usage()
    print("\n" + "="*50 + "\n")
    
    create_telemetry_template()
    print("\n" + "="*50 + "\n")
    
    advanced_customization_example()
    print("\n" + "="*50 + "\n")
    
    performance_optimization_tips()
    print("\n" + "="*50 + "\n")
    
    testing_and_validation()
    print("\n" + "="*50 + "\n")
    
    integrate_with_mavlink()
    print("\n" + "="*50 + "\n")
    
    # Create sample files
    print("Creating sample telemetry data...")
    sample_telemetry = create_sample_telemetry(duration_seconds=60, fps=30)
    save_telemetry_to_file(sample_telemetry, "sample_telemetry.json")
    
    create_telemetry_template()
    
    print("\nSetup complete! You can now:")
    print("1. Place your drone video file in the same directory")
    print("2. Run: python drone_hud_overlay.py your_video.mp4")
    print("3. Or use the sample telemetry: python drone_hud_overlay.py your_video.mp4 -t sample_telemetry.json")