import cv2
import numpy as np
import json
import time
import math
from datetime import datetime
import argparse
import os

class DroneHUDOverlay:
    def __init__(self, video_path, output_path=None, telemetry_data=None):
        self.video_path = video_path
        self.output_path = output_path or f"hud_overlay_{os.path.basename(video_path)}"
        self.telemetry_data = telemetry_data or {}
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # HUD colors (BGR format for OpenCV)
        self.colors = {
            'primary': (0, 255, 0),      # Green
            'secondary': (255, 170, 0),   # Orange/Cyan
            'warning': (0, 170, 255),     # Orange/Yellow
            'error': (0, 0, 255),         # Red
            'background': (0, 0, 0),      # Black
            'white': (255, 255, 255)      # White
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1
        
        # Initialize telemetry simulation
        self.start_time = time.time()
        self.frame_count = 0
        
    def simulate_telemetry(self, frame_number):
        """Simulate realistic telemetry data"""
        elapsed_time = frame_number / self.fps
        
        return {
            'altitude': 240 + math.sin(elapsed_time / 5) * 20,
            'ground_speed': 15 + math.sin(elapsed_time / 3) * 3,
            'vertical_speed': math.sin(elapsed_time / 4) * 3,
            'heading': (elapsed_time * 10) % 360,
            'roll': math.sin(elapsed_time / 6) * 10,
            'pitch': math.sin(elapsed_time / 7) * 5,
            'battery': max(20, 85 - (elapsed_time / 10)),
            'gps_satellites': 12,
            'distance_home': 1.2 + elapsed_time * 0.1,
            'wind_speed': 8.2,
            'wind_direction': 270,
            'temperature': 23.5,
            'pressure': 1013.2,
            'mission_time': elapsed_time,
            'current_waypoint': min(8, int(elapsed_time / 30) + 1),
            'total_waypoints': 8
        }
    
    def draw_crosshair(self, frame):
        """Draw central crosshair"""
        center_x, center_y = self.width // 2, self.height // 2
        radius = 50
        
        # Outer circle
        cv2.circle(frame, (center_x, center_y), radius, self.colors['primary'], 2)
        
        # Cross lines
        cv2.line(frame, (center_x, center_y - radius - 20), 
                (center_x, center_y - radius + 20), self.colors['primary'], 2)
        cv2.line(frame, (center_x - radius - 20, center_y), 
                (center_x - radius + 20, center_y), self.colors['primary'], 2)
        cv2.line(frame, (center_x, center_y + radius - 20), 
                (center_x, center_y + radius + 20), self.colors['primary'], 2)
        cv2.line(frame, (center_x + radius - 20, center_y), 
                (center_x + radius + 20, center_y), self.colors['primary'], 2)
    
    def draw_attitude_indicator(self, frame, roll, pitch):
        """Draw attitude indicator"""
        center_x = 100
        center_y = self.height // 2
        radius = 60
        
        # Outer circle
        cv2.circle(frame, (center_x, center_y), radius, self.colors['primary'], 2)
        
        # Horizon line
        roll_rad = math.radians(roll)
        line_length = radius + 20
        
        x1 = int(center_x - line_length * math.cos(roll_rad))
        y1 = int(center_y - line_length * math.sin(roll_rad))
        x2 = int(center_x + line_length * math.cos(roll_rad))
        y2 = int(center_y + line_length * math.sin(roll_rad))
        
        cv2.line(frame, (x1, y1), (x2, y2), self.colors['primary'], 2)
        
        # Center dot
        cv2.circle(frame, (center_x, center_y), 3, self.colors['primary'], -1)
    
    def draw_compass(self, frame, heading):
        """Draw compass"""
        center_x = self.width - 100
        center_y = self.height // 2
        radius = 60
        
        # Outer circle
        cv2.circle(frame, (center_x, center_y), radius, self.colors['primary'], 2)
        
        # Compass needle
        heading_rad = math.radians(heading - 90)  # Adjust for north pointing up
        needle_length = radius - 10
        
        needle_x = int(center_x + needle_length * math.cos(heading_rad))
        needle_y = int(center_y + needle_length * math.sin(heading_rad))
        
        cv2.line(frame, (center_x, center_y), (needle_x, needle_y), self.colors['error'], 3)
        
        # N marker
        cv2.putText(frame, 'N', (center_x - 8, center_y - radius - 10), 
                   self.font, 0.5, self.colors['primary'], 1)
    
    def draw_speed_tape(self, frame, speed):
        """Draw speed tape on the left side"""
        x = 30
        y_start = self.height // 2 - 150
        y_end = self.height // 2 + 150
        width = 60
        
        # Background
        cv2.rectangle(frame, (x - 10, y_start), (x + width, y_end), 
                     (0, 0, 0, 180), -1)
        cv2.rectangle(frame, (x - 10, y_start), (x + width, y_end), 
                     self.colors['primary'], 1)
        
        # Speed markings
        for i in range(-10, 11):
            speed_val = int(speed) + i * 2
            if speed_val >= 0:
                y_pos = self.height // 2 - i * 15
                if y_start <= y_pos <= y_end:
                    cv2.line(frame, (x, y_pos), (x + 15, y_pos), self.colors['primary'], 1)
                    cv2.putText(frame, str(speed_val), (x + 20, y_pos + 5), 
                               self.font, 0.4, self.colors['primary'], 1)
        
        # Current speed indicator
        cv2.rectangle(frame, (x - 5, self.height // 2 - 8), 
                     (x + width - 5, self.height // 2 + 8), self.colors['primary'], 2)
    
    def draw_altitude_tape(self, frame, altitude):
        """Draw altitude tape on the right side"""
        x = self.width - 90
        y_start = self.height // 2 - 150
        y_end = self.height // 2 + 150
        width = 60
        
        # Background
        cv2.rectangle(frame, (x, y_start), (x + width + 10, y_end), 
                     (0, 0, 0, 180), -1)
        cv2.rectangle(frame, (x, y_start), (x + width + 10, y_end), 
                     self.colors['primary'], 1)
        
        # Altitude markings
        for i in range(-10, 11):
            alt_val = int(altitude) + i * 10
            if alt_val >= 0:
                y_pos = self.height // 2 - i * 15
                if y_start <= y_pos <= y_end:
                    cv2.line(frame, (x + width - 15, y_pos), (x + width, y_pos), 
                            self.colors['primary'], 1)
                    cv2.putText(frame, str(alt_val), (x + 5, y_pos + 5), 
                               self.font, 0.4, self.colors['primary'], 1)
        
        # Current altitude indicator
        cv2.rectangle(frame, (x + 5, self.height // 2 - 8), 
                     (x + width + 5, self.height // 2 + 8), self.colors['primary'], 2)
    
    def draw_panel(self, frame, x, y, width, height, title, data):
        """Draw information panel"""
        # Background with transparency effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), 
                     self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), 
                     self.colors['primary'], 1)
        
        # Title
        cv2.putText(frame, title, (x + 10, y + 20), 
                   self.font, 0.5, self.colors['secondary'], 1)
        
        # Data rows
        row_height = 18
        for i, (label, value) in enumerate(data.items()):
            y_pos = y + 40 + i * row_height
            cv2.putText(frame, f"{label}:", (x + 10, y_pos), 
                       self.font, 0.4, self.colors['secondary'], 1)
            cv2.putText(frame, str(value), (x + width - 80, y_pos), 
                       self.font, 0.4, self.colors['primary'], 1)
    
    def draw_battery_bar(self, frame, x, y, battery_level):
        """Draw battery indicator"""
        bar_width = 100
        bar_height = 15
        
        # Background
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), 
                     self.colors['background'], -1)
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), 
                     self.colors['primary'], 1)
        
        # Battery fill
        fill_width = int(bar_width * battery_level / 100)
        color = self.colors['primary'] if battery_level > 30 else self.colors['error']
        cv2.rectangle(frame, (x + 1, y + 1), 
                     (x + fill_width - 1, y + bar_height - 1), color, -1)
        
        # Battery percentage text
        cv2.putText(frame, f"{battery_level:.0f}%", (x + bar_width + 10, y + 12), 
                   self.font, 0.4, self.colors['primary'], 1)
    
    def draw_warning_banner(self, frame, message):
        """Draw warning banner at top"""
        banner_height = 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, banner_height), 
                     self.colors['error'], -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        text_size = cv2.getTextSize(message, self.font, 0.7, 2)[0]
        text_x = (self.width - text_size[0]) // 2
        cv2.putText(frame, message, (text_x, 25), 
                   self.font, 0.7, self.colors['error'], 2)
    
    def draw_flight_mode(self, frame, mode):
        """Draw flight mode indicator"""
        text_size = cv2.getTextSize(mode, self.font, 0.8, 2)[0]
        x = (self.width - text_size[0]) // 2
        y = 60
        
        # Background
        cv2.rectangle(frame, (x - 20, y - 25), (x + text_size[0] + 20, y + 10), 
                     self.colors['background'], -1)
        cv2.rectangle(frame, (x - 20, y - 25), (x + text_size[0] + 20, y + 10), 
                     self.colors['primary'], 2)
        
        cv2.putText(frame, mode, (x, y), self.font, 0.8, self.colors['primary'], 2)
    
    def draw_waypoints(self, frame, current_wp, total_wp):
        """Draw mission waypoints"""
        # Example waypoint positions (you can modify based on actual mission data)
        waypoints = [
            (int(self.width * 0.6), int(self.height * 0.3)),
            (int(self.width * 0.4), int(self.height * 0.7))
        ]
        
        for i, (x, y) in enumerate(waypoints):
            color = self.colors['warning'] if i < current_wp else self.colors['secondary']
            cv2.circle(frame, (x, y), 8, color, -1)
            cv2.circle(frame, (x, y), 10, self.colors['white'], 2)
    
    def process_frame(self, frame, telemetry):
        """Process a single frame with HUD overlay"""
        # Draw main HUD elements
        self.draw_crosshair(frame)
        self.draw_attitude_indicator(frame, telemetry['roll'], telemetry['pitch'])
        self.draw_compass(frame, telemetry['heading'])
        self.draw_speed_tape(frame, telemetry['ground_speed'])
        self.draw_altitude_tape(frame, telemetry['altitude'])
        
        # Draw information panels
        flight_data = {
            'Altitude': f"{telemetry['altitude']:.1f} m",
            'Speed': f"{telemetry['ground_speed']:.1f} m/s",
            'V-Speed': f"{telemetry['vertical_speed']:+.1f} m/s",
            'Heading': f"{telemetry['heading']:.0f}°",
            'Dist Home': f"{telemetry['distance_home']:.1f} km"
        }
        self.draw_panel(frame, 20, 20, 250, 130, "FLIGHT DATA", flight_data)
        
        systems_data = {
            'GPS': f"{telemetry['gps_satellites']} SAT",
            'Battery': f"{telemetry['battery']:.0f}%",
            'Temp': f"{telemetry['temperature']:.1f}°C",
            'Pressure': f"{telemetry['pressure']:.1f} hPa"
        }
        self.draw_panel(frame, self.width - 270, 20, 250, 110, "SYSTEMS", systems_data)
        
        mission_data = {
            'Waypoint': f"{telemetry['current_waypoint']}/{telemetry['total_waypoints']}",
            'Mission Time': f"{int(telemetry['mission_time']//60):02d}:{int(telemetry['mission_time']%60):02d}",
            'Wind Speed': f"{telemetry['wind_speed']:.1f} m/s",
            'Wind Dir': f"{telemetry['wind_direction']:.0f}°"
        }
        self.draw_panel(frame, 20, self.height - 130, 280, 110, "MISSION", mission_data)
        
        # Battery bar in bottom right panel
        self.draw_battery_bar(frame, self.width - 270, self.height - 40, telemetry['battery'])
        
        # Flight mode
        modes = ['AUTONOMOUS', 'LOITER', 'RTL', 'GUIDED']
        current_mode = modes[int(telemetry['mission_time'] / 10) % len(modes)]
        self.draw_flight_mode(frame, current_mode)
        
        # Warning banner for low battery
        if telemetry['battery'] < 30:
            self.draw_warning_banner(frame, "⚠ LOW BATTERY WARNING ⚠")
        
        # Draw waypoints
        self.draw_waypoints(frame, telemetry['current_waypoint'], telemetry['total_waypoints'])
        
        return frame
    
    def process_video(self):
        """Process the entire video with HUD overlay"""
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        print(f"Processing video: {self.video_path}")
        print(f"Output: {self.output_path}")
        print(f"Resolution: {self.width}x{self.height}, FPS: {self.fps}")
        print(f"Total frames: {self.total_frames}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Get telemetry data (simulated or from file)
            telemetry = self.simulate_telemetry(self.frame_count)
            
            # Process frame with HUD overlay
            hud_frame = self.process_frame(frame, telemetry)
            
            # Write frame
            out.write(hud_frame)
            
            # Progress indicator
            if self.frame_count % 30 == 0:
                progress = (self.frame_count / self.total_frames) * 100
                print(f"Progress: {progress:.1f}%")
            
            self.frame_count += 1
        
        # Clean up
        self.cap.release()
        out.release()
        print(f"Processing complete! Output saved to: {self.output_path}")

def load_telemetry_from_file(file_path):
    """Load telemetry data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Telemetry file not found: {file_path}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Add HUD overlay to drone video')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('-o', '--output', help='Output video file path')
    parser.add_argument('-t', '--telemetry', help='Telemetry data JSON file')
    
    args = parser.parse_args()
    
    # Load telemetry data if provided
    telemetry_data = None
    if args.telemetry:
        telemetry_data = load_telemetry_from_file(args.telemetry)
    
    try:
        # Create HUD overlay processor
        hud_processor = DroneHUDOverlay(
            video_path=args.input_video,
            output_path=args.output,
            telemetry_data=telemetry_data
        )
        
        # Process the video
        hud_processor.process_video()
        
    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    # Example usage without command line arguments
    # Uncomment and modify the following lines for direct usage:
    
    # hud_processor = DroneHUDOverlay(
    #     video_path="input_drone_video.mp4",
    #     output_path="output_with_hud.mp4"
    # )
    # hud_processor.process_video()
    
    main()