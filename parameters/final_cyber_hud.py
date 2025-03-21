import cv2
import numpy as np

def draw_scan_lines(frame):
    """ Adds animated scan lines to the HUD. """
    overlay = frame.copy()
    height, width, _ = frame.shape
    for y in range(0, height, 10):
        cv2.line(overlay, (0, y), (width, y), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    return frame

def draw_holographic_grid(frame):
    """ Draws a futuristic wireframe grid overlay. """
    overlay = frame.copy()
    height, width, _ = frame.shape
    for x in range(0, width, 40):
        cv2.line(overlay, (x, 0), (x, height), (0, 255, 255), 1)
    for y in range(0, height, 40):
        cv2.line(overlay, (0, y), (width, y), (0, 255, 255), 1)
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
    return frame

def draw_compass(frame, heading):
    """ Draws a cyberpunk-style compass at the top left. """
    center = (80, 80)
    radius = 50
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, (255, 0, 255), 2, cv2.LINE_AA)
    
    angle_rad = np.deg2rad(-heading)
    needle_x = int(center[0] + radius * np.sin(angle_rad))
    needle_y = int(center[1] - radius * np.cos(angle_rad))
    cv2.line(frame, center, (needle_x, needle_y), (255, 0, 255), 3, cv2.LINE_AA)
    
    cv2.putText(frame, f"{heading:.0f}°", (center[0] - 20, center[1] + radius + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return frame

def draw_altitude_indicator(frame, altitude):
    """ Draws an altitude indicator at the top right corner. """
    height, width, _ = frame.shape
    bar_x, bar_y, bar_width, bar_height = width - 50, 50, 20, 200
    fill_height = int(bar_height * (altitude / 100))
    overlay = frame.copy()
    cv2.rectangle(overlay, (bar_x, bar_y + bar_height - fill_height), (bar_x + bar_width, bar_y + bar_height), (0, 255, 255), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    cv2.putText(frame, f"Alt: {altitude:.0f}m", (bar_x - 60, bar_y + bar_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame

def draw_cyberpunk_speed_meter(frame, center, speed, max_speed, label, color):
    """ Draws a cyberpunk-style circular speed meter with a neon glow. """
    radius = 60
    angle_range = 270  
    min_angle = -135
    max_angle = 135
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, color, 2, cv2.LINE_AA)
    cv2.circle(overlay, center, radius-2, (0, 0, 0), -1)
    
    needle_angle = np.deg2rad(min_angle + (speed / max_speed) * angle_range)
    needle_x = int(center[0] + (radius - 10) * np.cos(needle_angle))
    needle_y = int(center[1] - (radius - 10) * np.sin(needle_angle))
    cv2.line(frame, center, (needle_x, needle_y), color, 3, cv2.LINE_AA)
    
    speed_display = f"{speed:.1f} m/s"
    cv2.putText(frame, speed_display, (center[0] - 35, center[1] + radius + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)
    return frame

def draw_cyberpunk_battery(frame, percentage):
    """ Draws a neon-style battery bar. """
    bar_x, bar_y, bar_width, bar_height = 500, 20, 120, 25
    fill_width = int(bar_width * (percentage / 100))
    color = (0, 255, 255) if percentage > 30 else (0, 0, 255)
    overlay = frame.copy()
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    cv2.putText(frame, f"Battery: {percentage:.0f}%", (bar_x - 70, bar_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# Load test video
video_path = "./input/in2.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Simulated data
    drone_speed = np.random.uniform(0, 50)
    wind_speed = np.random.uniform(0, 30)
    battery_level = np.random.uniform(20, 100)
    heading = np.random.uniform(0, 360)
    altitude = np.random.uniform(0, 100)
    
    # Draw UI elements
    frame = draw_scan_lines(frame)
    frame = draw_holographic_grid(frame)
    frame = draw_cyberpunk_speed_meter(frame, (100, 400), drone_speed, 50, "Drone Speed", (255, 0, 255))
    frame = draw_cyberpunk_speed_meter(frame, (250, 400), wind_speed, 30, "Wind Speed", (0, 255, 255))
    frame = draw_compass(frame, heading)
    frame = draw_cyberpunk_battery(frame, battery_level)
    frame = draw_altitude_indicator(frame, altitude)
    
    cv2.imshow("Drone Camera: LIVE", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
