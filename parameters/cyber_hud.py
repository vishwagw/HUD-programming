import cv2
import numpy as np

def draw_cyberpunk_speed_meter(frame, center, speed, max_speed, label, color):
    """ Draws a cyberpunk-style circular speed meter with a neon glow. """
    radius = 60
    angle_range = 270  
    min_angle = -135
    max_angle = 135

    overlay = frame.copy()
    
    # Outer neon circle
    cv2.circle(overlay, center, radius, color, 2, cv2.LINE_AA)
    cv2.circle(overlay, center, radius-2, (0, 0, 0), -1)
    
    # Labels (0, mid, max speed)
    for val in [0, max_speed // 2, max_speed]:
        angle = np.deg2rad(min_angle + (val / max_speed) * angle_range)
        label_x = int(center[0] + (radius - 15) * np.cos(angle))
        label_y = int(center[1] - (radius - 15) * np.sin(angle))
        cv2.putText(frame, str(val), (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Needle
    needle_angle = np.deg2rad(min_angle + (speed / max_speed) * angle_range)
    needle_x = int(center[0] + (radius - 10) * np.cos(needle_angle))
    needle_y = int(center[1] - (radius - 10) * np.sin(needle_angle))
    cv2.line(frame, center, (needle_x, needle_y), color, 3, cv2.LINE_AA)
    
    # Digital speed display
    speed_display = f"{speed:.1f} m/s"
    cv2.putText(frame, speed_display, (center[0] - 35, center[1] + radius + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)
    
    # Apply overlay
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
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
    
    # Draw UI elements with neon colors
    frame = draw_cyberpunk_speed_meter(frame, (100, 400), drone_speed, 50, "Drone Speed", (255, 0, 255))
    frame = draw_cyberpunk_speed_meter(frame, (250, 400), wind_speed, 30, "Wind Speed", (0, 255, 255))
    frame = draw_cyberpunk_battery(frame, battery_level)
    
    cv2.imshow("Cyberpunk Drone HUD", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
