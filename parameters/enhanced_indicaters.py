# more indicaters with colored arhitecture

import cv2
import numpy as np

def draw_speed_meter(frame, center, speed, max_speed, label):
    """ Draws a circular speed meter with enhanced visuals. """
    radius = 50
    angle_range = 270  
    min_angle = -135
    max_angle = 135

    overlay = frame.copy()
    
    # Draw semi-transparent circle
    cv2.circle(overlay, center, radius, (50, 50, 50), -1)  # Dark background
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)  # Apply transparency

    # Draw labels (0, mid, max speed)
    for i, val in enumerate([0, max_speed // 2, max_speed]):
        angle = np.deg2rad(min_angle + (val / max_speed) * angle_range)
        label_x = int(center[0] + (radius - 15) * np.cos(angle))
        label_y = int(center[1] - (radius - 15) * np.sin(angle))
        cv2.putText(frame, str(val), (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Calculate needle position
    needle_angle = np.deg2rad(min_angle + (speed / max_speed) * angle_range)
    needle_x = int(center[0] + (radius - 10) * np.cos(needle_angle))
    needle_y = int(center[1] - (radius - 10) * np.sin(needle_angle))

    # Draw needle with glow effect
    cv2.line(frame, center, (needle_x, needle_y), (0, 0, 255), 3)
    cv2.line(frame, (center[0] - 1, center[1] - 1), (needle_x, needle_y), (0, 0, 100), 1)  # Shadow effect

    # Label the meter
    cv2.putText(frame, label, (center[0] - 30, center[1] + radius + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Digital speed display with a glowing effect
    speed_display = f"{speed:.1f} m/s"
    cv2.putText(frame, speed_display, (center[0] - 30, center[1] + radius + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)

    return frame

def draw_battery_indicator(frame, percentage):
    """ Draws a gradient battery indicator in the top-right corner. """
    bar_x, bar_y, bar_width, bar_height = 500, 20, 100, 20
    fill_width = int(bar_width * (percentage / 100))

    # Gradient effect
    overlay = frame.copy()
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Draw battery outline
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

    # Display percentage with glow effect
    cv2.putText(frame, f"Battery: {percentage:.0f}%", (bar_x - 50, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

def draw_altitude_display(frame, altitude):
    """ Draws an altitude digital display with a semi-transparent background. """
    overlay = frame.copy()
    cv2.rectangle(overlay, (480, 380), (600, 420), (50, 50, 50), -1)  # Dark box
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    text = f"Altitude: {altitude:.1f} m"
    cv2.putText(frame, text, (500, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
    return frame

def draw_heading_display(frame, heading):
    """ Draws a heading (yaw) display at the top-center with a modern font. """
    text = f"Heading: {heading:.0f}°"
    cv2.putText(frame, text, (250, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    return frame

# Load test video (Replace with 0 for webcam)
video_path = "./input/in2.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Simulated data (Replace with real drone data)
    drone_speed = np.random.uniform(0, 50)
    wind_speed = np.random.uniform(0, 30)
    altitude = np.random.uniform(10, 100)  
    battery_level = np.random.uniform(20, 100)  
    heading = np.random.uniform(0, 360)

    # Draw UI elements with visual enhancements
    frame = draw_speed_meter(frame, (100, 400), drone_speed, 50, "Drone Speed")
    frame = draw_speed_meter(frame, (250, 400), wind_speed, 30, "Wind Speed")
    frame = draw_battery_indicator(frame, battery_level)
    frame = draw_altitude_display(frame, altitude)
    frame = draw_heading_display(frame, heading)

    cv2.imshow("Enhanced Drone HUD", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
