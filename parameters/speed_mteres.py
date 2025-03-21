# indicaters for drone (ground) speed and wind speed:

import cv2
import numpy as np

# function for create meters:
def draw_speed_meters(frame, center, speed, max_speed, label):
    # design:
    radius = 50
    angle_range = 270  # Full scale from -135° to +135°
    min_angle = -135  # Leftmost position
    max_angle = 135  # Rightmost position

    # Draw outer circle
    cv2.circle(frame, center, radius, (255, 255, 255), 2)

    # Draw labels (0, mid, max speed)
    for i, val in enumerate([0, max_speed // 2, max_speed]):
        angle = np.deg2rad(min_angle + (val / max_speed) * angle_range)
        label_x = int(center[0] + (radius - 15) * np.cos(angle))
        label_y = int(center[1] - (radius - 15) * np.sin(angle))
        cv2.putText(frame, str(val), (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Calculate needle position
    needle_angle = np.deg2rad(min_angle + (speed / max_speed) * angle_range)
    needle_x = int(center[0] + (radius - 10) * np.cos(needle_angle))
    needle_y = int(center[1] - (radius - 10) * np.sin(needle_angle))

     # Draw needle
    cv2.line(frame, center, (needle_x, needle_y), (0, 0, 255), 2)

    # Label the meter
    cv2.putText(frame, label, (center[0] - 25, center[1] + radius + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

# Load test video (Replace with 0 for webcam)
video_path = "./input/in2.mp4"  # Change this to your video file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Simulated speeds (Replace with real drone data)
    drone_speed = np.random.randint(0, 50)  # Simulated drone speed (0-50 m/s)
    wind_speed = np.random.randint(0, 30)   # Simulated wind speed (0-30 m/s)

    # Draw speed meters in bottom left
    frame = draw_speed_meters(frame, (100, 400), drone_speed, 50, "Drone Speed")
    frame = draw_speed_meters(frame, (250, 400), wind_speed, 30, "Wind Speed")

    cv2.imshow("Drone HUD", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()