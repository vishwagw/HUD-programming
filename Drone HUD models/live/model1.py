from dronekit import connect, VehicleMode
import time
import cv2
import numpy as np


# Connect to the drone (e.g., via telemetry, USB, or SITL)
vehicle = connect("127.0.0.1:14550", wait_ready=True)

def get_drone_telemetry():
    return {
        "altitude": vehicle.location.global_relative_frame.alt,
        "speed": vehicle.groundspeed,
        "heading": vehicle.heading,
        "mode": vehicle.mode.name
    }

def draw_hud(frame, telemetry):
    height, width, _ = frame.shape

    # Draw Altitude
    cv2.putText(frame, f"Alt: {telemetry['altitude']:.1f}m", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw Speed
    cv2.putText(frame, f"Speed: {telemetry['speed']:.1f}m/s", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw Compass Heading
    cv2.putText(frame, f"Heading: {telemetry['heading']}°", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw Flight Mode
    cv2.putText(frame, f"Mode: {telemetry['mode']}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

cap = cv2.VideoCapture(0)  # Adjust for your camera input

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    telemetry = get_drone_telemetry()
    frame = draw_hud(frame, telemetry)

    cv2.imshow("Drone HUD", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
vehicle.close()
