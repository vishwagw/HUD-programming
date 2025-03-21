import cv2
import numpy as np


def draw_compass(frame, heading):
    """ Draws a compass in the top-left corner of the frame. """
    center = (100, 100)  # Top-left position
    radius = 50

    # Draw compass circle
    cv2.circle(frame, center, radius, (255, 255, 255), 2)

    # Direction labels
    directions = {'N': (100, 50), 'E': (150, 100), 'S': (100, 150), 'W': (50, 100)}
    for text, pos in directions.items():
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Convert heading to radians (adjusted for compass orientation)
    angle = np.deg2rad(-heading + 90)

    # Calculate arrow end point
    arrow_length = 40
    end_x = int(center[0] + arrow_length * np.cos(angle))
    end_y = int(center[1] - arrow_length * np.sin(angle))

    # Draw arrow
    cv2.arrowedLine(frame, center, (end_x, end_y), (0, 0, 255), 2, tipLength=0.2)

    return frame

# Load test video (Replace with 0 for webcam)
video_path = "./input/in3.mp4"  # Change this to your video file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    heading = np.random.randint(0, 360)  # Simulated heading (Replace with real drone data)
    frame = draw_compass(frame, heading)

    cv2.imshow("Drone HUD", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
