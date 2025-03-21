#adding a compass to drone HUD screen:

import cv2
import numpy as np

# function for draw compass:
def draw_compass(frame, heading):
    # position and size:
    center = (100, 100) # top left corner
    radius = 50

    # Draw compass circle
    cv2.circle(frame, center, radius, (255, 255, 255), 2)

    # Draw direction labels
    directions = {'N': (100, 50), 'E': (150, 100), 'S': (100, 150), 'W': (50, 100)}
    for text, pos in directions.items():
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Convert heading to radians
    angle = np.deg2rad(-heading + 90)  # Adjust to match compass orientation

    # Calculate arrow end point
    arrow_length = 40
    end_x = int(center[0] + arrow_length * np.cos(angle))
    end_y = int(center[1] - arrow_length * np.sin(angle))

    # Draw arrow
    cv2.arrowedLine(frame, center, (end_x, end_y), (0, 0, 255), 2, tipLength=0.2)

    return frame

# Example usage
frame = np.zeros((300, 300, 3), dtype=np.uint8)
heading = 45  # Example heading (change dynamically in real implementation)
frame = draw_compass(frame, heading)

cv2.imshow("Compass HUD", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()