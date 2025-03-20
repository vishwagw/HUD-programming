import sys
import cv2
import numpy as np
import math
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

class DroneHUD(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window setup
        self.setWindowTitle("Cyberpunk Drone HUD")
        self.setGeometry(100, 100, 800, 600)

        # Video label
        self.video_label = QLabel(self)
        self.video_label.setGeometry(0, 0, 800, 600)

        # Timer for video updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Dummy Data (Replace with actual drone data)
        self.drone_speed = 50  # km/h
        self.altitude = 120  # meters
        self.battery = 80  # %
        self.heading = 90  # degrees (Compass)

        # Start Video Capture
        vid_p = './input/in3.mp4'
        self.cap = cv2.VideoCapture(vid_p)

    def draw_hud(self, frame):
        h, w, _ = frame.shape

        # Cyberpunk Grid Overlay
        grid = np.zeros_like(frame, dtype=np.uint8)
        for i in range(0, w, 40):
            cv2.line(grid, (i, 0), (i, h), (0, 255, 0), 1)
        for j in range(0, h, 40):
            cv2.line(grid, (0, j), (w, j), (0, 255, 0), 1)
        frame = cv2.addWeighted(frame, 1, grid, 0.3, 0)

        # Draw Compass
        compass_center = (100, 100)
        cv2.circle(frame, compass_center, 50, (0, 255, 255), 2)
        angle_rad = math.radians(self.heading)
        compass_x = int(compass_center[0] + 40 * math.sin(angle_rad))
        compass_y = int(compass_center[1] - 40 * math.cos(angle_rad))
        cv2.line(frame, compass_center, (compass_x, compass_y), (0, 255, 255), 2)
        cv2.putText(frame, f"Heading: {self.heading}°", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw Speed Meter
        cv2.putText(frame, f"Speed: {self.drone_speed} km/h", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw Altitude Indicator
        cv2.putText(frame, f"Altitude: {self.altitude} m", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw Battery Status
        battery_color = (0, 255, 0) if self.battery > 50 else (0, 0, 255)
        cv2.rectangle(frame, (700, 50), (750, 100), battery_color, -1)
        cv2.putText(frame, f"Battery: {self.battery}%", (650, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, battery_color, 2)

        return frame

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (800, 600))
            frame = self.draw_hud(frame)

            # Convert frame to Qt format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.cap.release()

# Run the HUD
if __name__ == "__main__":
    app = QApplication(sys.argv)
    hud = DroneHUD()
    hud.show()
    sys.exit(app.exec_())
