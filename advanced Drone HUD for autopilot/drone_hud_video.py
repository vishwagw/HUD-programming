import pygame
import cv2
import numpy as np
import random

# Initialize Pygame
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Drone HUD Interface")
font = pygame.font.SysFont("monospace", 20)
clock = pygame.time.Clock()

# Load video file
video_path = "test1.mp4"  # Replace with your video file path
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print("Error: Could not open video file.")
    exit()

# Convert OpenCV frame to Pygame surface
def frame_to_surface(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame = np.transpose(frame, (1, 0, 2))  # Rotate if needed
    return pygame.surfarray.make_surface(frame)

# Draw HUD elements
def draw_hud(frame):
    # Draw video frame
    if frame is not None:
        surface = frame_to_surface(frame)
        surface = pygame.transform.scale(surface, (width, height))
        screen.blit(surface, (0, 0))
    else:
        screen.fill((50, 50, 100))  # Telemetry data
    altitude = random.uniform(0, 100)
    speed = random.uniform(0, 50)
    battery = random.uniform(0, 100)
    gps = (random.uniform(-90, 90), random.uniform(-180, 180))
    roll = random.uniform(0, 360)
    pitch = random.uniform(-90, 90)
    yaw = random.uniform(0, 360)

    # Draw telemetry overlay (top-left)
    telemetry_bg = pygame.Surface((200, 120), pygame.SRCALPHA)
    telemetry_bg.fill((0, 0, 0, 128))  # Semi-transparent black
    screen.blit(telemetry_bg, (10, 10))
    
    telemetry_texts = [
        f"Altitude: {altitude:.1f} m",
        f"Speed: {speed:.1f} km/h",
        f"Battery: {battery:.0f}%",
        f"GPS: {gps[0]:.3f}, {gps[1]:.3f}"
    ]
    for i, text in enumerate(telemetry_texts):
        text_surface = font.render(text, True, (0, 255, 0))
        screen.blit(text_surface, (20, 20 + i * 25))

    # Draw orientation overlay (top-right)
    orientation_bg = pygame.Surface((200, 100), pygame.SRCALPHA)
    orientation_bg.fill((0, 0, 0, 128))
    screen.blit(orientation_bg, (width - 210, 10))
    
    orientation_texts = [
        f"Roll: {roll:.1f}°",
        f"Pitch: {pitch:.1f}°",
        f"Yaw: {yaw:.1f}°"
    ]
    for i, text in enumerate(orientation_texts):
        text_surface = font.render(text, True, (0, 255, 0))
        screen.blit(text_surface, (width - 200, 20 + i * 25))

    # Draw crosshair
    center = (width // 2, height // 2)
    pygame.draw.circle(screen, (0, 255, 0), center, 40, 2)
    pygame.draw.line(screen, (0, 255, 0), (center[0] - 10, center[1]), (center[0] + 10, center[1]), 2)
    pygame.draw.line(screen, (0, 255, 0), (center[0], center[1] - 10), (center[0], center[1] + 10), 2)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Read video frame
    ret, frame = video_capture.read()
    if not ret:
        # Loop video when it ends
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read video frame.")
            break

    # Draw HUD
    draw_hud(frame)
    pygame.display.flip()
    clock.tick(30)  # 30 FPS

# Cleanup
video_capture.release()
pygame.quit()