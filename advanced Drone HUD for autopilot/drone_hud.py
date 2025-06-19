import pygame
import platform
import asyncio
import random
from pygame import Vector2

# Initialize Pygame
def setup():
    global screen, font, width, height
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Drone HUD Interface")
    font = pygame.font.SysFont("monospace", 20)

# Draw HUD elements
def draw_hud():
    # Simulated camera feed (placeholder: colored background)
    screen.fill((50, 50, 100))  # Blue-gray placeholder for video feed

    # Telemetry data
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
    center = Vector2(width / 2, height / 2)
    pygame.draw.circle(screen, (0, 255, 0), center, 40, 2)
    pygame.draw.line(screen, (0, 255, 0), (center.x - 10, center.y), (center.x + 10, center.y), 2)
    pygame.draw.line(screen, (0, 255, 0), (center.x, center.y - 10), (center.x, center.y + 10), 2)

# Update game loop
def update_loop():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return
    draw_hud()
    pygame.display.flip()

# Main loop for Pyodide compatibility
async def main():
    setup()
    while True:
        update_loop()
        await asyncio.sleep(1.0 / 60)  # 60 FPS

# Run based on platform
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())