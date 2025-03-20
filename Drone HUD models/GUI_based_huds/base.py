import tkinter as tk
from tkinter import ttk
import random

class DroneHUD(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Drone HUD")
        self.geometry("400x400")

        self.compass_label = ttk.Label(self, text="Compass: N")
        self.compass_label.pack(pady=10)

        self.speed_label = ttk.Label(self, text="Speed: 0 m/s")
        self.speed_label.pack(pady=10)
        
        self.altitude_label = ttk.Label(self, text="Altitude: 0 m")
        self.altitude_label.pack(pady=10)

        self.battery_label = ttk.Label(self, text="Battery: 100%")
        self.battery_label.pack(pady=10)

        self.update_hud()

    def update_hud(self):
        compass_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        self.compass_label.config(text=f"Compass: {random.choice(compass_directions)}")

        speed = random.randint(0, 100)
        self.speed_label.config(text=f"Speed: {speed} m/s")

        altitude = random.randint(0, 500)
        self.altitude_label.config(text=f"Altitude: {altitude} m")

        battery = random.randint(0, 100)
        self.battery_label.config(text=f"Battery: {battery}%")

        self.after(1000, self.update_hud)

if __name__ == "__main__":
    app = DroneHUD()
    app.mainloop()