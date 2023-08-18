"""Guard against tampering by watching for mouse movement"""

from tkinter import *
import time
import os

COMMAND = "google-chrome https://www.youtube.com/embed/4YnetcRAdpw?autoplay=1&"
activated = False


def activate():
    """Activation function for when mouse movement is detected"""
    global activated
    if not activated:
        # activated = True
        os.system(COMMAND)


root = Tk()


def get_position():
    return (root.winfo_pointerx(), root.winfo_pointery())


print("Activating in 3s")
time.sleep(3)

print("Monitoring mouse movement now")

# main watching loop
while True:
    current_pos = get_position()
    time.sleep(0.5)
    if current_pos != get_position():
        print("Mouse movement detected")
        activate()
        current_pos = get_position()
        time.sleep(5)
