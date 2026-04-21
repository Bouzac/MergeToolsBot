import time

import pyautogui
import keyboard

import boardManager

if __name__ == "__main__":
    while True:
        time.sleep(0.25)
        if keyboard.is_pressed("r"):
            print(f"current pose = {pyautogui.position()}")