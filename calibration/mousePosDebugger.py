import time

import pyautogui
import keyboard

import boardManager

if __name__ == "__main__":
    while True:
        time.sleep(0.1)
        if keyboard.is_pressed("r"):
            print("---------------------------------------------")
            print(f"drop_color = {boardManager.get_drop_color()}")
            print(f"gift_green = {boardManager.get_green_gift_color()}")
            print(f"gift_red = {boardManager.get_red_gift_color()}")
            print(f"cant_buy_color = {boardManager.get_buy_color()}")
            print(f"next Color: {boardManager.get_next_button_color()}")
            print(f"unlocked_color = {boardManager.get_unlocked_color()}")
            print(f"Current Pose: {pyautogui.position()}")
            print("---------------------------------------------")