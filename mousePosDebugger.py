import time

import pyautogui

import boardManager

if __name__ == "__main__":
    while True:
        time.sleep(1)
        print(boardManager.get_drop_color())