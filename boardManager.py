import time

import cv2
import mss
import pyautogui
import numpy as np

from boardHelper import isolate_board, read_number_with_pytorch, cnn_model

pyautogui.PAUSE = 0.02

debugPos = False

MONITOR = {"top": 0, "left": 0, "width": 1920, "height": 1080}
BOARD_COORDINATES = {"Top": 176, "Left": 196, "Bottom": 845, "Right": 1411}

DROP_COORDS = (791, 737)
NEXT_COORDS = (806, 564)
BUY_COORDS = (780, 694)
GGIFT_COORDS = (673, 684)
RGIFT_COORDS = (942, 670)
START_COORDS = (800, 711)
UNLOCKED_COORDS = (803, 605)



def move_tool(initial, final):
    initial_x, initial_y = initial
    final_x, final_y = final

    pyautogui.moveTo(initial_x, initial_y)

    pyautogui.mouseDown(initial_x, initial_y)

    pyautogui.moveTo(final_x, final_y)

    pyautogui.mouseUp(final_x, final_y)

def click_unlocked_button():
    pyautogui.moveTo(UNLOCKED_COORDS[0], UNLOCKED_COORDS[1])
    pyautogui.click(UNLOCKED_COORDS[0], UNLOCKED_COORDS[1])

def get_drop_color():
    return pyautogui.pixel(DROP_COORDS[0], DROP_COORDS[1])

def get_buy_color():
    return pyautogui.pixel(BUY_COORDS[0], BUY_COORDS[1])

def get_red_gift_color():
    return pyautogui.pixel(RGIFT_COORDS[0], RGIFT_COORDS[1])

def go_outside_board():
    pyautogui.moveTo(400, 300)

def get_green_gift_color():
    return pyautogui.pixel(GGIFT_COORDS[0], GGIFT_COORDS[1])

def get_next_button_color():
    return pyautogui.pixel(NEXT_COORDS[0], NEXT_COORDS[1])

def click_next_button():
    pyautogui.moveTo(NEXT_COORDS[0], NEXT_COORDS[1])
    pyautogui.click(NEXT_COORDS[0], NEXT_COORDS[1])

def restart():
    pyautogui.moveTo(NEXT_COORDS[0], NEXT_COORDS[1])
    pyautogui.click(NEXT_COORDS[0], NEXT_COORDS[1])

def drop():
    betterClick(DROP_COORDS)

def get_green_gift():
    x, y = GGIFT_COORDS
    betterClick((x + 10, y + 10))

def get_unlocked_color():
    return pyautogui.pixel(UNLOCKED_COORDS[0], UNLOCKED_COORDS[1])

def get_red_gift():
    x, y = RGIFT_COORDS
    betterClick((x + 10, y + 10))

def buy():
    pyautogui.moveTo(BUY_COORDS[0], BUY_COORDS[1])
    pyautogui.click(BUY_COORDS[0], BUY_COORDS[1])

def betterClick(coords):
    x, y = coords
    pyautogui.moveTo(x, y)
    pyautogui.click(x, y)

def clickAchievements(coords):
    x, y = coords
    pyautogui.moveTo(x + 10, y)

    while True:
        time.sleep(0.1)
        if pyautogui.pixel(x, y) == (245, 245, 245) or pyautogui.pixel(x, y) == (245, 226, 0):
            break

        pyautogui.click(x + 5, y)

def bulkBuyRGift():
    with mss.mss() as sct:
        time.sleep(1)
        screenshot = sct.grab(MONITOR)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        roi_x = 908
        roi_y = 290
        roi_w = 36
        roi_h = 11

        gems_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        gems_number = read_number_with_pytorch(gems_frame, cnn_model)

        gifts_to_buy = int(gems_number / 50)

        print(gifts_to_buy)

        pyautogui.moveTo(923, 464)

        for _ in range (gifts_to_buy):
            pyautogui.click()

def buy_in_shop_sequence():
    betterClick((246, 215))
    betterClick((1378, 204))
    clickAchievements((811, 345))
    clickAchievements((800, 461))
    clickAchievements((799, 620))
    clickAchievements((802, 695))
    betterClick((668, 209))
    time.sleep(0.1)
    betterClick((230, 271))
    pyautogui.click(230, 271)
    bulkBuyRGift()
    betterClick((658, 209))
    pyautogui.PAUSE = 0.1
    drop()
    pyautogui.PAUSE = 0.02

if __name__ == "__main__":
    if debugPos:
        while True:
            time.sleep(0.5)
            print(pyautogui.position())

    buy_in_shop_sequence()