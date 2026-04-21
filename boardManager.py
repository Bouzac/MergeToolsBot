import time

import pyautogui

debugPos = False

TRASH_COORDS = (-1183,350)
DROP_COORDS = (-1111,740)
NEXT_COORDS = (-1111,580)
BUY_COORDS = (-1111,700)
GGIFT_COORDS = (-1253,683)
RGIFT_COORDS = (-990,683)

BACK_COORDS = (-1670,219)

SHOP_COORDS = (-1686, 282)
SHOP_RGIFT_COORDS = (-999,472)

SHOP_ACHIEVEMENTS_BACK_COORDS = (-1257,210)

ACHIEVEMENTS_COORDS = (-537,214)
ACHIEVEMENT_1_COORDS = (-1106,359)
ACHIEVEMENT_2_COORDS = (-1106,478)
ACHIEVEMENT_3_COORDS = (-1106,597)
ACHIEVEMENT_4_COORDS = (-1106,707)

START_COORDS = (-1116,727)

def move_tool(initial, final):
    initial_x, initial_y = initial
    final_x, final_y = final

    pyautogui.mouseDown(initial_x, initial_y)

    pyautogui.moveTo(final_x, final_y)

    pyautogui.mouseUp(final_x, final_y)
    time.sleep(0.05)

def get_drop_color():
    return pyautogui.pixel(DROP_COORDS[0], DROP_COORDS[1])

def get_buy_color():
    return pyautogui.pixel(BUY_COORDS[0], BUY_COORDS[1])

def get_red_gift_color():
    return pyautogui.pixel(RGIFT_COORDS[0], RGIFT_COORDS[1])

def go_outside_board():
    pyautogui.moveTo(1, 1)

def get_green_gift_color():
    return pyautogui.pixel(GGIFT_COORDS[0], GGIFT_COORDS[1])

def get_next_button_color():
    return pyautogui.pixel(NEXT_COORDS[0], NEXT_COORDS[1])

def click_next_button():
    pyautogui.click(NEXT_COORDS[0], NEXT_COORDS[1])

def restart():
    pyautogui.click(NEXT_COORDS[0], NEXT_COORDS[1])

def drop():
    pyautogui.click(DROP_COORDS[0], DROP_COORDS[1])

def get_green_gift():
    pyautogui.click(GGIFT_COORDS[0], GGIFT_COORDS[1])

def get_unlocked_color():
    return pyautogui.pixel(NEXT_COORDS[0], NEXT_COORDS[1])

def get_red_gift():
    pyautogui.click(RGIFT_COORDS[0], RGIFT_COORDS[1])

def buy():
    pyautogui.click(BUY_COORDS[0], BUY_COORDS[1])

if __name__ == "__main__":
    if debugPos:
        while True:
            time.sleep(0.5)
            print(pyautogui.position())

    get_green_gift()