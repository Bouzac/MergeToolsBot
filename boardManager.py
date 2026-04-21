import time

import pyautogui

debugPos = False

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
    pyautogui.moveTo(100, 100)

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
    pyautogui.moveTo(DROP_COORDS[0], DROP_COORDS[1])
    pyautogui.click(DROP_COORDS[0], DROP_COORDS[1])

def get_green_gift():
    pyautogui.moveTo(GGIFT_COORDS[0], GGIFT_COORDS[1])
    pyautogui.click(GGIFT_COORDS[0], GGIFT_COORDS[1])

def get_unlocked_color():
    return pyautogui.pixel(UNLOCKED_COORDS[0], UNLOCKED_COORDS[1])

def get_red_gift():
    pyautogui.moveTo(RGIFT_COORDS[0], RGIFT_COORDS[1])
    pyautogui.click(RGIFT_COORDS[0], RGIFT_COORDS[1])

def buy():
    pyautogui.moveTo(BUY_COORDS[0], BUY_COORDS[1])
    pyautogui.click(BUY_COORDS[0], BUY_COORDS[1])

def betterClick(coords):
    x, y = coords
    pyautogui.moveTo(x, y)
    pyautogui.click(x, y)

def clickTenTimes(coords):
    for r in range(10):
        betterClick(coords)

def buy_in_shop_sequence():
    betterClick((246, 215))
    betterClick((1378, 204))
    clickTenTimes((811, 353))
    clickTenTimes((800, 470))
    clickTenTimes((799, 588))
    clickTenTimes((802, 695))
    betterClick((668, 209))
    betterClick((230, 271))
    clickTenTimes((923, 464))
    betterClick((658, 209))

if __name__ == "__main__":
    if debugPos:
        while True:
            time.sleep(0.5)
            print(pyautogui.position())
    time.sleep(3)
    buy_in_shop_sequence()