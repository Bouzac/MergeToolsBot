import time
import cv2
import keyboard
import mss
import pyautogui
import numpy as np

from boardHelper import read_number_with_pytorch, cnn_model, MONITOR, MONITOR_SIZE_FACTOR_X, \
    MONITOR_SIZE_FACTOR_Y

pyautogui.PAUSE = 0.02
debugPos = True


# ==========================================
# --- HELPER FUNCTIONS POUR LE SCALING ---
# ==========================================
def s_coord(x, y):
    """ Met à l'échelle un tuple (x, y) et force des entiers (pixels) """
    return (int(x * MONITOR_SIZE_FACTOR_X) - 1366, int(y * MONITOR_SIZE_FACTOR_Y))

# ==========================================
# --- COORDONNÉES GLOBALES ---
# ==========================================
# L'utilisation de s_coord() rend le code beaucoup plus propre et sûr
DROP_COORDS = s_coord(922, 968)
NEXT_COORDS = s_coord(916, 617)
BUY_COORDS = s_coord(917, 840)
GGIFT_COORDS = s_coord(720, 864)
RGIFT_COORDS = s_coord(1200, 859)
START_COORDS = s_coord(916, 897)
UNLOCKED_COORDS = s_coord(915, 662)


def move_tool(initial, final):
    initial_x, initial_y = initial
    final_x, final_y = final

    pyautogui.moveTo(initial_x, initial_y)
    pyautogui.mouseDown(initial_x, initial_y)
    pyautogui.moveTo(final_x, final_y)
    pyautogui.mouseUp(final_x, final_y)


def click_unlocked_button():
    betterClick(UNLOCKED_COORDS)


def get_drop_color():
    return pyautogui.pixel(DROP_COORDS[0], DROP_COORDS[1])


def get_buy_color():
    return pyautogui.pixel(BUY_COORDS[0], BUY_COORDS[1])


def get_red_gift_color():
    return pyautogui.pixel(RGIFT_COORDS[0], RGIFT_COORDS[1])


def go_outside_board():
    betterClick(s_coord(400, 300))  # Ajout du scale ici


def get_green_gift_color():
    return pyautogui.pixel(GGIFT_COORDS[0], GGIFT_COORDS[1])


def get_next_button_color():
    return pyautogui.pixel(NEXT_COORDS[0], NEXT_COORDS[1])


def click_next_button():
    betterClick(NEXT_COORDS)


def restart():
    betterClick(NEXT_COORDS)


def drop():
    betterClick(DROP_COORDS)


def get_green_gift():
    x, y = GGIFT_COORDS
    # Attention au décalage : le +10 doit aussi être mis à l'échelle idéalement,
    # mais si c'est juste un petit offset de sécurité, ça peut rester +10.
    betterClick((x + 10, y + 10))


def get_unlocked_color():
    return pyautogui.pixel(UNLOCKED_COORDS[0], UNLOCKED_COORDS[1])


def get_red_gift():
    x, y = RGIFT_COORDS
    betterClick((x + 10, y + 10))


def buy():
    betterClick(BUY_COORDS)


def betterClick(coords):
    x, y = coords
    pyautogui.moveTo(x, y)
    pyautogui.click(x, y)


def clickAchievements(coords):
    x, y = coords

    pyautogui.moveTo(x + 10, y)

    while True:
        time.sleep(0.1)
        current_pixel = pyautogui.pixel(x, y)
        if current_pixel == (245, 245, 245) or current_pixel == (245, 226, 0):
            break

        pyautogui.click(x + 10, y)


def bulkBuyRGift():
    with mss.mss() as sct:
        time.sleep(1)
        screenshot = sct.grab(MONITOR)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        cv2.imshow("frame", frame)

        cv2.waitKey(100)

        roi_x, roi_y = s_coord(1130, 229)
        roi_w, roi_h = 40, 11

        gems_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        gems_number = read_number_with_pytorch(gems_frame, cnn_model)

        gifts_to_buy = int(gems_number / 50) if gems_number >= 50 else 0

        print(f"Gifts to buy: {gifts_to_buy}")

        if gifts_to_buy > 0:
            pyautogui.moveTo(s_coord(1107, 397))  # Ajout du scale ici
            for _ in range(gifts_to_buy):
                pyautogui.click()


def buy_in_shop_sequence():
    betterClick(s_coord(80, 98))
    time.sleep(0.1)
    betterClick(s_coord(1860, 84))

    clickAchievements(s_coord(930, 323))
    clickAchievements(s_coord(924, 514))
    clickAchievements(s_coord(924, 705))
    clickAchievements(s_coord(924, 897))

    betterClick(s_coord(700, 81))

    time.sleep(0.5)

    betterClick(s_coord(80, 194))

    bulkBuyRGift()

    betterClick(s_coord(690, 77))

    time.sleep(0.5)

    pyautogui.PAUSE = 0.1
    drop()
    pyautogui.PAUSE = 0.02


if __name__ == "__main__":
    if debugPos:
        while True:
            time.sleep(0.1)
            if keyboard.is_pressed('r'):
                x, y = pyautogui.position()
                base_x = int((x + 1336) / MONITOR_SIZE_FACTOR_X)
                base_y = int(y / MONITOR_SIZE_FACTOR_Y)
                print(f"Current: ({x}, {y}) | Base (1.0x): ({base_x}, {base_y})")

    bulkBuyRGift()