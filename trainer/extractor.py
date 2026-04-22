import os
import cv2
import numpy as np
import mss
import time

import pyautogui

import boardManager

from boardHelper import split_into_digits

MONITOR = {"top": 0, "left": 0, "width": 1920, "height": 1080}
BOARD_COORDINATES = {"Top": 176, "Left": 196, "Bottom": 845, "Right": 1411}
EXTRACTION_FOLDER = r"D:\code project\MergeToolsBot\extraction"
DATASET_FOLDER = r"D:\code project\MergeToolsBot\dataset"

os.makedirs(EXTRACTION_FOLDER, exist_ok=True)

def isolate_board(frame):
    return frame[
        BOARD_COORDINATES["Top"]:BOARD_COORDINATES["Bottom"], BOARD_COORDINATES["Left"]:BOARD_COORDINATES["Right"]]


def main():
    saved_images = []
    MAX_PIXEL_DIFFERENCE = 2

    for filename in os.listdir(EXTRACTION_FOLDER):
        img = cv2.imread(os.path.join(EXTRACTION_FOLDER, filename), cv2.IMREAD_UNCHANGED)
        saved_images.append(img)

    for i in range(10):
        for filename in os.listdir(os.path.join(DATASET_FOLDER, str(i))):
            img = cv2.imread(os.path.join(DATASET_FOLDER, str(i), filename), cv2.IMREAD_UNCHANGED)
            saved_images.append(img)

    print(f"[*] Extractor running. Images will be saved to '{EXTRACTION_FOLDER}/'.")
    print(f"[*] Tolerance set to {MAX_PIXEL_DIFFERENCE} pixels. Press 'q' to stop.")

    with mss.mss() as sct:
        while True:
            time.sleep(0.5)

            screenshot = sct.grab(MONITOR)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            pixel_y = boardManager.DROP_COORDS[0]
            pixel_x = boardManager.DROP_COORDS[1]

            couleur_pixel = boardManager.get_drop_color()

            couleur_attendue_bgr = (255, 171, 98)

            if couleur_pixel != couleur_attendue_bgr:
                print("En attente du tableau...")
                time.sleep(1)
                continue

            board_image = isolate_board(frame)

            gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cells = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if 1000 < w * h < 15000:
                    cells.append({'x': x, 'y': y, 'w': w, 'h': h})

            for cell in cells:
                x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']

                roi_x = x + 39
                roi_y = y + 41
                roi_w = 12
                roi_h = 11

                number_crop = board_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

                gray_crop = cv2.cvtColor(number_crop, cv2.COLOR_BGR2GRAY)
                _, binarized_crop = cv2.threshold(gray_crop, 230, 255, cv2.THRESH_BINARY)

                # --- 1. SÉPARER LES CHIFFRES ICI ---
                individual_digits = split_into_digits(binarized_crop)

                # --- 2. TRAITER CHAQUE CHIFFRE SÉPARÉMENT ---
                for digit_img in individual_digits:
                    
                    white_pixels = cv2.countNonZero(digit_img)

                    # J'ai ajusté la limite basse à 5 au lieu de 10 car 
                    # certains chiffres fins (comme le 1) auront très peu de pixels.
                    if white_pixels < 5 or white_pixels > 50:
                        continue  # Rejeté, c'est du bruit

                    is_duplicate = False

                    for saved_img in saved_images:
                        # TRÈS IMPORTANT : On ne peut comparer que des images de la même taille !
                        # Comme on a découpé, un "1" n'aura pas la même largeur qu'un "8".
                        if saved_img.shape == digit_img.shape:
                            diff = cv2.absdiff(digit_img, saved_img)
                            different_pixels = np.count_nonzero(diff)

                            if different_pixels <= MAX_PIXEL_DIFFERENCE:
                                is_duplicate = True
                                break

                    if not is_duplicate:
                        filename = f"img_{int(time.time() * 1000)}.png"
                        filepath = os.path.join(EXTRACTION_FOLDER, filename)
                        cv2.imwrite(filepath, digit_img)
                        saved_images.append(digit_img)
                        print(f"Saved new unique digit: {filename} (Shape: {digit_img.shape}, White px: {white_pixels})")

            cv2.imshow("Extractor Active", cv2.resize(board_image, (400, 300)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()