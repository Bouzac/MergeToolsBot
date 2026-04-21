import os
import cv2
import numpy as np
import mss
import time

import pyautogui

import boardManager

MONITOR = {"top": 0, "left": -1920, "width": 1920, "height": 1080}
BOARD_COORDINATES = {"Top": 167, "Left": 195, "Bottom": 809, "Right": 1421}
EXTRACTION_FOLDER = "extraction"

os.makedirs(EXTRACTION_FOLDER, exist_ok=True)


def isolate_board(frame):
    return frame[
        BOARD_COORDINATES["Top"]:BOARD_COORDINATES["Bottom"], BOARD_COORDINATES["Left"]:BOARD_COORDINATES["Right"]]


def main():
    # Instead of hashes, we store the actual NumPy arrays of the images we've saved
    saved_images = []

    # Tolerated difference (in pixels).
    # If less than 15 pixels are different, we consider it a duplicate.
    # You can tweak this number! Higher = less images saved. Lower = more images saved.
    MAX_PIXEL_DIFFERENCE = 15

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

            # Remplacez B, G, R par la couleur exacte de ce pixel (Rappel : OpenCV utilise le format BGR, pas RGB !)
            couleur_attendue_bgr = (255, 177, 109)  # Exemple arbitraire

            if couleur_pixel != couleur_attendue_bgr:
                # Si la couleur ne correspond pas, on n'est pas sur le tableau !
                print("En attente du tableau...")
                time.sleep(1)  # On attend un peu plus longtemps pour économiser le CPU
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

                roi_x = x + 42
                roi_y = y + 40
                roi_w = 9
                roi_h = 11

                number_crop = board_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

                gray_crop = cv2.cvtColor(number_crop, cv2.COLOR_BGR2GRAY)
                _, binarized_crop = cv2.threshold(gray_crop, 200, 255, cv2.THRESH_BINARY)

                # ==========================================
                # --- NEW: THE SANITY CHECK (GATEKEEPER) ---
                # ==========================================

                # 1. Check pixel density
                white_pixels = cv2.countNonZero(binarized_crop)
                total_pixels = roi_w * roi_h  # 24 * 11 = 264 pixels

                # If less than 10 pixels are white, it's mostly empty shadow.
                # If more than 120 pixels are white, it's covered by a bright hover effect.
                # (You may need to tweak 10 and 120 slightly based on your game's font thickness)
                if white_pixels < 10 or white_pixels > 50:
                    continue  # Reject immediately! It's garbage.

                # 2. Check structure height
                # Find the bounding box of whatever white pixels are in the image
                #contours_crop, _ = cv2.findContours(binarized_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #if not contours_crop:
                #    continue

                # ==========================================
                # --- OLD: FUZZY MATCHING (FOR DUPLICATES) ---
                # ==========================================
                is_duplicate = False

                for saved_img in saved_images:
                    diff = cv2.absdiff(binarized_crop, saved_img)
                    different_pixels = np.count_nonzero(diff)

                    if different_pixels <= 3:  # Your tolerance
                        is_duplicate = True
                        break

                if not is_duplicate:
                    filename = f"img_{int(time.time() * 1000)}.png"
                    filepath = os.path.join(EXTRACTION_FOLDER, filename)
                    cv2.imwrite(filepath, binarized_crop)
                    saved_images.append(binarized_crop)
                    print(f"Saved new unique digit: {filename} (White pixels: {white_pixels})")

            cv2.imshow("Extractor Active", cv2.resize(board_image, (400, 300)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()