import cv2
import numpy as np
import mss
import time
import math

points = {
    "DROP": [750, 818],
    "NEXT": [869, 580],
    "BUY": [732, 719],
    "GGIFT": [634, 721],
    "RGIFT": [902, 710],
    "START": [864, 727]
}

MONITOR = {"top": 0, "left": 0, "width": 1920, "height": 1080}

dragging_point = None
current_left = 0
current_top = 0

def nothing(x):
    pass

def mouse_events(event, x, y, flags, param):
    global dragging_point, points, current_left, current_top

    # Coordonnées absolues basées sur le décalage du crop
    abs_x = x + current_left
    abs_y = y + current_top

    if event == cv2.EVENT_LBUTTONDOWN:
        for name, coord in points.items():
            dist = math.hypot(abs_x - coord[0], abs_y - coord[1])
            if dist < 15:
                dragging_point = name
                break

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_point is not None:
            points[dragging_point] = [abs_x, abs_y]

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point = None

def main():
    global current_left, current_top, points

    print("Mets-toi sur ton jeu ! Capture de l'écran dans...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("Capture ! Tu peux maintenant calibrer.")

    # On prend UNE SEULE capture statique
    with mss.mss() as sct:
        screenshot = sct.grab(MONITOR)
        original_frame = np.array(screenshot)
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGRA2BGR)

    # WINDOW_NORMAL permet de redimensionner les fenêtres si elles prennent trop de place
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL) 
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL) 
    cv2.setMouseCallback("Calibration", mouse_events)

    cv2.createTrackbar("Top", "Trackbars", 0, 1080, nothing)
    cv2.createTrackbar("Left", "Trackbars", 0, 1920, nothing)
    cv2.createTrackbar("Bottom", "Trackbars", 1080, 1080, nothing)
    cv2.createTrackbar("Right", "Trackbars", 1920, 1920, nothing)

    while True:
        # On travaille sur une copie de la photo statique pour ne pas superposer les dessins
        frame = original_frame.copy()

        current_top = cv2.getTrackbarPos("Top", "Trackbars")
        current_left = cv2.getTrackbarPos("Left", "Trackbars")
        bot = cv2.getTrackbarPos("Bottom", "Trackbars")
        right = cv2.getTrackbarPos("Right", "Trackbars")

        # Dessiner les points
        for name, coord in points.items():
            cv2.circle(frame, tuple(coord), 5, (0, 0, 255), -1)
            cv2.putText(frame, name, (coord[0] - 20, coord[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if current_top < bot and current_left < right:
            croped_frame = frame[current_top:bot, current_left:right]
            cv2.imshow("Calibration", croped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n--- Infos de Crop ---")
            print(f"Crop final : top={current_top}, left={current_left}, bottom={bot}, right={right}")
            
            print("\n--- Nouvelles coordonnées à copier-coller ---")
            for name, coord in points.items():
                print(f"{name}_COORDS = ({coord[0]}, {coord[1]})")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()