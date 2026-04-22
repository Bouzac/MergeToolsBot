import cv2
import numpy as np
import mss
import time

MONITOR = {"top": 0, "left": 0, "width": 1920, "height": 1080}

# Point de départ par défaut (l'ancre / le coin de ta case)
anchor_x, anchor_y = 960, 540 

def mouse_events(event, x, y, flags, param):
    global anchor_x, anchor_y
    # Un clic gauche définit le coin supérieur gauche de la case
    if event == cv2.EVENT_LBUTTONDOWN:
        anchor_x, anchor_y = x, y

def nothing(x):
    pass

def main():
    print("Mets-toi sur ton jeu ! Capture dans...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("Capture ! Tu peux calibrer ta case.")

    with mss.mss() as sct:
        screenshot = sct.grab(MONITOR)
        original_frame = np.array(screenshot)
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGRA2BGR)

    cv2.namedWindow("Calibration Case (Ecran)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration Case (Ecran)", mouse_events)
    cv2.namedWindow("ZOOM (Le chiffre a lire)", cv2.WINDOW_NORMAL)

    # Trackbars pour tes 4 variables
    # On met tes valeurs actuelles (32, 40, 21, 11) par défaut
    cv2.createTrackbar("roi_x (Offset Gauche)", "Calibration Case (Ecran)", 32, 100, nothing)
    cv2.createTrackbar("roi_y (Offset Haut)", "Calibration Case (Ecran)", 40, 100, nothing)
    cv2.createTrackbar("roi_w (Largeur)", "Calibration Case (Ecran)", 21, 100, nothing)
    cv2.createTrackbar("roi_h (Hauteur)", "Calibration Case (Ecran)", 11, 100, nothing)

    while True:
        frame = original_frame.copy()

        roi_x = cv2.getTrackbarPos("roi_x (Offset Gauche)", "Calibration Case (Ecran)")
        roi_y = cv2.getTrackbarPos("roi_y (Offset Haut)", "Calibration Case (Ecran)")
        roi_w = cv2.getTrackbarPos("roi_w (Largeur)", "Calibration Case (Ecran)")
        roi_h = cv2.getTrackbarPos("roi_h (Hauteur)", "Calibration Case (Ecran)")

        # Sécurité pour éviter les largeurs/hauteurs à zéro
        roi_w = max(1, roi_w)
        roi_h = max(1, roi_h)

        # 1. Dessiner le point d'ancrage (Le coin de la case détectée par ton contour)
        cv2.circle(frame, (anchor_x, anchor_y), 3, (255, 0, 0), -1)
        cv2.putText(frame, "Coin de la case (Clic Gauche)", (anchor_x - 50, anchor_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 2. Calculer les coordonnées du crop du chiffre
        crop_x1 = anchor_x + roi_x
        crop_y1 = anchor_y + roi_y
        crop_x2 = crop_x1 + roi_w
        crop_y2 = crop_y1 + roi_h

        # 3. Dessiner le rectangle autour du chiffre
        cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 2)
        cv2.imshow("Calibration Case (Ecran)", frame)

        # 4. Afficher le zoom
        # On s'assure que le rectangle ne sort pas de l'écran pour éviter un crash
        if 0 <= crop_y1 < crop_y2 <= frame.shape[0] and 0 <= crop_x1 < crop_x2 <= frame.shape[1]:
            croped_frame = original_frame[crop_y1:crop_y2, crop_x1:crop_x2]
            # Zoom x8 pour bien voir les pixels
            zoomed_frame = cv2.resize(croped_frame, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
            cv2.imshow("ZOOM (Le chiffre a lire)", zoomed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n================================================")
            print("--- OFFSETS DE LA CASE PRÊTS ---")
            print("Remplace ces lignes dans ta fonction get_grid_matrix :\n")
            print(f"roi_x = {anchor_x + roi_x}")
            print(f"roi_y = {anchor_y + roi_y}")
            print(f"roi_w = {roi_w}")
            print(f"roi_h = {roi_h}")
            print("================================================\n")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()