from typing import List

import cv2
import numpy as np
import torch

from trainer.train import ChiffreCNN

MONITOR_SIZE_FACTOR_X = 1366 / 1920
MONITOR_SIZE_FACTOR_Y = 768 / 1080

MONITOR = {"top": 0, "left": -1366, "width": int(1920 * MONITOR_SIZE_FACTOR_X), "height": int(1080 * MONITOR_SIZE_FACTOR_Y)}
BOARD_COORDINATES = {"Top": 0, "Left": 0, "Bottom": int(1920 * MONITOR_SIZE_FACTOR_X), "Right": int(1080 * MONITOR_SIZE_FACTOR_Y)}

try:
    device = torch.device("cpu")
    cnn_model = ChiffreCNN().to(device)
    cnn_model.load_state_dict(torch.load(r"modele_chiffres.pth", map_location=device, weights_only=True))
    cnn_model.eval()
except FileNotFoundError:
    print("Erreur du chargement du model")

def isolate_board(frame):
    return frame[
        BOARD_COORDINATES["Top"]:BOARD_COORDINATES["Bottom"], BOARD_COORDINATES["Left"]:BOARD_COORDINATES["Right"]]

def split_into_digits(binarized_img: np.ndarray, max_digit_width: int = 9) -> List[np.ndarray]:
    """
    Extrait les chiffres d'une image binarisée.
    Optimisé en performance et gère automatiquement les chiffres collés.
    """
    contours, _ = cv2.findContours(binarized_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filtre anti-bruit (on ignore les particules de moins de 10 pixels de surface)
        if w * h >= 10:
            bounding_boxes.append((x, y, w, h, c))

    # Trier de gauche à droite
    bounding_boxes.sort(key=lambda b: b[0])

    digits = []

    for x, y, w, h, c in bounding_boxes:

        # --- OPTIMISATION ---
        # On découpe TOUT DE SUITE la petite zone. On ne travaille plus sur l'image entière.
        roi_crop = binarized_img[y:y + h, x:x + w]

        # On crée un mini-masque de la taille du chiffre
        mask = np.zeros_like(roi_crop)

        # Astuce de pro : On décale les coordonnées du contour pour qu'il
        # rentre parfaitement dans notre mini-masque (où x=0 et y=0)
        c_shifted = c - [x, y]
        cv2.drawContours(mask, [c_shifted], -1, 255, thickness=cv2.FILLED)

        # Application du masque sur la petite zone
        isolated_digit = cv2.bitwise_and(roi_crop, mask)

        # --- GESTION DES CHIFFRES COLLÉS ---
        # Si la largeur du bloc est suspecte (ex: > 9 pixels), on le casse en deux !
        if w > max_digit_width:
            # On compte les pixels blancs dans chaque colonne
            colonnes_blanches = np.sum(isolated_digit == 255, axis=0)

            # On cherche la "vallée" de pixels noirs au centre du bloc (entre 30% et 70%)
            debut_milieu = int(w * 0.3)
            fin_milieu = int(w * 0.7)
            zone_recherche = colonnes_blanches[debut_milieu:fin_milieu]

            if len(zone_recherche) > 0:
                # On trouve la colonne avec le moins de pixels blancs (le point de fusion)
                cut_x = debut_milieu + np.argmin(zone_recherche)

                # Coupure au scalpel !
                digit_1 = isolated_digit[:, :cut_x]
                digit_2 = isolated_digit[:, cut_x:]

                # On ajoute les deux moitiés si elles ne sont pas vides
                if cv2.countNonZero(digit_1) > 3: digits.append(digit_1)
                if cv2.countNonZero(digit_2) > 3: digits.append(digit_2)
                continue  # On passe au contour suivant

        # Si la largeur est normale, on ajoute simplement le chiffre
        digits.append(isolated_digit)

    return digits

def pad_to_target_size(img, target_h=11, target_w=9):
    h, w = img.shape[:2]
    if h > target_h or w > target_w:
        img = cv2.resize(img, (min(w, target_w), min(h, target_h)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    top = (target_h - h) // 2
    bottom = target_h - h - top
    left = (target_w - w) // 2
    right = target_w - w - left

    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded_img

def read_number_with_pytorch(roi_image, model):
    gray_crop = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    _, binarized_crop = cv2.threshold(gray_crop, 230, 255, cv2.THRESH_BINARY)

    digits = split_into_digits(binarized_crop)

    if not digits:
        return 0

    final_number_str = ""

    for digit_img in digits:
        if cv2.countNonZero(digit_img) < 5:
            continue

        padded_digit = pad_to_target_size(digit_img, 11, 9)

        img_array = padded_digit.astype(np.float32) / 255.0

        tensor_img = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor_img)

        _, predicted = torch.max(outputs.data, 1)
        final_number_str += str(predicted.item())

    return int(final_number_str) if final_number_str else 0