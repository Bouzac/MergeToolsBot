import os
import time

import torch
import torch.nn as nn

import cv2
import numpy as np
import mss

import boardManager
from boardManager import get_drop_color

# ==========================================
# --- 1. ARCHITECTURE DU MODÈLE PYTORCH ---
# ==========================================
class ChiffreCNN(nn.Module):
    def __init__(self):
        super(ChiffreCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3) # Peut rester ici, désactivé par .eval()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x) # Dropout ignoré pendant l'inférence
        return x

# Initialisation (On utilise le CPU car pour des images 11x9, c'est instantané)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = ChiffreCNN().to(device)
cnn_model.load_state_dict(torch.load("modele_chiffres.pth", map_location=device, weights_only=True))
cnn_model.eval() # TRÈS IMPORTANT : Désactive le Dropout pour l'inférence

# ==========================================
# --- 2. CONSTANTES ET VARIABLES ---
# ==========================================
MONITOR = {"top": 0, "left": 0, "width": 1920, "height": 1080}
BOARD_COORDINATES = {"Top": 176, "Left": 196, "Bottom": 845, "Right": 1411}

restart_red = (255, 67, 1)
next_green = (0, 197, 41)
drop_color = (255, 171, 98)
gift_green = (255, 235, 102)
gift_red = (244, 127, 135)
cant_buy_color = (166, 89, 40)
unlocked_color = (0, 239, 28)

# ==========================================
# --- 3. FONCTIONS DE TRAITEMENT D'IMAGE ---
# ==========================================
def isolate_board(frame):
    return frame[
        BOARD_COORDINATES["Top"]:BOARD_COORDINATES["Bottom"], BOARD_COORDINATES["Left"]:BOARD_COORDINATES["Right"]]

def split_into_digits(binarized_img):
    contours, _ = cv2.findContours(binarized_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= 5: 
            bounding_boxes.append((x, y, w, h, c))
            
    bounding_boxes.sort(key=lambda b: b[0])
    digits = []
    
    for x, y, w, h, c in bounding_boxes:
        mask = np.zeros_like(binarized_img)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        isolated_digit = cv2.bitwise_and(binarized_img, mask)
        digit_crop = isolated_digit[:, x:x+w]
        digits.append(digit_crop)
        
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

# ==========================================
# --- 4. LECTURE PAR RÉSEAU DE NEURONES ---
# ==========================================
def read_number_with_pytorch(roi_image, model):
    gray_crop = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    _, binarized_crop = cv2.threshold(gray_crop, 210, 255, cv2.THRESH_BINARY)

    # 1. Découpage en chiffres séparés
    digits = split_into_digits(binarized_crop)
    
    if not digits:
        return 0

    final_number_str = ""

    # 2. Inférence pour chaque chiffre
    for digit_img in digits:
        if cv2.countNonZero(digit_img) < 5:
            continue

        # Formatage 11x9
        padded_digit = pad_to_target_size(digit_img, 11, 9)

        # PyTorch attend des valeurs de 0 à 1 en Float32
        img_array = padded_digit.astype(np.float32) / 255.0
        
        # Ajout des dimensions [Batch, Canal, H, W] -> [1, 1, 11, 9]
        tensor_img = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor_img)
        
        _, predicted = torch.max(outputs.data, 1)
        final_number_str += str(predicted.item())

    return int(final_number_str) if final_number_str else 0

# ==========================================
# --- 5. LOGIQUE DU JEU ---
# ==========================================
def get_grid_matrix(board_image, model):
    gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 1000 < w * h < 15000:
            cells.append({'x': x, 'y': y, 'w': w, 'h': h})

    if not cells:
        return []

    cells.sort(key=lambda c: c['y'])
    rows = []
    current_row = [cells[0]]

    for i in range(1, len(cells)):
        if abs(cells[i]['y'] - current_row[-1]['y']) < (cells[i]['h'] / 2):
            current_row.append(cells[i])
        else:
            if len(current_row) >= 4:
                rows.append(current_row)
            current_row = [cells[i]]
    rows.append(current_row)

    matrix = []
    for row in rows:
        row.sort(key=lambda c: c['x'])
        matrix_row = []

        for cell in row:
            x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']
            
            # --- OFFSETS DE LA CASE ---
            roi_x = x + 32
            roi_y = y + 40
            roi_w = 21
            roi_h = 11

            number_crop = board_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            
            # APPEL DU RÉSEAU DE NEURONES ICI
            text = read_number_with_pytorch(number_crop, model)
            
            center_local_x = x + int(w / 2)
            center_local_y = y + int(h / 2)

            screen_x = MONITOR["left"] + BOARD_COORDINATES["Left"] + center_local_x
            screen_y = MONITOR["top"] + BOARD_COORDINATES["Top"] + center_local_y

            matrix_row.append({
                "niveau": text,
                "x": screen_x,
                "y": screen_y
            })

        matrix.append(matrix_row)

    # Debug visuel
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']
            cv2.rectangle(board_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            numero_trouve = str(matrix[i][j]["niveau"])
            text_x = x + int(w / 2) - 10
            text_y = y + int(h / 2) + 10
            cv2.putText(board_image, numero_trouve, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    board_image = cv2.resize(board_image, (400, 300))
    cv2.imshow("Debug PyTorch", board_image)

    return matrix

def trouver_pelles_a_fusionner(matrix):
    pelles_par_niveau = {}
    for row in matrix:
        for cell in row:
            niveau = cell["niveau"]
            if niveau > 0:
                if niveau not in pelles_par_niveau:
                    pelles_par_niveau[niveau] = []
                pelles_par_niveau[niveau].append(cell)

    pelles_en_double = {}
    for niveau, liste_cases in pelles_par_niveau.items():
        if len(liste_cases) >= 2 and niveau > 0:
            pelles_en_double[niveau] = liste_cases
    return pelles_en_double

def main():
    on_board = False

    with mss.mss() as sct:
        while True:
            # 1. Capture d'écran (Très rapide)
            if get_drop_color() == drop_color:
                on_board = True

            screenshot = sct.grab(MONITOR)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            detected_board = isolate_board(frame)

            if on_board:
                action_done = False # Traceur pour court-circuiter les étapes

                # --- ÉTAPE 1 : CADEAUX (Priorité absolue, coût CPU proche de 0) ---
                if boardManager.get_red_gift_color() == gift_red:
                    boardManager.get_red_gift()
                    boardManager.go_outside_board()
                    action_done = True
                elif boardManager.get_green_gift_color() == gift_green:
                    boardManager.get_green_gift()
                    boardManager.go_outside_board()
                    action_done = True

                # --- ÉTAPE 2 : ACHATS SPAM (Si pas de cadeaux) ---
                if not action_done:
                    bought_something = False
                    # On achète en boucle tant que le bouton est allumé
                    while boardManager.get_buy_color() != cant_buy_color:
                        boardManager.buy()
                        boardManager.go_outside_board()
                        bought_something = True
                    
                    if bought_something:
                        action_done = True

                # --- ÉTAPE 3 : LECTURE IA ET FUSIONS (Uniquement si le plateau est "calme") ---
                if not action_done:
                    # On appelle le CNN uniquement si on n'a rien cliqué d'autre
                    grid_matrix = get_grid_matrix(detected_board, cnn_model)
                    fusions_possibles = trouver_pelles_a_fusionner(grid_matrix)

                    if fusions_possibles:
                        action_done = True
                        
                        # OPTIMISATION MAJEURE : On fusionne en rafale (paires multiples)
                        for niveau, pelles in fusions_possibles.items():
                            # Avance de 2 en 2 pour prendre 0&1, puis 2&3, puis 4&5...
                            for i in range(0, len(pelles) - 1, 2):
                                pelle_1 = (pelles[i]["x"], pelles[i]["y"])
                                pelle_2 = (pelles[i+1]["x"], pelles[i+1]["y"])
                                boardManager.move_tool(pelle_1, pelle_2)

                                # Check sécurité anti-popup
                                if boardManager.get_unlocked_color() == unlocked_color:
                                    boardManager.click_unlocked_button()

                # --- ÉTAPE 4 : DROP (Dernier recours pour générer du contenu) ---
                if not action_done:
                    on_board = False
                    boardManager.drop()
            else:
                # --- ÉTAPE 5 : NAVIGATION DE FIN DE NIVEAU ---
                current_color = boardManager.get_next_button_color()
                if current_color == restart_red:
                    boardManager.click_next_button()
                    on_board = True
                elif current_color == next_green:
                    boardManager.click_next_button()
                    boardManager.drop()
                    on_board = True

            # OpenCV interface
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()