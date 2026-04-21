import os
import time

import torch
import torch.nn as nn

import cv2
import numpy as np
import mss

import boardManager
from boardManager import get_drop_color

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
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = ChiffreCNN()
cnn_model.load_state_dict(torch.load("modele_chiffres.pth", weights_only=True))
cnn_model.eval()

MONITOR = {"top": 0, "left": -1920, "width": 1920, "height": 1080}
BOARD_COORDINATES = {"Top": 167, "Left": 195, "Bottom": 809, "Right": 1421}

restart_red = (250, 65, 0)
next_green = (0, 192, 38)

gift_red = (239, 235, 236)
gift_green = (255, 235, 101)

unlocked_color = (41, 255, 65)

drop_color = (245, 170, 105)

cant_buy_color = (166, 92, 42)

def isolate_board(frame):
    return frame[
        BOARD_COORDINATES["Top"]:BOARD_COORDINATES["Bottom"], BOARD_COORDINATES["Left"]:BOARD_COORDINATES["Right"]]


def load_templates(folder_path):
    templates = {str(i): [] for i in range(10)}

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            digit_str = filename[0]

            if digit_str.isdigit():
                path = os.path.join(folder_path, filename)
                template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                # _, template = cv2.threshold(template, 200, 255, cv2.THRESH_BINARY)

                if template is not None:
                    templates[digit_str].append(template)

    return templates


def read_number_with_pytorch(roi_image, model):
    gray_crop = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    _, binarized_crop = cv2.threshold(gray_crop, 200, 255, cv2.THRESH_BINARY)

    white_pixels = cv2.countNonZero(binarized_crop)
    if white_pixels < 10:
        return 0

    # PyTorch s'attend au format [Batch, Canal, Hauteur, Largeur] et aux pixels entre 0 et 1 (float32)
    # 1. Convertir en numpy float32 et normaliser
    img_array = binarized_crop.astype(np.float32) / 255.0

    # 2. Convertir en Tenseur PyTorch
    tensor_img = torch.from_numpy(img_array)

    # 3. Ajouter les dimensions Batch et Canal : (11, 24) -> (1, 1, 11, 24)
    tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)

    # 4. Prédiction (Sans calculer les gradients pour être ultra-rapide)
    with torch.no_grad():
        outputs = model(tensor_img)

    # 5. Récupérer l'index du neurone qui a le plus haut score
    _, predicted = torch.max(outputs.data, 1)

    return int(predicted.item())

def read_number_from_roi(roi_image, templates):
    gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    _, gray_roi = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)

    import time
    cv2.imwrite(f"extraction/img_{time.time()}.png", gray_roi)

    matches = []
    threshold = 0.85

    for digit_str, template_list in templates.items():
        for template in template_list:
            res = cv2.matchTemplate(gray_roi, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)

            for pt in zip(*loc[::-1]):
                matches.append({
                    "digit": digit_str,
                    "x": pt[0],
                    "score": res[pt[1], pt[0]]
                })

    if not matches:
        return 0

    matches.sort(key=lambda m: m['x'])

    best_matches = []
    current_group = [matches[0]]
    group_start_x = matches[0]['x']

    for i in range(1, len(matches)):
        match = matches[i]

        if match['x'] - group_start_x <= 3:
            current_group.append(match)
        else:
            best_in_group = max(current_group, key=lambda m: m['score'])
            best_matches.append(best_in_group)

            current_group = [match]
            group_start_x = match['x']

    if current_group:
        best_in_group = max(current_group, key=lambda m: m['score'])
        best_matches.append(best_in_group)

    final_number_str = "".join([m['digit'] for m in best_matches])

    return int(final_number_str) if final_number_str else 0


def get_grid_matrix(board_image, templates):
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
            roi_x = x + 32
            roi_y = y + 40
            roi_w = 21
            roi_h = 11

            number_crop = board_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            text = read_number_from_roi(number_crop, templates)
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

    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']

            cv2.rectangle(board_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            numero_trouve = str(matrix[i][j]["niveau"])

            text_x = x + int(w / 2) - 10
            text_y = y + int(h / 2) + 10

            cv2.putText(board_image, numero_trouve, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    board_image = cv2.resize(board_image, (400, 300))
    cv2.imshow("Debug Contours", board_image)

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
        if len(liste_cases) >= 2:
            pelles_en_double[niveau] = liste_cases

    return pelles_en_double

def main():
    templates = load_templates("numbers")

    on_board = True

    with mss.mss() as sct:
        while True:
            if get_drop_color() == drop_color:
                on_board = True
                time.sleep(0.05)
                boardManager.go_outside_board()

            screenshot = sct.grab(MONITOR)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            detected_board = isolate_board(frame)

            if on_board:
                drop = True
                can_buy = False

                if boardManager.get_buy_color() != cant_buy_color:
                    can_buy = True

                while can_buy:
                    boardManager.buy()
                    time.sleep(0.05)
                    boardManager.go_outside_board()
                    time.sleep(0.05)
                    if boardManager.get_buy_color() == cant_buy_color:
                        can_buy = False

                if boardManager.get_red_gift_color() == gift_red:
                    boardManager.get_red_gift()
                    time.sleep(0.05)
                    boardManager.go_outside_board()
                    time.sleep(0.05)
                elif boardManager.get_green_gift_color() == gift_green:
                    boardManager.get_green_gift()
                    drop = False
                    time.sleep(0.05)
                    boardManager.go_outside_board()
                    time.sleep(0.05)


                grid_matrix = get_grid_matrix(detected_board, templates)

                fusions_possibles = trouver_pelles_a_fusionner(grid_matrix)

                if fusions_possibles:
                    drop = False
                    for niveau, pelles in fusions_possibles.items():
                        pelle_1 = (pelles[0]["x"], pelles[0]["y"])
                        pelle_2 = (pelles[1]["x"], pelles[1]["y"])
                        boardManager.move_tool(pelle_1, pelle_2)
                        if boardManager.get_unlocked_color() == unlocked_color:
                            boardManager.click_next_button()

                if drop:
                    on_board = False
                    time.sleep(0.05)
                    boardManager.drop()
                    time.sleep(0.05)
            else:
                current_color = boardManager.get_next_button_color()
                if current_color == restart_red:
                    boardManager.click_next_button()
                    on_board = True
                    time.sleep(0.05)
                elif current_color == next_green:
                    boardManager.click_next_button()
                    time.sleep(0.05)
                    boardManager.drop()
                    time.sleep(0.05)
                    on_board = True

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()