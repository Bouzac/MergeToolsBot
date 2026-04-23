import torch
import torch.nn as nn

import boardManager
from boardHelper import read_number_with_pytorch, ChiffreCNN, cnn_model, MONITOR, BOARD_COORDINATES, isolate_board

from boardManager import *

restart_red = (255, 94, 48)
next_green = (0, 234, 67)

drop_color = (255, 170, 97)
no_gift_green = (105, 178, 120)
no_gift_red = (193, 190, 191)
cant_buy_color = (159, 144, 65)

air = (3, 255, 234)

unlocked_color = (79, 255, 98)

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
            roi_x = x + 40
            roi_y = y + 48
            roi_w = 24
            roi_h = 11

            number_crop = board_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

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

    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']
            cv2.rectangle(board_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            numero_trouve = str(matrix[i][j]["niveau"])
            text_x = x + int(w / 2) - 10
            text_y = y + int(h / 2) + 10
            cv2.putText(board_image, numero_trouve, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    board_image = cv2.resize(board_image, (int(0.5 * board_image.shape[1]), int(0.5 * board_image.shape[0])))
    cv2.imshow("Debug PyTorch", board_image)

    return matrix


def optimize_tools(matrix):
    # Sécurité : on s'assure que la matrice n'est pas vide
    if not matrix:
        return

    # La rangée du bas est la dernière ligne de la matrice (-1)
    rangee_du_bas = matrix[-1]

    # On s'assure qu'il y a bien au moins 4 cases dans cette rangée
    if len(rangee_du_bas) < 4:
        return

    # Nos 4 "places de parking" VIP en bas à gauche
    cases_vip = [
        rangee_du_bas[0],
        rangee_du_bas[1],
        rangee_du_bas[2],
        rangee_du_bas[3]
    ]

    # Récupérer tous les outils valides
    all_tools = []
    for row in matrix:
        for cell in row:
            if cell["niveau"] > 0:
                all_tools.append(cell)

    # Trier du plus fort au plus faible
    top_tools = sorted(all_tools, key=lambda t: t['niveau'], reverse=True)

    # On gare nos 4 meilleurs outils (ou moins si on en a moins de 4 sur le plateau)
    for i in range(min(4, len(top_tools))):
        outil = top_tools[i]
        case_cible = cases_vip[i]

        # Si l'outil n'est pas DÉJÀ à sa place exacte, on le bouge
        if outil["x"] != case_cible["x"] or outil["y"] != case_cible["y"]:
            pos_actuelle = (outil["x"], outil["y"])
            pos_cible = (case_cible["x"], case_cible["y"])

            boardManager.move_tool(pos_actuelle, pos_cible)

def trouver_pelles_a_fusionner(matrix):
    pelles_par_niveau = {}
    for row in matrix:
        for cell in row:
            niveau = cell["niveau"]
            if 0 < niveau < 65:
                if niveau not in pelles_par_niveau:
                    pelles_par_niveau[niveau] = []
                pelles_par_niveau[niveau].append(cell)

    pelles_en_double = {}
    for niveau, liste_cases in pelles_par_niveau.items():
        if len(liste_cases) >= 2 and niveau > 0:
            pelles_en_double[niveau] = liste_cases
    return pelles_en_double

def main():
    debug_cells = False
    iteration = 0

    with mss.mss() as sct:
        while True:
            on_board = False

            if boardManager.get_unlocked_color() == unlocked_color:
                boardManager.click_unlocked_button()

            # 1. Capture d'écran (Très rapide)
            if get_drop_color() == drop_color:
                on_board = True

            screenshot = sct.grab(MONITOR)
            frame = np.array(screenshot)
            detected_board = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if debug_cells:
                time.sleep(0.1)

                get_grid_matrix(detected_board, cnn_model)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                continue

            if debug_cells:
                return

            if on_board:
                if iteration == 5:
                    iteration = 0
                    boardManager.buy_in_shop_sequence()

                action_done = False # Traceur pour court-circuiter les étapes

                # --- ÉTAPE 1 : CADEAUX (Priorité absolue, coût CPU proche de 0) ---
                if boardManager.get_red_gift_color() != no_gift_red and boardManager.get_red_gift_color() != air:
                    boardManager.get_red_gift()
                    action_done = True
                elif boardManager.get_green_gift_color() != no_gift_green and boardManager.get_green_gift_color() != air:
                    boardManager.get_green_gift()
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

                # --- ÉTAPE 4 : DROP (Dernier recours pour générer du contenu) ---
                if not action_done:
                    grid_matrix = get_grid_matrix(detected_board, cnn_model)

                    # if endgame:
                    #     LEVEL_THRESHOLD = 10
                    #     DISCARD_COORDS = (738, 273)
                    #
                    #     highest_level = max((cell["niveau"] for row in grid_matrix for cell in row), default=0)
                    #
                    #     # 1. Vérifie la valeur max détectée
                    #     print(f"DEBUG: Niveau maximum détecté = {highest_level}")
                    #
                    #     for row in grid_matrix:
                    #         for cell in row:
                    #             niveau = cell["niveau"]
                    #
                    #             if 0 < niveau < (highest_level - LEVEL_THRESHOLD):
                    #                 # 2. Vérifie quels objets il cible avant le clic
                    #                 print(f"DEBUG: Suppression objet niveau {niveau} en X:{cell['x']} Y:{cell['y']}")
                    #
                    #                 boardManager.move_tool((cell["x"], cell["y"]), DISCARD_COORDS)
                    #
                    #                 # 3. Ajoute une petite pause pour laisser le jeu respirer
                    #                 time.sleep(0.3)

                    optimize_tools(grid_matrix)
                    on_board = False
                    iteration += 1
                    boardManager.drop()
            else:
                # --- ÉTAPE 5 : NAVIGATION DE FIN DE NIVEAU ---
                current_color = boardManager.get_next_button_color()
                if current_color == restart_red:
                    boardManager.click_next_button()
                    boardManager.go_outside_board()
                    while not on_board:
                        time.sleep(0.1)
                        pyautogui.click()
                        if get_drop_color() == drop_color:
                            on_board = True
                elif current_color == next_green:
                    boardManager.click_next_button()
                    boardManager.drop()
                    boardManager.go_outside_board()
                    while not on_board:
                        time.sleep(0.1)
                        pyautogui.click()
                        if get_drop_color() == drop_color:
                            on_board = True
                else:
                    time.sleep(0.1)
                    drop()
                    if get_drop_color() == drop_color:
                        on_board = True

            # OpenCV interface
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()