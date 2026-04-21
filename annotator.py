import os
import cv2
import shutil

EXTRACT_DIR = "extraction"
DATASET_DIR = "dataset"

# --- NOUVEAU : Dimensions cibles ---
TARGET_HEIGHT = 11
TARGET_WIDTH = 9

def setup_directories():
    """Crée les dossiers 0 à 9 et la corbeille s'ils n'existent pas."""
    for i in range(10):
        os.makedirs(os.path.join(DATASET_DIR, str(i)), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "trash"), exist_ok=True)


def pad_to_target_size(img, target_h, target_w):
    """
    Centre l'image dans une boîte noire de taille target_h x target_w.
    """
    h, w = img.shape[:2]

    # Sécurité : Si l'image extraite est accidentellement plus grande que la cible,
    # on la force à rentrer (évite les crashs).
    if h > target_h or w > target_w:
        img = cv2.resize(img, (min(w, target_w), min(h, target_h)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # Calcul des marges pour centrer le chiffre
    top = (target_h - h) // 2
    bottom = target_h - h - top
    left = (target_w - w) // 2
    right = target_w - w - left

    # Ajout des bordures noires (0, 0, 0)
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img


def main():
    setup_directories()

    if not os.path.exists(EXTRACT_DIR):
        print(f"Erreur : Le dossier '{EXTRACT_DIR}' n'existe pas. Lancez l'extracteur d'abord.")
        return

    files = sorted([f for f in os.listdir(EXTRACT_DIR) if f.endswith(".png")])

    if not files:
        print("Aucune image trouvée dans le dossier d'extraction.")
        return

    print(f"Trouvé {len(files)} images à annoter.")
    print("Contrôles : [0-9] = Annoter | [d] = Corbeille | [z] = Annuler (Undo) | [q] = Quitter")

    history = []  
    index = 0  

    while index < len(files):
        filename = files[index]
        filepath = os.path.join(EXTRACT_DIR, filename)

        img = cv2.imread(filepath)

        if img.shape[1] < 5:
            index += 1
            continue

        if img is None:
            index += 1
            continue

        # --- NOUVEAU : On applique le padding AVANT l'affichage et la sauvegarde ---
        padded_img = pad_to_target_size(img, TARGET_HEIGHT, TARGET_WIDTH)

        # On grossit l'image paddée pour la voir clairement
        view_img = cv2.resize(padded_img, None, fx=15, fy=15, interpolation=cv2.INTER_NEAREST)

        cv2.imshow(f"Annoteur ({index + 1}/{len(files)})", view_img)

        key = cv2.waitKey(0)
        char = chr(key & 0xFF)

        if char.isdigit():
            dest = os.path.join(DATASET_DIR, char, filename)
            # Au lieu de shutil.move, on ÉCRIT la nouvelle image paddée et on SUPPRIME l'ancienne
            cv2.imwrite(dest, padded_img)
            os.remove(filepath)
            
            history.append((filepath, dest))
            print(f"Classé '{filename}' comme : {char}")
            index += 1

        elif char == 'd':
            dest = os.path.join(DATASET_DIR, "trash", filename)
            cv2.imwrite(dest, padded_img)
            os.remove(filepath)
            
            history.append((filepath, dest))
            print(f"Mis à la corbeille : '{filename}'")
            index += 1

        elif char == 'z':
            if history:
                last_filepath, last_dest = history.pop()
                # On ramène l'image paddée dans le dossier extraction
                shutil.move(last_dest, last_filepath)

                print(f"ANNULÉ : '{os.path.basename(last_filepath)}' a été remis dans la file d'attente.")
                index -= 1
            else:
                print("Rien à annuler ! Vous êtes au début.")

        elif char == 'q':
            print("Fermeture de l'annoteur...")
            break

        else:
            print(f"Touche ignorée : '{char}'. Utilisez 0-9, d, z, ou q.")

    cv2.destroyAllWindows()
    print("Session d'annotation terminée !")


if __name__ == "__main__":
    main()