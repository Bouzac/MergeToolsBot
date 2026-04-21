import os
import cv2
import shutil

EXTRACT_DIR = "extraction"
DATASET_DIR = "dataset"


def setup_directories():
    """Crée les dossiers 0 à 9 et la corbeille s'ils n'existent pas."""
    for i in range(10):
        os.makedirs(os.path.join(DATASET_DIR, str(i)), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "trash"), exist_ok=True)


def main():
    setup_directories()

    if not os.path.exists(EXTRACT_DIR):
        print(f"Erreur : Le dossier '{EXTRACT_DIR}' n'existe pas. Lancez l'extracteur d'abord.")
        return

    # On trie les fichiers pour qu'ils apparaissent toujours dans le même ordre
    files = sorted([f for f in os.listdir(EXTRACT_DIR) if f.endswith(".png")])

    if not files:
        print("Aucune image trouvée dans le dossier d'extraction.")
        return

    print(f"Trouvé {len(files)} images à annoter.")
    print("Contrôles : [0-9] = Annoter | [d] = Corbeille | [z] = Annuler (Undo) | [q] = Quitter")

    history = []  # Historique pour le rollback : stocke des tuples (nom_fichier, chemin_destination)
    index = 0  # Position actuelle dans la liste des fichiers

    while index < len(files):
        filename = files[index]
        filepath = os.path.join(EXTRACT_DIR, filename)

        # Lecture de l'image
        img = cv2.imread(filepath)
        if img is None:
            # Si l'image ne peut pas être lue, on passe à la suivante
            index += 1
            continue

        # L'image est toute petite, on la grossit x15 sans lisser les pixels (INTER_NEAREST)
        view_img = cv2.resize(img, None, fx=15, fy=15, interpolation=cv2.INTER_NEAREST)

        # Le titre de la fenêtre affiche votre progression
        cv2.imshow(f"Annoteur ({index + 1}/{len(files)})", view_img)

        key = cv2.waitKey(0)
        char = chr(key & 0xFF)

        if char.isdigit():
            # Déplacer vers dataset/X
            dest = os.path.join(DATASET_DIR, char, filename)
            shutil.move(filepath, dest)
            history.append((filename, dest))  # Sauvegarder l'action dans l'historique
            print(f"Classé '{filename}' comme : {char}")
            index += 1  # Avancer à l'image suivante

        elif char == 'd':
            # Déplacer vers la corbeille
            dest = os.path.join(DATASET_DIR, "trash", filename)
            shutil.move(filepath, dest)
            history.append((filename, dest))
            print(f"Mis à la corbeille : '{filename}'")
            index += 1

        elif char == 'z':
            # ROLLBACK (Annulation)
            if history:
                # Récupérer la dernière action
                last_filename, last_dest = history.pop()

                # Remettre le fichier dans le dossier d'extraction original
                original_path = os.path.join(EXTRACT_DIR, last_filename)
                shutil.move(last_dest, original_path)

                print(f"ANNULÉ : '{last_filename}' a été remis dans la file d'attente.")
                index -= 1  # Reculer l'index pour réafficher cette image
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