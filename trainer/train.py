import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- PARAMÈTRES ---
DATASET_DIR = "../dataset"
IMG_HEIGHT = 11  # Fixé par ton annotation
IMG_WIDTH = 9    # Fixé par ton annotation
BATCH_SIZE = 16  # Réduit un peu pour la stabilité sur petit dataset
EPOCHS = 250     # 100 suffisent généralement pour des chiffres si simples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"training on {device}")

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
        # Le calcul (32 * 2 * 2) = 128 est correct pour 11x9
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

def main():
    # 2. Préparation des données
    # On ajoute un seuillage pour transformer les nuances de gris en 0 ou 1 pur
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # BINARISATION : On force les pixels > 0.5 à 1, sinon 0 (comme en inférence)
        transforms.Lambda(lambda x: (x > 0.5).float()),
        # DATA AUGMENTATION : On simule des petits décalages de 1 pixel
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])

    # Chargement
    if not os.path.exists(DATASET_DIR):
        print(f"Erreur : Le dossier {DATASET_DIR} est vide !")
        return

    full_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=transform)
    
    # Division en Train (80%) et Validation (20%) pour vérifier que le modèle n'apprend pas par coeur
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[*] Dataset chargé : {len(full_dataset)} images.")
    print(f"[*] Classes : {full_dataset.classes}")

    model = ChiffreCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n[*] Début de l'entraînement...")
    best_accuracy = 0.0 # On garde en mémoire le meilleur score

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        printed = False
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            
            # Affichage classique
            if (epoch + 1) % 10 == 0 and printed == False:
                printed = True
                print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f} - Accuracy Val: {accuracy:.2f}%")

            # --- NOUVEAU : Sauvegarde intelligente ---
            if accuracy >= best_accuracy and accuracy > 80.0: # On sauvegarde si c'est le meilleur ET qu'il est décent
                best_accuracy = accuracy
                torch.save(model.state_dict(), "../modele_chiffres.pth")
                # print(f"  -> Nouveau record ({accuracy:.2f}%) ! Modèle sauvegardé.") 
                # (Tu peux décommenter le print ci-dessus pour le voir en direct)

    print(f"\n[SUCCESS] Entraînement terminé. Le meilleur modèle (Précision: {best_accuracy:.2f}%) a été conservé !")

    # 5. Sauvegarde
    torch.save(model.state_dict(), "../modele_chiffres.pth")
    print("\n[SUCCESS] Modèle sauvegardé !")

if __name__ == "__main__":
    main()