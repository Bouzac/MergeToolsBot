import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- PARAMÈTRES ---
DATASET_DIR = "dataset"
IMG_HEIGHT = 11
IMG_WIDTH = 9
BATCH_SIZE = 32
EPOCHS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Architecture du Réseau de Neurones (Micro-CNN)
class ChiffreCNN(nn.Module):
    def __init__(self):
        super(ChiffreCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()

        # --- L'ARME ANTI-MÉMORISATION ---
        self.dropout = nn.Dropout(0.5)  # Désactive 50% des neurones au hasard
        # --------------------------------

        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))

        # On applique le dropout juste avant la décision finale
        x = self.dropout(x)

        x = self.fc2(x)
        return x


def main():
    print("[*] Configuration de PyTorch...")

    # 2. Préparation des données (Transformations)
    # 2. Préparation des données (Transformations)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),

        # --- LA MAGIE EST ICI (DATA AUGMENTATION) ---
        # Décale l'image aléatoirement jusqu'à 10% horizontalement et verticalement
        # Ça force le modèle à reconnaître un "1" même s'il n'est pas parfaitement centré !
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # --------------------------------------------

        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])

    # Chargement du dossier (ignore automatiquement les fichiers non-images)
    dataset = datasets.ImageFolder(root=DATASET_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"[*] Classes détectées : {dataset.classes}")

    # 3. Initialisation du modèle, de la perte et de l'optimiseur
    model = ChiffreCNN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. La boucle d'entraînement (Typique de PyTorch)
    print("\n[*] Début de l'entraînement...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            # Remettre les gradients à zéro
            optimizer.zero_grad()

            # Faire une prédiction (Forward pass)
            outputs = model(images)

            # Calculer l'erreur (Loss)
            loss = criterion(outputs, labels)

            # Corriger les poids (Backward pass & Optimize)
            loss.backward()
            optimizer.step()

            # Statistiques
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] - Erreur: {running_loss / len(dataloader):.4f} - Précision: {accuracy:.2f}%")

    # 5. Sauvegarde des poids du modèle
    model_path = "modele_chiffres.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n[SUCCESS] Poids du modèle sauvegardés sous '{model_path}' !")


if __name__ == "__main__":
    main()