import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold

from models import create_model


# ===============================
#   Dataset for MFCC features
# ===============================

class MFCCDataset(Dataset):
    def __init__(self, root_dir, classes):
        self.root_dir = root_dir
        self.classes = classes
        self.files = []

        # Collect all paths + labels
        for idx, cls in enumerate(classes):
            folder = os.path.join(root_dir, cls)
            for f in os.listdir(folder):
                if f.endswith(".npy"):
                    self.files.append((os.path.join(folder, f), idx))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        mfcc = np.load(file_path)          # shape (40, 500)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)
        mfcc = mfcc.unsqueeze(0)           # (1, 40, 500)
        return mfcc, label


# ===============================
#     CLASS NAMES (edit here)
# ===============================

classes = [
    "cat", "cow", "crow", "dog", "frog",
    "hen", "not_animals", "pig", "rooster", "sheep"
]

num_classes = len(classes)

dataset = MFCCDataset("data/processed", classes)
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fold_accuracies = []


# ===============================
#           TRAINING
# ===============================
def train_one_fold(train_idx, val_idx, fold):

    print(f"\n========== Fold {fold+1} / {k} ==========")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

    model = create_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for mfcc, labels in train_loader:
            mfcc, labels = mfcc.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(mfcc)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Fold {fold+1} | Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for mfcc, labels in val_loader:
            mfcc, labels = mfcc.to(device), labels.to(device)

            outputs = model(mfcc)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Fold {fold+1} Accuracy: {accuracy:.2f}%\n")

    return model, accuracy


# ===============================
#      RUN 10-FOLD TRAINING
# ===============================

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    model, acc = train_one_fold(train_idx, val_idx, fold)
    fold_accuracies.append(acc)

# ===============================
#      FINAL RESULTS + SAVE
# ===============================
avg_acc = sum(fold_accuracies) / k
print("====================================")
print("K-FOLD CROSS VALIDATION RESULTS")
print("====================================")

for i, acc in enumerate(fold_accuracies):
    print(f"Fold {i+1} Accuracy: {acc:.2f}%")

print(f"\nAverage Accuracy: {avg_acc:.2f}%")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/best_model.pth")

print("\nSaved final model → models/best_model.pth")
