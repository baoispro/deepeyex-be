import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import matplotlib.pyplot as plt

# ===== 1. Config =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # ===== 2. Data transforms =====
    transform = transforms.Compose([
        transforms.Resize((300, 300)),  # EfficientNet-B3 input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ===== 3. Load datasets =====
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    NUM_CLASSES = len(train_dataset.classes)
    print(f"📂 Classes: {train_dataset.classes}")

    # ===== 4. Model EfficientNet-B3 =====
    model = timm.create_model("efficientnet_b3", pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # ===== 5. Loss & Optimizer =====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_acc_history, test_acc_history = [], []

    # ===== 6. Training loop =====
    for epoch in range(EPOCHS):
        # Train phase
        model.train()
        train_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        train_acc_history.append(train_acc)

        # Test phase
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100. * correct / total
        test_acc_history.append(test_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # ===== 7. Save model =====
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "efficientnet_b3.pth"))
    print("✅ Model saved!")

    # ===== 8. Plot accuracy =====
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(test_acc_history, label='Test Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
