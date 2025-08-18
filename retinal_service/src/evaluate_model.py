import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import confusion_matrix, classification_report

# ==== CONFIG ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models/best_efficientnet_b3_ver2.pth")
DATA_DIR = os.path.join(BASE_DIR, "data/test")
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 1. Load transforms ====
test_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==== 2. Load dataset ====
test_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
class_names = test_dataset.classes

# ==== 3. Load model ====
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)  # chỉ chứa state_dict
num_classes = len(class_names)

model = timm.create_model("efficientnet_b3", pretrained=False)
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, num_classes)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# ==== 4. Evaluate ====
y_true, y_pred = [], []
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

accuracy = 100.0 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")

# ==== 5. Báo cáo chi tiết ====
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
