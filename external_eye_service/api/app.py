import io
import os
import json
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import timm
import numpy as np
import uvicorn

# -------- CONFIG (chỉnh đường dẫn nếu cần) --------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_efficientnet_b3_ver2.pth")  # đổi theo file thực tế
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "data", "train")  # dùng để suy class nếu cần
CLASS_JSON = os.path.join(BASE_DIR, "models", "class_names.json")  # tuỳ chọn
IMG_SIZE = 300

# -------- Device --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Transforms (giữ giống lúc train/test) --------
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------- Helper: load checkpoint / state_dict & class names --------
def load_state_and_labels(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")

    # Lấy state_dict đúng cách (hỗ trợ nhiều format lưu)
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            # Có thể file chính là state_dict (thường là dict mapping param->tensor)
            # hoặc checkpoint chứa trực tiếp state_dict keys
            # Kiểm tra: nếu keys giống tên layer thì xem là state_dict
            keys = list(checkpoint.keys())
            # heuristics: nếu key có 'conv' hoặc 'classifier' hoặc 'bn' => state_dict
            if any(k.startswith(("module.","conv","bn","classifier","head","blocks")) for k in keys[:5]):
                state_dict = checkpoint
            else:
                # cố fallback: nếu checkpoint có 'model' và là dict
                state_dict = checkpoint.get("model_state", checkpoint.get("model", checkpoint))
    else:
        state_dict = checkpoint

    # strip "module." nếu lưu bằng DataParallel
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_key] = v

    # Lấy class names
    class_names = None
    if isinstance(checkpoint, dict):
        class_names = checkpoint.get("class_names") or checkpoint.get("classes")
    if class_names is None and os.path.exists(CLASS_JSON):
        try:
            with open(CLASS_JSON, "r", encoding="utf-8") as f:
                class_names = json.load(f)
        except Exception:
            class_names = None
    if class_names is None and os.path.isdir(TRAIN_DATA_DIR):
        # ImageFolder uses sorted folder names as classes
        class_names = sorted([d for d in os.listdir(TRAIN_DATA_DIR) if os.path.isdir(os.path.join(TRAIN_DATA_DIR, d))])

    if class_names is None:
        raise ValueError("Không tìm được danh sách class. Hãy cung cấp models/class_names.json hoặc đảm bảo data/train tồn tại.")

    return new_state, class_names

# -------- Build model based on number of classes --------
def build_model(num_classes: int):
    model = timm.create_model("efficientnet_b3", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

# -------- Load model once at startup --------
try:
    state_dict, CLASS_NAMES = load_state_and_labels(MODEL_PATH)
    model = build_model(len(CLASS_NAMES))
    model.load_state_dict(state_dict, strict=False)  # strict=False nếu có mismatch nhẹ
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Model loaded. Classes: {CLASS_NAMES}")
except Exception as e:
    # vẫn tạo model rỗng để API không crash ngay, nhưng /predict sẽ lỗi nếu model chưa load
    model = None
    CLASS_NAMES = []
    load_error = str(e)
    print(f"[WARN] Không thể load model: {e}")

# -------- FastAPI app --------
app = FastAPI(title="Retinal Disease Classifier")

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": model is not None}

@app.get("/labels")
def labels():
    return {"classes": CLASS_NAMES}

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    """
    Upload an image file (form field 'file') and get top_k predictions.
    """
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded. Error: {load_error}")

    # read image bytes
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không thể mở file ảnh. Hãy upload file ảnh hợp lệ (jpg/png).")

    # preprocess
    input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    # predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    # top-k
    topk_idx = probs.argsort()[::-1][:top_k]
    results = []
    for idx in topk_idx:
        results.append({
            "label": CLASS_NAMES[idx],
            "index": int(idx),
            "probability": float(probs[idx])
        })

    return JSONResponse({"predictions": results, "top1": results[0] if results else None})

# Optional: endpoint to save uploaded image for auditing
@app.post("/predict_and_save")
async def predict_and_save(file: UploadFile = File(...), folder: str = "uploads", top_k: int = 3):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded. Error: {load_error}")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # save
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, file.filename)
    with open(save_path, "wb") as f:
        f.write(contents)

    # reuse predict
    input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    topk_idx = probs.argsort()[::-1][:top_k]
    results = [{"label": CLASS_NAMES[i], "probability": float(probs[i])} for i in topk_idx]
    return {"saved_to": save_path, "predictions": results}

if __name__ == "__main__":
    # Chạy server FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8083, reload=True)