import os
import shutil
import random

# Lấy đường dẫn thư mục hiện tại (nơi chứa file script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn tương đối
RAW_DIR = os.path.join(BASE_DIR, "raw-images")
PROCESSED_DIR = os.path.join(BASE_DIR, "data")

# Tỷ lệ chia
TRAIN_RATIO = 0.8

# Xóa thư mục processed cũ nếu có
if os.path.exists(PROCESSED_DIR):
    shutil.rmtree(PROCESSED_DIR)

# Tạo thư mục train/test
for split in ["train", "test"]:
    os.makedirs(os.path.join(PROCESSED_DIR, split), exist_ok=True)

# Duyệt từng loại bệnh
for disease_class in os.listdir(RAW_DIR):
    class_path = os.path.join(RAW_DIR, disease_class)
    if not os.path.isdir(class_path):
        continue

    # Lấy danh sách ảnh
    images = os.listdir(class_path)
    random.shuffle(images)

    # Chia train/test
    split_idx = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    for img in train_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(PROCESSED_DIR, "train", disease_class)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)

    for img in test_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(PROCESSED_DIR, "test", disease_class)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)

print("✅ Hoàn tất chia ảnh!")
