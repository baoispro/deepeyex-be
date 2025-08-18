import os
from PIL import Image

def remove_corrupt_images(folder_path):
    total_files = 0
    removed_files = 0
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_files += 1
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Kiểm tra file hợp lệ
                except Exception as e:
                    print(f"❌ Xóa file lỗi: {file_path} ({e})")
                    os.remove(file_path)
                    removed_files += 1
    
    print(f"✅ Hoàn tất: {removed_files}/{total_files} file bị xóa trong thư mục {folder_path}")

# ----------- Chạy cho cả train và test -----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn tương đối
RAW_DIR = os.path.join(BASE_DIR, "raw-images")
PROCESSED_DIR = os.path.join(BASE_DIR, "data")
train_dir = os.path.join(PROCESSED_DIR, "train")
test_dir = os.path.join(PROCESSED_DIR, "test")

remove_corrupt_images(train_dir)
remove_corrupt_images(test_dir)
