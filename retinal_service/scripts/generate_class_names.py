import os, json

base_dir = os.path.dirname(os.path.dirname(__file__))
train_dir = os.path.join(base_dir, "data", "train")
classes = sorted(os.listdir(train_dir))  # lấy danh sách thư mục con
models_dir = os.path.join(base_dir, "models")
output_path = os.path.join(models_dir, "class_names.json")
class_dict = {i: cls for i, cls in enumerate(classes)}

with open(output_path, "w") as f:
    json.dump(class_dict, f, indent=4)

print("Đã tạo class_names.json:", class_dict)
