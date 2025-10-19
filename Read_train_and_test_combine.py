import json

json_path = "Datasets\\Fisheye8K_all_including_train&test\\train_test\\train_test.json"

with open(json_path, "r") as f:
    data = json.load(f)

num_images = len(data.get("images", []))
num_annotations = len(data.get("annotations", []))
num_labels = len(data.get("categories", []))

print("Thống kê train_test.json:")
print(f"Số ảnh (images): {num_images}")
print(f"Số annotation (annotations): {num_annotations}")
print(f"Số nhãn (categories/labels): {num_labels}")
