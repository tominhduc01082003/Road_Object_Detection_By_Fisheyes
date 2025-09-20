import json

with open("..\\Road_Object_Detection_By_Fisheyes\\DataSets\\Fisheye8K_all_including_train&test\\train\\train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Các danh mục trong file JSON:")
for key in data.keys():
    print("-", key)

print("\nSố lượng phần tử trong mỗi danh mục:")
for key, value in data.items():
    if isinstance(value, list):
        print(f"{key}: {len(value)} phần tử")
    else:
        print(f"{key}: {type(value)}")  
    if key == "images":
        print("Ví dụ về một phần tử trong 'images':")
        print(value[0])
    elif key == "categories":
        print("Ví dụ về một phần tử trong 'categories':")
        print(value[0])

for ann in data["annotations"]:
    if ann["image_id"] == 141220:
        print("Annotation có id = 141220:")
        print(ann)
        break
