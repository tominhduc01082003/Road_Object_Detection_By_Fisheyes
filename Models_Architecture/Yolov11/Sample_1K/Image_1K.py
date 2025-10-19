import json
import os
import cv2
import numpy as np
from PIL import Image
import random

UNLABELED_IMAGES = "..\\..\\..\\DataSets\\Fisheyes1K\\Fisheyes1K_Eval"
PREDICTIONS_PATH = "..\\..\\..\\Predict\\Yolov11\\Eval_Model_1K\\predictions.json"
SAVE_DIR = "..\\..\\..\\Predict\\\Yolov11\\Eval_1K"

CLASS_NAMES = {
    0: "Bus",
    1: "Bike",
    2: "Car",
    3: "Pedestrian",
    4: "Truck"
}

def get_image_name_from_id(image_id, img_names):
    for img_name in img_names:
        try:
            if get_image_Id(img_name) == image_id:
                return img_name
        except:
            continue
    return None

def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId

def draw_predictions(image, predictions, class_names):
    for pred in predictions:
        xmin, ymin, width, height = pred['bbox']
        category_id = pred['category_id']
        score = pred['score']
        
        x1, y1 = int(xmin), int(ymin)
        x2, y2 = int(xmin + width), int(ymin + height)
        
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_names.get(category_id, 'Unknown')}: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

if not os.path.exists(PREDICTIONS_PATH):
    raise FileNotFoundError(f"File predictions.json không tồn tại: {PREDICTIONS_PATH}")

with open(PREDICTIONS_PATH, 'r') as f:
    predictions = json.load(f)

if not os.path.exists(UNLABELED_IMAGES):
    raise FileNotFoundError(f"Thư mục ảnh không nhãn không tồn tại: {UNLABELED_IMAGES}")

img_names = [f for f in os.listdir(UNLABELED_IMAGES) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"Số lượng ảnh trong {UNLABELED_IMAGES}: {len(img_names)}")

os.makedirs(SAVE_DIR, exist_ok=True)

num_images_to_draw = 20
image_ids = list(set(pred['image_id'] for pred in predictions))
if len(image_ids) < num_images_to_draw:
    print(f"Cảnh báo: Chỉ có {len(image_ids)} image_id trong predictions.json, sẽ vẽ tất cả.")
    selected_image_ids = image_ids
else:
    selected_image_ids = random.sample(image_ids, num_images_to_draw)

for image_id in selected_image_ids:
    img_name = get_image_name_from_id(image_id, img_names)
    if img_name is None:
        print(f"Không tìm thấy ảnh tương ứng với image_id: {image_id}")
        continue

    img_path = os.path.join(UNLABELED_IMAGES, img_name)
    try:
        img = np.array(Image.open(img_path).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
    except Exception as e:
        print(f"Lỗi khi đọc ảnh {img_path}: {e}")
        continue

    img_predictions = [pred for pred in predictions if pred['image_id'] == image_id]
    
    img_with_boxes = draw_predictions(img.copy(), img_predictions, CLASS_NAMES)
    
    save_path = os.path.join(SAVE_DIR, f"predicted_{img_name}")
    cv2.imwrite(save_path, img_with_boxes)
    print(f"Đã lưu ảnh với dự đoán tại: {save_path}")

print(f"Đã hoàn thành việc vẽ {len(selected_image_ids)} ảnh và lưu tại {SAVE_DIR}")