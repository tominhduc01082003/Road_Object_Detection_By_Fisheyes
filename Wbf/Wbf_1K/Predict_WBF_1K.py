import os
import yaml
import json
import cv2
import random
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

with open("..\\Config\\Config_Hyper.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

eval_dir = os.path.abspath(config["paths"]["eval_dir"])
vis_dir = os.path.abspath(config["paths"]["vis_dir"])
output_json = os.path.abspath(config["paths"]["output_json"])
os.makedirs(vis_dir, exist_ok=True)

model_paths = config["models"]
weights = config["params"]["weights"]
iou_thr = config["params"]["iou_thr"]
skip_thr = config["params"]["skip_thr"]
conf_pred = config["params"]["conf_pred"]
final_conf_thr = config["params"]["final_conf_thr"]

sceneList = ['M', 'A', 'E', 'N']

def get_image_id(img_name: str) -> int:
    base = img_name.split('.png')[0]
    cam_idx = int(base.split('_')[0].replace('camera', ''))
    scene_idx = sceneList.index(base.split('_')[1])
    frame_idx = int(base.split('_')[2])
    return int(f"{cam_idx}{scene_idx}{frame_idx}")

print("Đang tải YOLO models...")
models = {
    "yolov8m": YOLO(model_paths["yolov8m"]),
    "yolov8l": YOLO(model_paths["yolov8l"]),
    "yolov11x": YOLO(model_paths["yolov11x"]),
}
print("Load model thành công!\n")

image_files = sorted(
    [f for f in os.listdir(eval_dir) if f.lower().endswith((".jpg", ".png"))]
)
print(f"Tổng số ảnh tìm thấy: {len(image_files)}")

random.seed(42)
vis_images = random.sample(image_files, 10)
print(f"Lưu và hiển thị 10 ảnh: {vis_images}\n")

predictions = []

def merge_boxes_with_diff_labels(boxes, scores, labels, iou_threshold=0.7):
    merged_boxes, merged_scores, merged_labels = [], [], []
    used = set()
    for i, box_i in enumerate(boxes):
        if i in used:
            continue
        same_boxes = [i]
        for j, box_j in enumerate(boxes):
            if j <= i or j in used:
                continue
            # Tính IoU
            xi1, yi1, xi2, yi2 = box_i
            xj1, yj1, xj2, yj2 = box_j
            inter_x1, inter_y1 = max(xi1, xj1), max(yi1, yj1)
            inter_x2, inter_y2 = min(xi2, xj2), min(yi2, yj2)
            inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            area_i = (xi2 - xi1) * (yi2 - yi1)
            area_j = (xj2 - xj1) * (yj2 - yj1)
            union_area = area_i + area_j - inter_area
            iou = inter_area / union_area if union_area > 0 else 0

            if iou >= iou_threshold:
                same_boxes.append(j)
                used.add(j)
        best_idx = max(same_boxes, key=lambda k: scores[k])
        merged_boxes.append(boxes[best_idx])
        merged_scores.append(scores[best_idx])
        merged_labels.append(labels[best_idx])
        used.update(same_boxes)
    return merged_boxes, merged_scores, merged_labels


for img_name in tqdm(image_files, desc="Đang chạy dự đoán WBF trên 1K ảnh"):
    img_path = os.path.join(eval_dir, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue

    height, width = image.shape[:2]
    all_boxes, all_scores, all_labels = [], [], []

    for _, model in models.items():
        results = model.predict(
            img_path,
            conf=conf_pred,
            iou=iou_thr,
            augment=True,
            verbose=False
        )[0]

        if results.boxes is None:
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy().astype(int)

        boxes_norm = [
            [b[0] / width, b[1] / height, b[2] / width, b[3] / height]
            for b in boxes
        ]

        all_boxes.append(boxes_norm)
        all_scores.append(scores)
        all_labels.append(labels)

    if any(len(b) > 0 for b in all_boxes):
        boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_thr
        )

        boxes_fused = np.array(boxes_fused)
        boxes_fused[:, [0, 2]] *= width
        boxes_fused[:, [1, 3]] *= height

        boxes_final, scores_final, labels_final = merge_boxes_with_diff_labels(
            boxes_fused.tolist(),
            scores_fused.tolist(),
            labels_fused.tolist(),
            iou_threshold=0.7
        )

        image_id = get_image_id(img_name)

        for box, score, label in zip(boxes_final, scores_final, labels_final):
            if score < final_conf_thr:
                continue
            x1, y1, x2, y2 = map(float, box)
            w, h = x2 - x1, y2 - y1
            predictions.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x1, y1, w, h],
                "score": float(score)
            })

        if img_name in vis_images:
            vis_img = image.copy()
            for box, score, label in zip(boxes_final, scores_final, labels_final):
                if score < final_conf_thr:
                    continue
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                text = f"{label}:{score:.2f}"
                cv2.putText(vis_img, text, (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            save_path = os.path.join(vis_dir, f"WBF_{img_name}")
            cv2.imwrite(save_path, vis_img)

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=4)

print(f"\nLưu file kết quả thành công: {output_json}")
print(f"10 ảnh minh họa được lưu tại thư mục: {vis_dir}")
