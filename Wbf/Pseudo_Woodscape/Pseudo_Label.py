import os
import cv2
import yaml
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

with open("Config_Hyper.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

image_dir = cfg["paths"]["image_dir"]
label_dir = cfg["paths"]["label_dir"]
vis_dir = cfg["paths"]["vis_dir"]

os.makedirs(label_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

v8_m = YOLO(cfg["models"]["yolov8m"])
v8_l = YOLO(cfg["models"]["yolov8l"])
v11_x = YOLO(cfg["models"]["yolov11x"])

iou_thr = cfg["params"]["iou_thr"]
skip_thr = cfg["params"]["skip_thr"]
weights = cfg["params"]["weights"]
conf_pred = cfg["params"]["conf_pred"]
final_conf_thr = cfg["params"]["final_conf_thr"]

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Đang gán nhãn cho {len(image_files)} ảnh...")

vis_count = 0

for img_name in tqdm(image_files):
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Bỏ qua {img_name} (không đọc được).")
        continue
    h, w = img.shape[:2]

    preds_list = [
        v8_m.predict(img, conf=conf_pred, iou=0.5, verbose=False, augment=True),
        v8_l.predict(img, conf=conf_pred, iou=0.5, verbose=False, augment=True),
        v11_x.predict(img, conf=conf_pred, iou=0.5, verbose=False, augment=True)
    ]

    boxes_list, scores_list, labels_list = [], [], []

    for preds in preds_list:
        b, s, c = [], [], []
        for r in preds:
            if len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                if x2 <= x1 or y2 <= y1:
                    continue
                b.append([x1 / w, y1 / h, x2 / w, y2 / h])
                s.append(confs[i])
                c.append(classes[i])
        boxes_list.append(b)
        scores_list.append(s)
        labels_list.append(c)

    if not boxes_list or all(len(b) == 0 for b in boxes_list):
        continue

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=weights, iou_thr=iou_thr, skip_box_thr=skip_thr
    )

    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
    with open(label_path, "w") as f:
        for box, score, label in zip(boxes, scores, labels):
            if score < final_conf_thr:
                continue
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            w_box = x2 - x1
            h_box = y2 - y1
            f.write(f"{int(label)} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}\n")

    if vis_count < 10:
        vis_img = img.copy()
        for box, score, label in zip(boxes, scores, labels):
            if score < final_conf_thr:
                continue
            x1, y1, x2, y2 = (box * np.array([w, h, w, h])).astype(int)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, f"{int(label)} {score:.2f}",
                        (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)
        save_path = os.path.join(vis_dir, os.path.splitext(img_name)[0] + ".png")
        cv2.imwrite(save_path, vis_img)
        vis_count += 1
        if vis_count == 10:
            print("\nĐã lưu 10 ảnh minh họa tại thư mục Pre_Vis.")

print(f"\nHoàn tất! Nhãn đã được lưu tại: {label_dir}")
