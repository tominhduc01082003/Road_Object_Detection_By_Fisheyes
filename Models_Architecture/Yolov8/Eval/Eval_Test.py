import json
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_iou(box1, box2, img_size):
    """
    box1, box2: [x_center, y_center, width, height] in normalized coordinates
    img_size: (width, height) of the image
    Returns IoU value
    """
    w, h = img_size
    x1_min = (box1[0] - box1[2] / 2) * w
    y1_min = (box1[1] - box1[3] / 2) * h
    x1_max = (box1[0] + box1[2] / 2) * w
    y1_max = (box1[1] + box1[3] / 2) * h

    x2_min = (box2[0] - box2[2] / 2) * w
    y2_min = (box2[1] - box2[3] / 2) * h
    x2_max = (box2[0] + box2[2] / 2) * w
    y2_max = (box2[1] + box2[3] / 2) * h

    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0

def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId

def get_ground_truth(label_path, img_size):
    bboxes = []
    category_ids = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                bboxes.append([x_center, y_center, w, h])
                category_ids.append(int(class_id))
    return bboxes, category_ids

def calculate_precision_recall(predictions, ground_truths, img_size, iou_threshold=0.5):
    """
    predictions: List of {'bbox': [x_min, y_min, width, height], 'category_id': int, 'score': float}
    ground_truths: {'bboxes': [[x_center, y_center, w, h], ...], 'category_ids': [int, ...]}
    img_size: (width, height)
    """
    TP = 0
    FP = 0
    FN = 0
    matched_gt = set()

    for pred in sorted(predictions, key=lambda x: x['score'], reverse=True):
        pred_bbox = [
            (pred['bbox'][0] + pred['bbox'][2] / 2) / img_size[0],  # x_center
            (pred['bbox'][1] + pred['bbox'][3] / 2) / img_size[1],  # y_center
            pred['bbox'][2] / img_size[0],                          # width
            pred['bbox'][3] / img_size[1]                           # height
        ]
        pred_cat = pred['category_id']
        pred_score = pred['score']

        if pred_score < 0.25 or pred_cat == 0: 
            continue

        best_iou = 0
        best_gt_idx = -1
        for gt_idx, (gt_bbox, gt_cat) in enumerate(zip(ground_truths['bboxes'], ground_truths['category_ids'])):
            if gt_idx in matched_gt or gt_cat != pred_cat or gt_cat == 0:
                continue
            iou = calculate_iou(pred_bbox, gt_bbox, img_size)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            TP += 1
            matched_gt.add(best_gt_idx)
        else:
            FP += 1

    FN = len([cat for cat in ground_truths['category_ids'] if cat != 0]) - len(matched_gt)

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0

    return precision, recall, TP, FP, FN

def plot_images_with_boxes(images_to_plot, save_dir="..\\..\\..\\Predict\\Yolov8\\Image_Test"):
    os.makedirs(save_dir, exist_ok=True)
    for idx, item in enumerate(images_to_plot):
        img = item['image']
        img_name = os.path.basename(item['img_path'])
        ground_truth = item['ground_truth']
        predictions = item['predictions']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f"Image: {img_name}", fontsize=16)

        ax1.imshow(img)
        ax1.set_title("Ground Truth")
        for bbox, cat_id in zip(ground_truth['bboxes'], ground_truth['category_ids']):
            if cat_id != 0:
                x_center, y_center, w, h = bbox
                x_min = (x_center - w / 2) * img.width
                y_min = (y_center - h / 2) * img.height
                width = w * img.width
                height = h * img.height
                rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='g', facecolor='none')
                ax1.add_patch(rect)
                ax1.text(x_min, y_min - 5, f"Class {cat_id}", color='g', fontsize=12, weight='bold')

        ax2.imshow(img)
        ax2.set_title("Predictions")
        for pred in predictions:
            if pred['category_id'] != 0 and pred['score'] > 0.25:
                x_min, y_min, width, height = pred['bbox']
                rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax2.add_patch(rect)
                ax2.text(x_min, y_min - 5, f"Class {pred['category_id']} ({pred['score']:.2f})", 
                         color='r', fontsize=12, weight='bold')

        plt.savefig(os.path.join(save_dir, f"comparison_{img_name}"))
        plt.close()

def main():

    ROOT_DIR = "..\\..\\..\\DataSets\\Fisheye8K_all_including_train&test"
    TEST_DIR = os.path.join(ROOT_DIR, 'test')
    TEST_IMAGES = os.path.join(TEST_DIR, 'images')
    TEST_LABELS = os.path.join(TEST_DIR, 'labels')
    PREDICTIONS_PATH = "..\\..\\..\\Predict\\Yolov8\\Eval_Train_Test\\predictions.json"
    SAVE_DIR = "..\\..\\..\\Predict\\Yolov8\\Image_Test"

    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"File predictions.json không tồn tại tại: {PREDICTIONS_PATH}")
    if not os.path.exists(TEST_IMAGES) or not os.path.exists(TEST_LABELS):
        raise FileNotFoundError("Thư mục test/images hoặc test/labels không tồn tại.")

    with open(PREDICTIONS_PATH, 'r') as f:
        predictions = json.load(f)

    img_names = [f for f in os.listdir(TEST_IMAGES) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    pred_image_ids = set(pred['image_id'] for pred in predictions)
    
    images_to_plot = []
    total_precision = 0.0
    total_recall = 0.0
    valid_images = 0
    np.random.shuffle(img_names)
    for img_name in img_names:
        if len(images_to_plot) >= 5:
            break
        image_id = get_image_Id(img_name)
        if image_id in pred_image_ids:
            img_path = os.path.join(TEST_IMAGES, img_name)
            label_path = os.path.join(TEST_LABELS, img_name.replace(img_name.split('.')[-1], 'txt'))
            
            img = Image.open(img_path).convert('RGB')
            
            bboxes, category_ids = get_ground_truth(label_path, img.size)

            img_predictions = [pred for pred in predictions if pred['image_id'] == image_id]
            
            precision, recall, _, _, _ = calculate_precision_recall(img_predictions, 
                                                                {'bboxes': bboxes, 'category_ids': category_ids}, 
                                                                img.size)
            if precision > 0 or recall > 0:
                total_precision += precision
                total_recall += recall
                valid_images += 1

            images_to_plot.append({
                'image': img,
                'img_path': img_path,
                'ground_truth': {
                    'bboxes': bboxes,
                    'category_ids': category_ids
                },
                'predictions': img_predictions
            })

    if not images_to_plot:
        print("Không tìm thấy ảnh nào trong predictions.json khớp với tập test.")
        return

    avg_precision = total_precision / valid_images if valid_images > 0 else 0.0
    avg_recall = total_recall / valid_images if valid_images > 0 else 0.0
    print(f"Đã chọn {len(images_to_plot)} ảnh để hiển thị.")
    print(f"Precision trung bình: {avg_precision:.4f}")
    print(f"Recall trung bình: {avg_recall:.4f}")

    plot_images_with_boxes(images_to_plot, save_dir=SAVE_DIR)
    print(f"Đã lưu {len(images_to_plot)} ảnh tại {SAVE_DIR}")

if __name__ == "__main__":
    main()