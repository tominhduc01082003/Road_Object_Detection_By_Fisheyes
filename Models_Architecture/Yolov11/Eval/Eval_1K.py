import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from torchvision.ops import nms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import warnings
import os
import cv2

warnings.filterwarnings("ignore")

def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId

def scale_bbox_to_original(bbox, orig_width, orig_height, padded_width, padded_height):
    """Chuyển bbox [xmin, ymin, width, height] từ không gian padded về kích thước ảnh gốc."""
    scale_x = orig_width / padded_width
    scale_y = orig_height / padded_height
    xmin, ymin, width, height = bbox
    return [xmin * scale_x, ymin * scale_y, width * scale_x, height * scale_y]

def get_test_transform(orig_width, orig_height):
    padded_width = ((orig_width + 31) // 32) * 32
    padded_height = ((orig_height + 31) // 32) * 32
    return A.Compose([
        A.PadIfNeeded(min_height=padded_height, min_width=padded_width, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_id'])), padded_width, padded_height

class YOLODataset:
    def __init__(self, img_dir, label_dir=None, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not self.img_names:
            raise FileNotFoundError(f"Không tìm thấy ảnh trong {img_dir}")
        self.label_names = [f.replace(f.split('.')[-1], 'txt') for f in self.img_names]
        self.original_sizes = {}
        
        for img_name in self.img_names:
            img_path = os.path.join(img_dir, img_name)
            try:
                img = Image.open(img_path)
                self.original_sizes[img_name] = img.size
            except Exception as e:
                print(f"Lỗi khi đọc kích thước ảnh {img_path}: {e}")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Lỗi khi đọc ảnh {img_path}: {e}")
            return None, None, None, None

        bboxes = []
        category_ids = []
        if self.label_dir and os.path.exists(os.path.join(self.label_dir, self.label_names[idx])):
            try:
                with open(os.path.join(self.label_dir, self.label_names[idx]), 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x_center, y_center, w, h = map(float, parts)
                            bboxes.append([x_center, y_center, w, h])
                            category_ids.append(int(class_id))
            except Exception as e:
                print(f"Lỗi khi đọc nhãn {self.label_names[idx]}: {e}")

        orig_width, orig_height = self.original_sizes[self.img_names[idx]]
        transform, padded_width, padded_height = get_test_transform(orig_width, orig_height)

        if transform:
            try:
                transformed = transform(image=img, bboxes=bboxes, category_id=category_ids)
                img = transformed['image']
                bboxes = transformed['bboxes']
                category_ids = transformed['category_id']
            except Exception as e:
                print(f"Lỗi khi áp dụng transform cho ảnh {img_path}: {e}")
                return None, None, None, None

        target = []
        for bbox, cat_id in zip(bboxes, category_ids):
            x_center, y_center, w, h = bbox
            if w > 0 and h > 0 or cat_id == 0:
                target.append({
                    'bbox': [x_center, y_center, w, h],
                    'category_id': cat_id,
                    'image_id': get_image_Id(self.img_names[idx]),
                    'iscrowd': 0
                })

        if not target:
            target = [{'image_id': get_image_Id(self.img_names[idx]), 'bbox': [], 'category_id': 0, 'iscrowd': 0}]

        return img, target, orig_width, orig_height
    
def collate_fn(batch):
    images = [item[0] for item in batch if item[0] is not None]
    targets = [item[1] for item in batch if item[1] is not None]
    orig_sizes = [(item[2], item[3]) for item in batch if item[2] is not None]
    formatted_targets = []
    for target in targets:
        if not target or not any(t['bbox'] for t in target):
            formatted_targets.append({
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'image_id': torch.tensor([target[0]['image_id']], dtype=torch.int64),
                'area': torch.zeros(0, dtype=torch.float32),
                'iscrowd': torch.zeros(0, dtype=torch.int64)
            })
        else:
            boxes = []
            labels = []
            areas = []
            iscrowd = []
            image_id = target[0]['image_id']
            for t in target:
                bbox = t['bbox']
                if bbox:
                    x_center, y_center, w, h = bbox
                    x_min = x_center - w / 2
                    y_min = y_center - h / 2
                    x_max = x_center + w / 2
                    y_max = y_center + h / 2
                    if w > 0 and h > 0 or t['category_id'] == 0:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(t['category_id'])
                        areas.append(w * h)
                        iscrowd.append(t['iscrowd'])
            formatted_targets.append({
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor([image_id], dtype=torch.int64),
                'area': torch.as_tensor(areas, dtype=torch.float32),
                'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64)
            })
    return images, formatted_targets, orig_sizes

def save_predicted_images(images, outputs, img_names, orig_sizes, save_dir, max_images=10):
    os.makedirs(save_dir, exist_ok=True)
    for img, output, img_name, (orig_width, orig_height) in zip(images, outputs, img_names, orig_sizes):
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            if score > 0.00:
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(img_np, f"Class {label} {score:.2f}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        save_path = os.path.join(save_dir, f"pred_{img_name}")
        cv2.imwrite(save_path, img_np)
        tqdm.write(f"Debug: Lưu ảnh predict {save_path}, số box: {len(boxes)}")

def evaluate_model(model, data_loader, device, use_tta=False, save_dir=None):
    model.eval()
    predictions = []
    img_names = data_loader.dataset.img_names
    original_sizes = data_loader.dataset.original_sizes
    with torch.no_grad():
        for i, (images, targets, orig_sizes) in enumerate(tqdm(data_loader, desc="Đang đánh giá", dynamic_ncols=True)):
            images = [img.to(device) for img in images]
            orig_width, orig_height = orig_sizes[0] 
            padded_width = ((orig_width + 31) // 32) * 32
            padded_height = ((orig_height + 31) // 32) * 32
            model_imgsz = max(padded_width, padded_height)
            
            if use_tta:
                outputs = [tta_inference(model, img, device, model_imgsz) for img in images]
            else:
                outputs = model.predict(images, conf=0.2, iou=0.2, verbose=False, max_det=20, imgsz=model_imgsz)
                outputs = [{
                    'boxes': torch.tensor(out.boxes.xyxy),
                    'scores': torch.tensor(out.boxes.conf),
                    'labels': torch.tensor(out.boxes.cls, dtype=torch.int64)
                } for out in outputs]
            
            if save_dir and i < 10:
                save_predicted_images(images, outputs, img_names[i:i+1], [(orig_width, orig_height)], save_dir)
            
            for target, output, img_name in zip(targets, outputs, img_names[i:i+1]):
                image_id = target['image_id'].item()
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                orig_width, orig_height = original_sizes[img_name]
                tqdm.write(f"Debug: image_id {image_id}, số box: {len(boxes)}, kích thước gốc: {orig_width}x{orig_height}, model_imgsz: {model_imgsz}")
                
                for box, score, label in zip(boxes, scores, labels):
                    if score > 0.00:
                        xmin, ymin, xmax, ymax = box
                        bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]
                        bbox = scale_bbox_to_original(bbox, orig_width, orig_height, padded_width, padded_height)
                        predictions.append({
                            'image_id': int(image_id),
                            'category_id': int(label),
                            'bbox': bbox,
                            'score': float(score),
                        })

    return predictions


def tta_inference(model, image, device, model_imgsz, num_augs=3):
    outputs = []
    image = image.squeeze(0) if image.dim() == 4 else image
    image_np = image.permute(1, 2, 0).cpu().numpy() * 255.0
    image_np = image_np.astype(np.uint8)
    
    augs = [
        A.Compose([]),
        A.Compose([A.HorizontalFlip(p=1.0)]),
        A.Compose([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)]),
    ][:num_augs]
    
    for i, aug in enumerate(augs):
        aug_np = aug(image=image_np)['image']
        aug = ToTensorV2()(image=aug_np)['image'].float() / 255.0
        aug = aug.unsqueeze(0).to(device)
        pred = model.predict(aug, conf=0.2, iou=0.2, verbose=False, max_det=20, imgsz=model_imgsz)[0]
        boxes = pred.boxes.xyxy.clone()
        if i == 1:
            boxes[:, [0, 2]] = aug.shape[3] - boxes[:, [2, 0]]
        outputs.append({
            'boxes': boxes,
            'scores': torch.tensor(pred.boxes.conf),
            'labels': torch.tensor(pred.boxes.cls, dtype=torch.int64)
        })
    
    all_boxes = torch.cat([out['boxes'] for out in outputs])
    all_scores = torch.cat([out['scores'] for out in outputs])
    all_labels = torch.cat([out['labels'] for out in outputs])
    
    keep = nms(all_boxes, all_scores, iou_threshold=0.4)
    
    return {
        'boxes': all_boxes[keep],
        'scores': all_scores[keep],
        'labels': all_labels[keep]
    }

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Sử dụng device: {device}")

    ROOT_DIR = "..\\..\\..\\DataSets\\Fisheye8K_all_including_train&test"
    UNLABELED_IMAGES = "..\\..\\..\\DataSets\\Fisheyes1K\\Fisheyes1K_Eval"
    MODEL_PATH = "..\\..\\..\\Models_save\\Yolov11_Fisheye\\Yolov11x\\weights\\best.pt"
    SAVE_DIR = "..\\..\\..\\Predict\\Yolov11\\Eval_Model_1K"

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File mô hình best.pt không tồn tại: {MODEL_PATH}")
    if not os.path.exists(UNLABELED_IMAGES):
        raise FileNotFoundError(f"Thư mục ảnh không nhãn không tồn tại: {UNLABELED_IMAGES}")

    unlabeled_dataset = YOLODataset(UNLABELED_IMAGES, label_dir=None)
    
    if len(unlabeled_dataset) != 1000:
        print(f"Cảnh báo: Tập dữ liệu không nhãn có {len(unlabeled_dataset)} ảnh, không phải 1,000 ảnh.")
    
    sizes = set(unlabeled_dataset.original_sizes.values())
    print(f"Kích thước ảnh gốc: {sizes}")
    max_size = max(max(w, h) for w, h in sizes)
    print(f"Kích thước lớn nhất: {max_size}")

    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)

    model = YOLO(MODEL_PATH)
    model.to(device)
    print("Đánh giá trên tập không nhãn với TTA...")
    predictions = evaluate_model(model, unlabeled_loader, device, use_tta=True, save_dir=SAVE_DIR)

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, "predictions.json"), "w", encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"Predictions được lưu tại {os.path.join(SAVE_DIR, 'predictions.json')}")

if __name__ == "__main__":
    main()