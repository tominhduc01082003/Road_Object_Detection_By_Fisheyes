import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image
import os
import json
import random
import time
from tqdm import tqdm

def main():
    try:
        # ----------------------
        # ⚡ Device & Paths
        # ----------------------
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"⚡ Using device: {device}")

        DATA_DIR = "..\\Datasets\\Fisheye8K_all_including_train&test"
        TRAIN_DIR = os.path.join(DATA_DIR, 'train')
        TEST_DIR = os.path.join(DATA_DIR, 'test')
        TRAIN_JSON = os.path.join(TRAIN_DIR, 'train.json')
        TEST_JSON = os.path.join(TEST_DIR, 'test.json')
        TRAIN_IMAGES = os.path.join(TRAIN_DIR, 'images')
        TEST_IMAGES = os.path.join(TEST_DIR, 'images')

        # ----------------------
        # 📌 TRANSFORMS
        # ----------------------
        class ToTensor:
            def __call__(self, image, target):
                return F.to_tensor(image), target

        class RandomHorizontalFlip:
            def __init__(self, prob=0.5):
                self.prob = prob
            def __call__(self, image, target):
                if random.random() < self.prob:
                    image = F.hflip(image)
                    w = image.shape[-1]
                    boxes = target["boxes"]
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                    target["boxes"] = boxes
                return image, target

        class Normalize:
            def __init__(self, mean, std):
                self.mean = mean
                self.std = std
            def __call__(self, image, target):
                return F.normalize(image, mean=self.mean, std=self.std), target

        class Resize:
            def __init__(self, size):
                self.size = size if isinstance(size, tuple) else (size, size)
            def __call__(self, image, target):
                h0, w0 = image.shape[-2], image.shape[-1]
                image = F.resize(image, self.size)
                h1, w1 = image.shape[-2], image.shape[-1]
                scale_x = w1 / w0
                scale_y = h1 / h0
                boxes = target["boxes"]
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                target["boxes"] = boxes
                return image, target

        class Compose:
            def __init__(self, transforms):
                self.transforms = transforms
            def __call__(self, image, target):
                for t in self.transforms:
                    image, target = t(image, target)
                return image, target

        def get_train_transform(size=1280):
            return Compose([
                ToTensor(),
                RandomHorizontalFlip(0.5),
                Resize(size),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        def get_test_transform(size=1280):
            return Compose([
                ToTensor(),
                Resize(size),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # ----------------------
        # 📌 COCO DATASET
        # ----------------------
        class COCODataset(Dataset):
            def __init__(self, json_file, img_dir, transforms=None):
                with open(json_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                self.img_dir = img_dir
                self.transforms = transforms
                self.images = {img["id"]: img for img in self.data["images"]}
                self.image_ids = list(self.images.keys())
                self.annotations = {}
                for ann in self.data["annotations"]:
                    self.annotations.setdefault(ann["image_id"], []).append(ann)

            def __len__(self):
                return len(self.image_ids)

            def __getitem__(self, idx):
                img_id = self.image_ids[idx]
                info = self.images[img_id]
                img_path = os.path.join(self.img_dir, info["file_name"])
                img = Image.open(img_path).convert("RGB")
                anns = self.annotations.get(img_id, [])
                boxes, labels = [], []
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann["category_id"] + 1)
                if len(boxes) == 0:
                    boxes = [[0,0,1,1]]
                    labels = [0]
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}
                if self.transforms:
                    img, target = self.transforms(img, target)
                return img, target

        def collate_fn(batch):
            return tuple(zip(*batch))

        # ----------------------
        # 📌 MODEL
        # ----------------------
        def get_faster_rcnn_model(num_classes):
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            return model

        # ----------------------
        # 📌 TRAIN & EVAL
        # ----------------------
        def train_model(model, dataloader, optimizer, lr_scheduler, device, num_epochs=10):
            model.to(device)
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
                for images, targets in pbar:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    epoch_loss += losses.item()
                    pbar.set_postfix({"loss": f"{losses.item():.4f}"})
                lr_scheduler.step()
                print(f"✅ Epoch {epoch+1} trung bình: {epoch_loss/len(dataloader):.4f}")
            return model

        def evaluate_model(model, dataloader, device):
            model.eval()
            results = []
            with torch.no_grad():
                for images, targets in tqdm(dataloader, desc="Evaluating"):
                    images = [img.to(device) for img in images]
                    outputs = model(images)
                    for i, output in enumerate(outputs):
                        boxes = output["boxes"].cpu().numpy()
                        scores = output["scores"].cpu().numpy()
                        labels = output["labels"].cpu().numpy()
                        img_id = targets[i]["image_id"].item()
                        for box, score, label in zip(boxes, scores, labels):
                            if score > 0.5 and label > 0:
                                x1, y1, x2, y2 = box
                                results.append({
                                    "image_id": img_id,
                                    "category_id": label-1,
                                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                                    "score": float(score)
                                })
            print(f"✅ Tổng số dự đoán: {len(results)}")
            return results

        # ----------------------
        # 📌 LOAD DATASET
        # ----------------------
        print("📂 Tạo dataset train...")
        train_dataset = COCODataset(TRAIN_JSON, TRAIN_IMAGES, transforms=get_train_transform(1280))
        train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=collate_fn, num_workers=0)

        print("📂 Tạo dataset test...")
        test_dataset = COCODataset(TEST_JSON, TEST_IMAGES, transforms=get_test_transform(1280))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

        # ----------------------
        # 📌 MODEL, OPTIMIZER, SCHEDULER
        # ----------------------
        with open(TRAIN_JSON, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        num_classes = max(cat["id"] for cat in train_data["categories"]) + 2
        print(f"🔢 Số lớp (bao gồm background): {num_classes}")

        model = get_faster_rcnn_model(num_classes)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        # ----------------------
        # 📌 TRAIN MODEL
        # ----------------------
        print("🚀 Bắt đầu huấn luyện...")
        model = train_model(model, train_loader, optimizer, lr_scheduler, device, num_epochs=30)

        # ----------------------
        # 📌 LƯU MODEL
        # ----------------------
        os.makedirs("..\\Models_save", exist_ok=True)
        torch.save(model.state_dict(), "..\\Models_save\\faster_rcnn_final.pth")
        print("💾 Model đã lưu tại ..\\Models_save\\faster_rcnn_final.pth")

        # ----------------------
        # 📌 EVALUATE MODEL
        # ----------------------
        results = evaluate_model(model, test_loader, device)
        os.makedirs("..\\Predict", exist_ok=True)
        with open("..\\Predict\\predictions.json", "w") as f:
            json.dump(results, f)
        print("💾 Dự đoán đã lưu tại ..\\Predict\\predictions.json")

    except Exception as e:
        print(f"❌ Lỗi trong main(): {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
