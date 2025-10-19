import os
import json
import random
from ultralytics import YOLO


DATA_YAML = r"..\\Config\\Config_Hyper.yaml"
SAVE_DIR = r"..\\..\\..\\Models_save\\Yolov8_Fisheye\\Yolov8l"
# SAVE_DIR = r"..\\..\\..\\Models_save\\Yolov8_Fisheye\\Yolov8m"
TEST_IMG_DIR = r"..\\..\\..\\DataSets\\Fisheyes1K\\Fisheyes1K_Eval"
PRED_FILE = r"..\\..\\..\\Predict\\Yolov8\\Eval_Model_1K\\predictions.json"
VIS_DIR = r"..\\..\\..\\Predict\\Yolov8\\Pre_Vis"
os.makedirs(VIS_DIR, exist_ok=True)

def train_model():
    model = YOLO("yolov8l.pt")
    # model = YOLO("yolov8m.pt")
    model.train(
        data=DATA_YAML,
        epochs=20,
        imgsz=960,
        batch=2,
        lr0=0.0007,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        device=0,
        freeze=10,              
        mosaic=0.9,
        mixup=0.3,
        copy_paste=0.3,
        erasing=0.3,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.1,
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
        box=7.5,
        cls=3.0,
        dfl=1.5,
        patience=15,
        project=SAVE_DIR,
        name="stage1_freeze_v8l",
        save=True,
        save_period=5,
        plots=True,
        resume=False
    )

    best_stage1 = os.path.join(SAVE_DIR, "stage1_freeze_v8l", "weights", "best.pt")
    model = YOLO(best_stage1)
    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=[640, 1280],   
        batch=2,
        lr0=0.0005,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        momentum=0.937,
        warmup_epochs=3,
        device=0,
        mosaic=0.8,
        mixup=0.2,
        copy_paste=0.2,
        erasing=0.2,
        degrees=10.0,
        translate=0.1,
        scale=0.4,
        shear=3.0,
        perspective=0.0003,
        fliplr=0.5,
        flipud=0.05,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        box=7.5,
        cls=3.0,
        dfl=1.5,
        patience=40,
        project=SAVE_DIR,
        name="stage2_unfreeze_v8l",
        save=True,
        save_period=5,
        plots=True,
        resume=False
    )

    return model

sceneList = ['M', 'A', 'E', 'N']
def get_image_id(img_name):
    img_name = img_name.split('.png')[0]
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    return int(f"{cameraIndx}{sceneIndx}{frameIndx}")

def predict_on_1K(model_path, test_dir, out_file, vis_dir):
    model = YOLO(model_path)
    predictions = []

    all_imgs = [f for f in os.listdir(test_dir) if f.endswith(".png")]
    sample_imgs = random.sample(all_imgs, min(10, len(all_imgs)))

    for img_name in all_imgs:
        img_path = os.path.join(test_dir, img_name)

        results = model.predict(
            img_path,
            conf=0.01,    
            iou=0.5,
            augment=True,
            verbose=False
        )

        image_id = get_image_id(img_name)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w, h = x2 - x1, y2 - y1
                category_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                predictions.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, w, h],
                    "score": score
                })

        if img_name in sample_imgs:
            save_path = os.path.join(vis_dir, img_name)
            results[0].save(filename=save_path)

    with open(out_file, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Lưu file predictions ở {out_file}")
    print(f"lưu 10 ảnh ở {vis_dir}")

if __name__ == "__main__":
    trained_model = train_model()
    best_model_path = os.path.join(SAVE_DIR, "stage2_unfreeze_v8l", "weights", "best.pt")
    predict_on_1K(best_model_path, TEST_IMG_DIR, PRED_FILE, VIS_DIR)
