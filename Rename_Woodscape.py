import os
import random
from PIL import Image

woodscape_dir = r"DataSets\Fisheye8K_all_including_train&test\Woodscape\images"
train_test_dir = r"DataSets\Fisheye8K_all_including_train&test\train_test\images"
output_dir = os.path.join(os.path.dirname(woodscape_dir), "images_converted")

scene_list = ['M', 'A', 'E', 'N']
camera_index = 30        
frame_index = 1
max_frames_per_camera = 400

os.makedirs(output_dir, exist_ok=True)

def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId

existing_names = {f.lower() for f in os.listdir(train_test_dir) if f.lower().endswith(".png")}
existing_names.update({f.lower() for f in os.listdir(output_dir) if f.lower().endswith(".png")})

used_names = set(existing_names)
used_ids = set()

image_files = [f for f in os.listdir(woodscape_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
image_files.sort()

renamed_count = 0

print(f"Đang xử lý {len(image_files)} ảnh từ Woodscape...")

for file in image_files:
    old_path = os.path.join(woodscape_dir, file)

    try:
        img = Image.open(old_path).convert("RGB")
    except Exception as e:
        print(f"Không đọc được ảnh {file}: {e}")
        continue

    while True:
        if frame_index > max_frames_per_camera:
            camera_index += 1
            frame_index = 1

        scene_letter = random.choice(scene_list)
        new_name = f"camera{camera_index}_{scene_letter}_{frame_index}.png"
        new_id = get_image_Id(new_name)

        if new_name.lower() not in used_names and new_id not in used_ids:
            break
        else:
            frame_index += 1

    new_path = os.path.join(output_dir, new_name)
    img.save(new_path, format="PNG")

    used_names.add(new_name.lower())
    used_ids.add(new_id)
    renamed_count += 1
    frame_index += 1

    print(f"Đổi: {file} → {new_name} (ID={new_id})")

print(f"\nHoàn tất! Đã đổi và lưu {renamed_count} ảnh vào: {output_dir}")
print(f"Tổng số camera đã dùng: {camera_index - 29}")
