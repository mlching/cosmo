import os
import json
import cv2
from tqdm import tqdm
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

with open("./textcap/TextCaps_0.1_train.json", "r") as f:
    cap = json.load(f)
    cap = cap["data"]
with open("./textocr/TextOCR_0.1_train.json", "r") as f:
    ocr = json.load(f)

ocr_imgs = list(ocr["imgs"].keys())


print("cap data length", len(cap))
print("img length", len(ocr_imgs))



results = []

for img in tqdm(ocr_imgs, total=len(ocr_imgs)):
    captions = []
    for image in cap:
        if image["image_id"] == img:
            captions.append(image["caption_str"])
    frame = cv2.imread(f'./textcap/train_val_images/train_images/{img}.jpg')
    prediction = model(frame, conf=0.5)[0]
    data = prediction.boxes.data.tolist()
    objects = []
    texts = []
    for i in data:
        objects.append(f"{prediction.names[i[5]]}: {i[:4]}")
    for i in list(ocr["anns"].values()):
        if img == i["image_id"]:
            texts.append(f'{i["utf8_string"]}: {i["bbox"]}')
    results.append({"img_id": img, "captions": captions, "objects": objects, "texts": texts})

with open("train_objects_0.5.json", "w") as f:
    json.dump(results, f)

"""""


# Find elements present in list1 but not in list2
diff1 = list(set(cap_imgs) - set(ocr_imgs))

# Find elements present in list2 but not in list1
diff2 = list(set(ocr_imgs) - set(cap_imgs))

print("Elements in list1 but not in list2:", diff1, len(diff1))
print("Elements in list2 but not in list1:", diff2)
"""""
