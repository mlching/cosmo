import os
import json
import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

with open("./textcap/TextCaps_0.1_train.json", "r") as f:
    cap = json.load(f)
    cap = cap["data"]
with open("./textocr/TextOCR_0.1_train.json", "r") as f:
    ocr = json.load(f)

ocr_imgs = list(ocr["imgs"].keys())

results = []

for image in cap:
    if image["image_id"] in ocr_imgs:
        frame = cv2.imread(f'./textcap/train_val_images/train_images/{image["image_id"]}.jpg')
        prediction = model(frame, conf=0.25)[0]
        data = prediction.boxes.data.tolist()
        objects = []
        texts = []
        for i in data:
            objects.append(f"{prediction.names[i[5]]}: {i[:4]}")
        for i in list(ocr["anns"].values()):
            if image["image_id"] == i["image_id"]:
                texts.append(f'{i["utf8_string"]}: {i["bbox"]}')
        results.append({"img_id": image["image_id"], "captions": image["reference_strs"], "objects": objects, "texts": texts})

with open("train_objects_0.25.json", "w") as f:
    json.dump(results, f)

"""""


# Find elements present in list1 but not in list2
diff1 = list(set(cap_imgs) - set(ocr_imgs))

# Find elements present in list2 but not in list1
diff2 = list(set(ocr_imgs) - set(cap_imgs))

print("Elements in list1 but not in list2:", diff1, len(diff1))
print("Elements in list2 but not in list1:", diff2)
"""""
