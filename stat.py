import json
import statistics
from pycocotools.coco import COCO

with open('train_objects.json', 'r') as f:
    text = json.load(f)


text_stat = []
for i in text:
    text_stat.append(len(i['objects']))
print("Text mean:", statistics.mean(text_stat))
print("Text mode:", statistics.mode(text_stat))
print("Text variance:", statistics.variance(text_stat))

annotation_file = './coco2014/instances_train2014.json'
coco = COCO(annotation_file)
image_ids = coco.getImgIds()

coco_stat = []
for image_id in image_ids:
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
    num_objects = len(annotations)
    coco_stat.append(num_objects)
print("Coco mean:", statistics.mean(coco_stat))
print("Coco mode:", statistics.mode(coco_stat))
print("Coco variance:", statistics.variance(coco_stat))