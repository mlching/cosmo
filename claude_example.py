import json
import os
import requests
from tqdm import tqdm

url = "https://xqtd520qidong.com/v1/chat/completions"

guide = """You are an AI visual assistant that can analyze a single image. You receive five sentences, each describing 
the same image you are observing. In addition, specific object and text locations within the image are given, along with detailed coordinates. 
These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. 
These values correspond to the top left x, top left y, bottom right x, and bottom right y.
The task is to use the provided caption and bounding box information, create a plausible question about the image, and provide the answer 
in detail in two dictionaries seperated by a comma in the format of {"from": "human", "value": (question)}, {"from": "gpt", "value": answer} only, 
your response should include two dictionaries only and no any sentence outside of the two dictionaries.
Create complex reasoning questions beyond describing the scene and do not include easy description questions like where the object/text is .
To answer such questions, one should require first understanding the visual content, then based on the background knowledge 
or reasoning, either explain why the things are happening that way, or provide guides and help to user's request without mentioning the data provided.  
Make the question challenging by not including the visual content details in the question so that the user needs to reason about that first.
Instead of directly mentioning the bounding box coordinates, utilize this data to explain the scene using natural language. 
Include details like object/texts counts, position of the objects/texts, relative position between the objects/texts.  
The task is to simulates a human user ask gpt about a question based on an image, where gpt has no captions or bounding boxes or text annotation information 
like we have provided to you, all gpt has is an image without any annotation, so do not mention those data when creating gpt's answer. 
The only data provided by the user is an image and gpt's answer is based entirely on that image. """

def question(guide, user_prompt):
    headers = {
        "Authorization": "Bearer sk-Ui7lbp28l9ThiddOEcF7FdD560A847F994C9FcDe5d7aC199",
        "content-type": "application/json"
    }
    data = {
        "messages": [
            {
                "role": "user",
                "content": guide + "<data>" + user_prompt + "</data>",

            }
        ],
        "model": "claude-2",
        "max_tokens_to_sample": 1000,
        "stream": True
    }
    #print(data)
    response = requests.post(url, headers=headers, json=data, stream=True)
    #print('Claude>>> ', end='')

    text = ''
    for message_chunk in response.iter_lines():
        message_chunk = message_chunk.decode('utf-8')
        if message_chunk.strip() == '':
            continue
        if message_chunk.strip() == 'data: [DONE]':
            continue
        try:
            message_json = json.loads(message_chunk[6:])  # Skip the 'data: ' prefix
            if "stop_reason" in message_json and message_json["stop_reason"] is not None:
                break
            message = message_json["choices"][0]["delta"]["content"]
            #print(message, end='', flush=True)
            text += message
        except:
            print("Can not write the message", message_chunk)
    return text
    print()
#print(question(guide, """["captions": ["A monitor with vNES in the top corner is mostly black but has white writing telling us the Wii will not be working on this system.", "Toshiba screen that says the letters VNES on it.", "a slide that has the word because on it", "the letters NES at the top of s creeen", "A black screen has vNES in the upper left corner."], "objects": ["tv: [33.5830078125, 7.982079982757568, 1022.7388916015625, 656.841552734375]"], "texts": ["TOSHIBA: [502.04, 593.93, 59.5, 11.8]", "WWW: [711.77, 519.44, 68.8, 25.0]", "\u00a1: [931.05, 507.49, 14.8, 23.4]", "Because: [356.72, 378.41, 77.65, 18.46]", "of: [439.07, 378.06, 19.85, 18.11]", "a: [461.71, 382.42, 13.58, 13.93]", "lack: [480.36, 377.88, 38.0, 17.7]", "of: [522.01, 378.1, 20.1, 17.69]", "the: [546.02, 377.99, 29.52, 18.49]", "Java: [579.26, 378.49, 42.6, 17.73]", "Standard: [625.19, 375.76, 79.27, 20.61]", "Edition: [709.92, 377.43, 58.96, 18.64]", "environment,: [773.73, 376.82, 109.59, 20.77]", "vNES: [388.87, 402.16, 50.78, 19.55]", "is: [443.85, 402.42, 16.07, 18.18]", "not: [464.92, 403.32, 30.47, 16.98]", "currently: [499.7, 402.72, 76.8, 21.8]", "supported: [579.9, 400.92, 88.0, 22.4]", "on: [672.65, 405.2, 26.39, 14.25]", "the: [701.95, 401.9, 29.95, 18.08]", "Nintendo: [737.04, 400.71, 76.8, 19.0]", "Wii.: [819.45, 400.56, 31.67, 19.4]", "vNES: [96.29, 107.4, 285.13, 76.0]", ".: [437.35, 3.89, 4.7, 5.3]", ".: [444.25, 3.59, 14.5, 6.4]", ".: [460.35, 3.69, 6.0, 5.6]", ".: [637.92, 4.44, 5.8, 5.9]", ".: [645.62, 4.34, 14.5, 6.0]", ".: [661.92, 4.64, 5.4, 5.8]", ".: [339.02, 729.37, 5.4, 6.5]", ".: [337.72, 736.97, 5.3, 5.0]", ".: [334.52, 742.17, 7.1, 6.6]", ".: [503.72, 713.91, 10.5, 12.0]", ".: [513.22, 714.21, 10.1, 10.8]", ".: [920.27, 0.02, 19.5, 5.0]", ".: [62.03, 38.03, 41.6, 10.5]"]"""))

with open ("val_objects_0.5.json", "r") as f:
    custom = json.load(f)

visual = []
for i in tqdm(range(len(custom)), total=len(custom)):
    prompt = "captions: " + repr(custom[i]["captions"]) + "objects: " + repr(custom[i]["objects"]) + "texts: " + repr(custom[i]["texts"])
    response = question(guide, prompt)
    #print(response)
    visual.append({"id": custom[i]["img_id"], "image": f'./textcap/train_val_images/train_images/{custom[i]["img_id"]}.jpg', "conversations": [response]})
    #print(visual)
    if (i + 1) % 50== 0:
        with open('visual.json', 'w') as f:
            json.dump(visual, f)
    #break

with open('visual.json', 'w') as f:
    json.dump(visual, f)