import json
import tiktoken
from tqdm import tqdm
import statistics

with open("llava_v1_5_mix665k.json", "r") as f:
    data = json.load(f)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

number = []
for i in tqdm(data, total=len(data)):
    try:
        if i["image"].split("/")[0] == "coco":
            number.append(num_tokens_from_string(repr(i["conversations"]), "gpt-4"))
    except KeyError:
        continue

print("total number of conversations:", len(number))
print("LLaVA mean:", statistics.mean(number))
print("LLaVA median:", statistics.median(number))
print("LLaVA mode:", statistics.mode(number))
print("LLaVA variance:", statistics.variance(number))
