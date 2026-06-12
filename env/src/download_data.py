import os
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import requests

def download_single_image(url: str) -> np.ndarray:
    pass
    
if not os.path.exists("data/"):
    os.makedirs("data/")
    os.makedirs("data/images")
    os.makedirs("data/captions")
    

print("hello")
tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
print("hello2")
ds = load_dataset("stanford-vision-lab/gpic", split="val", data_files={"val": "hf://datasets/stanford-vision-lab/gpic/val/*.tar"})

N_EXAMPLES = 1000
BATCH_SIZE = 50
print("hello3")
for _ in range(N_EXAMPLES // BATCH_SIZE):
    head = ds.take(BATCH_SIZE)
    print(head)
    break
    