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
print("hello2")
ds = load_dataset("laion/laion2B-en-aesthetic", split="train")

N_EXAMPLES = 1000
BATCH_SIZE = 50
print("hello3")
for _ in range(N_EXAMPLES // BATCH_SIZE):
    head = ds.take(BATCH_SIZE)
    print(head)
    break
    