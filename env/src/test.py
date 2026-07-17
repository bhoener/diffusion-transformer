import os
import sys
sys.path.append("src/")

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    CLIPTextModelWithProjection,
    T5EncoderModel,
)

HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}

def log(message: str) -> None:
    print(message, flush=True)


def load_hf_asset(asset_cls, model_id: str, name: str):
    source = "local Hugging Face cache" if HF_LOCAL_FILES_ONLY else "Hugging Face"
    log(f"loading {name} from {source}")
    try:
        return asset_cls.from_pretrained(model_id, local_files_only=HF_LOCAL_FILES_ONLY)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {name} ({model_id}) from {source}. "
            "Set HF_LOCAL_FILES_ONLY=0 if you need to download missing files."
        ) from exc


t5_tokenizer = load_hf_asset(AutoTokenizer, "google/t5-v1_1-small", "t5 tokenizer")

captions = [
    "the quick brown fox jumps over the lazy dog",
    "humpty dumpty sat on a wall",
    "testing123",
]

t5_out = t5_tokenizer(
    captions,
    padding=True,
    truncation=True,
    return_tensors="pt",
)
input_ids = t5_out.input_ids
attn_mask = t5_out.attention_mask
print(input_ids)
print(attn_mask)

img_tokens = 10

attn_mask_single_stream = torch.cat((torch.ones(attn_mask.size(0), img_tokens), attn_mask), dim=-1)

import matplotlib.pyplot as plt
plt.imshow(attn_mask_single_stream)
plt.show()