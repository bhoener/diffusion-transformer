import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import DiT
from src.autoencoder import Autoencoder

img_size = 256
n_channels = 3
batch_size = 8
steps = 30000

cooldown_steps = 4000
lr = 3e-4


# VAE
latent_channels = (128, 256, 512, 512)
z_channels = 32
kernel_size = 3
padding = 1
resnet_blocks_per_layer = 2
resnet_kernel_size = 3
resnet_stride = 1
resnet_padding = 1

divergence_weight = 1e-4

# DiT
d_model = 768
n_heads = 12
n_layers_multi_stream = 6
n_layers_single_stream = 6
patch_size = 16

muon_lr = 1e-2
adam_lr = 3e-4




from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)
print("Image size:", image.height, image.width)  # [480, 640]

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m")
model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m", device_map="auto")

inputs = processor(images=image, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state

# 4. Parse the output tokens
# [CLS] token represents the global embedding for the entire image
cls_token = hidden_states[:, 0, :] 

# Patch tokens represent specific regional/local features
patch_tokens = hidden_states[:, 1 + model.config.num_register_tokens:, :]

print("Global Embedding Shape:", cls_token.shape)
print("Patch Tokens Shape:", patch_tokens.shape)

