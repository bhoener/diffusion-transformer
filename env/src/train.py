import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5EncoderModel, AutoTokenizer, CLIPTextModelWithProjection
from transformers import AutoImageProcessor, AutoModel

from src.model import DiT
from src.autoencoder import Autoencoder

device = torch.device("cpu")
torch.set_float32_matmul_precision("high")

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

ae = Autoencoder(
    latent_channels=latent_channels,
    z_channels=z_channels,
    kernel_size=kernel_size,
    padding=padding,
    resnet_blocks_per_layer=resnet_blocks_per_layer,
    resnet_kernel_size=resnet_kernel_size,
    resnet_stride=resnet_stride,
    resnet_padding=resnet_padding,
    n_channels=n_channels,
)

ae = ae.to(device)

# DiT
d_model = 768
n_heads = 12
n_layers_multi_stream = 6
n_layers_single_stream = 6
patch_size = 2
n_timesteps = 100

muon_lr = 1e-2
adam_lr = 3e-4


encoder = T5EncoderModel.from_pretrained("google/t5-v1_1-small", device_map=device)
tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")

clip_encoder = CLIPTextModelWithProjection.from_pretrained(
    "openai/clip-vit-large-patch14", device_map=device
)
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")


dit = DiT(
    encoder_model=encoder,
    clip_encoder_model=clip_encoder,
    d_model=d_model,
    n_heads=n_heads,
    n_layers_multi_stream=n_layers_multi_stream,
    n_layers_single_stream=n_layers_single_stream,
    patch_size=patch_size,
    w=img_size // 2**(len(ae.encoder.channels) - 1),
    h=img_size // 2**(len(ae.encoder.channels) - 1),
    n_timesteps=n_timesteps,
    n_channels=z_channels,
)

dit = dit.to(device)

from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url).resize((img_size, img_size))
print("Image size:", image.height, image.width)  # [480, 640]

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov3-vith16plus-pretrain-lvd1689m"
)
model = AutoModel.from_pretrained(
    "facebook/dinov3-vits16-pretrain-lvd1689m", device_map="auto"
)

inputs = processor(images=image, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state

# 4. Parse the output tokens
# [CLS] token represents the global embedding for the entire image
cls_token = hidden_states[:, 0, :]

# Patch tokens represent specific regional/local features
patch_tokens = hidden_states[:, 1 + model.config.num_register_tokens :, :]

print("Global Embedding Shape:", cls_token.shape)
print("Patch Tokens Shape:", patch_tokens.shape)

patch_size = model.config.patch_size
print("Patch size:", patch_size)

B, N, D = patch_tokens.size()
H = W = int(N**0.5)  # h/w in patches
patch_tokens = patch_tokens.permute(0, 2, 1).contiguous().view(D, H, W)
print(patch_tokens.size())

import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

to_tensor = ToTensor()

img_x = patch_tokens.permute(1, 2, 0)[:, :, :3]
print(img_x.size())
plt.imshow(img_x.detach().cpu().numpy())
plt.show()

proj_conv = nn.Conv2d(d_model, model.config.hidden_size, kernel_size=3, stride=1, padding=1)

latent, _, _ = ae.encode(to_tensor(image).to(device).unsqueeze(0))

print(latent.size())

dit_out = dit(latent, torch.randint(0, 32128, (1, 16)).to(device), torch.randint(0, 32128, (1, 32)).to(device), torch.randint(0, 100,  (1, 1)).to(device))

print(dit_out.size())