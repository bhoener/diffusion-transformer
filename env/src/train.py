import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5EncoderModel, AutoTokenizer, CLIPTextModelWithProjection
from transformers import AutoImageProcessor, AutoModel

import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

from src.model import DiT
from src.autoencoder import Autoencoder

to_tensor = ToTensor()
to_image = ToPILImage()

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


# training config

batch_size = 8
num_steps = 1000
lr_ae = 3e-4
lr_dit = 1e-2


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
    w=img_size // 2 ** (len(ae.encoder.channels) - 1),
    h=img_size // 2 ** (len(ae.encoder.channels) - 1),
    n_timesteps=n_timesteps,
    n_channels=z_channels,
)

dit = dit.to(device)

image = to_tensor(Image.open("src/test_img2.png").convert("RGB").resize((img_size, img_size))).to(device)

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov3-vith16plus-pretrain-lvd1689m"
)
model = AutoModel.from_pretrained(
    "facebook/dinov3-vits16-pretrain-lvd1689m", device_map=device
)


proj_conv = nn.Conv2d(
    d_model, model.config.hidden_size, kernel_size=3, stride=1, padding=1
)

pre_bn = nn.BatchNorm2d(z_channels)

# -------------------------------- training step -------------------------------

ts = torch.rand(batch_size, 1)

print(image.size())
inputs = processor(image, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state

# Patch tokens represent specific regional/local features
patch_tokens = hidden_states[:, 1 + model.config.num_register_tokens :, :]

print("Patch Tokens Shape:", patch_tokens.shape)

patch_size = model.config.patch_size
print("Patch size:", patch_size)

B, N, D = patch_tokens.size()
H = W = int(N**0.5)  # h/w in patches
patch_tokens = patch_tokens.permute(0, 2, 1).contiguous().view(B, D, H, W)
print(patch_tokens.size())

latent, _, _ = ae.encode(image.unsqueeze(0))

# weird magic for stopgrad
# we detach the latent
# then run the dit on it
# then we get the grad for latent
# and then do backward on the un-detached latent
# and then can do backward normally
latent_det = latent.detach() 
latent_det.requires_grad = True

epsilon = torch.randn_like(latent)

interpolated = ts * latent + (1-ts) * epsilon

print(latent.size())

dit_out, repa_out = dit(
    latent_det,
    torch.randint(0, 32128, (1, 16)).to(device),
    torch.randint(0, 32128, (1, 32)).to(device),
    torch.randint(0, 100, (1, 1)).to(device),
    repa_layer=4,
)

N_dit = repa_out.size(1)

H_dit = W_dit = img_size // patch_size

print(repa_out.size())

repa_out = repa_out.permute(0, 2, 1).view(B, d_model, H_dit, W_dit)

print(patch_tokens.size(), repa_out.size())

repa_out = F.adaptive_avg_pool2d(repa_out, (H, W))

print(repa_out.size())

projected = proj_conv(repa_out)

projected = projected / (projected.norm() + 1e-7)
patch_tokens = patch_tokens / (patch_tokens.norm() + 1e-7)

alignment_loss = 1 - F.cosine_similarity(projected, patch_tokens, dim=-1).mean() # max 2 if antiparallel, likely ~1

print("alignment loss:", alignment_loss) 

diffusion_loss = ((dit_out - (latent_det - epsilon)) ** 2).mean()

