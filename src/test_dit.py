import sys
import os
from pathlib import Path

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

# config
GENERATION_STEPS = 30
MODEL_PATH = "../saved_models/dit/upbeat-feather-158/"
TIMESHIFT_ALPHA = 4.63
ddp=True

from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

from transformers import T5EncoderModel, AutoTokenizer, CLIPTextModelWithProjection, AutoImageProcessor, AutoModel

PROJECT_DIR = Path(__file__).resolve().parents[1]
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "1").lower() not in {"0", "false", "no", "off"}


def log(message: str) -> None:
    print(message, flush=True)


def load_hf_model(model_cls, model_id: str, name: str, **kwargs):
    source = "local Hugging Face cache" if HF_LOCAL_FILES_ONLY else "Hugging Face"
    log(f"loading {name} from {source}")
    try:
        model = model_cls.from_pretrained(
            model_id,
            local_files_only=HF_LOCAL_FILES_ONLY,
            **kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {name} ({model_id}) from {source}. "
            "This script defaults to HF_LOCAL_FILES_ONLY=1 so Transformers does not "
            "probe the network or start its safetensors conversion thread during startup. "
            "Set HF_LOCAL_FILES_ONLY=0 if you need to download missing model files."
        ) from exc
    return model.to(device).eval()


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


log("loading t5")
encoder = load_hf_model(
    T5EncoderModel,
    "google/t5-v1_1-small",
    "t5 encoder",
    use_safetensors=False,
)

log("loading t5 tokenizer")
t5_tokenizer = load_hf_asset(AutoTokenizer, "google/t5-v1_1-small", "t5 tokenizer")

log("loading clip")
clip_encoder = load_hf_model(
    CLIPTextModelWithProjection,
    "openai/clip-vit-large-patch14",
    "clip text encoder",
)
log("loading clip tokenizer")
clip_tokenizer = load_hf_asset(AutoTokenizer, "openai/clip-vit-large-patch14", "clip tokenizer")

log("encoder models setup")

import matplotlib.pyplot as plt
from torchvision import transforms

from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

from model import DiT
from autoencoder import Autoencoder
from discriminator import Discriminator


to_tensor = ToTensor()
to_image = ToPILImage()

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

img_size = 256
n_channels = 3


# VAE
latent_channels = (128, 256, 512, 512, 512)
z_channels = 32
kernel_size = 3
padding = 1
resnet_blocks_per_layer = 2
resnet_kernel_size = 3
resnet_stride = 1
resnet_padding = 1

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
ae.load_state_dict({k.replace("orig_mod." + ("module." if ddp else ""), ""): v for k, v in torch.load(os.path.join(MODEL_PATH, "ae.pth")).items()})

latent_size = img_size // 2 ** (len(ae.encoder.channels) - 1)

print("autoencoder created")

# DiT
d_model = 1152
n_heads = 36
n_layers_multi_stream = 14
n_layers_single_stream = 14
patch_size = 2
n_timesteps = 100
repa_layer=2

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
dit.load_state_dict({k.replace("_orig_mod." + ("module." if ddp else ""), ""): v for k, v in torch.load(os.path.join(MODEL_PATH, "dit_ema_gamma_16.97.pth")).items()})



# pre_bn = nn.BatchNorm2d(z_channels, affine=False).to(device)
# pre_bn.load_state_dict(torch.load(os.path.join(MODEL_PATH, "bn.pth")))
# print(pre_bn.state_dict())
# pre_bn.eval()

with torch.no_grad():
    while (prompt := input("Enter a prompt: ")).lower() not in {"q", "quit"}:
        tokens = t5_tokenizer.encode(prompt, return_tensors="pt").to(device)
        clip_tokens = clip_tokenizer.encode(prompt, return_tensors="pt").to(device)

        latent = torch.randn(1, z_channels, latent_size, latent_size).to(device) # ae.encode(to_tensor(Image.open("test_img2.png").convert("RGB").resize((img_size, img_size))).unsqueeze(0).to(device))[0]
        # latent = pre_bn(latent)

        for timestep in tqdm(range(GENERATION_STEPS + 1)):
            t = (TIMESHIFT_ALPHA * (timestep / GENERATION_STEPS)) / (1 + (TIMESHIFT_ALPHA - 1) * (timestep / GENERATION_STEPS))
            
            dit_out = dit(latent, tokens, clip_tokens, torch.tensor([t * (dit.n_timesteps - 1)]).long().view(-1, 1, 1, 1).to(device))
            latent = latent + 5 * dit_out / GENERATION_STEPS

        decoded = ae.decode(latent)

        to_image(decoded.view(3, img_size, img_size).detach().cpu()).save("output.png")