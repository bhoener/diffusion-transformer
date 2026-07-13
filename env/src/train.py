import os
import sys
import time
from pathlib import Path

import datasets
import lpips
import matplotlib.pyplot as plt
import torch
import torch._functorch.config
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from PIL import Image
from src.autoencoder import Autoencoder
from src.discriminator import Discriminator
from src.ema import EMA
from src.model import DiT
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    CLIPTextModelWithProjection,
    T5EncoderModel,
)

sys.path.append("src/")

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
# maybe remove, makes torch.compile happy
torch._functorch.config.donated_buffer = False


PROJECT_DIR = Path(__file__).resolve().parents[1]
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}


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
clip_tokenizer = load_hf_asset(
    AutoTokenizer, "openai/clip-vit-large-patch14", "clip tokenizer"
)

log("encoder models setup")


to_tensor = ToTensor()
to_image = ToPILImage()

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

lpips_loss = lpips.LPIPS(net="vgg")
lpips_loss = lpips_loss.to(device)

print("lpips loss setup")

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

latent_size = img_size // 2 ** (len(ae.encoder.channels) - 1)

print("autoencoder created")

# DiT
d_model = 512
n_heads = 8
n_layers_multi_stream = 4
n_layers_single_stream = 4
patch_size = 8
n_timesteps = 100
repa_layer = 2

# training config

batch_size = 4
grad_accum_steps = 4
num_steps = 12500
lr_ae = 3e-4
lr_dit_muon = 1e-2
lr_dit_adamw = 3e-4

repa_loss_weight = 0.5
reg_loss_weight = 0.5

repa_weight_vae = 1.5
repa_weight_dit = 0.5

mse_loss_weight = 1.0
lpips_loss_weight = 1.0
kl_loss_weight = 1e-4

log_every = 10
save_every = 1000

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
dit = torch.compile(dit)

ema = EMA(dit)

dino_processor = load_hf_asset(
    AutoImageProcessor,
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "dino image processor",
)
dino_model = load_hf_model(
    AutoModel,
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dino model",
)

print("dino model setup")

proj_conv = nn.Conv2d(
    d_model, dino_model.config.hidden_size, kernel_size=3, stride=1, padding=1
).to(device)

nn.init.zeros_(proj_conv.weight.data)

pre_bn = nn.BatchNorm2d(z_channels, affine=False).to(device)

optimizers = {
    "dit": [
        torch.optim.Muon([p for p in dit.parameters() if p.ndim == 2], lr=lr_dit_muon),
        torch.optim.AdamW(
            [p for p in dit.parameters() if p.ndim != 2], lr=lr_dit_adamw
        ),
    ],
    "vae": [torch.optim.AdamW([p for p in ae.parameters()], lr=lr_ae)],
    "proj_conv": [torch.optim.AdamW(proj_conv.parameters(), lr=lr_ae)],
}

data_files = str(PROJECT_DIR / "data" / "filtered_ds" / "*.arrow")
ds = load_dataset("arrow", data_files=data_files, split="train")

transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.ToTensor(),
    ]
)


def transform_batch(examples):
    out = {"pixel_values": [], "caption": []}
    for img, json in zip(examples["jpg"], examples["json"]):
        if img is not None and json is not None:
            out["pixel_values"].append(transform(img))
            out["caption"].append(json["caption"])
    return out


ds = ds.with_transform(transform_batch)

dl = torch.utils.data.DataLoader(
    ds, batch_size=batch_size, pin_memory=False, drop_last=True
)

iterator = iter(dl)
last_images = None

print("dataset created")

run = wandb.init(
    project="FlowMatching",
    config={
        "autoencoder": {
            "latent_channels": latent_channels,
            "z_channels": z_channels,
            "kernel_size": kernel_size,
            "resnet_blocks_per_layer": resnet_blocks_per_layer,
            "resnet_kernel_size": resnet_kernel_size,
            "resnet_stride": resnet_stride,
            "resnet_padding": resnet_padding,
            "n_channels": n_channels,
        },
        "dit": {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers_multi_stream": n_layers_multi_stream,
            "n_layers_single_stream": n_layers_single_stream,
            "patch_size": patch_size,
            "n_timesteps": n_timesteps,
        },
        "steps": num_steps,
        "batch_size": batch_size,
        "lr_ae": lr_ae,
        "lr_dit_muon": lr_dit_muon,
        "lr_dit_adamw": lr_dit_adamw,
        "repa_loss_weight": repa_loss_weight,
        "reg_loss_weight": reg_loss_weight,
        "mse_loss_weight": mse_loss_weight,
        "lpips_loss_weight": lpips_loss_weight,
        "kl_loss_weight": kl_loss_weight,
        "img_size": img_size,
    },
)

for step in range(num_steps + 1):
    # -------------------------------- training step -------------------------------
    for model_optims in optimizers.values():
        for optim in model_optims:
            optim.zero_grad()

    # for tracking total loss
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        # sample timesteps - TODO: maybe not from uniform dist
        ts = torch.rand(batch_size, 1, 1, 1).to(device)

        try:
            # load data
            batch = next(iterator)
            images = batch["pixel_values"].to(device)
            last_images = images
            captions = batch["caption"]
            dino_inputs = dino_processor(images, return_tensors="pt").to(device)
            clip_tokens = clip_tokenizer(
                captions,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            t5_tokens = t5_tokenizer(
                captions,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

        except StopIteration:
            iterator = iter(dl)
            continue

        # get dino latent
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = dino_model(**dino_inputs)
            hidden_states = outputs.last_hidden_state

        # remove non-patch tokens
        patch_tokens = hidden_states[:, 1 + dino_model.config.num_register_tokens :, :]

        # reshape, assume H=W (square)
        B, N, D = patch_tokens.size()
        H = W = int(N**0.5)
        patch_tokens = patch_tokens.permute(0, 2, 1).contiguous().view(B, D, H, W)

        # get clean latent, mean and var, do batchnorm for repa-e
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            latent, mu, logvar = ae.encode(images)
            latent_bn = pre_bn(latent)

        # detach normed latent to go throuh dit (so diffusion loss doesn't leak into encoder)
        latent_det = latent_bn.detach()
        latent_det.requires_grad = True

        # sample noise for diffusion interpolation
        epsilon = torch.randn_like(latent)

        # lerp
        interpolated = ts * latent_det + (1 - ts) * epsilon

        # get dit prediction for latent, also take a hidden state for alignment loss
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            dit_out, repa_out = dit(
                interpolated,
                tokens=t5_tokens,
                clip_tokens=clip_tokens,
                timesteps=torch.floor(ts.view(-1) * (dit.n_timesteps - 1))
                .long()
                .to(device),
                repa_layer=repa_layer,
            )

        # reshape to B, C, H, W for iREPA conv projection
        H_dit = W_dit = latent_size // patch_size
        repa_out = repa_out.permute(0, 2, 1).view(B, d_model, H_dit, W_dit)
        # pool so that H, W match that of dino - TODO: maybe try pooling after projection conv?
        repa_out = F.adaptive_avg_pool2d(repa_out, (H, W))

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            projected = proj_conv(repa_out)

        # repa alignment loss
        loss_alignment = (
            1 - F.cosine_similarity(projected, patch_tokens, dim=1).mean()
        )  # max 2 if antiparallel, likely ~1

        # following repa-e, there are different weights for vae and dit
        loss_alignment_vae = repa_loss_weight * repa_weight_vae * loss_alignment
        loss_alignment_dit = repa_loss_weight * repa_weight_dit * loss_alignment

        # normal flow matching loss
        loss_diffusion = ((dit_out - (latent_det - epsilon)) ** 2).mean()

        # manually get grad for the non-detached latent from alignment to backprop to vae
        grad_latent = torch.autograd.grad(
            loss_alignment_vae, latent_det, retain_graph=True
        )[0]

        # reconstruct the output for vae training, reg losses
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            recon = ae.decode(latent)

        # reg losses
        loss_mse = ((images - recon) ** 2).mean()
        loss_kl = (logvar.exp() + mu**2 - 1 - logvar).mean()
        loss_lpips = lpips_loss(images * 2 - 1, recon * 2 - 1).mean()

        loss_reg = (
            mse_loss_weight * loss_mse
            + kl_loss_weight * loss_kl
            + lpips_loss_weight * loss_lpips
        )

        # weird trick to give non-detached latent the same grad as detached
        loss_repa = (latent_bn * grad_latent.detach()).sum()

        # total loss

        loss = (
            loss_diffusion + loss_alignment_dit + reg_loss_weight * loss_reg + loss_repa
        )

        loss = loss / grad_accum_steps

        loss.backward()

        loss_accum += loss.detach()

    norm_ae = torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
    norm_dit = torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)

    for model_optims in optimizers.values():
        for optim in model_optims:
            optim.step()

    ema.update()

    if step % log_every == 0:
        print(
            f"step: {step:8d} | loss: {loss_accum.item():8.4f} | align: {loss_alignment.item():8.4f} | diff: {loss_diffusion.item():8.4f} | reg: {loss_reg.item():8.4f} | mse: {loss_mse.item():8.4f} | kl: {loss_kl.item():8.4f} | lpips: {loss_lpips.item():8.4f} | norm ae: {norm_ae.item():8.4f} | norm dit: {norm_dit.item():8.4f}"
        )

    if (step % save_every == 0 and step > 0) or step == num_steps:
        root = f"../saved_models/dit/{run.name}"
        if not os.path.exists(root):
            os.makedirs(root)

        torch.save(dit.state_dict(), os.path.join(root, "dit.pth"))
        torch.save(ae.state_dict(), os.path.join(root, "ae.pth"))
        torch.save(pre_bn.state_dict(), os.path.join(root, "bn.pth"))
        for model_type, optims in optimizers.items():
            for optim in optims:
                torch.save(
                    optim.state_dict(),
                    os.path.join(
                        root, f"opt_{model_type}_{optim.__class__.__name__}.pth"
                    ),
                )
        ema.checkpoint(root, step)

    # log everything to wandb. everything other than total loss isn't accumulated because i don't want to do it
    wandb.log(
        {
            "step": step,
            "loss/total": loss_accum.item(),
            "loss/alignment": loss_alignment.item(),
            "loss/diffusion": loss_diffusion.item(),
            "loss/reg/total": loss_reg.item(),
            "loss/reg/mse": loss_mse.item(),
            "loss/reg/kl": loss_kl.item(),
            "loss/reg/lpips": loss_lpips.item(),
            "norm_ae": norm_ae.item(),
            "norm_dit": norm_dit.item(),
        }
    )

if last_images is not None:
    plt.imshow(
        ae(last_images[0].unsqueeze(0))[-1]
        .permute(0, 2, 3, 1)
        .view(img_size, img_size, 3)
        .detach()
        .cpu()
        .numpy()
    )
    plt.show()
