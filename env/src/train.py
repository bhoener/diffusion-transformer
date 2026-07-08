import sys
import os
from pathlib import Path

sys.path.append("src/")

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._functorch.config

import  time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
# maybe remove
torch._functorch.config.donated_buffer = False

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

import datasets
from datasets import load_dataset

import matplotlib.pyplot as plt
from torchvision import transforms

from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

from src.model import DiT
from src.autoencoder import Autoencoder
from src.discriminator import Discriminator

import lpips


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

latent_size = img_size // 2 ** (len(ae.encoder.channels) - 1)

print("autoencoder created")

# DiT
d_model = 512
n_heads = 8
n_layers_multi_stream = 4
n_layers_single_stream = 4
patch_size = 8
n_timesteps = 100
repa_layer=2

# training config

batch_size = 4
grad_accum_steps=4
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
kl_loss_weight = 1.0

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

pre_bn = nn.BatchNorm2d(z_channels, affine=False).to(device)

optimizers = {
    "dit": [torch.optim.Muon([p for p in dit.parameters() if p.ndim == 2], lr=lr_dit_muon),
            torch.optim.AdamW([p for p in dit.parameters() if p.ndim != 2], lr=lr_dit_adamw)],
    "vae": [torch.optim.AdamW([p for p in ae.parameters()], lr=lr_ae)]
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

dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, pin_memory=False, drop_last=True)

iterator = iter(dl)
last_images = None

print("dataset created")

import wandb

run = wandb.init(project="FlowMatching", config={
    "autoencoder": {
        "latent_channels": latent_channels,
        "z_channels": z_channels,
        "kernel_size": kernel_size,
        "resnet_blocks_per_layer":resnet_blocks_per_layer,
        "resnet_kernel_size":resnet_kernel_size,
        "resnet_stride":resnet_stride,
        "resnet_padding":resnet_padding,
        "n_channels":n_channels,
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
    "lr_ae" : lr_ae,
    "lr_dit_muon" : lr_dit_muon,
    "lr_dit_adamw" : lr_dit_adamw,

    "repa_loss_weight" : repa_loss_weight,
    "reg_loss_weight" : reg_loss_weight,

    "mse_loss_weight" : mse_loss_weight,
    "lpips_loss_weight" : lpips_loss_weight,
    "kl_loss_weight" : kl_loss_weight,
    "img_size": img_size,
})

for step in range(num_steps + 1):
    # -------------------------------- training step -------------------------------
    for model_optims in optimizers.values():
        for optim in model_optims:
            optim.zero_grad()

    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        ts = torch.rand(batch_size, 1, 1, 1).to(device)

        try:
            #t0 = time.time()
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
            #print("time taken to load batch:", time.time() - t0)

        except StopIteration:
            iterator = iter(dl)
            continue

        #t0 = time.time()
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = dino_model(**dino_inputs)
            hidden_states = outputs.last_hidden_state
        #print("dino model runtime:", time.time() - t0)

        # Patch tokens represent specific regional/local features
        patch_tokens = hidden_states[:, 1 + dino_model.config.num_register_tokens:, :]

        B, N, D = patch_tokens.size()
        H = W = int(N**0.5)
        patch_tokens = patch_tokens.permute(0, 2, 1).contiguous().view(B, D, H, W)

        #t0 = time.time()
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            latent, mu, logvar = ae.encode(images)
        #print("encode runtime:", time.time() - t0)

        # weird magic for stopgrad
        # we detach the latent
        # then run the dit on it
        # then we get the grad for latent
        # and then do backward on the un-detached latent
        # and then can do backward normally
        latent_det = pre_bn(latent).detach()
        latent_det.requires_grad = True

        epsilon = torch.randn_like(latent)

        interpolated = ts * latent + (1 - ts) * epsilon

        #t0 = time.time()
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            dit_out, repa_out = dit(
                latent_det,
                tokens=t5_tokens,
                clip_tokens=clip_tokens,
                timesteps=torch.floor(ts.view(-1) * (dit.n_timesteps - 1)).long().to(device),
                repa_layer=repa_layer,
            )
        #print("dit runtime:", time.time() - t0)

        H_dit = W_dit = latent_size // patch_size

        repa_out = repa_out.permute(0, 2, 1).view(B, d_model, H_dit, W_dit)

        repa_out = F.adaptive_avg_pool2d(repa_out, (H, W))

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            projected = proj_conv(repa_out)

        projected = projected / (projected.norm() + 1e-7)
        patch_tokens = patch_tokens / (patch_tokens.norm() + 1e-7)

        loss_alignment = 1 - F.cosine_similarity(projected, patch_tokens, dim=-1).mean()  # max 2 if antiparallel, likely ~1
        loss_alignment = loss_alignment / grad_accum_steps

        loss_alignment_vae = repa_loss_weight * repa_weight_vae * loss_alignment
        loss_alignment_dit = repa_loss_weight * repa_weight_dit * loss_alignment

        loss_diffusion = ((dit_out - (latent_det - epsilon)) ** 2).mean()

        grad_latent = torch.autograd.grad(loss_alignment_vae, latent_det, retain_graph=True)[0]

        latent.backward(grad_latent, retain_graph=True)

        # regularization losses

        #t0 = time.time()
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            recon = ae.decode(latent)
        #print("decode runtime:", time.time() - t0)

        loss_mse = ((images - recon) ** 2).mean()

        loss_kl = - kl_loss_weight * (1 + logvar - mu** 2 - logvar.exp()).mean()
        loss_lpips = lpips_loss(images * 2 - 1, recon * 2 - 1).mean()

        loss_reg = mse_loss_weight * loss_mse + kl_loss_weight * loss_kl + lpips_loss_weight * loss_lpips

        # total loss

        loss = loss_diffusion + loss_alignment_dit + reg_loss_weight * loss_reg


        loss = loss / grad_accum_steps
        #t0 = time.time()
        loss.backward()
        #print("backward runtime:", time.time() - t0)

        loss_accum += loss.detach()
    norm_ae = torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
    norm_dit = torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)

    for model_optims in optimizers.values():
        for optim in model_optims:
            optim.step()

    if step % log_every == 0:
        print(
            f"step: {step:8d} | loss: {loss_accum.item():8.4f} | align: {loss_alignment.item():8.4f} | diff: {loss_diffusion.item():8.4f} | reg: {loss_reg.item():8.4f} | mse: {loss_mse.item():8.4f} | kl: {loss_kl.item():8.4f} | lpips: {loss_lpips.item():8.4f} | norm ae: {norm_ae.item():8.4f} | norm dit: {norm_dit.item():8.4f}")
    if step % save_every == 0:
        if not os.path.exists(f"../saved_models/dit/{run.name}"):
            os.makedirs(f"../saved_models/dit/{run.name}")

        torch.save(dit.state_dict(), f"../saved_models/dit/{run.name}/dit.pth")
        torch.save(ae.state_dict(), f"../saved_models/dit/{run.name}/ae.pth")
        torch.save(pre_bn.state_dict(), f"../saved_models/dit/{run.name}/bn.pth")
        for model_type, optims in optimizers.items():
            for optim in optims:
                torch.save(optim.state_dict(), f"../saved_models/dit/{run.name}/opt_{model_type}_{optim.__class__.__name__}.pth")


    wandb.log({"step": step,
               "loss/total": loss_accum.item(),
               "loss/alignment": loss_alignment.item(),
               "loss/diffusion": loss_diffusion.item(),
               "loss/reg/total": loss_reg.item(),
               "loss/reg/mse": loss_mse.item(),
               "loss/reg/kl": loss_kl.item(),
               "loss/reg/lpips": loss_lpips.item(),
               "norm_ae": norm_ae.item(),
               "norm_dit": norm_dit.item(),
               })

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
