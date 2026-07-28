import contextlib
import os
import time

import lpips
import torch
import torch._functorch.config

# ddp
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    CLIPTextModelWithProjection,
    T5EncoderModel,
)

import wandb
from autoencoder import Autoencoder

#from discriminator import Discriminator
from ema import EMA
from model import DiT


def main() -> None:
    DO_BENCHMARK = False

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = ".inductor_cache"
    # for h100
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)

    def ddp_setup() -> int:
        ddp_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl")
        return ddp_rank

    def ddp_cleanup() -> None:
        dist.destroy_process_group()

    ddp = "LOCAL_RANK" in os.environ

    world_size = int(os.environ["WORLD_SIZE"])
    torch.set_num_threads(world_size)

    torch.set_num_interop_threads(world_size)
    torch.multiprocessing.set_start_method("spawn", force=True)

    if ddp:
        ddp_rank = ddp_setup()

        ddp_rank = int(os.environ["LOCAL_RANK"])
        if ddp_rank == 0:
            print(f"ddp set up | world size: {world_size}")

        import logging

        logging.getLogger().setLevel(logging.CRITICAL)

    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

    device = torch.device(
        f"cuda:{ddp_rank if ddp else 0}" if torch.cuda.is_available() else "cpu"
    )
    torch.cuda.set_device(device)
    print("hello from", device)
    torch.set_float32_matmul_precision("high")
    # maybe remove, makes torch.compile happy
    torch._functorch.config.donated_buffer = False

    HF_LOCAL_FILES_ONLY = False

    def log(*args) -> None:
        if ddp and ddp_rank != 0:
            return
        print(*args, flush=True)

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
            return asset_cls.from_pretrained(
                model_id, local_files_only=HF_LOCAL_FILES_ONLY
            )
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

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    lpips_loss = lpips.LPIPS(net="vgg")
    lpips_loss = lpips_loss.to(device)

    log("lpips loss setup")

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
    if ddp:
        ae = DDP(ae, device_ids=[ddp_rank])
    ae = torch.compile(ae)

    latent_size = img_size // 2 ** (
        len(ae.module.encoder.channels if ddp else ae.encoder.channels) - 1
    )

    log("autoencoder created")

    # DiT
    d_model = 768
    n_heads = 12
    n_layers_multi_stream = 6
    n_layers_single_stream = 6
    patch_size = 8
    n_timesteps = 100
    repa_layer = 2

    # training config

    batch_size = 32  # prob change to 32 for h100
    grad_accum_steps = 1
    num_steps = 50000
    cooldown_frac = 0.1
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

    pad_to_seqlen = 64

    dit = DiT(
        encoder_model=encoder,
        clip_encoder_model=clip_encoder,
        d_model=d_model,
        n_heads=n_heads,
        n_layers_multi_stream=n_layers_multi_stream,
        n_layers_single_stream=n_layers_single_stream,
        patch_size=patch_size,
        w=img_size
        // 2 ** (len(ae.module.encoder.channels if ddp else ae.encoder.channels) - 1),
        h=img_size
        // 2 ** (len(ae.module.encoder.channels if ddp else ae.encoder.channels) - 1),
        n_timesteps=n_timesteps,
        n_channels=z_channels,
    )

    dit = dit.to(device)
    if ddp:
        dit = DDP(dit, device_ids=[ddp_rank])
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
    dino_model = torch.compile(dino_model)

    log("dino model setup")

    proj_conv = nn.Conv2d(
        d_model, dino_model.config.hidden_size, kernel_size=3, stride=1, padding=1
    ).to(device)
    nn.init.zeros_(proj_conv.weight.data)
    if ddp:
        proj_conv = DDP(proj_conv, device_ids=[ddp_rank])

    pre_bn = nn.BatchNorm2d(z_channels, affine=False).to(device)

    optimizers = {
        "dit": [
            torch.optim.Muon(
                [p for p in dit.parameters() if p.ndim == 2], lr=lr_dit_muon
            ),
            torch.optim.AdamW(
                [p for p in dit.parameters() if p.ndim != 2],
                lr=lr_dit_adamw,
                fused=True,
            ),
        ],
        "vae": [torch.optim.AdamW(ae.parameters(), lr=lr_ae, fused=True)],
        "proj_conv": [torch.optim.AdamW(proj_conv.parameters(), lr=lr_ae, fused=True)],
    }

    # TODO: use streaming? 200k images might not be enough
    ds = load_dataset("arrow", data_files="data/filtered_ds/*.arrow", split="train")

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
        ]
    )

    def transform_batch(examples):
        out = {
            "pixel_values": [],
            "t5_tokens": [],
            "t5_attn_mask": [],
            "clip_tokens": [],
        }
        for img, json in zip(examples["jpg"], examples["json"]):
            if img is not None and json is not None:
                out["pixel_values"].append(transform(img))
                t5_out = t5_tokenizer(
                    json["caption"],
                    padding="max_length",
                    max_length=pad_to_seqlen,
                    truncation=True,
                    return_tensors="pt",
                )

                out["t5_tokens"].append(t5_out.input_ids)
                out["t5_attn_mask"].append(t5_out.attention_mask.bool())
                out["clip_tokens"].append(
                    clip_tokenizer(
                        json["caption"],
                        padding="max_length",
                        max_length=pad_to_seqlen,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids
                )
        return out

    ds = ds.map(
        transform_batch,
        batched=True,
        remove_columns=[
            col
            for col in ds.column_names
            if col not in {"pixel_values", "t5_tokens", "t5_attn_mask", "clip_tokens"}
        ],
    )
    ds = ds.with_format("torch")

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(ds) if ddp else None,
        persistent_workers=True,
        num_workers=world_size,
        prefetch_factor=2,
        in_order=False,
    )

    iterator = iter(dl)

    log("dataset created")

    if not ddp or ddp_rank == 0:
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

    def get_lr(it: int, lr_base: float) -> float:
        if 1 - (it / num_steps) < cooldown_frac:
            cooldown_prog = 1 - (num_steps - it) / (num_steps * cooldown_frac)
            return (1 - cooldown_prog) * lr_base
        else:
            return lr_base

    def load_data() -> tuple[torch.Tensor, ...]:
        batch = next(iterator)
        images = batch["pixel_values"].to(device)
        dino_inputs = dino_processor(
            images, return_tensors="pt", do_rescale=False, device=device
        )
        # TODO - pre-tokenize
        clip_tokens = batch["clip_tokens"].squeeze(1).to(device)
        t5_tokens = batch["t5_tokens"].squeeze(1).to(device)
        t5_attn_mask = batch["t5_attn_mask"].squeeze(1).to(device)
        return images, dino_inputs, clip_tokens, t5_tokens, t5_attn_mask

    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
        if torch.cuda.is_bf16_supported()
        else contextlib.nullcontext()
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
            ts = torch.rand(batch_size, device=device)

            try:
                # load data
                t0 = time.time()
                images, dino_inputs, clip_tokens, t5_tokens, t5_attn_mask = load_data()
                if DO_BENCHMARK:
                    log("load data: ", time.time() - t0)
            except StopIteration:
                log("Dataloader restarting")
                t0 = time.time()
                iterator = iter(dl)
                images, dino_inputs, clip_tokens, t5_tokens, t5_attn_mask = load_data()
                if DO_BENCHMARK:
                    log("load data: ", time.time() - t0)

            t0 = time.time()
            # get dino latent
            with torch.no_grad():
                with ctx:
                    outputs = dino_model(**dino_inputs)
                hidden_states = outputs.last_hidden_state

            # remove non-patch tokens
            patch_tokens = hidden_states[
                :, 1 + dino_model.config.num_register_tokens :, :
            ]

            # reshape, assume H=W (square)
            B, N, D = patch_tokens.size()
            H = W = int(N**0.5)
            patch_tokens = patch_tokens.permute(0, 2, 1).contiguous().view(B, D, H, W)
            if DO_BENCHMARK:
                log("dino runtime:", time.time() - t0)

            t0 = time.time()
            # get clean latent, mean and var, do batchnorm for repa-e
            with ctx:
                latent, mu, logvar, recon = ae(images)
                latent_bn = pre_bn(latent)
            if DO_BENCHMARK:
                log("vae runtime:", time.time() - t0)

            # detach normed latent to go throuh dit (so diffusion loss doesn't leak into encoder)
            latent_det = latent_bn.detach()
            latent_det.requires_grad = True

            # sample noise for diffusion interpolation
            epsilon = torch.randn_like(latent)

            # lerp
            interpolated = (
                ts[:, None, None, None] * latent_det
                + (1 - ts[:, None, None, None]) * epsilon
            )

            # get dit prediction for latent, also take a hidden state for alignment loss
            t0 = time.time()
            with ctx:
                dit_out, repa_out = dit(
                    interpolated,
                    tokens=t5_tokens,
                    attn_mask=t5_attn_mask,
                    clip_tokens=clip_tokens,
                    timesteps=torch.floor(
                        ts.view(-1)
                        * (dit.module.n_timesteps if ddp else dit.n_timesteps - 1)
                    )
                    .long()
                    .to(device),
                    repa_layer=repa_layer,
                )
            if DO_BENCHMARK:
                log("dit runtime:", time.time() - t0)

            # reshape to B, C, H, W for iREPA conv projection
            H_dit = W_dit = latent_size // patch_size
            repa_out = repa_out.permute(0, 2, 1).view(B, d_model, H_dit, W_dit)
            # pool so that H, W match that of dino - TODO: maybe try pooling after projection conv?
            repa_out = F.adaptive_avg_pool2d(repa_out, (H, W))

            with ctx:
                projected = proj_conv(repa_out)

            t0 = time.time()
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
                loss_diffusion
                + loss_alignment_dit
                + reg_loss_weight * loss_reg
                + loss_repa
            )

            loss = loss / grad_accum_steps

            if DO_BENCHMARK:
                log("loss calculation time:", time.time() - t0)

            t0 = time.time()
            loss.backward()
            if DO_BENCHMARK:
                log("backward time:", time.time() - t0)

            loss_accum += loss.detach()

        norm_ae = torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
        norm_dit = torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)

        for model_optims in optimizers.values():
            for optim in model_optims:
                for param_group in optim.param_groups:
                    param_group["lr"] = get_lr(step, optim.defaults["lr"])
                optim.step()

        t0 = time.time()
        ema.update()
        if DO_BENCHMARK:
            log("ema update time:", time.time() - t0)

        if step % log_every == 0:
            log(
                f"step: {step:8d} | loss: {loss_accum.item():8.4f} | align: {loss_alignment.item():8.4f} | diff: {loss_diffusion.item():8.4f} | reg: {loss_reg.item():8.4f} | mse: {loss_mse.item():8.4f} | kl: {loss_kl.item():8.4f} | lpips: {loss_lpips.item():8.4f} | norm ae: {norm_ae.item():8.4f} | norm dit: {norm_dit.item():8.4f}"
            )

        if ((step % save_every == 0 and step > 0) or step == num_steps) and (
            ddp_rank == 0 or not ddp
        ):
            root = f"../saved_models/dit/{run.name}"
            if not os.path.exists(root):
                os.makedirs(root)

            torch.save(
                dit.module.state_dict() if ddp else dit.state_dict(),
                os.path.join(root, "dit.pth"),
            )
            torch.save(
                ae.module.state_dict() if ddp else ae.state_dict(),
                os.path.join(root, "ae.pth"),
            )
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
        if not ddp or ddp_rank == 0:
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
                    "lr_mult": get_lr(step, 1.0),
                }
            )

    ddp_cleanup()


if __name__ == "__main__":
    main()
