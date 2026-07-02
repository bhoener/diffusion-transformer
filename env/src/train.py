import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5EncoderModel, AutoTokenizer, CLIPTextModelWithProjection
from transformers import AutoImageProcessor, AutoModel
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

lpips_loss = lpips.LPIPS(net="vgg")
lpips_loss = lpips_loss.to(device)

print("lpips loss setup")

img_size = 256
n_channels = 3
batch_size = 8
steps = 1000

cooldown_steps = 4000

log_every = 10


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

print("autoencoder created")

# DiT
d_model = 768
n_heads = 12
n_layers_multi_stream = 6
n_layers_single_stream = 6
patch_size = 2
n_timesteps = 100


# training config

batch_size = 8
num_steps = 1000
lr_ae = 3e-4
lr_dit_muon = 1e-2
lr_dit_adamw = 3e-4

repa_loss_weight = 0.1
reg_loss_weight = 0.5

mse_loss_weight = 0.5
lpips_loss_weight = 1.0
kl_loss_weight = 1e-4

print("downloading t5")

encoder = T5EncoderModel.from_pretrained("google/t5-v1_1-small", device_map=device)

print("downloading t5 tokenizer")
t5_tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")

print("downloading clip")

clip_encoder = CLIPTextModelWithProjection.from_pretrained(
    "openai/clip-vit-large-patch14", device_map=device
)
print("downloading clip tokenizer")
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

print("encoder models setup")

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

dino_processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov3-vith16plus-pretrain-lvd1689m"
)
dino_model = AutoModel.from_pretrained(
    "facebook/dinov3-vits16-pretrain-lvd1689m", device_map=device
)

print("dino model setup")

proj_conv = nn.Conv2d(
    d_model, dino_model.config.hidden_size, kernel_size=3, stride=1, padding=1
).to(device)

pre_bn = nn.BatchNorm2d(z_channels).to(device)

optimizers = {
    "DiT": [torch.optim.Muon([p for p in dit.parameters() if p.ndim == 2], lr=lr_dit_muon), torch.optim.AdamW([p for p in dit.parameters() if p.ndim != 2], lr=lr_dit_adamw)],
    "VAE": [torch.optim.AdamW([p for p in ae.parameters()], lr=lr_ae)]
}

ds = load_dataset("arrow", data_files={"data/filtered_ds/*.arrow"}, split="train")

transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.RGB(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

def transform_batch(examples):
    out = {"pixel_values" : [], "caption": []}
    for img, cap in zip(examples["jpg"], examples["caption"]):
        if img is not None and cap is not None:
            out["pixel_values"].append(transform(img))
            out["caption"].append(cap)
    return out

ds = ds.with_transform(transform_batch)

dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, pin_memory=False)

iterator = iter(dl)

print("dataset created")

for step in range(num_steps):
    # -------------------------------- training step -------------------------------

    ts = torch.rand(batch_size, 1, 1, 1).to(device)


    try:
        batch = next(iterator)
        images = batch["pixel_values"].to(device)
        captions = batch["caption"]
        dino_inputs = dino_processor(images, return_tensors="pt").to(device)
        clip_tokens = clip_tokenizer.encode(captions, return_tensors="pt").to(device)
        t5_tokens = tokenizer.encode(captions, return_tensors="pt").to(device)
        
    except StopIteration:
        iterator = iter(dl)
        continue
    
    with torch.no_grad():
        outputs = dino_model(**dino_inputs)
        hidden_states = outputs.last_hidden_state

    # Patch tokens represent specific regional/local features
    patch_tokens = hidden_states[:, 1 + dino_model.config.num_register_tokens :, :]

    patch_size = dino_model.config.patch_size

    B, N, D = patch_tokens.size()
    H = W = int(N**0.5)  # h/w in patches
    patch_tokens = patch_tokens.permute(0, 2, 1).contiguous().view(B, D, H, W)

    latent, mu, logvar = ae.encode(images)

    # weird magic for stopgrad
    # we detach the latent
    # then run the dit on it
    # then we get the grad for latent
    # and then do backward on the un-detached latent
    # and then can do backward normally
    latent_det = pre_bn(latent).detach() 
    latent_det.requires_grad = True

    epsilon = torch.randn_like(latent)

    interpolated = ts * latent + (1-ts) * epsilon

    dit_out, repa_out = dit(
        latent_det,
        tokens=t5_tokens,
        clip_tokens=clip_tokens,
        timesteps=torch.floor(ts.view(-1) * (dit.n_timesteps - 1)).to(device),
        repa_layer=4,
    )

    N_dit = repa_out.size(1)

    H_dit = W_dit = img_size // patch_size

    repa_out = repa_out.permute(0, 2, 1).view(B, d_model, H_dit, W_dit)

    repa_out = F.adaptive_avg_pool2d(repa_out, (H, W))

    projected = proj_conv(repa_out)

    projected = projected / (projected.norm() + 1e-7)
    patch_tokens = patch_tokens / (patch_tokens.norm() + 1e-7)

    loss_alignment = 1 - F.cosine_similarity(projected, patch_tokens, dim=-1).mean() # max 2 if antiparallel, likely ~1

    loss_diffusion = ((dit_out - (latent_det - epsilon)) ** 2).mean()

    grad_latent = torch.autograd.grad(loss_alignment, latent_det, retain_graph=True)[0]

    latent.backward(grad_latent, retain_graph=True)

    # regularization losses

    recon = ae.decode(latent)

    loss_mse = ((images - recon) ** 2).mean()

    loss_kl = (logvar.exp() + mu**2 - 1 - logvar).mean()
    
    loss_lpips = lpips_loss(images * 2 - 1, recon * 2 - 1)

    loss_reg = mse_loss_weight * loss_mse + kl_loss_weight * loss_kl + lpips_loss_weight * loss_lpips

    # total loss

    loss = loss_diffusion + repa_loss_weight * loss_alignment + reg_loss_weight * loss_reg

    if step % log_every == 0:
        print(f"step: {step:8d} | loss: {loss.item():8.4f} | align: {loss_alignment.item():8.4f} | diff: {loss_diffusion.item():8.4f} | reg: {loss_reg.item():8.4f} | mse: {loss_mse.item():8.4f} | kl: {loss_kl.item():8.4f} | lpips: {loss_lpips.item():8.4f}")

    for model_optims in optimizers.values():
        for optim in model_optims:
            optim.zero_grad()
    loss.backward()
    for model_optims in optimizers.values():
        for optim in model_optims:
            optim.step()


plt.imshow(ae(images[0].unsqueeze(0))[-1].permute(0, 2, 3, 1).view(img_size, img_size, 3).detach().cpu().numpy())
plt.show()