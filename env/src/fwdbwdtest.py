import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from transformers import T5EncoderModel, AutoTokenizer, CLIPTextModelWithProjection
import wandb
import random
import matplotlib.pyplot as plt
import os
from model import DiT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder = T5EncoderModel.from_pretrained("google/t5-v1_1-small", device_map=device)
tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")

clip_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", device_map=device)
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

img_size = (32, 32)
n_channels = 3
d_model = 768
n_heads = 12
n_layers_multi_stream = 6
n_layers_single_stream = 6
patch_size = 16
w, h = img_size
muon_lr = 1e-2
adam_lr = 3e-4
steps = 5000
bsz = 16

model = DiT(
    encoder_model=encoder,
    clip_encoder_model=clip_encoder,
    d_model=d_model,
    n_heads=n_heads,
    n_layers_multi_stream=n_layers_multi_stream,
    n_layers_single_stream=n_layers_single_stream,
    patch_size=patch_size,
    w=w,
    h=h,
    n_channels=n_channels,
)

model = model.to(device)

print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

optimizers = [
    torch.optim.Muon(
        [param for param in model.parameters() if param.ndim == 2], lr=muon_lr
    ),
    torch.optim.AdamW(
        [param for param in model.parameters() if param.ndim != 2], lr=adam_lr
    ),
]



to_tensor = ToTensor()
to_image = ToPILImage()

src = Image.open("src/test_img.jpg").resize((32, 32))
input_img = to_tensor(src).unsqueeze(0).to(device)


input_tokens = torch.randint(0, tokenizer.vocab_size, (bsz, 16)).to(device)
input_tokens_clip = torch.randint(0, clip_tokenizer.vocab_size, (bsz, 18)).to(device)

run = wandb.init(project="FlowMatching", config={
    "d_model" : d_model,
    "n_heads": n_heads,
    "n_layers_multi_stream": n_layers_multi_stream,
    "n_layers_single_stream": n_layers_single_stream,
    "patch_size": patch_size,
    "w": w,
    "h": h,
    "n_channels": n_channels,
    "muon_lr": muon_lr,
    "adam_lr": adam_lr,
    "steps": steps,
    "batch_size": bsz,
    "dataset": "dummy",
})

for step in range(steps):
    epsilon = torch.rand(bsz, n_channels, w, h).to(device)

    ts = torch.rand(bsz).to(device)

    out = model(
        ts[:, None, None, None] * input_img + (1 - ts[:, None, None, None]) * epsilon,
        input_tokens,
        input_tokens_clip,
        torch.floor(ts * (model.n_timesteps - 1)).long(),
    )

    loss = ((out - (input_img - epsilon)) ** 2).mean()

    loss.backward()


    max_grad_std = 0
    max_grad_name = ""

    for name, p in model.named_parameters():
        if p.grad is not None:
            if p.grad.std() > max_grad_std:
                max_grad_std = p.grad.std().item()
                max_grad_name = name
            # fig, ax = plt.subplots()
            # ax.hist(p.grad.view(-1).detach().cpu().numpy(), bins=20)
            # fig.savefig(os.path.join("src/gradients", f"{name}.png"))
            # plt.close()

            # print(f"name: {name} | grad mean: {p.grad.mean()} | grad std: {p.grad.std()} | grad max: {p.grad.max()}")

    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    run.log({"loss": loss.item(), "norm": norm.item()})

    if step % 10 == 0:
        print(f"step: {step} | norm: {norm:.4f} | loss: {loss.item():.4f}")

        print(max_grad_name, max_grad_std)

    for optim in optimizers:
        optim.step()
        optim.zero_grad()


with torch.no_grad():
    max_iters = 10
    x_in = torch.rand(bsz, n_channels, w, h).to(device)
    for step, t in enumerate(torch.linspace(0, 1, max_iters)):
        pred = model(
            x_in,
            input_tokens,
            input_tokens_clip,
            torch.floor(t * (model.n_timesteps - 1)).long().to(device),
        )
        x_in = x_in + (1 / max_iters) * pred

torchvision.utils.save_image(x_in, "src/out.png")

# plt.imshow(torch.cat((x_in.squeeze(0), input_img.squeeze(0)), dim=1).permute(1, 2, 0).cpu().numpy())
# plt.show()
# print(input_img.size())
# print(ts)
# plt.imshow(torch.cat((input_img * ts[:, None, None, None] + (1-ts[:, None, None, None]) * epsilon, epsilon)).permute(0, 2, 3, 1).contiguous().view(-1, 32, 4).detach().cpu().numpy())
# plt.show()
