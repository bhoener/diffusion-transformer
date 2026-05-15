import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from transformers import T5EncoderModel, AutoTokenizer
import random
import matplotlib.pyplot as plt
import os
from model import DiT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

encoder = T5EncoderModel.from_pretrained("google/t5-v1_1-small", device_map=device)
tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")

img_size = (32, 32)

model = DiT(
    encoder_model=encoder,
    d_model=768,
    n_heads=12,
    n_layers_multi_stream=6,
    n_layers_single_stream=6,
    patch_size=16,
    w=img_size[0],
    h=img_size[1],
)

print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

optimizers = [
    torch.optim.Muon([param for param in model.parameters() if param.ndim == 2], lr=1e-2),
    torch.optim.AdamW([param for param in model.parameters() if param.ndim != 2], lr=3e-4)
]

steps = 1000

to_tensor = ToTensor()
to_image = ToPILImage()

src = Image.open("src/test_img.jpg").convert("RGBA").resize((32, 32))
input_img = to_tensor(src).unsqueeze(0).to(device)


bsz = 4
std_min = 0.0
std_max = 1.0

for step in range(steps):
    
    input_tokens = torch.randint(0, 32128, (bsz, 16)).to(device)
    epsilon = torch.randn(bsz, 4, 32, 32).to(device) * torch.tensor([random.uniform(std_min, std_max) for _ in range(bsz)])[:, None, None, None].to(device)

    out = model(input_img + epsilon, input_tokens, 1)

    loss = ((epsilon/epsilon.std(dim=0, keepdim=True) - out)**2).mean()
    
    loss.backward()
    
    max_grad_std = 0
    max_grad_name = ""
    
    for name, p in model.named_parameters():
        if p.grad is not None:
            if p.grad.std() > max_grad_std:
                max_grad_std = p.grad.std()
                max_grad_name = name
            # fig, ax = plt.subplots()
            # ax.hist(p.grad.view(-1).detach().cpu().numpy(), buckets=20)
            # fig.savefig(os.path.join("src/gradients", f"{name}.png"))
            

            # print(f"name: {name} | grad mean: {p.grad.mean()} | grad std: {p.grad.std()} | grad max: {p.grad.max()}")
    
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    if step % 1 == 0:
        print(f"step: {step} | norm: {norm:.4f} | loss: {loss.item():.4f}")
        
        print(max_grad_name, max_grad_std)
    
    
    for optim in optimizers:
        optim.step()
        optim.zero_grad()
        
    break



print(input_img.size())
plt.imshow(torch.cat((input_img + epsilon, epsilon)).permute(0, 2, 3, 1).contiguous().view(-1, 32, 4).detach().cpu().numpy())
plt.show()


 

