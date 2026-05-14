import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, AutoTokenizer
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

steps = 100

input_img = torch.randn(2, 4, 32, 32).to(device)
epsilon = torch.randn(2, 4, 32, 32).to(device)

for step in range(steps):
    
    input_tokens = torch.randint(0, 32128, (2, 16)).to(device)
    out = model(input_img, input_tokens, 1)


    loss = ((epsilon - out)**2).mean()

    print("loss:", loss.item())
    
    loss.backward()
    
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    print(f"norm: {norm:.4f}")
    
    for optim in optimizers:
        optim.step()
        optim.zero_grad()

# for name, p in model.named_parameters():
#     if p.grad is not None:
#         print(f"name: {name} | grad mean: {p.grad.mean()} | grad std: {p.grad.std()}")
 

