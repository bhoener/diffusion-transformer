import  sys

import torch
from autoencoder import Autoencoder
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import matplotlib.pyplot as plt
img_size = 256
latent_channels = (128, 256, 512, 512, 512)
z_channels = 32
kernel_size = 3
padding = 1
resnet_blocks_per_layer = 2
resnet_kernel_size = 3
resnet_stride = 1
resnet_padding = 1
n_channels = 3

ddp = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
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

to_tensor = ToTensor()
to_image = ToPILImage()

src = Image.open("test_img2.png").convert("RGB").resize((img_size, img_size))
input_img = to_tensor(src).unsqueeze(0).to(device)

ae.load_state_dict({k.replace("_orig_mod." + "module." if ddp else "", "") : v for k, v in torch.load("../saved_models/dit/jumping-paper-122/ae.pth").items()})

print(input_img.size())

latent, mu, logvar = ae.encode(input_img)

out = ae.decode(mu)

print(f"Compression: {input_img.numel() / latent.numel():.2f}x")

to_image(
    torch.cat((input_img.squeeze(0), out.squeeze(0)), dim=1)
    .detach()
    .cpu()).save("vae_output.png")