import torch
from src.autoencoder import Autoencoder
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
img_size = 256
patch_size = 32
latent_channels = 16
num_downsamples = 4
num_resnet_blocks = 4
resnet_stride = 1
n_channels = 3



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
ae = Autoencoder(
    patch_size=patch_size,
    latent_channels=latent_channels,
    num_downsamples=num_downsamples,
    num_resnet_blocks=num_resnet_blocks,
    resnet_stride=resnet_stride,
    n_channels=n_channels,
)

to_tensor = ToTensor()

src = Image.open("src/test_img2.png").resize((img_size, img_size))
input_img = to_tensor(src).unsqueeze(0).to(device)

ae.load_state_dict(torch.load("saved_models/vae/elated-sea-2.pth"))

print(input_img.size())
print(ae(input_img)[-1][0, :, 0, 0])

plt.imshow(
    torch.cat((input_img.squeeze(0), ae(input_img)[-1].squeeze(0)), dim=1)
    .permute(1, 2, 0)
    .detach()
    .cpu()
    .numpy()
)
plt.show()