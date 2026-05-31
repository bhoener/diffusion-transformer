import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, patch_size: int, n_channels: int = 4):
        """
        AutoEncoder
        d_model (int): model dimension
        patch_size (int): patch_size to split into
        n_channels (int): image channels

        input size: (B, C, X, Y)
        output size: (B, T, D)
        """
        super().__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels

        self.conv = nn.Conv2d(
            n_channels, n_channels, kernel_size=patch_size, stride=patch_size
        )


        self.act = nn.Tanh()

        self.conv_t = nn.ConvTranspose2d(
            n_channels, n_channels, kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.act(self.conv(x))
        return latent, F.sigmoid(self.conv_t(latent))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    ae = Autoencoder(4)
    print(ae(torch.randn(6, 4, 32, 32))[0].size())

    from torchvision.transforms import ToTensor, ToPILImage
    from PIL import Image
    import matplotlib.pyplot as plt
    
    to_tensor = ToTensor()
    to_image = ToPILImage()

    src = Image.open("src/test_img.jpg").convert("RGBA").resize((64, 64))
    input_img = to_tensor(src).unsqueeze(0).to(device)

    steps = 10000

    optim = torch.optim.AdamW(ae.parameters(), lr=0.01)

    for step in range(steps):
        _, pred = ae(input_img)

        loss = ((input_img - pred) ** 2).mean()

        optim.zero_grad()
        loss.backward()
        
        optim.step()

        print(f"step: {step} | loss: {loss.item():.4f}")

    plt.imshow(torch.cat((input_img.squeeze(0), pred.squeeze(0)), dim=1).permute(1, 2, 0).detach().cpu().numpy())
    plt.show()

if __name__ == "__main__":
    main()
