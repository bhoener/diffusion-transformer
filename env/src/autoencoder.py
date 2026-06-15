import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, dim: int, stride: int = 1):
        super().__init__()
        self.dim = dim
        self.stride = stride

        self.l1 = nn.Conv2d(dim, dim, stride, stride)

        self.act = nn.SiLU()

        self.l2 = nn.Conv2d(dim, dim, stride, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.l2(self.act(self.l1(x)))
        return x


class Autoencoder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        latent_channels: int,
        num_downsamples: int = 4,
        num_resnet_blocks: int = 4,
        resnet_stride: int = 1,
        n_channels: int = 3,
    ):
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
        self.num_downsamples = num_downsamples
        self.latent_channels = latent_channels
        self.num_resnet_blocks = num_resnet_blocks
        self.resnet_stride = resnet_stride

        self.preprocess_conv = nn.Conv2d(
            n_channels,
            latent_channels,
            kernel_size=1,
            stride=1,
        )

        self.encoder_resnet_blocks = nn.ModuleList(
            ResNetBlock(latent_channels, resnet_stride)
            for _ in range(num_resnet_blocks)
        )

        self.downsamples = nn.ModuleList(
            nn.Conv2d(latent_channels, latent_channels, 2, 2)
            for _ in range(num_downsamples)
        )

        self.mu_proj = nn.Conv2d(latent_channels, latent_channels, 1, 1)
        self.log_sigma_sq_proj = nn.Conv2d(latent_channels, latent_channels, 1, 1)

        self.upsamples = nn.ModuleList(
            nn.ConvTranspose2d(latent_channels, latent_channels, 2, 2)
            for _ in range(num_downsamples)
        )

        self.decoder_resnet_blocks = nn.ModuleList(
            ResNetBlock(latent_channels, resnet_stride)
            for _ in range(num_resnet_blocks)
        )

        self.channel_reduce_conv = nn.Conv2d(latent_channels, n_channels, 1, 1)
        

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x = self.preprocess_conv(x)

        for layer in self.encoder_resnet_blocks:
            x = layer(x)


        for i, layer in enumerate(self.downsamples):
            x = layer(x)
            if i < len(self.downsamples) - 1:
                x = F.silu(x)

        mu = self.mu_proj(x)
        sigma = self.log_sigma_sq_proj(0.5 * x).exp()

        noise = torch.randn_like(x).to(x.device)

        x = mu + sigma * noise

        latent = x

        for i, layer in enumerate(self.upsamples):
            x = layer(x)
            x = F.silu(x)

        for layer in self.decoder_resnet_blocks:
            x = layer(x)

        x = self.channel_reduce_conv(x)
        
        return latent, mu, sigma, x

def main():
    img_size = 256
    batch_size = 8
    steps = 999
    patch_size = 32
    latent_channels = 16
    num_downsamples = 2
    num_resnet_blocks = 4
    resnet_stride = 1
    n_channels = 3
    divergence_weight = 0.01

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
    print(ae(torch.randn(6, 3, img_size, img_size))[0].size())

    from torchvision.transforms import ToTensor, ToPILImage
    import torchvision.transforms.v2 as transforms
    from datasets import load_dataset
    from PIL import Image
    import matplotlib.pyplot as plt
    import wandb

    run = wandb.init(
        project="VAE",
        config={
            "img_size": img_size,
            "batch_size": batch_size,
            "steps": steps,
            "patch_size": patch_size,
            "latent_channels": latent_channels,
            "num_downsamples": num_downsamples,
            "num_resnet_blocks": num_resnet_blocks,
            "resnet_stride": resnet_stride,
            "n_channels": n_channels,
            "divergence_weight": divergence_weight,
        },
    )

    to_tensor = ToTensor()
    to_image = ToPILImage()

    src = Image.open("src/test_img2.png").convert("RGB").resize((img_size, img_size))
    input_img = to_tensor(src).unsqueeze(0).to(device)

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
        images = [transform(img) for img in examples["jpg"] if img is not None]
        return {"pixel_values": images}

    ds = ds.with_transform(transform_batch)

    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, pin_memory=False)

    iterator = iter(dl)

    optim = torch.optim.AdamW(ae.parameters(), lr=3e-4)

    for step in range(steps):
        try:
            x = next(iterator)["pixel_values"].to(device)
            latent, mu, sigma, pred = ae(x)

            recon_loss = ((x - pred) ** 2).mean()

            divergence_loss = divergence_weight * (sigma + mu**2 - 1 - sigma.log()).mean()

            loss = recon_loss + divergence_loss

            optim.zero_grad()
            loss.backward()

            optim.step()

            run.log({"loss": loss.item(), "recon_loss": recon_loss.item(), "divergence_loss": divergence_loss.item()})
            if step % 10 == 0:
                print(f"step: {step} | loss: {loss.item():.4f}")

            if step > 0 and step % 1000 == 0:
                torch.save(ae.state_dict(), f"saved_models/vae/{run.name}.pth")
        except StopIteration:
            print("Dataloader resterting")
            iterator = iter(ds)

    plt.imshow(
        torch.cat((input_img.squeeze(0), ae(input_img)[-1].squeeze(0)), dim=1)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )
    plt.show()


if __name__ == "__main__":
    main()
