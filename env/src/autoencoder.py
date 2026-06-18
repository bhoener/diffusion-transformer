import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ResNetBlock(nn.Module):
    def __init__(
        self, dim: int, kernel_size: int = 3, stride: int = 1, padding: int = 1
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.norm1 = nn.BatchNorm2d(dim)
        self.l1 = nn.Conv2d(dim, dim, kernel_size, stride, padding=padding)

        self.act = nn.SiLU()

        self.norm2 = nn.BatchNorm2d(dim)
        self.l2 = nn.Conv2d(dim, dim, kernel_size, stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.l1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.l2(h)
        return x + h


class DownBlock(nn.Module):
    def __init__(
        self,
        num_resnet_blocks: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        resnet_kernel_size: int = 3,
        resnet_stride: int = 1,
        resnet_padding: int = 1,
    ):
        super().__init__()
        self.num_resnet_blocks = num_resnet_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 2

        self.resnet_blocks = nn.ModuleList(
            ResNetBlock(
                dim=in_channels,
                kernel_size=resnet_kernel_size,
                stride=resnet_stride,
                padding=resnet_padding,
            )
            for _ in range(num_resnet_blocks)
        )

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.resnet_blocks:
            x = layer(x)
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        num_resnet_blocks: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        resnet_kernel_size: int = 3,
        resnet_stride: int = 1,
        resnet_padding: int = 1,
    ):
        super().__init__()
        self.num_resnet_blocks = num_resnet_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 2

        self.resnet_blocks = nn.ModuleList(
            ResNetBlock(
                dim=in_channels,
                kernel_size=resnet_kernel_size,
                stride=resnet_stride,
                padding=resnet_padding,
            )
            for _ in range(num_resnet_blocks)
        )

        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            output_padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.resnet_blocks:
            x = layer(x)
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        
        self.norm = nn.GroupNorm(32, d)
        
        self.wq = nn.Conv2d(d, d, kernel_size=1)
        self.wk = nn.Conv2d(d, d, kernel_size=1)
        self.wv = nn.Conv2d(d, d, kernel_size=1)
        
        self.wo = nn.Conv2d(d, d, kernel_size=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, c, W, H = x.size()
        
        x = self.norm(x)
        
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        
        Q = rearrange(Q, "B c W H -> B 1 (W H) c").contiguous()
        K = rearrange(K, "B c W H -> B 1 (W H) c").contiguous()
        V = rearrange(V, "B c W H -> B 1 (W H) c").contiguous()
        
        attn_scores = F.scaled_dot_product_attention(Q, K, V)
        
        out = self.wo(rearrange(attn_scores, "B 1 (W H) c -> B c W H", W = W, H=H).contiguous())
        return x + out
    
class MidBlock(nn.Module):
    def __init__(self, d: int, n_layers: int = 1):
        super().__init__()
        self.d = d
        self.n_layers = n_layers
        
        self.resnets = nn.ModuleList(ResNetBlock(dim=d) for _ in range(n_layers))
        self.attns = nn.ModuleList(AttentionBlock(d=d) for _ in range(n_layers))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attention, resnet in zip(self.attns, self.resnets):
            x = attention(x)
            x = resnet(x)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        resnet_blocks_per_layer: int,
        channels: tuple[int], # (128, 256, 512, 512), (128, 128, 256, 512, 512)
        z_channels: int = 32,
        kernel_size: int = 3,
        padding: int = 1,
        resnet_kernel_size: int = 3,
        resnet_stride: int = 1,
        resnet_padding: int = 1,
        mid_block_layers: int = 1,
    ):
        super().__init__()
        self.resnet_blocks_per_layer = resnet_blocks_per_layer
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = 2

        self.downsamples = nn.ModuleList(
            DownBlock(
                num_resnet_blocks=resnet_blocks_per_layer,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                resnet_kernel_size=resnet_kernel_size,
                resnet_stride=resnet_stride,
                resnet_padding=resnet_padding,
            )
            for in_channels, out_channels in zip((channels[0],) + channels, channels)
        )
        
        self.mid_block = MidBlock(d=channels[-1], n_layers=mid_block_layers)
        
        self.proj_conv = nn.Conv2d(channels[-1], z_channels, kernel_size=3, stride=1, padding=1)
        self.mu_proj = nn.Conv2d(z_channels, z_channels, kernel_size=3, stride=1, padding=1)
        self.log_sigma_sq_proj = nn.Conv2d(z_channels, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        for layer in self.downsamples:
            x = layer(x)
        
        x = self.mid_block(x)
        
        x = self.proj_conv(x)
        return self.mu_proj(x), self.log_sigma_sq_proj(x)

class Decoder(nn.Module):
    def __init__(
        self,
        resnet_blocks_per_layer: int,
        channels: tuple[int],
        z_channels: int = 32,
        kernel_size: int = 3,
        padding: int = 1,
        resnet_kernel_size: int = 3,
        resnet_stride: int = 1,
        resnet_padding: int = 1,
        mid_block_layers: int = 1,
    ):
        super().__init__()
        self.resnet_blocks_per_layer = resnet_blocks_per_layer
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = 2
        
        self.in_proj = nn.Conv2d(z_channels, channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_block = MidBlock(d=channels[-1], n_layers=mid_block_layers)

        in_channel_list = channels[::-1]
        out_channel_list = in_channel_list[1:] + (in_channel_list[-1],)
        self.upsamples = nn.ModuleList(
            UpBlock(
                num_resnet_blocks=resnet_blocks_per_layer,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                resnet_kernel_size=resnet_kernel_size,
                resnet_stride=resnet_stride,
                resnet_padding=resnet_padding,
            )
            for in_channels, out_channels in zip(in_channel_list, out_channel_list)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)

        x = self.mid_block(x)

        for layer in self.upsamples:
            x = layer(x)
        return x


class Autoencoder(nn.Module):
    def __init__(
        self,
        latent_channels: tuple[int],
        z_channels: int = 32,
        kernel_size: int = 3,
        padding: int = 1,
        resnet_blocks_per_layer: int = 2,
        resnet_kernel_size: int = 3,
        resnet_stride: int = 1,
        resnet_padding: int = 1,
        n_channels: int = 3,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.latent_channels = latent_channels
        self.z_channels = z_channels
        self.resnet_blocks_per_layer = resnet_blocks_per_layer
        self.resnet_kernel_size = resnet_kernel_size

        self.preprocess_conv = nn.Conv2d(
            n_channels,
            latent_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.encoder = Encoder(
            channels=latent_channels,
            z_channels=z_channels,
            resnet_blocks_per_layer=resnet_blocks_per_layer,
            kernel_size=resnet_kernel_size,
            padding=resnet_padding,
            resnet_kernel_size=resnet_kernel_size,
            resnet_stride=resnet_stride,
            resnet_padding=resnet_padding,
        )


        self.decoder = Decoder(
            channels=latent_channels,
            z_channels=z_channels,
            resnet_blocks_per_layer=resnet_blocks_per_layer,
            kernel_size=resnet_kernel_size,
            padding=padding,
            resnet_kernel_size=resnet_kernel_size,
            resnet_stride=resnet_stride,
            resnet_padding=resnet_padding,
        )

        self.channel_reduce_conv = nn.Conv2d(
            latent_channels[0], n_channels, 3, stride=1, padding=1
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x = self.preprocess_conv(x)

        mu, logvar = self.encoder(x)

        std = (0.5 * logvar).exp()

        noise = torch.randn_like(mu).to(x.device)

        return mu + std * noise, mu, logvar
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)

        x = self.channel_reduce_conv(x)
        
        return F.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        latent, mu, logvar = self.encode(x)
        
        out = self.decode(latent)

        return latent, mu, logvar, out

def main():
    img_size = 256
    batch_size = 8
    steps = 10000
    latent_channels = (128, 256, 512, 512)
    z_channels = 32
    kernel_size = 3
    padding = 1
    resnet_blocks_per_layer = 2
    resnet_kernel_size = 3
    resnet_stride = 1
    resnet_padding = 1
    n_channels = 3
    divergence_weight = 0
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    ae.train()
    
    torch.set_float32_matmul_precision("high")
    ae = ae.to(device)
    ae = torch.compile(ae)
    
    print(ae(torch.randn(6, 3, img_size, img_size).to(device))[-1].size())
    print(f"Total Parameters: {(sum(p.numel() for p in ae.parameters())  / 1e6) :.1f}M")

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
            "latent_channels": latent_channels,
            "z_channels": z_channels,
            "kernel_size": kernel_size,
            "padding": padding,
            "resnet_blocks_per_layer": resnet_blocks_per_layer,
            "resnet_kernel_size": resnet_kernel_size,
            "resnet_stride": resnet_stride,
            "resnet_padding": resnet_padding,
            "n_channels": n_channels,
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

    wandb.watch(ae, log_freq=100)

    for step in range(steps):
        try:
            x = next(iterator)["pixel_values"].to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                latent, mu, logvar, pred = ae(x)

            recon_loss = ((x - pred).abs()).mean()

            divergence_loss = (
                divergence_weight * (logvar.exp() + mu**2 - 1 - logvar).mean()
            )

            loss = recon_loss + divergence_loss

            optim.zero_grad()
            loss.backward()
            
            norm = torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            optim.step()

            run.log(
                {
                    "loss": loss.item(),
                    "recon_loss": recon_loss.item(),
                    "divergence_loss": divergence_loss.item(),
                    "norm": norm.item(),
                }
            )
            
            if step % 10 == 0:
                print(f"step: {step:8d} | loss: {loss.item():8.4f} | recon_loss: {recon_loss.item():8.4f} | divergence_loss: {divergence_loss.item():8.4f} | norm: {norm.item():8.4f}")

            if step > 0 and step % 1000 == 0:
                torch.save(ae.state_dict(), f"saved_models/vae/{run.name}.pth")
        except StopIteration:
            print("Dataloader resterting")
            iterator = iter(ds)
    wandb.finish()
    ae.eval()

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
