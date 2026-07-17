import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(
        self, channels: tuple[int], hidden_size: int = 512, pool_size: int = 8
    ):
        super().__init__()
        self.channels = channels
        self.hidden_size = hidden_size
        self.pool_size = pool_size

        self.in_conv = nn.Conv2d(
            channels[0], channels[1], kernel_size=5, stride=1, padding=2
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2, padding=2)
                for in_ch, out_ch in zip(channels[1:], channels[2:])
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_ch) for out_ch in channels[2:]])

        self.lin = nn.Linear(channels[-1] * pool_size * pool_size, hidden_size)

        self.lin_bn = nn.BatchNorm1d(hidden_size)

        self.relu = nn.ReLU()

        self.out_proj = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        x = self.in_conv(x)

        x = self.relu(x)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size)).view(B, -1)

        x = self.lin(x)
        x = self.lin_bn(x)
        x = self.relu(x)

        x = self.out_proj(x)

        x = self.sigmoid(x)

        return x


def main():
    disc = Discriminator((32, 128, 245, 256), 512)

    z = torch.randn(2, 32, 32, 32)

    print(disc(z).size())


if __name__ == "__main__":
    main()
