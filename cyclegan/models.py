"""CycleGAN networks: a ResNet generator and a 70x70 PatchGAN discriminator.

Architecture follows Zhu et al., "Unpaired Image-to-Image Translation using
Cycle-Consistent Adversarial Networks" (ICCV 2017).
"""
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two 3x3 convs with a skip connection. Keeps resolution and channels."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """Encoder -> residual blocks -> decoder.

    c7s1-64, d128, d256, R256 x n_blocks, u128, u64, c7s1-3
    Output goes through tanh, so images live in [-1, 1].
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base: int = 64, n_blocks: int = 9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base, 7),
            nn.InstanceNorm2d(base),
            nn.ReLU(inplace=True),
        ]
        # Downsample twice: 64 -> 128 -> 256 channels, H/4 x W/4
        ch = base
        for _ in range(2):
            layers += [
                nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ch * 2),
                nn.ReLU(inplace=True),
            ]
            ch *= 2
        layers += [ResidualBlock(ch) for _ in range(n_blocks)]
        # Upsample back to full resolution
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ch // 2),
                nn.ReLU(inplace=True),
            ]
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, out_channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """PatchGAN: outputs a grid of real/fake scores, one per ~70x70 patch.

    Judging local patches instead of the whole image pushes the generator
    toward realistic texture and color everywhere, with fewer parameters.
    """

    def __init__(self, in_channels: int = 3, base: int = 64):
        super().__init__()

        def block(cin, cout, stride, norm=True):
            layers = [nn.Conv2d(cin, cout, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, base, 2, norm=False),
            *block(base, base * 2, 2),
            *block(base * 2, base * 4, 2),
            *block(base * 4, base * 8, 1),
            nn.Conv2d(base * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


def init_weights(module: nn.Module, std: float = 0.02):
    """Gaussian init N(0, 0.02) for conv layers, as in the paper."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
