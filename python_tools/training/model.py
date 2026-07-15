"""Lightweight U-Net for GRACE stripe-noise denoising.

Architecture (base_filters=32, bilinear, 64×64 input):
  inc:     1→32   (64×64)
  down1:  32→64   (32×32)
  down2:  64→128  (16×16)
  down3: 128→256   (8×8)
  down4: 256→256   (4×4)  bottleneck
  up1:   256+256→128  (8×8)
  up2:   128+128→64   (16×16)
  up3:    64+64 →32   (32×32)
  up4:    32+32 →16   (64×64)
  outc:   16→1
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    """Conv2d → BN → ReLU → Conv2d → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
        mid_ch = mid_ch or out_ch
        super().__init__(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class Down(nn.Sequential):
    """MaxPool → DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))


class Up(nn.Module):
    """Upsample → Concat with skip → DoubleConv.

    Args:
        dec_ch:  channels of the incoming decoder feature map (before upsampling)
        skip_ch: channels of the skip connection from encoder
        out_ch:  desired output channels after DoubleConv
        bilinear: use bilinear upsampling (True) or transposed conv (False)
    """
    def __init__(self, dec_ch: int, skip_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            cat_ch = dec_ch + skip_ch
        else:
            self.up = nn.ConvTranspose2d(dec_ch, dec_ch // 2, kernel_size=2, stride=2)
            cat_ch = (dec_ch // 2) + skip_ch
        self.conv = DoubleConv(cat_ch, out_ch)

    def forward(self, x_dec: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_dec = self.up(x_dec)
        # Pad spatial dims if needed (odd input sizes)
        dh = x_skip.size(2) - x_dec.size(2)
        dw = x_skip.size(3) - x_dec.size(3)
        if dh > 0 or dw > 0:
            x_dec = nn.functional.pad(
                x_dec, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x_skip, x_dec], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """UNet for GRACE stripe-noise prediction.

    Input:  (B, 1, H, W)  raw EWH grid
    Output: (B, 1, H, W)  predicted noise (filtered = raw - noise)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 32, bilinear: bool = True):
        super().__init__()
        bf = base_filters
        factor = 2 if bilinear else 1

        # Encoder
        self.inc   = DoubleConv(in_channels, bf)
        self.down1 = Down(bf,       bf * 2)
        self.down2 = Down(bf * 2,   bf * 4)
        self.down3 = Down(bf * 4,   bf * 8)
        self.down4 = Down(bf * 8,   bf * 16 // factor)   # bottleneck

        # Decoder – args: (dec_ch, skip_ch, out_ch)
        self.up1 = Up(bf * 16 // factor, bf * 8,   bf * 8 // factor,  bilinear)
        self.up2 = Up(bf * 8 // factor,  bf * 4,   bf * 4 // factor,  bilinear)
        self.up3 = Up(bf * 4 // factor,  bf * 2,   bf * 2 // factor,  bilinear)
        self.up4 = Up(bf * 2 // factor,  bf,       bf,                bilinear)

        self.outc = nn.Conv2d(bf, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)                                    # bf × H×W
        x2 = self.down1(x1)                                 # bf*2 × H/2×W/2
        x3 = self.down2(x2)                                 # bf*4 × H/4×W/4
        x4 = self.down3(x3)                                 # bf*8 × H/8×W/8
        x5 = self.down4(x4)                                 # bottleneck

        # Decoder with skip connections
        x = self.up1(x5, x4)                                # bf*8/f × H/8×W/8
        x = self.up2(x, x3)                                 # bf*4/f × H/4×W/4
        x = self.up3(x, x2)                                 # bf*2/f × H/2×W/2
        x = self.up4(x, x1)                                 # bf × H×W

        return self.outc(x)
