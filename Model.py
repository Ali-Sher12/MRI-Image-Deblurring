import torch
import torch.nn as nn

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )

class ResUNet3D(nn.Module):
    """
    Lightweight 3D U-Net for MRI deblurring.
    Predicts a residual correction that gets added back to the blurred input:
        sharp_pred = blurred + residual
    Designed for patch-based training (e.g. 64^3 or 96^3 patches) on a
    single free-tier Colab GPU.
    """
    def __init__(self, in_channels=1, out_channels=1, base_ch=16):
        super().__init__()

        # Encoder
        self.enc1 = conv_block(in_channels, base_ch)          # 16
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv_block(base_ch, base_ch * 2)          # 32
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)      # 64
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = conv_block(base_ch * 4, base_ch * 8)  # 128

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        self.out_conv = nn.Conv3d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        input_volume = x  # keep for residual connection

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        residual = self.out_conv(d1)
        return input_volume + residual   # sharp_pred = blurred + residual