"""
Description: This file contains the implementation of the UNetASPP model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=512):
        super(ASPP, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv3x3_6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.conv3x3_12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        self.conv3x3_18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False)
        self.conv1x1_out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        size = x.size()
        pool = self.avg_pool(x)
        pool = self.conv1x1_pool(pool)
        pool = self.bn(pool)
        pool = self.relu(F.interpolate(pool, size=(size[2], size[3]), mode='bilinear', align_corners=True))

        conv1x1_1 = self.conv1x1_1(x)
        conv1x1_1 = self.bn(conv1x1_1)
        conv1x1_1 = self.relu(conv1x1_1)

        conv3x3_6 = self.conv3x3_6(x)
        conv3x3_6 = self.bn(conv3x3_6)
        conv3x3_6 = self.relu(conv3x3_6)

        conv3x3_12 = self.conv3x3_12(x)
        conv3x3_12 = self.bn(conv3x3_12)
        conv3x3_12 = self.relu(conv3x3_12)

        conv3x3_18 = self.conv3x3_18(x)
        conv3x3_18 = self.bn(conv3x3_18)
        conv3x3_18 = self.relu(conv3x3_18)

        out = torch.cat([pool, conv1x1_1, conv3x3_6, conv3x3_12, conv3x3_18], dim=1)
        out = self.conv1x1_out(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class UNetASPP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetASPP, self).__init__()

        # Encoder (contracting path)
        self.encoder1 = self.contract_block(in_channels, 64, 2)
        self.encoder2 = self.contract_block(64, 128, 2)
        self.encoder3 = self.contract_block(128, 256, 2)
        self.encoder4 = self.contract_block(256, 512, 2)

        # ASPP module
        self.aspp = ASPP(512)

        # Decoder (expanding path)
        self.decoder1 = self.expand_block(512, 256, 2)
        self.decoder2 = self.expand_block(512, 128, 2)
        self.decoder3 = self.expand_block(256, 64, 2)
        self.decoder4 = self.expand_block(128, 32, 2)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def contract_block(self, in_channels, out_channels, pool_size=2):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        )
        return block

    def expand_block(self, in_channels, out_channels, upsample_scale=2):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=upsample_scale, stride=upsample_scale),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # ASPP
        aspp_out = self.aspp(enc4)
        
        # Decoder
        dec1 = self.decoder1(aspp_out)
        dec1 = torch.cat([dec1, enc3], dim=1)
        dec2 = self.decoder2(dec1)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec3 = self.decoder3(dec2)
        dec3 = torch.cat([dec3, enc1], dim=1)
        dec4 = self.decoder4(dec3)

        # Final convolution
        final_output = self.final_conv(dec4)
        return torch.sigmoid(final_output)