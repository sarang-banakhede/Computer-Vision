"""
Description: This file contains the implementation of the Attention U-Net model.
"""
import torch
import torch.nn as nn

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1),
                                  nn.BatchNorm2d(num_features = out_channels),
                                  nn.ReLU(inplace = True),
                                  nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, padding = 1),
                                  nn.BatchNorm2d(num_features = out_channels),
                                  nn.ReLU(inplace = True))
    
    def forward(self, x):
        x = self.conv(x)
        return x
    
class Encoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_Block, self).__init__()

        self.conv = Conv_Block(in_channels = in_channels, out_channels = out_channels)
        self.pool = nn.MaxPool2d(kernel_size = (2,2))

    def forward(self, x):
        s = self.conv(x)
        output = self.pool(s)
        return s, output
    
class attention_gate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag = attention_gate(in_c, out_c)
        self.c1 = Conv_Block(in_c[0]+out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x

class attention_unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = Encoder_Block(1, 64)
        self.e2 = Encoder_Block(64, 128)
        self.e3 = Encoder_Block(128, 256)

        self.b1 = Conv_Block(256, 512)

        self.d1 = decoder_block([512, 256], 256)
        self.d2 = decoder_block([256, 128], 128)
        self.d3 = decoder_block([128, 64], 64)
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b1 = self.b1(p3)

        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        output = self.output(d3)
        return torch.sigmoid(output)