import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, padding=p)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DrawerNet(nn.Module):
    def __init__(self, in_ch: int = 17, width: int = 64, depth: int = 6, out_ch: int = 10):
        """
        in_ch: number of input planes (board_to_planes)
        out_ch: number of glyph channels to predict (== len(GLYPH_CHANNELS))
        """
        super().__init__()
        self.in_ch = in_ch
        self.width = width
        self.depth = depth
        self.out_ch = out_ch

        layers = [ConvBlock(in_ch, width)]
        for _ in range(depth - 1):
            layers.append(ConvBlock(width, width))
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, out_ch, kernel_size=1)

        # lightweight init for the head
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)  # logits [B, out_ch, 8, 8]
        return x
