import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + x)

class DrawerNet(nn.Module):
    """
    Tiny CNN -> 10-channel sigmoid outputs for glyph maps (8x8).
    """
    def __init__(self, in_ch=17, width=64, depth=4, out_ch=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        self.head = nn.Conv2d(width, out_ch, 1)
        self.out_ch = out_ch

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        logits = self.head(h)          # [B,10,8,8]
        return logits

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
