import torch
import torch.nn as nn


class Flattener(nn.Module):
    def __init__(self):
        super(Flattener, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class SimpleDetector(nn.Module):
    def __init__(self, final_dim=1024):
        super(SimpleDetector, self).__init__()
        self.final_dim = final_dim
        self.downsample = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flattener(),
            nn.Dropout(p=0.1),
            nn.Linear(2048, self.final_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, img_feats):
        return self.downsample(img_feats)
