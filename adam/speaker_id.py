import torch
from torch import nn


class SpeakerVoiceEmbedder(nn.Module):
    def __init__(self, in_dim: int):
        super(SpeakerVoiceEmbedder, self).__init__()
        self.base = nn.Sequential(nn.Linear(in_dim, 16))

    def forward(self, x):
        out = self.base(x)
        return out / torch.norm(out, p=2, dim=0)
        # out_length = torch.norm(out, p=2, dim=1).detach()
        return (out.T / out_length.T).T
