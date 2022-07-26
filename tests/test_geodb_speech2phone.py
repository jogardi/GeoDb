import unittest

import torch
from torch import nn, optim

from adam.resources import adam_res


class SpeakerVoiceEmbedder(nn.Module):
    def __init__(self, in_dim: int):
        super(SpeakerVoiceEmbedder, self).__init__()
        self.base = nn.Sequential(nn.Linear(in_dim, 16))

    def forward(self, x):
        out = self.base(x)
        out_length = torch.norm(out, p=2, dim=1).detach()
        return (out.T / out_length.T).T


class MyTestCase(unittest.TestCase):
    def test_something(self):
        data, labels = adam_res.speech2phone_labeled(512)
        data_flat = data.view(data.shape[0], -1)
        ntrain = 450
        model = SpeakerVoiceEmbedder(data_flat.shape[1])
        model.train()
        opt = optim.AdamW(model.parameters(), lr=0.001)
        for ii in range(1000):
            embedding = model(data_flat)
            for sample_i in range(ntrain, len(embedding)):
                dists = torch.norm(embedding[:ntrain] - embedding[sample_i], p=2)
                nearest_neighbor_i = dists.argmin()
                print("rr", labels[sample_i] == labels[nearest_neighbor_i])

            # loss = nn.CrossEntropyLoss()(model(data_flat), labels)
            opt.step()
            opt.zero_grad()


if __name__ == "__main__":
    unittest.main()
