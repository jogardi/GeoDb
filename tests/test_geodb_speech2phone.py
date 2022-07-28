import unittest

import torch
from torch import optim

from adam.resources import adam_res
from adam.speaker_id import SpeakerVoiceEmbedder


class MyTestCase(unittest.TestCase):
    def test_untrained(self):
        data, labels = adam_res.speech2phone_labeled(512)
        data_flat = data.view(data.shape[0], -1)
        ntrain = 450
        model = SpeakerVoiceEmbedder(data_flat.shape[1])
        model.train()
        opt = optim.AdamW(model.parameters(), lr=0.001)
        tot = 0
        tot_corrrect = 0
        for ii in range(1000):
            embedding = model(data_flat)
            for sample_i in range(ntrain, len(embedding)):
                dists = torch.norm(
                    embedding[:ntrain] - embedding[sample_i].T, p=2, dim=1
                )
                nearest_neighbor_i = dists.argmin()
                is_correct = labels[sample_i] == labels[nearest_neighbor_i]
                if is_correct:
                    dists[nearest_neighbor_i].backward(retain_graph=True)
                    tot_corrrect += 1
                else:
                    (-dists[nearest_neighbor_i]).backward(retain_graph=True)
                tot += 1

            # loss = nn.CrossEntropyLoss()(model(data_flat), labels)
            opt.step()
            opt.zero_grad()
            if ii % 2 == 0:
                print(f"Accuracy: ii, {tot_corrrect / tot}")
        print(f"Accuracy: {tot_corrrect / tot}")


if __name__ == "__main__":
    unittest.main()
