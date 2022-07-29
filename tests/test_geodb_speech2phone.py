import datetime
from torch.utils import data as tdata
import unittest
import numpy as np

import torch
from torch import optim

from adam import utils
from adam.resources import adam_res
from adam.speaker_id import SpeakerVoiceEmbedder


class MyTestCase(unittest.TestCase):
    def test_untrained(self):
        data, labels = adam_res.speech2phone_labeled_all()
        rand_idxs = torch.randperm(len(data))
        ntrain = 1800
        data = data[rand_idxs]
        labels = labels[rand_idxs]

        print("num speakers", len(set(labels.tolist())))
        data_flat = data.view(data.shape[0], -1)
        model = SpeakerVoiceEmbedder(data_flat.shape[1])
        model.train()
        opt = optim.AdamW(model.parameters(), lr=0.001)
        tot = 0
        tot_corrrect = 0
        models_dir = utils.mkdir(f"{adam_res.models_dir}/model_speech2phone_embedder")
        for ii in range(1000):
            batch_idxs = np.random.choice(np.arange(0, len(data[:ntrain])), 64)
            batch_x = data_flat[batch_idxs]
            batch_y = labels[batch_idxs]
            embedding = model(batch_x)
            nmem = 50
            for sample_i in range(nmem, len(embedding)):
                dists = torch.norm(embedding[:nmem] - embedding[sample_i].T, p=2, dim=1)
                nearest_neighbor_i = dists.argmin()
                is_correct = batch_y[sample_i] == batch_y[nearest_neighbor_i]
                if is_correct:
                    dists[nearest_neighbor_i].backward(retain_graph=True)
                    tot_corrrect += 1
                else:
                    (-dists[nearest_neighbor_i]).backward(retain_graph=True)
                tot += 1

            opt.step()
            opt.zero_grad()
            if ii % 2 == 0:
                print(f"Accuracy: ii, {tot_corrrect / tot}")
            if ii % 100 == 0:
                t = datetime.datetime.now()
                torch.save(
                    model.state_dict(),
                    f"{models_dir}/model_{type(model).__name__}_{ii}_{t.strftime('ymdhms_%Y-%m-%d%H-%M-%S')}.pt",
                )
        print(f"train accuracy: {tot_corrrect / tot}")

        embedding = model(data_flat)
        tot = 0
        tot_corrrect = 0

        for sample_i in range(ntrain, len(data_flat)):
            dists = torch.norm(embedding[:ntrain] - embedding[sample_i].T, p=2, dim=1)
            nearest_neighbor_i = dists.argmin()
            if labels[sample_i] == labels[nearest_neighbor_i]:
                tot_corrrect += 1
            tot += 1
        print(f"test accuracy: {tot_corrrect / tot}")


if __name__ == "__main__":
    unittest.main()
