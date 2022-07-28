import torch.nn as nn
import numpy as np
from collections import OrderedDict
import itertools
from comet_ml import Experiment
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from geodb.cloned import cyclic_lr
from pykalman import KalmanFilter


class AutoMetricLearn:
    """
    Used tutorial on AdamW and CyclicLR at https://towardsdatascience.com/customer-case-study-building-an-end-to-end-speech-recognition-model-in-pytorch-with-assemblyai-473030e47c7c
    """

    # TODO experiment recoridng and blackbox optimization with comet.ml
    def __init__(self, get_lr, in_dim, emb_dim, exp: Experiment = None):
        self.model = MetricLearner(in_dim, emb_dim, num_dilations=4)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("using device", self.device)
        self.params = None
        self.model.to(self.device)
        self.optimizer = self.make_optimizer()

        self.lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=get_lr
        )
        # self.lr_scheduler = cyclic_lr.CyclicLR(
        #     self.optimizer,
        #     base_lr=1e-6,
        #     max_lr=1e-4,
        #     cycle_momentum=False,
        #     step_size_up=cycle_period // 2,
        # )
        self.exp = exp
        self.epoch = 0

        self.losses = []
        self.wait_time = self.prev_time = 500
        self.best_loss = None
        self.kf = None
        self.last_mean = None
        self.last_cov = None

    def make_optimizer(self):
        self.params = list(self.model.parameters())
        return optim.SGD(
            self.params, lr=0.1, weight_decay=1e-4, momentum=0.9, nesterov=True
        )

    def eval(self, inputs: torch.tensor) -> torch.tensor:
        return self.model(inputs)

    def step(self, loss):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        if self.exp is not None:
            self.exp.log_metric(
                "learning_rate", self.lr_scheduler.get_last_lr(), step=self.epoch
            )
        self.epoch += 1
        # reinitialize model with new layer here
        self.losses.append(loss.detach())
        if len(self.losses) == 10:
            self.kf = KalmanFilter(
                initial_state_mean=[self.losses[0], 0],
                observation_covariance=10 * np.eye(1),
                transition_matrices=np.array([[1, 1], [0, 1]]),
                transition_covariance=0.001 * np.eye(2),
            )
            means, vars = self.kf.filter(self.losses)
            self.last_mean, self.last_cov = means[-1], vars[-1]
        if self.kf is not None:
            self.last_mean, self.last_cov = self.kf.filter_update(
                self.last_mean, self.last_cov, loss
            )
            self.exp.log_metric("smoothed_loss", self.last_mean[0])
            if self.best_loss is None or self.last_mean[0] < 0.8 * self.best_loss:
                self.best_loss = self.last_mean[0]
                self.wait_time = self.prev_time = 500 + 0.5 * len(self.losses)
            print("wait time is", self.wait_time)
        self.wait_time -= 1
        if len(self.losses) > 10:
            if self.wait_time <= 0:
                # new_net = MetricLearner(self.model.in_dim, self.model.emb_dim, self.model.num_dilations + 1)
                # new_net.load_state_dict(self.model.state_dict(), strict=False)
                # self.model = new_net
                # self.model.to(self.device)
                # self.optimizer = self.make_optimizer()
                self.wait_time = self.prev_time = 500 + 4 * len(self.losses)
                self.exp.log_metric("dilations", new_net.num_dilations)
                print("wait time is", self.wait_time)
                self.losses = []

    def save(self, path: str):
        torch.save(self.model.state_dict(), path + ".pth")


class MetricLearner(nn.Module):
    def __init__(self, in_dim, emb_dim, num_dilations=0):
        super(MetricLearner, self).__init__()
        self.in_dim = in_dim
        self.soft_max = False
        self.emb_dim = emb_dim
        self.num_dilations = num_dilations

        self.num_dilations = num_dilations
        next_dim = 2 ** (int(np.log2(in_dim)) + 1)
        base_out_dim = next_dim * 2
        self.base = nn.Sequential(
            nn.Conv1d(in_dim, next_dim, 3),
            nn.BatchNorm1d(next_dim),
            nn.ReLU(),
            nn.Conv1d(next_dim, base_out_dim, 3),
            nn.BatchNorm1d(base_out_dim),
            nn.ReLU(),
        )
        print(self.base)

        def make_dilation(i: int) -> list:
            in_channels = base_out_dim * 2 ** i
            out_channels = 2 * in_channels
            return [
                nn.Conv1d(in_channels, out_channels, 3, dilation=2 * 2 ** i),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ]

        self.dilations = nn.Sequential(
            *itertools.chain(*map(make_dilation, range(num_dilations)))
        )
        print(self.dilations)
        self.conv_out_dim = base_out_dim * 2 ** num_dilations

        def make_dense_layer(i: int):
            in_dim = self.conv_out_dim // (2 ** i)
            out_dim = in_dim // 2
            return (
                f"dim_reduce_{num_dilations - i}",
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.Dropout(0.2),
                    nn.ReLU(),
                ),
            )

        self.dim_reduce = nn.Sequential(
            OrderedDict(map(make_dense_layer, range(num_dilations)))
        )
        print(self.dim_reduce)
        self.final_forward = nn.Sequential(nn.Linear(base_out_dim, self.emb_dim),)
        print(self.final_forward)

        # original
        #         self.transform = nn.Sequential(
        #             nn.Conv1d(2, 32, 3),
        #             nn.BatchNorm1d(32),
        #             nn.ReLU(),
        #             nn.Conv1d(32, 16, 3),
        #             nn.BatchNorm1d(16),
        #             nn.ReLU(),
        #             nn.MaxPool1d(kernel_size=4, stride=3)
        #         )
        #         self.final_forward = nn.Sequential(
        #             nn.Linear(272, 16),
        #             nn.BatchNorm1d(16),
        #             nn.Dropout(.4),
        #             nn.ReLU()
        #         )
        # self.transform = nn.Sequential(
        #     nn.Conv1d(2, 16, 3),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.Conv1d(16, 32, 3, dilation=2),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 64, 3, dilation=4),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 128, 3, dilation=4),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 256, 3, dilation=4),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     #             nn.Conv1d(32, 16, 3),
        #     #             nn.BatchNorm1d(16),
        #     #             nn.ReLU(),
        #     #             nn.Conv1d(64, 64, 3, dilation=4),
        #     #             nn.BatchNorm1d(64),
        #     #             nn.ReLU(),
        #     #             nn.Conv1d(64, 64, 3, dilation=4),
        #     #             nn.BatchNorm1d(64),
        #     #             nn.ReLU(),
        #     # nn.Conv1d(32, 16, 3),
        #     # nn.ReLU(),
        #     #             nn.MaxPool1d(4, stride=3)
        # )
        # for local bayes
        # self.final_forward = nn.Sequential(
        #     nn.Linear(256, 64),
        #     nn.BatchNorm1d(64),
        #     nn.Dropout(0.2),
        #     nn.ReLU(),
        #     nn.Linear(64, self.emb_dim),
        # )

        # for softmax

    #         self.final_forward = nn.Sequential(
    #             nn.Linear(256, 16),
    #             nn.BatchNorm1d(16),
    #             nn.Dropout(0.1),
    #             nn.ReLU(),
    #         )

    def forward(self, x: Variable) -> Variable:
        transformed = self.dilations(self.base(x.transpose(1, 2))).transpose(1, 2)
        maxed, _ = transformed.max(dim=1)

        out = self.final_forward(self.dim_reduce(maxed))
        if self.soft_max:
            return out
        else:
            # embed in hypersphere in a differentiable way
            out_length = torch.norm(out, p=2, dim=1).detach()
            return (out.T / out_length.T).T

        transformed = self.transform(x.transpose(1, 2)).transpose(1, 2)
        # original
        #         flattened = transformed.reshape((transformed.shape[0], -1))
        maxed, _ = transformed.max(dim=1)
        #         return self.final_forward(maxed)
        out = self.final_forward(maxed)
        #         out = self.final_forward(flattened)
        if self.soft_max:
            return out
        else:
            # embed in hypersphere in a differentiable way
            out_length = torch.norm(out, p=2, dim=1).detach()
            return (out.T / out_length.T).T
