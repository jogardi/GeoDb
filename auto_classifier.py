from geodb import auto_metric_learn, utils, _logging
import asyncio
import random
import numpy as np
from sklearn.neighbors import KDTree
import os
import threading
from pykalman import KalmanFilter
import itertools
import torch.optim as optim
import torch.multiprocessing as mp
from pyro.contrib.gp.kernels import RBF
import torch
from torch.utils import data as torch_data
from concurrent.futures import ThreadPoolExecutor
from torch.nn import functional as F
from torch.autograd import Variable
import itertools
import torch.nn as nn
from comet_ml import Experiment
from abc import ABC, abstractmethod
import math
import time
from torch.utils.cpp_extension import load

# import geodbcpp
import geodb

project_root = "/".join(geodb.__file__.split("/")[:-2])
geodbcpp = load(name="geodbcpp", sources=[f"{project_root}/geodbcpp/geodbcpp.cpp"])
loop = asyncio.get_event_loop()

log = _logging.getLogger(__name__)
emb_dim = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
kernel = RBF(emb_dim, lengthscale=0.5 * torch.ones((emb_dim,)))


class SoftmaxWrapper(nn.Module):
    def __init__(self):
        super(SoftmaxWrapper, self).__init__()
        self.final_forward = nn.Sequential(
            nn.Linear(emb_dim, len(num_per_class)), nn.LogSoftmax(dim=1)
        )

    def forward(self, x: Variable) -> Variable:
        return self.final_forward(x)


class FinalClassifier(ABC):
    @abstractmethod
    def get_loss(self, batch_indices: torch.Tensor, backpropagate=True) -> torch.Tensor:
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        pass


class SoftmaxClassifier(FinalClassifier):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.softmax_learner = SoftmaxWrapper()
        self.softmax_learner.to(device)
        self.params = list(self.softmax_learner.parameters())
        self.soft_optim = optim.SGD(self.params, lr=0.01, momentum=0.9, nesterov=True)
        self.train_x = train_x
        self.train_y = train_y

    def get_loss(
        self, embeded: torch.Tensor, y: torch.Tensor, backpropagate=True
    ) -> torch.Tensor:
        weights = torch.tensor([1 / num_in_class for num_in_class in num_per_class]).to(
            device
        )
        print("weights are", weights)
        loss = F.nll_loss(self.softmax_learner(embeded), y.long(), weight=weights)
        if backpropagate:
            loss.backward()
        return loss

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.softmax_learner(self.embedder.eval(x)), dim=1)
        # return torch.argmax(embeded, dim=1)

    def step(self):
        self.soft_optim.step()
        self.soft_optim.zero_grad()

    def save(self, path: str):
        torch.save(self.softmax_learner, path)

    def get_loss_train(self, batch_indices: torch.Tensor) -> torch.Tensor:
        return self.get_loss(
            self.embedder.eval(self.train_x[batch_indices].to(device)),
            self.train_y[batch_indices].to(device),
            True,
        )

    def get_loss_val(self, val_x: torch.Tensor, val_y: torch.Tensor) -> torch.Tensor:
        return self.get_loss(
            self.embedder.eval(val_x.to(device)), val_y.to(device), False
        )


def calc_cov(neighbors: torch.Tensor, kernel=kernel):
    centered = (neighbors - torch.mean(neighbors, dim=0)).cpu()
    # return (kernel(centered) + torch.eye(centered.shape[0])).to(device)
    return (centered @ centere + torch.eye(centered.shape[0])).to(device)


#     return (centered @ centered.T + torch.eye(centered.shape[0])).to(device)


def multivariate_normal(x, L):
    """
    Copied from
    https://github.com/t-vi/candlegp/blob/ecaed1303206ca43fca157cc0e45563c5284eb2e/candlegp/densities.py#L71
    L is the Cholesky decomposition of the covariance.
    x and mu are either vectors (ndim=1) or matrices. In the matrix case, we
    assume independence over the *columns*: the number of rows must match the
    size of L.
    """
    d = x
    if d.dim() == 1:
        d = d.unsqueeze(1)
    alpha, _ = torch.solve(d, L)
    alpha = alpha.squeeze(1)
    num_col = 1 if x.dim() == 1 else x.size(1)
    num_dims = x.size(0)
    # ret = -0.5 * num_dims * num_col * float(np.log(2 * np.pi))
    ret = -num_col * torch.diag(L).log().sum()
    # part = (alpha ** 2).sum()
    ret += -0.5 * (alpha ** 2).sum()
    return ret


def loss_for_neighbors(ex, np_class):
    neighbors, neighbor_labels, y = ex
    r = geodbcpp.loss_for_neighbors(
        neighbors, neighbor_labels, y, torch.tensor(np_class)
    )
    return r
    if len(neighbors) == 0:
        print("empty")
        return torch.tensor(0.0)

    def calc_cov(
        neighbors: torch.Tensor,
        kernel=RBF(emb_dim, lengthscale=0.1 * torch.ones((emb_dim,))),
    ):
        centered = neighbors - torch.mean(neighbors, dim=0)
        return kernel(centered) + torch.eye(centered.shape[0])

    cov = calc_cov(neighbors)
    assert torch.det(cov) > 1e-7
    neighbors_y = F.one_hot(
        neighbor_labels.long(), len(num_per_class)
    ).float()  # / torch.tensor(num_per_class).to(device)
    centered_y = neighbors_y - torch.mean(neighbors_y, dim=0)
    cov_chol = torch.cholesky(cov)
    p = multivariate_normal(centered_y, cov_chol)
    assert p is not None
    assert np_class[y] is not None
    r = -100 * p / (len(num_per_class) * np_class[y])
    assert r is not None
    # r.backward()
    return r


class LocalBayesClassifier:
    embedder = None

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, loss="local_gp"):
        self.train_x = train_x
        self.train_y = train_y
        self.loss = loss
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def batch_embed(self, x: torch.Tensor):
        embedded = torch.zeros((len(x), emb_dim))
        cbs = 1024
        for i in range(0, len(x) - cbs, cbs):
            on_gpu = x[i : i + cbs].to(device)
            embedded[i : i + cbs] = self.eval(on_gpu).cpu()
            del on_gpu
            if i % 200 == 0:
                torch.cuda.empty_cache()
        return embedded

    def eval(self, x: torch.Tensor):
        out = self.embedder.eval(x)
        out_length = torch.norm(out, p=2, dim=1).detach()
        return (out.T / out_length.T).T

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        x = self.eval(x)
        # labels = self.train_y[::2]
        # train_portion = self.train_x[::2]
        labels = self.train_y
        train_portion = self.train_x
        embedded = self.batch_embed(train_portion)
        tree = KDTree(embedded.detach())
        preds = torch.zeros(len(x)).to(device)
        for i in range(len(x)):
            neighbor_indices = torch.tensor(
                tree.query(np.array([x[i].detach().cpu().numpy()]), k=1024)[1][0]
            ).to(self.device)
            rand_ind = torch.randint(0, len(embedded), (2,)).to(device)
            neighbor_indices = torch.cat((neighbor_indices, rand_ind))

            neighborhood = torch.zeros(
                (len(neighbor_indices) + 1, embedded.shape[1])
            ).to(self.device)
            neighborhood[:-1] = embedded[neighbor_indices]
            neighborhood[-1] = x[i]
            #             neighborhood /= (torch.std(neighborhood, dim=0).detach() + 1e-4)
            #             cov = calc_cov(neighborhood, RBF(emb_dim, lengthscale=.5 * torch.std(neighborhood, dim=0).cpu()))
            cov = geodbcpp.calc_cov(neighborhood.cpu()).to(device)
            precision = torch.cholesky_inverse(torch.cholesky(cov))
            neighbors_y = (
                F.one_hot(labels[neighbor_indices].long(), len(num_per_class))
                .float()
                .to(device)
            )  # / torch.tensor(num_per_class).to(device)
            #             neighbors_y = (
            #                 F.one_hot(labels[neighbor_indices].long(), 5).float().to(device)
            #             )
            # preds[i] = torch.argmax(torch.sum(neighbors_y, dim=0))
            y_mean = torch.mean(neighbors_y, dim=0)
            centered_y = neighbors_y  # - y_mean
            pred_per_class = torch.zeros(len(num_per_class)).to(device)
            densities = cov[-1, :-1] @ precision[:-1, :-1] @ centered_y
            # densities = cov[-1, :-1] @ centered_y#[:, class_i]
            for class_i in range(len(num_per_class)):
                pred_per_class[class_i] = densities[class_i] / densities.sum()
            if i % 64 == 0:
                print(pred_per_class)
            preds[i] = torch.argmax(
                # pred_per_class.to(device)
                (pred_per_class / torch.tensor(num_per_class).to(device))
            )
            del neighborhood
        return preds

    def get_loss_train(self, batch_indices: torch.Tensor) -> torch.Tensor:
        other_mask = torch.ones((len(self.train_x),)).bool()
        other_mask[batch_indices] = False
        other_x, other_y = self.train_x[other_mask], self.train_y[other_mask]
        rand_indices = torch.randint(0, len(other_x), (10000,))

        return self.get_loss_for_batch(
            other_x[rand_indices],
            other_y[rand_indices],
            self.train_x[batch_indices],
            self.train_y[batch_indices],
            backpropagate=True,
        )

    def get_loss_val(self, val_x: torch.Tensor, val_y: torch.Tensor) -> torch.Tensor:
        print("shapes for val train", self.train_x.shape)
        return self.get_loss_for_batch(
            self.train_x, self.train_y, val_x, val_y, backpropagate=False
        )

    def get_loss_for_batch(
        self,
        db_x: torch.Tensor,
        db_y: torch.Tensor,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        backpropagate=True,
    ) -> torch.Tensor:
        begin_time = time.time()
        assert len(db_x) == len(db_y)
        assert len(batch_x) == len(batch_y)
        assert len(db_x) > len(batch_x)
        #         gpu_bx, gpu_by = batch_x.to(device), btch_y.to(device)
        all_emb_x = self.eval(batch_x.to(device))
        db_emb = self.batch_embed(db_x)
        start_construct = time.time()
        tree = KDTree(db_emb.detach())
        # print(f"took {time.time() - start_construct} for KDTree")
        num_done = 0
        before_loop = time.time()
        # print("before loop time is ", before_loop - begin_time)

        def gen_neighbors(ex):
            emb_x, y = ex

            close_ind = [
                r
                for r in tree.query(np.array([emb_x.detach().cpu().numpy()]), k=64)[1][
                    0
                ]
            ]
            diff_close_ind = [a for a in close_ind if db_y[a] != y]
            same_close_ind = [a for a in close_ind if db_y[a] == y][
                : int(3 * len(diff_close_ind))
            ]
            # rand_ind = torch.randint(0, len(db_x), (int(0.25 * len(diff_close_ind)),))
            rand_ind = torch.randint(0, len(db_x), (2,))
            ind = torch.cat(
                (rand_ind, torch.tensor(same_close_ind + diff_close_ind).long())
            )
            neighbors = db_emb[ind]
            neighbor_labels = db_y[ind]
            return neighbors, neighbor_labels, y

        neighbors_batches = map(gen_neighbors, zip(all_emb_x, batch_y))
        print("num batches", len(all_emb_x))
        with ThreadPoolExecutor(max_workers=1) as executor:
            loss_results = loop.run_until_complete(
                asyncio.gather(
                    *[
                        loop.run_in_executor(
                            executor, loss_for_neighbors, batch, num_per_class
                        )
                        for batch in neighbors_batches
                    ]
                )
            )
        rand_ind = torch.randint(0, len(db_x), (1024,))
        total_loss = loss_for_neighbors(
            (db_emb[rand_ind], db_y[rand_ind], torch.tensor(0)), num_per_class
        )
        # assert not torch.isnan(total_loss).item()
        total_loss += sum(loss_results)
        assert not torch.isnan(total_loss).item()
        # print("loop time is", time.time() - before_loop)
        if total_loss == 0:
            print("loss is zero")
        total_loss /= len(batch_x)
        assert not torch.isnan(total_loss).item()
        # print("num_done", num_done)
        if backpropagate and total_loss > 0:
            start_derive = time.time()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.embedder.model.parameters(), 0.25)
            for g in self.embedder.model.parameters():
                assert not torch.isnan(g.grad).sum().item()
            # print("derive time is ", time.time() - start_derive)
        # print("total loss time is ", time.time() - begin_time)
        return total_loss

    def step(self):
        pass

    def save(self, path: str):
        # TODO save embeddings
        pass


def train_val_split(data: torch_data.Dataset):
    indices = list(range(len(data)))
    random.shuffle(indices)
    indices = torch.tensor(indices).long()
    return data[indices[512:]], data[indices[:512]]


num_per_class = None
tot_val = None


class LabeledLearner:
    """
    This should be refactored to have the softmax and local bayes implementations more indenpendent and not use
    inheritance.
    """

    def __init__(
        self,
        data: torch_data.Dataset,
        num_classes: int,
        batch_size=512,
        exp: Experiment = None,
        model_save_loc: str = None,
        final_classifier="local_bayes",
    ):
        if exp is not None:
            self.log = _logging.comet_logger(__name__, exp)
        else:
            self.log = log
        epochs_before_switch = 800000
        cycle_period = 40
        current_lr = 0.01
        (train_x, train_y), (val_x, val_y) = train_val_split(data)
        val_x, val_y = val_x.to(device), val_y.to(device)
        log.info(f"num_val, num_train: {len(val_x)}, {train_x.shape}")
        global tot_val, num_per_class
        tot_val = len(val_x)
        num_per_class = num_classes * [None]
        for class_i in range(num_classes):
            num_counted = 0
            for idx in range(len(val_y)):
                if val_y[idx] == class_i:
                    num_counted += 1
            if num_per_class[class_i] is None:
                num_per_class[class_i] = num_counted

        print("num per class", num_per_class)
        auto_learner = auto_metric_learn.AutoMetricLearn(
            lambda epoch: current_lr, train_x.shape[2], emb_dim, exp
        )
        val_losses = []
        classifier_impl = None
        classifier_impl = LocalBayesClassifier(train_x, train_y, loss="local_gp")
        # classifier_impl = SoftmaxClassifier(train_x, train_y)
        # auto_learner.model.soft_max = True

        accs = []

        while True:
            with torch.no_grad():
                classifier_impl.embedder = auto_learner
                auto_learner.model.eval()
                is_correct = (
                    classifier_impl.classify(val_x).int().to(device) == val_y.int()
                ).cpu()
                auto_learner.model.train()
                tot_acc = 0
                for class_i in range(num_classes):
                    num_right = 0
                    num_counted = 0
                    for idx in range(len(val_y)):
                        if val_y[idx] == class_i:
                            num_counted += 1
                            if is_correct[idx]:
                                num_right += 1
                    if num_per_class[class_i] is None:
                        num_per_class[class_i] = num_counted
                    c_acc = float(num_right) / num_counted
                    log.info(f"accuracy for class {class_i} is {c_acc}")
                    tot_acc += c_acc
                accuracy = tot_acc / num_classes
                print(f"tot acc is {tot_acc/num_classes}")

                self.log.info(f"val_acc is {tot_acc/num_classes}")
                accs.append(tot_acc / num_classes)

                val_loss = classifier_impl.get_loss_val(val_x, val_y)
                if exp is not None:
                    exp.log_metric("val_acc", accuracy, step=auto_learner.epoch)
                    exp.log_metric(
                        "val_loss",
                        val_loss.cpu().numpy().item(),
                        step=auto_learner.epoch,
                    )
                val_losses.append(val_loss)
                if (
                    auto_learner.epoch > epochs_before_switch
                    and exp is not None
                    or model_save_loc is not None
                ):
                    if model_save_loc is not None:
                        utils.mkdir(model_save_loc)
                    else:
                        model_save_loc = "/tmp"
                    model_dir = f"{model_save_loc}_{auto_learner.epoch}"
                    utils.mkdir(model_dir)
                    auto_learner.save(f"{model_dir}/feature_extractor")
                    classifier_impl.save(f"{model_dir}/final_classifier.pth")
                    print("starting save")
                    try:
                        exp.log_asset(model_dir, step=auto_learner.epoch)
                    except Exception as e:
                        print("did not work", e)
                    exp.log_asset_folder(
                        model_dir, step=auto_learner.epoch, log_file_name=True
                    )
                    # if (
                    #     len(val_losses) >= 20
                    #     and val_losses[-1] >= val_losses[-2] * 1.02
                    #     and val_losses[-2] >= val_losses[-3] * 1.02
                    # ):
                    #     break
            for _ in range(cycle_period):
                batch_indices = torch.tensor(
                    random.sample(range(len(train_x)), batch_size)
                )
                loss = classifier_impl.get_loss_train(batch_indices)
                if exp is not None:
                    exp.log_metric(
                        "loss",
                        loss.detach().cpu().numpy().item(),
                        step=auto_learner.epoch,
                    )
                self.log.info(f"on {auto_learner.epoch} and loss is {loss}")

                auto_learner.step(loss.detach().cpu())
                classifier_impl.step()
                torch.cuda.empty_cache()

            #             if len(accs) >= 3:
            #                 kf = KalmanFilter(
            #                     transition_matrices=np.array([[1, 1], [0, 1]]),
            #                     transition_covariance=0.05 * np.eye(2),
            #                 )
            #                 means, vars = kf.smooth(accs)
            #                 snr = np.sqrt(means[-1, 1] / np.sqrt(vars[-1, 1, 1]))
            #                 if exp is not None:
            #                     exp.log_metric("snr", snr, step=auto_learner.epoch)
            #                 current_lr = 0.2 * snr
            #                 if len(accs) > 10 and snr <= 1e-8:
            #                     print("done")
            #                     break

            torch.cuda.empty_cache()
            if auto_learner.epoch > epochs_before_switch:
                classifier_impl = LocalBayesClassifier(
                    train_x, train_y, loss="local_gp"
                )
                classifier_impl.embedder = auto_metric_learn
                # auto_learner.soft_max = False

        print(val_losses)

    def triplet_loss(self, embeddings: torch.Tensor, labels: torch.Tensor):
        def get_label(i):
            return labels[i]

        indices = sorted(range(len(embeddings)), key=get_label)
        # for itertools.groupby(indices, key=get_label)
        # triplets = itertools.product(len(embeddings), repeat=3)
        # def loss_for_triplet(triplet):
        #     i, j, k = triplet
        #     torch.dot(embeddings[i], embeddings[j]
        # torch.mean(torch.Tensor([torch.max()]))
