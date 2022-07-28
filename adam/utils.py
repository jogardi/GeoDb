import io
from typing import *
import itertools
import os
import zipfile

import requests
import torch
from torch.nn import functional as F
from typing import Tuple, Callable
from torch.utils import data as torch_data


def mkdir(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def download_extract_zip_to(link: str, path: str):
    zip_bytes = requests.get(link).content
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip:
        zip.extractall(path)


def dict_groups(xs: Iterable, key: Callable) -> Dict:
    return {
        key: list(group) for key, group in itertools.groupby(sorted(xs, key=key), key)
    }


def mfcc_from_path(path: str) -> torch.Tensor:
    import librosa

    y, sr = librosa.load(path)
    return torch.from_numpy(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))


Embedder = Callable[[torch.Tensor], torch.Tensor]


def train_val_loaders_from_tensors(
    data: torch_data.Dataset, batch_size: int = 128
) -> Tuple[torch_data.DataLoader, torch_data.DataLoader]:
    val_size = min(int(0.15 * len(data)), 2048)
    train_set, val_set = torch_data.random_split(data, [len(data) - val_size, val_size])
    return (
        torch_data.DataLoader(train_set, batch_size, shuffle=True),
        torch_data.DataLoader(val_set, val_size, shuffle=True),
    )


def to_one_hot(y: torch.tensor, num_classes):
    return F.one_hot(y.long(), num_classes=num_classes).float()


def standardize(arr: torch.Tensor) -> torch.Tensor:
    """
    Standardizes along the sepcified axis. Axis is 0 by default and in that case,
    each column will be standardized.
    """
    arr_reshaped = arr.reshape((-1, arr.shape[-1]))
    means = torch.mean(arr_reshaped, dim=0)
    return (
        (arr_reshaped - means)
        / (torch.std(arr_reshaped, dim=0) + 0.001 * torch.ones(arr.shape[-1:]))
    ).reshape(arr.shape)


def each_with_time(arr: torch.Tensor) -> torch.Tensor:
    assert len(arr.shape) == 2
    new_arr = torch.zeros((arr.shape[0], arr.shape[1], 2))
    new_arr[:, :, 0] = torch.linspace(0, 1, steps=arr.shape[1]).repeat(
        (arr.shape[0], 1)
    )
    new_arr[:, :, 1] = arr
    return new_arr


def with_time(arr: torch.tensor) -> torch.tensor:
    if len(arr.shape) == 2:
        dim = arr.shape[1]
    else:
        dim = 1
    new_arr = torch.zeros((arr.shape[0], dim + 1))
    new_arr[:, 0] = torch.linspace(0, 1, steps=arr.shape[0])
    new_arr[:, 1:] = arr.reshape((arr.shape[0], -1))
    return new_arr


def batch_embed(x: torch.Tensor, embedder: Embedder, emb_dim, device) -> torch.Tensor:
    embedded = torch.zeros((len(x), emb_dim))
    cbs = 1024
    for i in range(0, len(x), cbs):
        on_gpu = x[i : i + cbs].to(device)
        embedded[i : i + cbs] = embedder(on_gpu).cpu()
        del on_gpu
        if i % 200 == 0:
            print("on i", i)
            torch.cuda.empty_cache()
    return embedded
