import io
from typing import *
import itertools
import os
import zipfile

import librosa
import requests
import torch


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
    y, sr = librosa.load(path)
    return torch.from_numpy(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
