import itertools
from functools import cached_property
from typing import *
from glob import glob

import joblib
from pydub import AudioSegment
from pathlib import Path

import torch
from tqdm import tqdm

from adam import _logging
from adam import utils
import os

log = _logging.getLogger(__name__)


def speaker_id_in_path(path: str) -> int:
    return int(Path(path).stem.split("-")[0])


def speech2phone_labelednp(paths):
    return list(tqdm(map(utils.mfcc_from_path, paths), total=len(paths)))


class Resources:
    adam_dir = utils.mkdir(os.path.expanduser("~/.adam"))
    data_dir = utils.mkdir(f"{adam_dir}/data")
    cache_dir = utils.mkdir(f"{adam_dir}/cache")
    memcache = joblib.Memory(f"{cache_dir}/joblib", verbose=1)

    def __init__(self):
        global speech2phone_labelednp
        speech2phone_labelednp = self.memcache.cache(speech2phone_labelednp)

    @cached_property
    def speech2phonev1(self) -> str:
        out_path = f"{self.data_dir}/speech2phonev1"
        if os.path.exists(out_path):
            log.debug(f"speech2phonev1 dataset is already cached in {out_path}")
            return out_path
        else:
            log.info("downloading speech2phonev1 dataset")
            utils.download_extract_zip_to(
                "https://www.dl.dropboxusercontent.com/s/zx6qsx5ucike92w/Speech2Phone-Dataset-V1.zip",
                out_path,
            )
            return out_path

    @cached_property
    def speech2phone_per_speaker(self) -> Dict[str, torch.Tensor]:

        paths_per_speaker = utils.dict_groups(
            self.speech2phoneprocessed(), key=speaker_id_in_path,
        )

        return {
            speaker: torch.stack(list(map(utils.mfcc_from_path, paths)))
            for speaker, paths in paths_per_speaker.items()
        }

    def speech2phoneprocessed(self):
        return glob(
            f"{self.speech2phonev1}/Speech2Phone-Dataset-V1/preprocessed/X/*.wav"
        )

    def speech2phone_labeled(self, num: int) -> Tuple[torch.Tensor, torch.Tensor]:
        paths = list(itertools.islice(self.speech2phoneprocessed(), num))
        labels = torch.tensor(list(map(speaker_id_in_path, paths)))
        unique_labels = torch.unique(labels).tolist()
        fixed_labels = torch.tensor([unique_labels.index(label) for label in labels])
        return torch.stack(speech2phone_labelednp(paths)), fixed_labels


adam_res = Resources()
