import requests
import os
import tarfile
import io
from adam import _logging, padova_data
from adam.resources import adam_res
from typing import List
import numpy as np
import cv2

log = _logging.getLogger(__name__)

data_dir = adam_res.data_dir


def load_padova() -> str:
    """
    Load dataset described at http://signet.dei.unipd.it/human-sensing/
    """
    create_data_dir_if_necessary()
    out_path = f"{data_dir}/padova"
    if not os.path.exists(out_path):
        log.info("downloading padova data")
        download_extract_tar_to(
            "http://signet.dei.unipd.it/wearables/IDNet_dataset.tar.gz", out_path
        )
    return out_path


def load_caltech256() -> str:
    create_data_dir_if_necessary()
    out_path = f"{data_dir}/caltech256"
    if not os.path.exists(out_path):
        log.info("downloading caltech256")
        download_extract_tar_to(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar",
            out_path,
        )
    return f"{out_path}/256_ObjectCategories"


def load_caltech_as_matrices(
    num_per_class: int = None, with_color=False, num_classes=256
) -> List[List]:
    def imread(path):
        if with_color:
            return cv2.imread(path)
        else:
            return cv2.imread(path, 0)

    data_path = load_caltech256()

    def limit_paths(paths):
        if num_per_class is not None:
            return paths[:num_per_class]
        else:
            return paths

    def get_images_at_path(path):
        return [
            imread(f"{path}/{img_path}") for img_path in limit_paths(os.listdir(path))
        ]

    if num_per_class is not None:
        original_get_imgs = get_images_at_path
        get_images_at_path = lambda path: original_get_imgs(path)[:num_per_class]
    imgs_per_class: List[List] = []
    class_dirs = [path for path in os.listdir(data_path) if path[0].isdigit()]
    labels = sorted(class_dirs, key=lambda path: int(path.split(".")[0]))
    for class_i, class_dir in enumerate(
        [f"{data_path}/{label}" for label in labels[:num_classes]]
    ):
        if os.path.isdir(class_dir):
            if class_i % 30 == 0:
                log.info(f'on class: {class_dir.split("/")[-1]}')
            imgs_per_class.append(get_images_at_path(class_dir))
    return imgs_per_class


def load_padova_as_arrs() -> List[np.ndarray]:
    """
    :return: A list of numpy arrays where each numpy array has all data for one user.
    The array columns are ax, ay, ax, gx, gy, gz
    """
    padova_path = load_padova()
    return padova_data.parse_padova(padova_path)


def load_kaggle_ecg():
    """
    Load dataset from https://www.kaggle.com/shayanfazeli/heartbeat
    """
    out_path = f"{data_dir}/heartbeat"
    if os.path.exists(out_path):
        log.debug(f"kaggle ecg dataset is already cached in {out_path}")
        return out_path

    create_data_dir_if_necessary()

    download_extract_tar_to(
        "https://drive.google.com/uc?id=1b4OQz19dlZCIHVKG-UV-ifmDQztrXz4L&export=download",
        data_dir,
    )
    return out_path


# ---- internal methods ----
def download_extract_tar_to(url: str, output_path: str):
    tar_bytes = requests.get(url).content
    log.debug("finished downloading tar. Now beginning extraction...")
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        tar.extractall(output_path)


def create_data_dir_if_necessary():
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
