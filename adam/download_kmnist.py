import requests
from adam.datasets import data_dir
from adam import utils


def download_kmnist():
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = (
            lambda x, total, unit: x
        )  # If tqdm doesn't exist, replace it with a function that does nothing
        print(
            "**** Could not import tqdm. Please install tqdm for download progressbars! (pip install tqdm) ****"
        )

    download_dict = {
        "1) Kuzushiji-MNIST (10 classes, 28x28, 70k examples)": {
            "1) MNIST data format (ubyte.gz)": [
                "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz",
                "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz",
                "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz",
                "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz",
            ],
            "2) NumPy data format (.npz)": [
                "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz",
                "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz",
                "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz",
                "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz",
            ],
        },
        "2) Kuzushiji-49 (49 classes, 28x28, 270k examples)": {
            "1) NumPy data format (.npz)": [
                "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz",
                "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz",
                "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz",
                "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz",
            ]
        },
        "3) Kuzushiji-Kanji (3832 classes, 64x64, 140k examples)": {
            "1) Folders of images (.tar)": [
                "http://codh.rois.ac.jp/kmnist/dataset/kkanji/kkanji.tar"
            ]
        },
    }
    kmnist_dir = utils.prep_dir(f"{data_dir}/kmnist")
    # Download a list of files
    def download_list(url_list):
        for url in url_list:
            path = f"{data_dir}/kmnist/{url.split('/')[-1]}"
            print(url)
            r = requests.get(url, stream=True)
            with open(path, "wb") as f:
                total_length = int(r.headers.get("content-length"))
                print(
                    "Downloading {} - {:.1f} MB".format(path, (total_length / 1024000))
                )

                for chunk in tqdm(
                    r.iter_content(chunk_size=1024),
                    total=int(total_length / 1024) + 1,
                    unit="KB",
                ):
                    if chunk:
                        f.write(chunk)
        print("All dataset files downloaded!")

    download_list(["http://codh.rois.ac.jp/kmnist/dataset/kkanji/kkanji.tar"])
    return kmnist_dir
