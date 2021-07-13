"""
Plan here is to create a server that can search either in ECG or images.
That's it really
"""
import numpy as np
from typing import Dict, List, Callable, Union
import torch
from sklearn.neighbors import KDTree
from operator import itemgetter
from sklearn.decomposition import PCA
from torch.utils.cpp_extension import load
import geodb

project_root = "/".join(geodb.__file__.split("/")[:-2])
geodbcpp = load(name="geodbcpp", sources=[f"{project_root}/geodbcpp/geodbcpp.cpp"])

ManifoldLink = str
"""
The path to the user's data relative to the root path for that user. Sometimes followed by a
f":{p}" where  p is the position within the file
"""
Sim = float
"""
float in the range [0, 1] that measures similarity
"""

from geodb.utils import Embedder, batch_embed

device = "cuda" if torch.cuda.is_available() else "cpu"


class Dataset:
    def __init__(
        self,
        links: List[ManifoldLink],
        data: torch.Tensor,
        annotations: Union[List, torch.Tensor],
    ):
        self.links = links
        self.index_of_link = {link: i for i, link in enumerate(links)}
        self.data = data
        self.annotations = annotations

    def __len__(self):
        return len(self.links)


class SearchableKernel:
    """
    This is the core abstration behind NeuralDB
    """

    def __init__(self, embedder: Embedder, embedder_dim: int, dataset: Dataset):
        self.db_x: torch.Tensor = batch_embed(
            dataset.data, embedder, embedder_dim, device
        ).detach()
        self.tree = KDTree(self.db_x.cpu().numpy())
        self.dataset = dataset

    def dlkpca(self, neighbor_coordinates: torch.Tensor) -> torch.Tensor:
        """
        does DLKPCA (Deep Local Kernelized Principle Component Analysis)
        The key point of it is that the result is invariant to the original representation of
        the data because it depends only on the kernel
        """
        cov = geodbcpp.calc_cov(neighbor_coordinates)
        lower_dim_rep = torch.tensor(PCA(n_components=3).fit_transform(cov))
        # lower_dim_rep[:, 3:] = 255 * torch.sigmoid(lower_dim_rep[:, 3:])
        return lower_dim_rep

    def search(
        self,
        query_link: ManifoldLink,
        radius: float = None,
        k: int = None,
        max_points=2 ** 8,
    ) -> Dict:
        query_i = self.dataset.index_of_link[query_link]
        if radius is not None:
            assert k is None
            neighbor_inds = self.tree.query_radius(
                [self.db_x[query_i].cpu().numpy()], radius
            )[0]
            if len(neighbor_inds) > max_points:
                neighbor_inds = np.random.choice(
                    neighbor_inds, max_points, replace=False
                )
            elif len(neighbor_inds) < 5:
                return self.search(query_link, 1.5 * radius, k, max_points)
        else:
            assert k is not None
            neighbor_inds = self.tree.query([self.db_x[query_i].cpu().numpy()], k)[1][0]
        print("num_neighbors", len(neighbor_inds))
        neighbors = self.db_x[neighbor_inds]
        lower_dim_rep = (self.dlkpca(neighbors)).tolist()
        links = itemgetter(*neighbor_inds)(self.dataset.links)
        return {
            "results": list(zip(links, lower_dim_rep)),
            # "raw_data": self.dataset.data[neighbor_inds].tolist(),
            "new_radius": radius,
        }

    def sim(self, a: ManifoldLink, b: ManifoldLink) -> Sim:
        a_i, b_i = self.dataset.index_of_link[a], self.dataset.index_of_link[b]
        return geodbcpp.cal_cov(torch.stack([self.db_x[a_i], self.db_x[b_i]]))[1, 0]
