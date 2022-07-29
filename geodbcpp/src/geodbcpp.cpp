#include "geodbcpp.h"
#include <math.h>
#include <tuple>
#include <chrono>

#define _USE_MATH_DEFINES

#include <math.h>
#include <thread>
#include <future>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <ATen/ATen.h>
#include <torch/nn/functional.h>
#include <torch/extension.h>
#include <glog/logging.h>

using namespace std;
namespace F = torch::nn::functional;

struct GLogInit {
    GLogInit() {
        google::InitGoogleLogging("GeoDb");
    }
};
namespace {
    static GLogInit glog_init;
}

torch::Tensor multivariate_normal(torch::Tensor d, torch::Tensor L) {
    int num_col = 0;
    if (d.dim() == 1) {
        d = d.unsqueeze(1);
        num_col = 1;
    } else {
        num_col = d.size(1);
    }
    int num_dims = d.size(0);
//    torch::Tensor ret = -0.5 * num_dims * num_col * at::log(2 * boost::math::constants::pi<float>());
    torch::Tensor ret = -num_col * at::diag(L).log().sum();
    return ret;
}


torch::Tensor calc_cov(const torch::Tensor &neighbors, const GeoDbConfig &conf) {
    int num = neighbors.size(0);
    torch::Tensor centered = neighbors - at::mean(neighbors, 0);
    auto cov = torch::einsum("ik,jk->ij", {centered, centered});
    if (conf.kernel == GeoDbKernel::EXP) {
        cov = torch::exp(-cov * conf.scale);
    }
    return cov + conf.alpha * torch::eye(num);
}


torch::Tensor calc_pcs(const torch::Tensor &neighbors, const GeoDbConfig &conf) {
    auto cov = calc_cov(neighbors, conf);
//    std::cout << "didn't break " << cov.sizes() << std::endl;
    auto precision = cov.inverse();
    auto pcs = torch::einsum("ik,kl,jl->ij", {cov, precision, cov});
//    auto cov = centered.matmul(centered.t());
//    auto r2 = X2 - 2 * cov + X2.t();
    return pcs;
//    return at::exp(-.5 * r2.clamp(0)) + 1 * torch::eye(num);
    //return centered.matmul(centered.transpose(0, 1)) + torch::eye(num);
//    torch::Tensor cov = torch::zeros({num, num});
//    for (int i = 0; i < num; i++) {
//        for (int j = 0; j <= i; j++) {
//            torch::Tensor dist = at::pow(centered[i] - centered[j], 2).sum();
//            cov[i][j] = at::exp(-.5 * dist / pow(.1, 2));
//            if (i == j) {
//                cov[i][j] += 1;
//            } else {
//                cov[j][i] = cov[i][j];
//            }
//        }
//    }
//    return cov;
}

torch::Tensor estimate_densities(const torch::Tensor &neighbors, const torch::Tensor &ys, const GeoDbConfig &conf) {
    torch::Tensor cov = conf.use_pcs ? calc_pcs(neighbors, conf) : calc_cov(neighbors, conf);
    assert(at::det(cov).item<float>() > pow(10, -7));
//    auto chol = torch::linalg::cholesky(cov);
    for (int i = 0; i < cov.size(0); i++) {
        cov[i][i] = 0;
    }
    int num_col = 0;
    //torch::Tensor transformed = at::cholesky_solve(ys, chol);
    return cov.matmul(ys) + conf.min_density;
}


torch::Tensor loss_for_neighbors_block(const torch::Tensor &neighbors, const torch::Tensor &neighbor_labels,
                                       const torch::Tensor &np_class, const GeoDbConfig &conf) {
    if (neighbors.size(0) == 0) {
        cout << "empty" << endl;
        return torch::zeros({1})[0];
    }

    torch::Tensor neighbors_y = (at::one_hot(neighbor_labels, np_class.size(0)).to(
            torch::kFloat32));// / at::sqrt(np_class.to(torch::kFloat32));
    auto densities = estimate_densities(neighbors, neighbors_y, conf);
    //densities = (densities.transpose(0, 1) / (densities.sum(1).mul(np_class))).transpose(0, 1);
//    densities /=  np_class;
    densities = (densities.transpose(0, 1) / (densities.sum(1))).transpose(0, 1);
    assert(at::isnan(densities).sum().item<float>() == 0);
    auto rr = torch::nn::functional::cross_entropy(densities, neighbor_labels,
                                                   F::CrossEntropyFuncOptions().ignore_index(-100).reduction(
                                                           torch::kMean));
    if (rr.isnan().any().item<bool>()) {
        exit(1);
    }
    return rr;
    // torch::Tensor centered_y = neighbors_y// - at::mean(neighbors_y, 0);
    // torch::Tensor cov_chol = at::cholesky(cov);
    // torch::Tensor p = multivariate_normal(centered_y, cov_chol);
    // return -100 * p / (5 * np_class[y.item<int64_t>()]);
    // return -100 * p / (5;
}

void loss_for_neighbors_async(const torch::Tensor &neighbors, const torch::Tensor &neighbor_labels,
                              const torch::Tensor &np_class, std::promise<torch::Tensor> *promObj) {
    std::cout << "not implemented" << std::endl;
    exit(1);
//    promObj->set_value(loss_for_neighbors_block(neighbors, neighbor_labels, np_class));
}

torch::Tensor
loss_for_neighbors(const torch::Tensor &neighbors, const torch::Tensor &neighbor_labels, const torch::Tensor &np_class,
                   const GeoDbConfig &conf) {
    py::gil_scoped_release release;
    return loss_for_neighbors_block(neighbors, neighbor_labels, np_class, conf);
}
//def loss_for_neighbors(ex, np_class):
//print("loss for neighbors")
//neighbors, neighbor_labels, y = ex
//if len(neighbors) == 0:
//print("empty")
//return torch.tensor(0.0)
//
//def calc_cov(
//        neighbors: torch.Tensor,
//kernel=RBF(emb_dim, lengthscale=0.1 * torch.ones((emb_dim,))),
//):
//centered = neighbors - torch.mean(neighbors, dim=0)
//return kernel(centered) + torch.eye(centered.shape[0])
//
//cov = calc_cov(neighbors)
//assert torch.det(cov) > 1e-7
//neighbors_y = F.one_hot(
//        neighbor_labels.long(), 5
//).float()  # / torch.tensor(num_per_class).to(device)
//centered_y = neighbors_y - torch.mean(neighbors_y, dim=0)
//cov_chol = torch.cholesky(cov)
//p = multivariate_normal(centered_y, cov_chol)
//assert p is not None
//        assert np_class[y] is not None
//        r = -100 * p / (5 * np_class[y])
//assert r is not None
//        r.backward()
//return r.detach()

//namespace py = pybind11;

//PYBIND11_PLUGIN(geodbcpp) {
//    py::module m("geodbcpp", "pybind11 geodbcpp plugin");
