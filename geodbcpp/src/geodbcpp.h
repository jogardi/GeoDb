#pragma once

#include <torch/extension.h>
#include <functional>

enum GeoDbKernel {LINEAR, EXP};
struct GeoDbConfig {
    float alpha = 1;
    float min_density = .00001;
    float scale = .5;
    bool use_pcs = true;
    GeoDbKernel kernel = LINEAR;
};

torch::Tensor calc_cov(const torch::Tensor &neighbors, const GeoDbConfig &conf);
/**
 * Calculate the partial correlation matrix of the data for a gaussian process
 * @param neighbors
 * @return
 */
torch::Tensor calc_pcs(const torch::Tensor& neighbors, const GeoDbConfig &conf);

torch::Tensor multivariate_normal(torch::Tensor d, torch::Tensor L);



torch::Tensor estimate_densities(const torch::Tensor &neighbors, const torch::Tensor &ys, const GeoDbConfig &conf);

torch::Tensor loss_for_neighbors_block(const torch::Tensor &neighbors, const torch::Tensor &neighbor_labels,
                                       const torch::Tensor &np_class, const GeoDbConfig &conf);

torch::Tensor
loss_for_neighbors(const torch::Tensor &neighbors, const torch::Tensor &neighbor_labels, const torch::Tensor &np_class, const GeoDbConfig &conf);
