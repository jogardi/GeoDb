#pragma once

#include <torch/extension.h>

torch::Tensor multivariate_normal(torch::Tensor d, torch::Tensor L);

torch::Tensor calc_cov(torch::Tensor neighbors);

torch::Tensor estimate_densities(torch::Tensor neighbors, torch::Tensor ys);

torch::Tensor loss_for_neighbors_block(const torch::Tensor& neighbors, const torch::Tensor& neighbor_labels, const torch::Tensor& y,
                                       torch::Tensor np_class);

torch::Tensor
loss_for_neighbors(const torch::Tensor& neighbors, torch::Tensor neighbor_labels, torch::Tensor y, torch::Tensor np_class);
