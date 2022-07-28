#include "gtest/gtest.h"
#include "geodbcpp.h"
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
namespace py = pybind11;

template <typename T>
torch::Tensor as_ten(py::array_t<T> arr) {
    return torch::from_blob(arr.mutable_data(), {arr.shape()[0]}, torch::dtype(torch::kLong));
}
std::string dwds = "Users/joeyg/Downloads";

TEST(test_geo, test_geo1) {
    std::ifstream dataf(dwds + "/speech2phone512labels.pt", std::ios::binary);
    std::vector datavec(std::istreambuf_iterator<char>{dataf}, {});
    auto a = torch::pickle_load(datavec);

}
