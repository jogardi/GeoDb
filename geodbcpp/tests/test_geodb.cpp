#include "gtest/gtest.h"
#include "geodbcpp.h"
#include <vector>
//#include <torch/torch.h>

std::string dwds = "Users/joeyg/Downloads";

TEST(test_geo, test_geo1) {
    std::ifstream dataf(dwds + "/speech2phone512labels.pt", std::ios::binary);
    std::vector<char> datavec(std::istreambuf_iterator<char>{dataf}, {});
//    std::cout << "PyTorch version: "
//    << TORCH_VERSION_MAJOR << "."
//    << TORCH_VERSION_MINOR << "."
//    << TORCH_VERSION_PATCH << std::endl;
//    auto a = torch::pickle_load(datavec);

}
