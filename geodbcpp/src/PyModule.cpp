#include <pybind11/stl.h>
#include "geodbcpp.cpp"

PYBIND11_MODULE(geodbcpp, m) {

    m.def("multivariate_normal", &multivariate_normal, "multivariate_normal");
    m.def("calc_cov", &calc_cov, "calc_cov");
    m.def("loss_for_neighbors", &loss_for_neighbors, "loss_for_neighbors");
    m.def("loss_for_neighbors_block", &loss_for_neighbors_block, "loss_for_neighbors_block");

//    return m.ptr();
}
