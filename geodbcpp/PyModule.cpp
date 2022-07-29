#include "src/geodbcpp.h"
#include <torch/extension.h>

PYBIND11_MODULE(geodbcpp, m) {
    m.def("multivariate_normal", &multivariate_normal, "multivariate_normal");
    m.def("calc_cov", &calc_cov, "calc_cov");
    m.def("loss_for_neighbors", &loss_for_neighbors, "loss_for_neighbors");
    m.def("loss_for_neighbors_block", &loss_for_neighbors_block, "loss_for_neighbors_block");
    m.def("calc_pcs", &calc_pcs, "calc_pcs");
    py::enum_<GeoDbKernel>(m, "Kernel")
        .value("LINEAR", GeoDbKernel::LINEAR)
        .value("EXP", GeoDbKernel::EXP)
        .export_values();
    py::class_<GeoDbConfig>(m, "GeoDbConfig")
            .def(py::init())
            .def_readwrite("min_density", &GeoDbConfig::min_density)
            .def_readwrite("kernel", &GeoDbConfig::kernel)
            .def_readwrite("use_pcs", &GeoDbConfig::use_pcs)
            .def_readwrite("scale", &GeoDbConfig::scale)
            .def_readwrite("alpha", &GeoDbConfig::alpha);
}
