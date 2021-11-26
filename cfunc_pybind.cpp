#include "cfunc_pybind.hpp"

void init_cfunc_pybind(pybind11::module_ &m){
    // Add submodule
    auto cfunc_module = m.def_submodule("cfunc", "Wrapper to common C/C++ functions");

    // For struct_size member in QudaGaugeParam
    cfunc_module.def("sizeof", 
        [](const pybind11::object &obj) {
            try {
                QudaGaugeParam o = obj.cast<QudaGaugeParam>();
                return sizeof(o);
            } catch(const pybind11::cast_error &e) {
                throw pybind11::cast_error("sizeof is only implemented for QudaGaugeParam");
            };
        }
    );
}