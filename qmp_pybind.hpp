#pragma once

// Header only


// Python bindings for QMP that is needed for QUDA
inline void init_qio_field_pybind(pybind11::module_ &m, bool has_qmp_comms) {
    auto subm = m.def_submodule("qmp");
}

