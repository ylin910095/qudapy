#pragma once

// Header only

// Python bindings for QMP within the conext of QUDA
inline void init_qmp_field_pybind(pybind11::module_ &m, bool has_qmp_comms) {
    auto subm = m.def_submodule("qmp");

    subm.def("QMP_get_number_of_nodes", &QMP_get_number_of_nodes);
    subm.def("QMP_get_node_number", &QMP_get_node_number);
    subm.def("QMP_get_logical_coordinates", 
        [has_qmp_comms]() {
            if (!has_qmp_comms) {
                throw std::runtime_error("QMP_COMMS does not exist");
            }
            const int* tmp; 
            std::array<int, 4> grid_corr;
            tmp = QMP_get_logical_coordinates();
            std::copy(&tmp[0], &tmp[4], grid_corr.begin());
            return grid_corr;
        }
    );
}

