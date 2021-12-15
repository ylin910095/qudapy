#pragma once

// Header only

inline void init_communicator_quda_pybind(pybind11::module_ &m)
{
    //Add submodule
    auto subm = m.def_submodule("communicator_quda", "Wrapper to communicator_quda.h");

    subm.def("comm_finalize", &comm_finalize);
}