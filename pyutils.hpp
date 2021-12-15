#pragma once

#include <pybind11/pybind11.h>

#include "quda.h"
#include "gauge_field.h" 

// Header only

inline int lex_rank_from_coords_t(const int *coords, void *fdata) {
    int rank = coords[0];
    int* grid_dims = (int*) fdata;
    for (int i = 1; i < 4; i++) { rank = grid_dims[i] * rank + coords[i]; }
    return rank;
}

// Copy gaugeSmeared to gaugePrecise
// QUDA API does not include copyExtendedGauge so we have to 
// implement it ourselves 
inline void init_pyutils(pybind11::module_ &m, bool has_qmp_comms) {   
    auto subm = m.def_submodule("pyutils"); 

    // Wrapper around initComms in tests/utils/host_utilities.cpp
    // to initiate communications either with QMP or MPI
    // TODO: Maybe separte into multiple files to expose finer details?
    subm.def(
        "init_comms",
        [has_qmp_comms](const std::vector<int> &comm_dims) {
            int rank;
            int argc = 1;
            char **argv = (char **)malloc(sizeof(char *));
            int n_dims = 4;
            if (has_qmp_comms) {
                QMP_thread_level_t tl;
                QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);

                // Make sure the QMP logical ordering matches QUDA's
                int map[] = {3, 2, 1, 0}; // [x,y,z,t], t fastest moving
                QMP_declare_logical_topology_map(comm_dims.data(), n_dims, map, n_dims);
                rank = QMP_get_node_number();
            } else {
                MPI_Init(&argc, &argv);
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            }

            QudaCommsMap func = lex_rank_from_coords_t;
            // We need to cast away the constantness of the comm_dims to use it
            // However, we are not really change it in anyway
            initCommsGridQuda(n_dims, comm_dims.data(), func, const_cast<int*>(comm_dims.data()));

            // Set the random number seed
            srand(17 * rank + 137);

            free(argv);
            initQuda(rank); 
        },
        R"pbdoc(
        Initilialize the topology and library.
        )pbdoc"
    );

    subm.def("copySmearedToPrecise", 
        []() {
            extern quda::cudaGaugeField* gaugeSmeared; // HACK. There must have a better way to do it.
            extern quda::cudaGaugeField* gaugePrecise;

            // Makeing sure we are actually accessing the global variable in interface_quda.cpp
            if (gaugeSmeared == nullptr) throw std::runtime_error("gaugeSmeared does not exist");
            quda::copyExtendedGauge(*gaugePrecise, *gaugeSmeared, QUDA_CUDA_FIELD_LOCATION);
        }
    );

    subm.def("setQudaVerbosityStdout", 
        [](QudaVerbosity verbosity) {
            setVerbosityQuda(verbosity, "QUDA: ", stdout);
        }
    );  
}

