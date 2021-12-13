#pragma once

#include <pybind11/pybind11.h>

#include "quda.h"
#include "gauge_field.h" 

// Header only

// Copy gaugeSmeared to gaugePrecise
// QUDA API does not include copyExtendedGauge so we have to 
// implement it ourselves 
inline void init_pyutils(pybind11::module_ &m)
{   
    auto csf = m.def_submodule("pyutils"); 
    csf.def("copySmearedToPrecise", 
        []() {
            extern quda::cudaGaugeField* gaugeSmeared; // HACK. There must have a better way to do it.
            extern quda::cudaGaugeField* gaugePrecise;

            // Makeing sure we are actually accessing the global variable in interface_quda.cpp
            if (gaugeSmeared == nullptr) throw std::runtime_error("gaugeSmeared does not exist");
            quda::copyExtendedGauge(*gaugePrecise, *gaugeSmeared, QUDA_CUDA_FIELD_LOCATION);
        }
    );

    csf.def("setQudaVerbosityStdout", 
        [](QudaVerbosity verbosity) {
            setVerbosityQuda(verbosity, "QUDA: ", stdout);
        }
    );  
}

