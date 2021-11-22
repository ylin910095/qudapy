#pragma once

#include <string>

#include <pybind11/stl.h> 
#include <pybind11/pybind11.h>

#include "qio_field.h"
#include "enum_quda.h"

void init_qio_field_pybind(pybind11::module_ &);

void init_qio_field_pybind(pybind11::module_ &m) 
{
    //Add submodule
    auto qio_module = m.def_submodule("qio_field", "Wrapper to qio_field.h");

    // QIO needs to be defined in QUDA compilation.
    // read_gauge_field has a slightly different signature than its 
    // QUDA counterpart (besides missing arg's). 
    // Instead of assigning the gauge field to 
    // an already allocated pointer, it returns a numpy array 
    // to the python side that are mapped to the buffer of the 
    // underlying array that can be passed around easily.
    qio_module.def("read_gauge_field",
    [](std::string filename, QudaPrecision prec, const std::array<int, 4> X,
       int gauge_site_size) 
    {
        void *gauge[4]; 
        // Allocate space on the host (always best to allocate and free in the same scope)
        size_t data_size = (prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
        int V = X[0]*X[1]*X[2]*X[3];
        size_t new_size = V * gauge_site_size * data_size;
        for (int dir = 0; dir < 4; dir++) gauge[dir] = ::operator new (new_size);

        // Read from QIO
        int argc = 0;
        read_gauge_field(filename.c_str(), gauge, prec, X.data(), argc, NULL);
    }
    );
}