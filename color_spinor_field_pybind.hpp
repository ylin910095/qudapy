#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/numpy.h> // for numpy array automatic conversion

// Quda headers
#include "quda.h"
#include "color_spinor_field.h"

// Binding headers
#include "utility.hpp"
#include "lattice_field_pybind.hpp"

namespace py = pybind11;
using namespace py::literals; // for _a

// Main function - bind things in quda::colorspinor:: under quda.colorspinor submodule in python
// and bind things in quda:: under quda module
void init_color_spinor_field_pybind(py::module_ &);