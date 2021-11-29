#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/numpy.h> // for numpy array automatic conversion

// Quda headers
#include "quda.h"
#include "lattice_field.h"

// Binding headers
#include "utility.hpp"

using namespace pybind11::literals; // for _a

// Main function - bind everything under quda module in python
// to mimic the namespace behavior quda::
void init_lattice_field_pybind(pybind11::module_ &);