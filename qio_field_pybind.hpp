#pragma once

#include <string>

#include <pybind11/stl.h> 
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // for numpy array automatic conversion

#include "quda.h"
#include "qio_field.h"

#include "utility.hpp"

// Declare utility functions
void init_qio_field_pybind(pybind11::module_ &, bool has_qio);