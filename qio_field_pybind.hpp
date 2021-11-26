#pragma once

#include <string>

#include <pybind11/stl.h> 
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // for numpy array automatic conversion

#include "quda.h"
#include "qio_field.h"

// Declare utility functions
void init_gauge_pointer_array(void *ptr[4], const void* gauge_ptr,
                              QudaPrecision prec, int local_volume, int site_size);

void init_qio_field_pybind(pybind11::module_ &);