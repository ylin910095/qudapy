#pragma once

#include <pybind11/pybind11.h>

#include "quda.h"

void init_cfunc_pybind(pybind11::module_ &);