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
// to mimic the namespace behavior
void init_lattice_field_pybind(pybind11::module_ &);

// Sub functions
void init_LatticeField(pybind11::module_ &);

void init_ColorSpinorField(pybind11::module_ &);
void init_cudaColorSpinorField(pybind11::module_ &); // not binded yet
void init_cpuColorSpinorField(pybind11::module_ &);

void init_EigValueSet(pybind11::module_ &); // not binded yet
void init_cudaEigValueSet(pybind11::module_ &); // not binded yet
void init_cpuColorSpinorField(pybind11::module_ &); // not binded yet

void init_GaugeField(pybind11::module_ &);
void init_cudaGaugeField(pybind11::module_ &); // not binded yet
void init_cpuGaugeField(pybind11::module_ &); 

void init_CloverField(pybind11::module_ &);
void init_cudaCloverField(pybind11::module_ &); // not binded yet
void init_cpuCloverField(pybind11::module_ &);
