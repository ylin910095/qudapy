#pragma once

#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // for std::vector automatic conversion
#include <pybind11/complex.h> // for std::complex automatic conversion
#include <pybind11/numpy.h> // for numpy array automatic conversion

#include "quda.h"

namespace py = pybind11;

// Declare AND define utility templates
template<typename Struct_Type, typename Attr_Type>
py::array_t<Attr_Type> attr_getter(py::object& self, int array_size, Attr_Type* data) 
{
    // Getter: allow python to read the value.
    // Also allow item assignment via 
    // inst.X[i] = value in python since it returns the
    // reference to the underlying C++ object.
    // Struct_Type::* attr is a pointer to the member variable.
    return py::array_t<Attr_Type, py::array::c_style>{array_size, data, self};
};


template<typename Struct_Type, typename Attr_Type>
void attr_setter(py::object& self, int array_size, Attr_Type* data,
                 const py::array_t<Attr_Type, py::array::c_style> a) // don't pass by reference for a - we want to copy the data
{
    // Setter: allow item assignment via
    // inst.X = [1,2,3,4] in python for example
    auto buf = a.request(); // accsing numpy properties and buffer
    if (buf.ndim != 1 || buf.shape[0] != array_size) {
        std::stringstream errmsg;
        errmsg << "Expected a 1D array of size " << array_size;
        throw py::value_error(errmsg.str());
    }
    Attr_Type* ptr = static_cast<Attr_Type*>(buf.ptr); // cast address 
    for (int i=0; i<array_size; i++) data[i] = ptr[i];
}


inline void check_c_constiguous(const py::object& obj) {
    if (!py::detail::check_flags(obj.ptr(), py::array::c_style)) {
        throw std::runtime_error("Array must be C-style contiguous");
    }
}

inline void check_quda_array_precision(const py::object& obj, QudaPrecision prec) {
    // Type and order checks
    if (prec == QUDA_SINGLE_PRECISION) {
        if (!py::isinstance<py::array_t<float>>(obj))
            throw std::runtime_error("Array precision does not match the provided QUDA precision (float32 required)");
    } else if (prec == QUDA_DOUBLE_PRECISION) {
        if (!py::isinstance<py::array_t<double>>(obj))
            throw std::runtime_error("Array precision does not match the provided QUDA precision (float64 required)");
    };
}
