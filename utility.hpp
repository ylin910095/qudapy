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

// Just a wrapper of check_c_constiguous and check_quda_array_precision
inline void check_precision_c_contiguous(const py::object& obj, QudaPrecision prec) {
    check_c_constiguous(obj);
    check_quda_array_precision(obj, prec);
}

// Check whether obj, which is a spinor numpy array from the python side, satifies 
// prec and local_volume requirement to prevent segfaults
inline void check_python_spinor_array(const py::object& obj, QudaPrecision prec, int local_volume) {
    check_c_constiguous(obj);
    check_quda_array_precision(obj, prec);
    auto array_obj = obj.cast<py::array>();
    auto buf = array_obj.request();
    if (buf.ndim != 1 && buf.ndim != 4)
            throw std::runtime_error("Number of dimensions must be one or five with the "
                                     "shape of (spatial, nspin, ncolor, complex)");
    std::vector<py::ssize_t> shape1d = {local_volume* 4 * 3 * 2};
    std::vector<py::ssize_t> shape4d = {local_volume, 4, 3, 2};
    if (buf.ndim == 1 && buf.shape != shape1d) {
        throw std::runtime_error("1D spinor array must have a shape of (local_volume*4*3*2)");
    } 
    if (buf.ndim == 4 && buf.shape != shape4d) {
        throw std::runtime_error("4D spinor array must have a shape of "
                                 "(local_volume, nspin=4, ncolor=3, complex=2)");
    }
}