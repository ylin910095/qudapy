#pragma once

#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // for std::vector automatic conversion
#include <pybind11/complex.h> // for std::complex automatic conversion
#include <pybind11/numpy.h> // for numpy array automatic conversion

#include "quda.h"


// Declare AND define utility templates
template<typename Struct_Type, typename Attr_Type>
pybind11::array_t<Attr_Type> attr_getter(pybind11::object& self, int array_size, Attr_Type* data) 
{
    // Getter: allow python to read the value.
    // Also allow item assignment via 
    // inst.X[i] = value in python since it returns the
    // reference to the underlying C++ object.
    // Struct_Type::* attr is a pointer to the member variable.
    return pybind11::array_t<Attr_Type>{array_size, data, self};
};

template<typename Struct_Type, typename Attr_Type>
void attr_setter(pybind11::object& self, int array_size, Attr_Type* data,
                 const pybind11::array_t<Attr_Type> a) // don't pass by reference for a - we want to copy the data
{
    // Setter: allow item assignment via
    // inst.X = [1,2,3,4] in python for example
    auto buf = a.request(); // accsing numpy properties and buffer
    if (buf.ndim != 1 || buf.shape[0] != array_size) {
        std::stringstream errmsg;
        errmsg << "Expected a 1D array of size " << array_size;
        throw pybind11::value_error(errmsg.str());
    }
    Attr_Type* ptr = static_cast<Attr_Type*>(buf.ptr); // cast address 
    for (int i=0; i<array_size; i++) data[i] = ptr[i];
}

