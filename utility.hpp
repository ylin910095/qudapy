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
template<typename Struct_Type, typename Attr_Type, int array_size>
pybind11::array_t<Attr_Type> attr_getter(pybind11::object& obj, 
                                         Attr_Type (Struct_Type::* attr)[array_size]) 
{
    // Getter: allow python to read the value.
    // Also allow item assignment via 
    // inst.X[i] = value in python since it returns the
    // reference to the underlying C++ object.
    // Struct_Type::* attr is a pointer to the member variable.
    Struct_Type& o = obj.cast<Struct_Type&>(); 
    return pybind11::array_t<Attr_Type>{array_size, o.*attr, obj};
};

template<typename Struct_Type, typename Attr_Type, int array_size>
void attr_setter(pybind11::object& obj, Attr_Type (Struct_Type::* attr)[array_size], 
                 const pybind11::array_t<Attr_Type> &a) 
{
    // Setter: allow item assignment via
    // inst.X = [1,2,3,4] in python for example
    Struct_Type &o = obj.cast<Struct_Type&>(); 
    auto buf = a.request(); // accsing numpy properties and buffer
    if (buf.ndim != 1 || buf.shape[0] != array_size) {
        std::stringstream errmsg;
        errmsg << "Expected a 1D array of size " << array_size;
        throw pybind11::value_error(errmsg.str());
    }
    Attr_Type* ptr = static_cast<Attr_Type*>(buf.ptr); // cast address 
    for (int i=0; i<array_size; i++) (o.*attr)[i] = ptr[i];
}