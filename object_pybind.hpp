#pragma once

// Header only

inline void init_lattice_field_pybind(pybind11::module_ &m)
{
    pybind11::class_<quda::Object>(m, "Object"); // just a simple wrap for the abstract class
}