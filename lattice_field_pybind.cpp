#include "lattice_field_pybind.hpp"

void init_lattice_field_pybind(pybind11::module_ &m) 
{
	pybind11::enum_<quda::QudaOffsetCopyMode>(m, "QudaOffsetCopyMode")
		.value("QUDA_SUCCESS", quda::QudaOffsetCopyMode::COLLECT)
		.value("QUDA_ERROR", quda::QudaOffsetCopyMode::DISPERSE)
		.export_values();

    // LatticeFieldParam
}