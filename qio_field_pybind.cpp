#include "qio_field_pybind.hpp"

namespace py = pybind11;

void init_gauge_pointer_array(void *ptr[4], const void* gauge_ptr,
                              QudaPrecision prec, int local_volume, int site_size) {
    if (prec == QUDA_DOUBLE_PRECISION) 
    {   
        for (int dir = 0; dir < 4; dir++) ptr[dir] = &((double *) gauge_ptr)[dir * local_volume * site_size];
    } else if (prec == QUDA_SINGLE_PRECISION)
    {  
        for (int dir = 0; dir < 4; dir++) ptr[dir] = &((float *) gauge_ptr)[dir * local_volume * site_size];
    } else {
        throw py::type_error("Unknown data precision");
    }
}

void init_qio_field_pybind(py::module_ &m, bool has_qio) {
    //Add submodule
    auto qio_module = m.def_submodule("qio_field", "Wrapper to qio_field.h");

    // QIO needs to be defined in QUDA compilation.
    // read_gauge_field has a slightly different signature than its 
    // QUDA counterpart (besides missing arg's). 
    // Instead of assigning the gauge field to 
    // an already allocated pointer, it returns a numpy array 
    // to the python side that are mapped to the buffer of the 
    // underlying array that can be passed around easily.

    // TODO: change X to numpy array and allow either 4 or 5 dimensions for 
    // domainwall fermions gauge field.
    qio_module.def("read_gauge_field",
    [has_qio](std::string filename, py::array &gauge,
       QudaPrecision prec, const std::array<int, 4> X, int gauge_site_size) {
        
        if (!has_qio) {
            throw std::runtime_error("HAVE_QIO is not enabled in qudapy compilation");
        }

        int local_volume = X[0]*X[1]*X[2]*X[3];
        int n_dir = 4;
        py::buffer_info buf = gauge.request();
        auto gauge_ptr = static_cast<void*>(buf.ptr);

        // Safety checks
        if (buf.ndim != 1 && buf.ndim != 5)
            throw std::runtime_error("Number of dimensions must be one or five with the "
                                     "shape of (ndir, spatial, ncolor, ncolor, complex)");
        std::vector<py::ssize_t> shape1d = {n_dir * local_volume * gauge_site_size};
        std::vector<py::ssize_t> shape5d = {n_dir, local_volume, 3, 3, 2};
        
        // Shape checks
        if (buf.ndim == 1 && buf.shape != shape1d) {
            throw std::runtime_error("Inconsistent numpy array shape");
        } else if (buf.ndim == 5 && buf.shape != shape5d) {
            throw std::runtime_error("Inconsistent numpy array shape");
        };

        // Check data ordering and precision
        check_precision_c_contiguous(gauge, prec);
        
        // Read the gauge field
        void *tmp[4]; // because QIO does not like *gauge directly for some reasons
        int argc = 1;   
        init_gauge_pointer_array(tmp, gauge_ptr, prec, local_volume, gauge_site_size);
        read_gauge_field(filename.c_str(), tmp, prec, X.data(), argc, NULL);
    });
}