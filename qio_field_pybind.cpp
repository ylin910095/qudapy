#include "qio_field_pybind.hpp"

void init_gauge_pointer_array(void *ptr[4], const void* gauge_ptr,
                              QudaPrecision prec, int local_volume, int site_size)
{
    if (prec == QUDA_DOUBLE_PRECISION) 
    {   
        for (int dir = 0; dir < 4; dir++) ptr[dir] = &((double *) gauge_ptr)[dir * local_volume * site_size];
    } else if (prec == QUDA_SINGLE_PRECISION)
    {  
        for (int dir = 0; dir < 4; dir++) ptr[dir] = &((float *) gauge_ptr)[dir * local_volume * site_size];
    } else {
        throw pybind11::type_error("Unknown data precision");
    }
}

void init_qio_field_pybind(pybind11::module_ &m) 
{
    //Add submodule
    auto qio_module = m.def_submodule("qio_field", "Wrapper to qio_field.h");

    // QIO needs to be defined in QUDA compilation.
    // read_gauge_field has a slightly different signature than its 
    // QUDA counterpart (besides missing arg's). 
    // Instead of assigning the gauge field to 
    // an already allocated pointer, it returns a numpy array 
    // to the python side that are mapped to the buffer of the 
    // underlying array that can be passed around easily.
    qio_module.def("read_gauge_field",
    [](std::string filename, pybind11::array &gauge,
       QudaPrecision prec, const std::array<int, 4> X,
       int gauge_site_size) 
    {
        pybind11::ssize_t data_size = (prec == QUDA_DOUBLE_PRECISION) ? sizeof(double): sizeof(float);
        int local_volume = X[0] * X[1] * X[2] * X[3];
        int n_dir = 4;
        pybind11::buffer_info buf = gauge.request();
        auto gauge_ptr = static_cast<void*>(buf.ptr);

        // Safety checks
        if (buf.ndim != 1 && buf.ndim != 4 && buf.ndim != 5)
            throw std::runtime_error("Number of dimensions must be one, four, or five with the "
                                     "shape of (ndir, spatial, ncolor, ncolor[, real_complex])");
        std::vector<pybind11::ssize_t> shape1d = {n_dir * local_volume * gauge_site_size};
        std::vector<pybind11::ssize_t> shape4d = {n_dir, local_volume, 3, 3};
        std::vector<pybind11::ssize_t> shape5d = {n_dir, local_volume, 3, 3, 2};
        
        if (buf.ndim == 1 && buf.shape != shape1d)
        {
            throw std::runtime_error("Inconsistent numpy array shape");
        } else if (buf.ndim == 4 && buf.shape != shape4d)
        {
            throw std::runtime_error("Inconsistent numpy array shape");
        } else if (buf.ndim == 5 && buf.shape != shape5d)
        {
            throw std::runtime_error("Inconsistent numpy array shape");
        }

        // Treat complex inputs in a special way - itemsize can be twice the data_size
        if (buf.ndim != 4 && buf.itemsize != data_size)
        {
            throw std::runtime_error("Inconsistent numpy array itemsize");
        } else if (buf.ndim == 4 && buf.itemsize != 2 * data_size)
        {
            throw std::runtime_error("Inconsistent numpy array itemsize");
        }

        std::vector<pybind11::ssize_t> strides4d = {
                                                    data_size * local_volume,
                                                    9 * data_size,
                                                    3 * data_size,
                                                    data_size
                                                   };
        std::vector<pybind11::ssize_t> strides5d = {
                                                    18 * data_size * local_volume,
                                                    2 * 9 * data_size,
                                                    2 * 3 * data_size,
                                                    2 * data_size,
                                                    data_size
                                                    };
        if (buf.ndim == 5 && buf.strides != strides5d) 
            throw std::runtime_error("Inconsistent numpy array strides");
        if (buf.ndim == 4 && buf.strides != strides4d) 
            throw std::runtime_error("Inconsistent numpy array strides");

        // Read the gauge field
        void *tmp[4]; // because QIO does not like *gauge directly for some reasons
        int argc = 1;   
        init_gauge_pointer_array(tmp, gauge_ptr, prec, local_volume, gauge_site_size);
        read_gauge_field(filename.c_str(), tmp, prec, X.data(), argc, NULL);
    });
}