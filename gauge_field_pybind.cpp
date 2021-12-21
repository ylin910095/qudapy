#include "gauge_field_pybind.hpp"

namespace py = pybind11; 

// Forward declaration - we only need the bare minimum to exchange ghosts on the host
void init_GaugeFieldParam(py::module_ &);
void init_cpuGaugeField(py::module_ &);

void init_gauge_field_pybind(py::module_ &m) {
    init_GaugeFieldParam(m);
    init_cpuGaugeField(m);
}

void init_GaugeFieldParam(py::module_ &m) {
    // GaugeFieldParam 
    py::class_<quda::GaugeFieldParam> cl(m, "GaugeFieldParam");
    
    // The only constructor that we will use for now
    cl.def(py::init( 
        [](py::array &gauge, QudaGaugeParam* param, QudaLinkType link_type_) { 
            // TODO: DO SOME CHECKS HERE
            py::buffer_info buf = gauge.request();
            void *h_gauge[4]; // convert from void*
            auto local_volume = param->X[0] * param->X[1] * param->X[2] * param->X[3];
            int gauge_site_size = 18; // 18 = 3 * 3 * (real + imag) for QDP ordering
            init_gauge_pointer_array(h_gauge, buf.ptr, param->cpu_prec, 
                                     local_volume, gauge_site_size);
            return new quda::GaugeFieldParam(h_gauge, *param, link_type_);
        }),   
        "gauge"_a, "param"_a, "link_type_"_a = QUDA_INVALID_LINKS // if link type is default(invalid), it will use param.type
    ); 
   
}

void init_cpuGaugeField(py::module_ &m) {
    // cpuGaugeField 
    py::class_<quda::cpuGaugeField> cl(m, "cpuGaugeField");

    // The only constructor that we will use for now
    // Using this constructor will automatically start the ghost exchange on the host
    // See cpu_gauge_field.cpp for details
    cl.def(py::init<const quda::GaugeFieldParam &>());

    // Inherited from GaugeField. But since we did not bind the base class, we need to manually
    // bind the methods here. The returned object is a list with nDim = 4 items for each dimension
    cl.def("Ghost", 
        [](const py::object &self){
            quda::cpuGaugeField& obj = self.cast<quda::cpuGaugeField&>(); 

            // Some safety checks
            if (obj.FieldOrder() != QUDA_QDP_GAUGE_ORDER) {
                throw py::type_error("The python binding only supports QUDA_QDP_GAUGE_ORDER");
            }
            if (obj.Nface() != 1) {
                throw py::type_error("The python binding only supports Nface=1");
            } 
            if (obj.Ndim() != 4) {
                throw py::type_error("The python binding only supports Ndim=4");
            } 
            if (obj.Precision() != QUDA_DOUBLE_PRECISION && obj.Precision() != QUDA_SINGLE_PRECISION) {
                throw py::type_error("The python binding only supports double or single precisions");
            }

            void** ghost_ptr;
            ghost_ptr = obj.Ghost();
            std::vector<py::array> ghost_list;
            
            // Initialize the return numpy array (no copying)
            // 2*obj.surfaceCB() = total surface volume (need a factor of two because sufaceCB
            // only returns the single parity surface volume)
            for (int dir = 0; dir < 4; dir++) {
                py::ssize_t ds = (obj.Precision() == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float); 
                std::vector<py::ssize_t> shape = {2*obj.SurfaceCB(dir), obj.Ncolor(), obj.Ncolor(), 2};
                std::vector<py::ssize_t> stride = {2*obj.Ncolor()*obj.Ncolor()*ds, 
                                                   2*obj.Ncolor()*ds, 
                                                   2*ds, ds}; // TODO: Fix the shape - they are wrong!

                py::array pyarray;
                // Convert to a 1D pointer
                if (obj.Precision() == QUDA_DOUBLE_PRECISION) {
                    pyarray = py::array(shape, stride, (double*) ghost_ptr[dir]);
                } else {
                    pyarray = py::array(shape, stride, (float*) ghost_ptr[dir]);
                }
                ghost_list.push_back(std::move(pyarray)); 
            }
            return ghost_list;
        }, py::return_value_policy::reference_internal
    ); // see https://pybind11.readthedocs.io/en/stable/advanced/functions.html
       // This prevent self(this) from being deallocated while the returned ghost is still in use
}
