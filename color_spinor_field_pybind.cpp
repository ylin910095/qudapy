#include "color_spinor_field_pybind.hpp"

void init_color_spinor_field_pybind(py::module_ &m)
{
    // quda.colorspinor submodule
    auto csf = m.def_submodule("colorspinor");
    csf.def("isNative", &quda::colorspinor::isNative);

    // quda module
    py::enum_<quda::MemoryLocation>(m, "MemoryLocation")
		.value("Device", quda::MemoryLocation::Device)
		.value("Host", quda::MemoryLocation::Host)
        .value("Remote", quda::MemoryLocation::Remote)
		.value("Shmem", quda::MemoryLocation::Shmem);

    // struct FullClover; // skip? Is it implemented?

    m.def("impliedParityFromMatPC", &quda::impliedParityFromMatPC);

    // Skip Composite types for now until I actually use them
    //struct CompositeColorSpinorFieldDescriptor {} 
    
    // ColorSpinorParam
    py::class_<quda::ColorSpinorParam, quda::LatticeFieldParam, std::unique_ptr<quda::ColorSpinorParam>> 
          cl(m, "ColorSpinorParam");

    // Constructors 
    cl.def(py::init<const quda::ColorSpinorField &>());

    cl.def(py::init( [](){ return new quda::ColorSpinorParam(); } ) );

    // Used to create cpu params
    cl.def(py::init(
        [](py::array &V, QudaInvertParam &inv_param, 
           const py::array_t<int> &X, const bool pc_solution, QudaFieldLocation location)
        {   
            py::buffer_info V_buf = V.request();
            py::buffer_info X_buf = X.request();
            auto V_ptr = static_cast<void*>(V_buf.ptr);
            auto X_ptr = static_cast<int*>(X_buf.ptr);
            return new quda::ColorSpinorParam(V_ptr, inv_param, X_ptr, pc_solution, location);
        }), 
        "V"_a, "inv_param"_a, "X"_a, "pc_solution"_a, "location"_a = QUDA_CPU_FIELD_LOCATION);
    
    // Normally used to create cuda param from a cpu param
    cl.def(py::init<quda::ColorSpinorParam &, QudaInvertParam &, QudaFieldLocation>());

    // Public members
    cl.def_readwrite("location", &quda::ColorSpinorParam::location);

    cl.def_readwrite("nColor", &quda::ColorSpinorParam::nColor);
    cl.def_readwrite("nSpin", &quda::ColorSpinorParam::nSpin);
    cl.def_readwrite("nVec", &quda::ColorSpinorParam::nVec); 

    cl.def_readwrite("twistFlavor", &quda::ColorSpinorParam::twistFlavor);

    cl.def_readwrite("siteOrder", &quda::ColorSpinorParam::siteOrder);

    cl.def_readwrite("fieldOrder", &quda::ColorSpinorParam::fieldOrder);
    cl.def_readwrite("gammaBasis", &quda::ColorSpinorParam::gammaBasis);
    cl.def_readwrite("create", &quda::ColorSpinorParam::create);

    cl.def_readwrite("pc_type", &quda::ColorSpinorParam::pc_type);

    cl.def_readwrite("suggested_parity", &quda::ColorSpinorParam::suggested_parity);

    // implement v and norm

    /* 
     void *v; // pointer to field
     void *norm;
    */

   cl.def_property("v", 
        [](py::object& self) {   
            // Python binding specific errors
            // Maybe it is trivial to include. But I did not check so just to be safe for now until needed
            auto &obj = self.cast<quda::ColorSpinorParam &>();
            if (obj.nVec != 1)
                throw py::value_error("ColorSpinorParam.v python interface is only implemented "
                                            "for nVec = 1");
            if (obj.Precision() != QUDA_SINGLE_PRECISION && obj.Precision() != QUDA_DOUBLE_PRECISION)
                throw py::value_error("ColorSpinorParam.v python interface is only implemented "
                                            "for single and double precision");
            if (obj.fieldOrder != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER)
                throw py::value_error("ColorSpinorParam.v python interface is only implemented "
                                            "for fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER (aka QDP++ order)");

            auto volume = 1;
            for (int i=0; i < obj.nDim; i++) volume *= obj.x[i]; 
            std::vector<int> array_size = {2 * volume * obj.nSpin * obj.nColor * obj.nVec}; // 1d

            switch (obj.Precision()) {  
                case QUDA_SINGLE_PRECISION:
                    return  py::array{py::dtype::of<float>(), array_size, obj.v, self};
                case QUDA_DOUBLE_PRECISION:
                    return  py::array{py::dtype::of<double>(), array_size, obj.v, self};
                default:
                    throw std::runtime_error("Well, I do not know how you get here");
            }
        }, 
        [](py::object &self, const py::array &field) {   
            // Python binding specific errors
            // Maybe it is trivial to include. But I did not check so just to be safe for now until needed
            auto &obj = self.cast<quda::ColorSpinorParam &>();
            if (obj.nVec != 1)
                throw py::value_error("ColorSpinorParam.v python interface is only implemented "
                                            "for nVec = 1");
            if (obj.Precision() != QUDA_SINGLE_PRECISION && obj.Precision() != QUDA_DOUBLE_PRECISION)
                throw py::value_error("ColorSpinorParam.v python interface is only implemented "
                                            "for single and double precision");
            if (obj.fieldOrder != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER)
                throw py::value_error("ColorSpinorParam.v python interface is only implemented "
                                            "for fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER (aka QDP++ order)");

            // Dimension checks
            pybind11::buffer_info buf = field.request();
            auto volume = 1;
            for (int i=0; i < obj.nDim; i++) volume *= obj.x[i]; 
            std::vector<py::ssize_t> shape1d = {volume * obj.nSpin * obj.nColor * obj.nVec * 2}; // 1d - float/double type
            std::vector<py::ssize_t> shape4d = {volume, obj.nSpin, obj.nColor, 2}; // 4d - float/double type

            // Shape checks
            if (buf.ndim == 1 && buf.shape != shape1d) {
                throw std::runtime_error("Inconsistent numpy array shape");
            } else if (buf.ndim == 4 && buf.shape != shape4d) {
                throw std::runtime_error("Inconsistent numpy array shape");
            }

            // Type checks
            if (obj.Precision() == QUDA_SINGLE_PRECISION) {
                if (!py::isinstance<py::array_t<float>>(field))
                    throw std::runtime_error("Inconsistent data type");
            } else if (obj.Precision() == QUDA_DOUBLE_PRECISION) {
                if (!py::isinstance<py::array_t<double>>(field))
                    throw std::runtime_error("Inconsistent data type");
            } else {
                throw std::runtime_error("Well, I do not know how you get here");
            } 
            // Set the data
            obj.v = static_cast<void*>(buf.ptr);
        }
    ); // v attribute
    

    cl.def_readwrite("is_composite", &quda::ColorSpinorParam::is_composite);
    cl.def_readwrite("composite_dim", &quda::ColorSpinorParam::composite_dim);
    cl.def_readwrite("is_component", &quda::ColorSpinorParam::is_component);
    cl.def_readwrite("component_id", &quda::ColorSpinorParam::component_id);

    cl.def("setPrecision", &quda::ColorSpinorParam::setPrecision,
           "precision"_a, "ghost_precision"_a = QUDA_INVALID_PRECISION,
           "force_native"_a = false);

    // Print utility
    cl.def("print", &quda::ColorSpinorParam::print);
}