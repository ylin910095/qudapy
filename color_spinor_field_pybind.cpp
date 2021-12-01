#include "color_spinor_field_pybind.hpp"

void check_color_spinor_param(const quda::ColorSpinorField &obj) {
            if (obj.Nvec() != 1)
                throw py::value_error("Python interface is only implemented "
                                            "for nVec = 1");
            if (obj.Precision() != QUDA_SINGLE_PRECISION && obj.Precision() != QUDA_DOUBLE_PRECISION)
                throw py::value_error("Python interface is only implemented "
                                            "for single and double precision");
            if (obj.FieldOrder() != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER)
                throw py::value_error("Python interface is only implemented "
                                      "for fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER (aka QDP++ order)");
}

// Check if the precision of field is consistent with the precision of param
void check_precision(const quda::ColorSpinorParam &sparam, const py::array &field) {
    if (sparam.Precision() == QUDA_SINGLE_PRECISION) {
        if (!py::isinstance<py::array_t<float>>(field))
            throw std::runtime_error("Inconsistent data type");
    } else if (sparam.Precision() == QUDA_DOUBLE_PRECISION) {
        if (!py::isinstance<py::array_t<double>>(field))
            throw std::runtime_error("Inconsistent data type");
    } else {
        throw std::runtime_error("Only single and double precisions are supported for the python binding");
    } 
}

/*
void check_precision(const py::array &field) {
    if (!py::isinstance<py::array_t<float>>(field) && !py::isinstance<py::array_t<double>>(field))
            throw std::runtime_error("Only single and double precisions are supported for the python binding");
}
*/

void check_lattice_site_index(const py::buffer_info &buf, 
                              const quda::ColorSpinorField &obj) {
    // Check checks
    if (buf.itemsize != sizeof(int)) {
        throw std::runtime_error("y must be a 1D array of ints");
    }
    std::vector<py::ssize_t> shape1d = {obj.Ndim()};
    if (buf.shape != shape1d) {
        throw std::runtime_error("y has the wrong dimensions");
    }
}

void init_ColorSpinorParam(py::module_ &m) {
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
            throw std::runtime_error("Don't use this to access the vector - use ColorSpinorField instead");    
        }, 
        [](py::object &self, const py::array &field) {   
            throw std::runtime_error("Don't use this to access the vector - use ColorSpinorField instead");
        }
    ); // v attribute

    // norm attribute not binded 

    cl.def_readwrite("is_composite", &quda::ColorSpinorParam::is_composite);
    cl.def_readwrite("composite_dim", &quda::ColorSpinorParam::composite_dim);
    cl.def_readwrite("is_component", &quda::ColorSpinorParam::is_component);
    cl.def_readwrite("component_id", &quda::ColorSpinorParam::component_id);

    cl.def("setPrecision", &quda::ColorSpinorParam::setPrecision,
           "precision"_a, "ghost_precision"_a = QUDA_INVALID_PRECISION,
           "force_native"_a = false);

    // Print utility
    cl.def("print", &quda::ColorSpinorParam::print);
} // init_ColorSpinorParam

void init_ColorSpinorField(py::module_ &m) {
    py::class_<quda::ColorSpinorField, quda::LatticeField, std::unique_ptr<quda::ColorSpinorField>> 
          cl(m, "ColorSpinorField");

    // No constructors -- it is an abstract class
    // Public members
    cl.def("Ncolor", &quda::ColorSpinorField::Ncolor);
    cl.def("Nspin", &quda::ColorSpinorField::Nspin);
    cl.def("Nvec", &quda::ColorSpinorField::Nvec);
    cl.def("TwistFlavor", &quda::ColorSpinorField::TwistFlavor);
    cl.def("Ndim", &quda::ColorSpinorField::Ndim);
    cl.def("X", 
        [](py::object& self)
        {    
            quda::ColorSpinorField& obj = self.cast<quda::ColorSpinorField&>(); 
            auto x = obj.X();
            return py::array_t<int>(QUDA_MAX_DIM, x);
        }, 
        py::return_value_policy::copy);

    cl.def("X", py::overload_cast<int>(&quda::ColorSpinorField::X, py::const_));
    cl.def("RealLength", &quda::ColorSpinorField::RealLength);
    cl.def("Length", &quda::ColorSpinorField::Length);
    cl.def("Stride", &quda::ColorSpinorField::Stride);
    cl.def("Volume", &quda::ColorSpinorField::Volume);
    cl.def("VolumeCB", &quda::ColorSpinorField::VolumeCB);
    cl.def("Pad", &quda::ColorSpinorField::Pad);
    cl.def("Bytes", &quda::ColorSpinorField::Bytes);
    cl.def("NormBytes", &quda::ColorSpinorField::NormBytes);
    cl.def("TotalBytes", &quda::ColorSpinorField::TotalBytes);
    cl.def("GhostBytes", &quda::ColorSpinorField::GhostBytes);
    cl.def("GhostFaceBytes", &quda::ColorSpinorField::GhostFaceBytes);
    cl.def("GhostNormBytes", &quda::ColorSpinorField::GhostNormBytes);
    cl.def("PrintDims", &quda::ColorSpinorField::PrintDims);

    // Implement these last
    // void* V() {return v;}
    // const void* V() const {return v;}
    cl.def("V", 
        [](py::object& self) {   
            // Python binding specific errors
            // Maybe it is trivial to include. But I did not check so just to be safe for now until needed
            quda::ColorSpinorField &obj = self.cast<quda::ColorSpinorField &>();

            // Safety check
            check_color_spinor_param(obj);

            std::vector<size_t> array_size = {obj.Length()}; // 1d, include both pad and ghost if on gpu (I think cpu won't have pad and ghost?)

            switch (obj.Precision()) {  
                case QUDA_SINGLE_PRECISION:
                    // Sanity check
                    assert (obj.Bytes() == sizeof(float) * obj.Length());
                    return  py::array{py::dtype::of<float>(), array_size, obj.V(), self};
                case QUDA_DOUBLE_PRECISION:
                    // Sanity check
                    assert (obj.Bytes() == sizeof(double) * obj.Length());
                    return  py::array{py::dtype::of<double>(), array_size, obj.V(), self};
                default:
                    throw std::runtime_error("Only single and double precisions are supported for the python binding");
            }
        }
    ); // V()


    // Norm not binded - they are for lower precision stuff?
    // void* Norm(){return norm;}
    // const void* Norm() const {return norm;}

    // Virtual - skip
    // virtual int full_dim(int d) const { return (d == 0 && siteSubset == 1) ? x[d] * 2 : x[d]; }

    // Ghost stuff - skip for now
    // void exchange(void **ghost, void **sendbuf, int nFace=1) const;

    // Virtual - skip 
    // virtual void exchangeGhost(QudaParity parity, int nFace, int dagger, const MemoryLocation *pack_destination=nullptr,
    //                            const MemoryLocation *halo_location=nullptr, bool gdr_send=false, bool gdr_recv=false,
    //                            QudaPrecision ghost_precision=QUDA_INVALID_PRECISION) const = 0;

    cl.def("isNative", &quda::ColorSpinorField::isNative);

    // Composite stuff - skip for now
    
    /*
     bool IsComposite() const { return composite_descr.is_composite; }
     bool IsComponent() const { return composite_descr.is_component; }
  
     int CompositeDim() const { return composite_descr.dim; }
     int ComponentId() const { return composite_descr.id; }
     int ComponentVolume() const { return composite_descr.volume; }
     int ComponentVolumeCB() const { return composite_descr.volumeCB; }
     int ComponentStride() const { return composite_descr.stride; }
     size_t ComponentLength() const { return composite_descr.length; }
     size_t ComponentRealLength() const { return composite_descr.real_length; }
  
     size_t ComponentBytes() const { return composite_descr.bytes; }
     size_t ComponentNormBytes() const { return composite_descr.norm_bytes; }
    */

    cl.def("PCType", &quda::ColorSpinorField::PCType);
    cl.def("SuggestedParity", &quda::ColorSpinorField::SuggestedParity);
    cl.def("setSuggestedParity", &quda::ColorSpinorField::setSuggestedParity);

    cl.def("SiteSubset", &quda::ColorSpinorField::SiteSubset);
    cl.def("SiteOrder", &quda::ColorSpinorField::SiteOrder);
    cl.def("FieldOrder", &quda::ColorSpinorField::FieldOrder);   
    cl.def("GammaBasis", &quda::ColorSpinorField::GammaBasis);

    cl.def("GhostFace", 
        [](py::object& self)
        {    
            quda::ColorSpinorField& o = self.cast<quda::ColorSpinorField&>(); 
            auto gf = o.GhostFace();
            return py::array_t<int>(QUDA_MAX_DIM, gf);
        });

    cl.def("GhostFaceCB", 
        [](py::object& self)
        {    
            quda::ColorSpinorField& o = self.cast<quda::ColorSpinorField&>(); 
            auto gfcb = o.GhostFaceCB();
            return py::array_t<int>(QUDA_MAX_DIM, gfcb);
        });

    cl.def("GhostOffset", &quda::ColorSpinorField::GhostOffset,
           "dim"_a, "dir"_a);

    // Skip complicated ghost stuff for now
    /*
     void* Ghost(const int i);
     const void* Ghost(const int i) const;
     void* GhostNorm(const int i);
     const void* GhostNorm(const int i) const;
  
     void* const* Ghost() const;
    */

    cl.def("getDslashConstant", &quda::ColorSpinorField::getDslashConstant);   

    // TODO
    /*  
     const ColorSpinorField& Even() const;
     const ColorSpinorField& Odd() const;
  
     ColorSpinorField& Even();
     ColorSpinorField& Odd();
    */


    // Skip component stuff for now
    /*
     ColorSpinorField& Component(const int idx) const;
     ColorSpinorField& Component(const int idx);
  
     CompositeColorSpinorField& Components(){
       return components;
     };
    */

    // Skip virtual
    /*
     virtual void Source(const QudaSourceType sourceType, const int st=0, const int s=0, const int c=0) = 0;
  
     virtual void PrintVector(unsigned int x) const = 0;
    */
    
    // Fun with unsigned ints
    cl.def("PrintVector", 
        [](py::object& self, int x_cb, int parity){
            quda::ColorSpinorField& o = self.cast<quda::ColorSpinorField&>(); 
            auto x_cb_us = static_cast<unsigned int>(x_cb);
            auto parity_us = static_cast<unsigned int>(parity);
            o.PrintVector(x_cb_us, parity_us);
        },
        "x_cb"_a, "parity"_a
    );   

    cl.def("LatticeIndex", 
        [](py::object& self, py::array &y, int i){
            quda::ColorSpinorField& obj = self.cast<quda::ColorSpinorField&>(); 
            pybind11::buffer_info buf = y.request();

            // Check checks
            check_lattice_site_index(buf, obj);
            
            auto butptr = static_cast<int*>(buf.ptr);
            obj.LatticeIndex(butptr, i);
        },
        "y"_a, "i"_a
    );     

    // Because integers are immutable in python, we have to return it 
    // instead of changing in place
    cl.def("OffsetIndex", 
        [](const py::object& self, py::array &y){
            quda::ColorSpinorField& obj = self.cast<quda::ColorSpinorField&>(); 
            pybind11::buffer_info buf = y.request();

            // Check checks
            check_lattice_site_index(buf, obj);
            
            auto butptr = static_cast<int*>(buf.ptr);
            int i;
            obj.OffsetIndex(i, butptr);
            return i;
        }
    );   

    cl.def_static("Create", py::overload_cast<const quda::ColorSpinorParam &>(&quda::ColorSpinorField::Create));     
    cl.def_static("Create",
      py::overload_cast<const quda::ColorSpinorField &, const quda::ColorSpinorParam &>(&quda::ColorSpinorField::Create));     

    // Don't need them yet - skip
    /*    
     ColorSpinorField *CreateAlias(const ColorSpinorParam &param);
  
     ColorSpinorField* CreateCoarse(const int *geoBlockSize, int spinBlockSize, int Nvec,
                                    QudaPrecision precision=QUDA_INVALID_PRECISION,
                                    QudaFieldLocation location=QUDA_INVALID_FIELD_LOCATION,
                                    QudaMemoryType mem_Type=QUDA_MEMORY_INVALID);
  
     ColorSpinorField* CreateFine(const int *geoblockSize, int spinBlockSize, int Nvec,
                                  QudaPrecision precision=QUDA_INVALID_PRECISION,
                                  QudaFieldLocation location=QUDA_INVALID_FIELD_LOCATION,
                                  QudaMemoryType mem_type=QUDA_MEMORY_INVALID);
    */
} // init_ColorSpinorField

void init_cpuColorSpinorField(py::module_ &m) {
    // ColorSpinorParam
    py::class_<quda::cpuColorSpinorField, quda::ColorSpinorField, std::unique_ptr<quda::cpuColorSpinorField>> 
          cl(m, "cpuColorSpinorField");
    
    // Constructors
    cl.def(py::init<const quda::cpuColorSpinorField &>());
    cl.def(py::init<const quda::ColorSpinorField &>());
    cl.def(py::init<const quda::ColorSpinorField &, const quda::ColorSpinorParam &>());
    cl.def(py::init<const quda::ColorSpinorParam &>());

    // Cpu ghosts... TODO
    /*
     static void* fwdGhostFaceBuffer[QUDA_MAX_DIM]; //cpu memory
     static void* backGhostFaceBuffer[QUDA_MAX_DIM]; //cpu memory
     static void* fwdGhostFaceSendBuffer[QUDA_MAX_DIM]; //cpu memory
     static void* backGhostFaceSendBuffer[QUDA_MAX_DIM]; //cpu memory
     static int initGhostFaceBuffer;
     static size_t ghostFaceBytes[QUDA_MAX_DIM];
    */

    // Not implementing overloaded operators yet

    cl.def("Source", &quda::cpuColorSpinorField::Source,
           "sourceType"_a, "st"_a = 0, "s"_a = 0, "c"_a = 0);   

    cl.def_static("Compare", &quda::cpuColorSpinorField::Compare,
                  "a"_a, "b"_a, "resolution"_a = 0);  

    // Fun with unsigned ints
    cl.def("PrintVector", 
        [](py::object& self, int x_cb){
            quda::cpuColorSpinorField& o = self.cast<quda::cpuColorSpinorField&>(); 
            auto x_cb_us = static_cast<unsigned int>(x_cb);
            o.PrintVector(x_cb_us);
        }
    );   

    cl.def("allocateGhostBuffer", &quda::cpuColorSpinorField::allocateGhostBuffer,
           "nFace"_a);   
    cl.def_static("freeGhostBuffer", &quda::cpuColorSpinorField::freeGhostBuffer);      

    // cpu ghosts - TODO
    /*  
     void packGhost(void **ghost, const QudaParity parity, const int nFace, const int dagger) const;
     void unpackGhost(void* ghost_spinor, const int dim,
                      const QudaDirection dir, const int dagger);
    */

    cl.def("copy", &quda::cpuColorSpinorField::copy);  
    cl.def("zero", &quda::cpuColorSpinorField::zero); 

    // Skip virtual
    /*
     virtual void copy_to_buffer(void *buffer) const;
  
     virtual void copy_from_buffer(void *buffer);
    */

    // cpu ghost - do later
    /*
     void exchangeGhost(QudaParity parity, int nFace, int dagger, const MemoryLocation *pack_destination=nullptr,
                        const MemoryLocation *halo_location=nullptr, bool gdr_send=false, bool gdr_recv=false,
                        QudaPrecision ghost_precision=QUDA_INVALID_PRECISION) const;
    */

    cl.def("backup", &quda::cpuColorSpinorField::backup);
    cl.def("restore", &quda::cpuColorSpinorField::restore); 
} // init_cpuColorSpinorField

    

void init_color_spinor_field_pybind(py::module_ &m)
{
    // quda.colorspinor submodule
    auto csf = m.def_submodule("colorspinor");
    csf.def("isNative", &quda::colorspinor::isNative);

    // struct FullClover; // skip? Is it implemented?
    m.def("impliedParityFromMatPC", &quda::impliedParityFromMatPC);

    // Skip Composite types for now until I actually use them
    //struct CompositeColorSpinorFieldDescriptor {} 

    py::enum_<quda::MemoryLocation>(m, "MemoryLocation")
		.value("Device", quda::MemoryLocation::Device)
		.value("Host", quda::MemoryLocation::Host)
        .value("Remote", quda::MemoryLocation::Remote)
		.value("Shmem", quda::MemoryLocation::Shmem);

    // ColorSpinorParam
    init_ColorSpinorParam(m);

    // DslashConstant  -- skip for now. Implement when needed

    // ColorSpinorField
    init_ColorSpinorField(m);

    // Skip initializing cudaColorSpinorField for now

    // cpuColorSpinorField
    init_cpuColorSpinorField(m);
}