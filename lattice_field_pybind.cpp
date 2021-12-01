#include "lattice_field_pybind.hpp"

namespace py = pybind11; 

void init_LatticeField(py::module_ &m); // decalred to be defined later

void init_lattice_field_pybind(py::module_ &m) 
{
	py::enum_<quda::QudaOffsetCopyMode>(m, "QudaOffsetCopyMode")
		.value("QUDA_SUCCESS", quda::QudaOffsetCopyMode::COLLECT)
		.value("QUDA_ERROR", quda::QudaOffsetCopyMode::DISPERSE);

    // LatticeFieldParam
    py::class_<quda::LatticeFieldParam, std::unique_ptr<quda::LatticeFieldParam>> 
          cl(m, "LatticeFieldParam");

    // Constructors
    cl.def(py::init( [](){ return new quda::LatticeFieldParam(); } ) );
    cl.def(py::init( 
        [](int nDim, const py::array x, int pad, QudaPrecision precision,
           QudaGhostExchange ghostExchange) { 
            py::buffer_info buf = x.request();
            if (buf.ndim != nDim) {
                throw py::value_error("nDim mismatch in x");
            }
            auto x_ptr = static_cast<int*>(buf.ptr);
            return new quda::LatticeFieldParam(nDim, x_ptr, pad, precision, ghostExchange); 
        }),   
        "nDim"_a, "x"_a, "pad"_a, "precision"_a, "ghostExchange"_a = QUDA_GHOST_EXCHANGE_PAD
    ); 
    cl.def(py::init<const QudaGaugeParam &>());
    cl.def(py::init<const quda::LatticeField &>());

    // Public members
    cl.def("Precision", &quda::LatticeFieldParam::Precision);
    cl.def("GhostPrecision", &quda::LatticeFieldParam::GhostPrecision);
    cl.def_readwrite("nDim", &quda::LatticeFieldParam::nDim);
    cl.def_property("x", 
        [](py::object& self) {
            quda::LatticeFieldParam &obj = self.cast<quda::LatticeFieldParam &>();
            return attr_getter<quda::LatticeFieldParam, int>(self, obj.nDim, &obj.x[0]);
        },
        [](py::object &self, const py::array_t<int> &a) {
            quda::LatticeFieldParam &obj = self.cast<quda::LatticeFieldParam &>();
            return attr_setter<quda::LatticeFieldParam, int>(self, obj.nDim, &obj.x[0], a);
        }
      );

    cl.def_readwrite("pad", &quda::LatticeFieldParam::pad);
    cl.def_readwrite("siteSubset", &quda::LatticeFieldParam::siteSubset);
    cl.def_readwrite("mem_type", &quda::LatticeFieldParam::mem_type);
    cl.def_readwrite("ghostExchange", &quda::LatticeFieldParam::ghostExchange);
    cl.def_property("r", 
        [](py::object& self) 
        {
            quda::LatticeFieldParam &obj = self.cast<quda::LatticeFieldParam &>();
            return attr_getter<quda::LatticeFieldParam, int>(self, obj.nDim, &obj.r[0]);
        },
        [](py::object &self, const py::array_t<int> &a) 
        {
            quda::LatticeFieldParam &obj = self.cast<quda::LatticeFieldParam &>();
            return attr_setter<quda::LatticeFieldParam, int>(self, obj.nDim, &obj.r[0], a);
        }
    );
    cl.def_readwrite("scale", &quda::LatticeFieldParam::scale);

    init_LatticeField(m);
}

void init_LatticeField(py::module_ &m) 
{
    // LatticeField, inherited from Object in object.h
    py::class_<quda::LatticeField, quda::Object, std::unique_ptr<quda::LatticeField>> 
          cl(m, "LatticeField");

    // No constructors -- it is an abstract class
    // Public members
    cl.def("allocateGhostBuffer", &quda::LatticeField::allocateGhostBuffer,
           "ghost_bytes"_a);
    cl.def_static("freeGhostBuffer", &quda::LatticeField::freeGhostBuffer); 
    cl.def("createComms", &quda::LatticeField::createComms,
           "no_comms_fill"_a=false, "bidir"_a=true);
    cl.def("destroyComms", &quda::LatticeField::destroyComms);
    cl.def("createIPCComms", &quda::LatticeField::createIPCComms);
    cl.def_static("destroyIPCComms", &quda::LatticeField::destroyIPCComms);
    //cl.def("ipcCopyComplete", &quda::LatticeField::ipcCopyComplete, 
    //       "dir"_a, "dim"_a); // inline func not implemented for this abstract class. Skip.
    //cl.def("ipcRemoteCopyComplete", &quda::LatticeField::ipcRemoteCopyComplete,
    //       "dir"_a, "dim"_a); // inline func not implemented for this abstract class. Skip.     

    /*
    These are cuda-related functions. Skip for now.
    const cudaEvent_t& getIPCCopyEvent(int dir, int dim) const;
    const cudaEvent_t& getIPCRemoteCopyEvent(int dir, int dim) const;
    */
    
    cl.def_readwrite_static("bufferIndex", &quda::LatticeField::bufferIndex);
    cl.def_readwrite_static("ghost_field_reset", &quda::LatticeField::ghost_field_reset);
    cl.def("Ndim", &quda::LatticeField::Ndim); 
    cl.def("X", 
        [](py::object& self)
        {    
            quda::LatticeField& o = self.cast<quda::LatticeField&>(); 
            auto x = o.X();
            return py::array_t<int>(QUDA_MAX_DIM, x);
        });

    // virtual int full_dim(int d) const = 0; // virtual. Not implemented

    cl.def("Volume", &quda::LatticeField::Volume);
    cl.def("VolumeCB", &quda::LatticeField::VolumeCB);
    cl.def("LocalVolume", &quda::LatticeField::LocalVolume);
    cl.def("LocalVolumeCB", &quda::LatticeField::LocalVolumeCB);

    // Overloaded functions
    cl.def("SurfaceCB", 
        [](py::object& self)
        {    
            quda::LatticeField& o = self.cast<quda::LatticeField&>(); 
            auto scb = o.SurfaceCB();
            return py::array_t<int>(QUDA_MAX_DIM, scb);
        });
    cl.def("SurfaceCB", 
        [](py::object& self, const int i) 
        {
            quda::LatticeField& o = self.cast<quda::LatticeField&>(); 
            return o.SurfaceCB(i);
        }
    );

    cl.def("Stride", &quda::LatticeField::Stride);
    cl.def("Pad", &quda::LatticeField::Pad);

    cl.def("R", 
        [](py::object& self)
        {    
            quda::LatticeField& o = self.cast<quda::LatticeField&>(); 
            auto r = o.R();
            return py::array_t<int>(QUDA_MAX_DIM, r);
        });
    
    cl.def("GhostExchange", &quda::LatticeField::GhostExchange);
    cl.def("Precision", &quda::LatticeField::Precision);
    cl.def("GhostPrecision", &quda::LatticeField::GhostPrecision);

    // overload_cast seems to have difficulty when the overloaded function has no argument?
    cl.def("Scale", 
        [](py::object& self) 
        {
            quda::LatticeField& o = self.cast<quda::LatticeField&>(); 
            return o.Scale();
        }
    );
    cl.def("Scale", py::overload_cast<double>(&quda::LatticeField::Scale));

    /*  // Virtual functions. Skip      
     virtual QudaSiteSubset SiteSubset() const { return siteSubset; }
     virtual QudaMemoryType MemType() const { return mem_type; }
    */
  
    cl.def("Nvec", &quda::LatticeField::Nvec);
    cl.def("Location", &quda::LatticeField::Location);
    cl.def("GBytes", &quda::LatticeField::GBytes);
    cl.def("checkField", &quda::LatticeField::checkField);

    /*  // Virtual functions. Skip    
     virtual void read(char *filename);
     virtual void write(char *filename);
    */

    /* // These functions are more complicated to implement now because of the void pointer. 
       // Skip for now until we actually use them. Some ideas of implementing them might be 
       // Adding a precision variable in the binding class. 
     void *myFace_h(int dir, int dim) const { return my_face_dim_dir_h[bufferIndex][dim][dir]; }
     void *myFace_hd(int dir, int dim) const { return my_face_dim_dir_hd[bufferIndex][dim][dir]; }
     void *myFace_d(int dir, int dim) const { return my_face_dim_dir_d[bufferIndex][dim][dir]; }
     void *remoteFace_d(int dir, int dim) const { return ghost_remote_send_buffer_d[bufferIndex][dim][dir]; }
     void *remoteFace_r() const { return ghost_recv_buffer_d[bufferIndex]; }
    */
     
     /* // Virtual functions. Skip    
     virtual void gather(int nFace, int dagger, int dir, qudaStream_t *stream_p = NULL) { errorQuda("Not implemented"); }
     virtual void commsStart(int nFace, int dir, int dagger = 0, qudaStream_t *stream_p = NULL, bool gdr_send = false,
                             bool gdr_recv = true)
     { errorQuda("Not implemented"); }
     virtual int commsQuery(int nFace, int dir, int dagger = 0, qudaStream_t *stream_p = NULL, bool gdr_send = false,
                            bool gdr_recv = true)
     { errorQuda("Not implemented"); return 0; }
     virtual void commsWait(int nFace, int dir, int dagger = 0, qudaStream_t *stream_p = NULL, bool gdr_send = false,
                            bool gdr_recv = true)
     { errorQuda("Not implemented"); }
     virtual void scatter(int nFace, int dagger, int dir)
     { errorQuda("Not implemented"); }
     */
     
     /* // Virtual or internal functions. Skip for now
     inline const char *VolString() const { return vol_string; }
     inline const char *AuxString() const { return aux_string; }
     virtual void backup() const { errorQuda("Not implemented"); }
     virtual void restore() const { errorQuda("Not implemented"); }
     virtual void prefetch(QudaFieldLocation mem_space, qudaStream_t stream = 0) const { ; }
     virtual bool isNative() const = 0;
     virtual void copy_to_buffer(void *buffer) const = 0;
     virtual void copy_from_buffer(void *buffer) = 0;
    */

    // Other functions in lattice_field.h are mostly internal functions.
    // Skip them for now until they are needed
}
