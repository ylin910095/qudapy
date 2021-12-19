# Disable automatic intialization/finalization by mpi4py
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
from qudapy import config, comm

import numpy as np
import quda
import atexit, copy

# Define constants
pi = np.pi
e = np.e
ncolors = 3
nspins = 4
ndims = 4

quda_precision = {
    np.complex64: quda.enum_quda.QUDA_SINGLE_PRECISION,
    np.complex128: quda.enum_quda.QUDA_DOUBLE_PRECISION,
    np.dtype("complex64"): quda.enum_quda.QUDA_SINGLE_PRECISION, # same as np.complex64
    np.dtype("complex128"): quda.enum_quda.QUDA_DOUBLE_PRECISION # same as np.complex128
}

quda_t_boundary = {
    "periodic": quda.enum_quda.QUDA_PERIODIC_T,
    "antiperiodic": quda.enum_quda.QUDA_ANTI_PERIODIC_T
}

def init(grid_size):
    """
    TODO: proper doc string...
    """
    comm.grid_size = grid_size # assign to the global variable grid_size
    if config._is_init:
        raise RuntimeError("QudaPy is already initialized")
    if MPI.Is_initialized(): 
        raise RuntimeError("MPI is already initialized")
    quda.pyutils.init_comms(grid_size)
    config._is_init = True

def finalize():
    """
    TODO: proper doc string...
    """
    # Not initialized
    if not config._is_init:
        return 
    quda.freeGaugeQuda()
    quda.freeCloverQuda()
    quda.endQuda()
    quda.finalize_comms()
    config._is_init = False
    assert MPI.Is_finalized(), "MPI is not finalized by QUDA"

class Gauge_Field(object):
    """
    TODO: proper doc string...
    """
    device_data = None

    def __init__(self, data=None, dims=None, param=None):
        if quda_precision[data.dtype] != param.cpu_prec:
            raise TypeError("data and param must have the same cpu precision")
        self.data = data
        self.param = param
        self.dims = dims
        self.loc = "host" 
        self.cls = type(self) # just for conveniences

    def to(self, dest):
        # Host or device
        if dest not in ["host", "device"]:
            raise ValueError("Unknown destination")

        # The same field is already on the device. Do nothing
        if self.loc == dest:
            return
        
        # Host to device
        if dest == "device":
            quda.loadGaugeQuda(self.data, self.param)
            self.cls.device_data = self
        # Device to host
        elif dest == "host":
            if self.cls.device_data is not self:
                raise ValueError("The data on the device does not belong to this instance")
            quda.saveGaugeQuda(self.data, self.param)
            # Set device data to be none
            # Technically the data still exisits on the device. However,
            # it is much easier to enforce a single location for an obejct
            self.cls.device_data = None 
        self.loc = dest

    def __str__(self):
        return str(self.data)

def load_gauge(gauge_file, dims, dtype=config._default_ctype, 
               anisotropy=False, t_boundary="periodic", tadpole_coeff=1.0, dest="host", **kwargs):
    """
    TODO: proper doc string...
    """
    if dtype != np.complex128 and dtype != np.complex64:
        raise TypeError("dtype must be np.complex64 or np.complex128")

    param = quda.newQudaGaugeParam() 

    param.anisotropy = int(not anisotropy) # 1 == NOT anisotropy. Is this correct?
    param.t_boundary = quda_t_boundary[t_boundary]
    param.X = np.copy(np.array(dims)/comm.grid_size) # local lattice dimensions
    param.cpu_prec = quda_precision[dtype]
    param.cuda_prec = quda_precision[dtype]
    param.cuda_prec_precondition = quda_precision[dtype]
    param.tadpole_coeff = tadpole_coeff

    # These are other default params
    param.type = quda.enum_quda.QUDA_SU3_LINKS
    param.gauge_order = quda.enum_quda.QUDA_QDP_GAUGE_ORDER
    param.reconstruct = quda.enum_quda.QUDA_RECONSTRUCT_NO # store all 18 real numbers explicitly
    param.reconstruct_precondition = quda.enum_quda.QUDA_RECONSTRUCT_NO
    param.mom_ga_pad = 0
    param.gauge_fix = quda.enum_quda.QUDA_GAUGE_FIXED_NO

    # For multi-GPU, ga_pad must be large enough to store a time-slice (in GPU, not in host)
    x_face_size = param.X[1]*param.X[2]*param.X[3] / 2
    y_face_size = param.X[0]*param.X[2]*param.X[3] / 2
    z_face_size = param.X[0]*param.X[1]*param.X[3] / 2
    t_face_size = param.X[0]*param.X[1]*param.X[2] / 2
    pad_size = max([x_face_size, y_face_size, z_face_size, t_face_size])
    param.ga_pad = int(pad_size)
    param.struct_size = quda.cfunc.sizeof(param)

    # Overriding params with entries in kwargs
    for key, value in kwargs.items():
        setattr(param, key, value)
    
    # I believe qio read only supports QDP_GAUGE_ORDER
    if param.gauge_order != quda.enum_quda.QUDA_QDP_GAUGE_ORDER:
        raise RuntimeError("The gauge field ordering must be QUDA_QDP_GAUGE_ORDER")

    # Allocate memory and read with QIO
    gauge_site_size = ncolors*ncolors*2 # storing all 3*3 complex numbers = 18 real numbers from SU(3) matrices
    if dtype == np.complex128:
        fdtype = np.double
    else:
        fdtype = np.float
    gauge_field = np.full((4, np.prod(param.X), ncolors, ncolors, 2), fill_value=np.nan, dtype=fdtype)
    quda.qio_field.read_gauge_field(gauge_file, gauge_field, quda_precision[dtype], 
                                    param.X, gauge_site_size) 
    gauge_field = gauge_field.view(dtype)[..., 0] # eliminate the last complex dimension after view
    gf = Gauge_Field(data=gauge_field, dims=dims, param=param)
    gf.to(dest)
    return gf

def plaq(gf: Gauge_Field, dtype=np.double, dest="device"):
    """
    TODO: proper doc string...
    """
    if not isinstance(gf, Gauge_Field):
        raise TypeError("gf is not an instance of Gauge_Field")

    # Use the exisiting copy on the gpu (gaugePrecise speicifcally in QUDA)
    # to measure the plaqutte if gf is None
    if gf.loc != "device":
        raise RuntimeError("The gauge field is not on the device")
    ret = np.full(3, np.nan, dtype=dtype)
    quda.plaqQuda(ret)
    gf.to(dest)
    assert np.nan not in ret
    return ret

def stout(gf: Gauge_Field, n, rho, dest="device"):
    """
    TODO: proper doc string...
    """
    if n != int(n):
        raise TypeError("n must ba an integer")
    if not isinstance(gf, Gauge_Field):
        raise TypeError("gf is not an instance of Gauge_Field")
    if gf.loc != "device":
        raise RuntimeError("The gauge field is not on the device")

    # Smear n steps, don't measure topological charge except for the first one (-1)
    # The result will be stored in gaugeSmeared in QUDA defined in interface_quda.cpp
    quda.performSTOUTnStep(n, rho, -1) 
    quda.pyutils.copySmearedToPrecise() # always keep the current gf in guagePrecise in QUDA
    gf.to(dest)
    return gf 

# Print usage messages from QUDA and finalize the communications upon the program exit
atexit.register(finalize)