
# Disable automatic intialization/finalization by mpi4py
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
from qudapy import config

import numpy as np
import quda
import atexit

# Define constants
pi = np.pi
e = np.e
ncolors = 3
nspins = 4
ndims = 4
_default_complex_type = np.complex128

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
    config.grid_size = grid_size # assign to the global variable grid_size
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
    def __init__(self, ndarray, param):
        if quda_precision[ndarray.dtype] != param.cpu_prec:
            raise TypeError("ndarray and param must have the same precision cpu precision")
        self.ndarray = ndarray
        self.param = param

    def __str__(self):
        return str(self.ndarray)

def load_gauge(gauge_file, dims, dtype=_default_complex_type, 
               anisotropy=False, t_boundary="periodic", tadpole_coeff=1.0, **kwargs):
    """
    TODO: proper doc string...
    """
    if dtype != np.complex128 and dtype != np.complex64:
        raise TypeError("dtype must be np.complex64 or np.complex128")

    param = quda.newQudaGaugeParam() 

    param.anisotropy = int(not anisotropy) # 1 == NOT anisotropy. Is this correct?
    param.t_boundary = quda_t_boundary[t_boundary]
    param.X = np.copy(np.array(dims)/config.grid_size) # local lattice dimensions
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
    gauge_site_size = 18 # storing all 3*3 complex numbers = 18 real numbers from SU(3) matrices
    gauge_field = np.full((4, np.prod(param.X), 3, 3, 2), fill_value=np.nan, dtype=np.double)
    quda.qio_field.read_gauge_field(gauge_file, gauge_field, quda_precision[dtype], 
                                    param.X, gauge_site_size) 
    gauge_field = gauge_field.view(dtype)[..., 0] # eliminate the last complex dimension after view
    return Gauge_Field(ndarray=gauge_field, param=param)

def plaq(gf: Gauge_Field=None, dtype=np.double) -> tuple:
    """
    TODO: proper doc string...
    """
    if (gf is not None) and (not isinstance(gf, Gauge_Field)):
        raise TypeError("gf has to be None or an instance of Gauge_Field")

    # Use the exisiting copy on the gpu (gaugePrecise speicifcally in QUDA)
    # to measure the plaqutte if gf is None
    if gf is not None:
        quda.loadGaugeQuda(gf.ndarray, gf.param) # load to gaugePrecise 
    ret = np.full(3, np.nan, dtype=dtype) 
    quda.plaqQuda(ret)
    assert np.nan not in ret
    return ret

def stout(n, rho, gf: Gauge_Field=None):
    """
    TODO: proper doc string...
    """
    if n != int(n):
        raise TypeError("n must ba an integer")
    if gf is not None and not isinstance(gf, Gauge_Field):
        raise TypeError("gf has to be None or an instance of Gauge_Field")
    # Use the exisiting copy on the gpu (gaugePrecise speicifcally in QUDA)
    # to measure the plaqutte

# Print usage messages from QUDA and finalize the communications upon program exit
atexit.register(finalize)