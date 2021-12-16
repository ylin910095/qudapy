
# Disable automatic intialization/finalization by mpi4py
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

import numpy as np
import quda
import atexit

# Define constants
pi = np.pi
e = np.e
ncolors = 3
nspins = 4
ndims = 4
grid_size = None # will be assigned by init
_is_init = False
_default_complex_type = np.complex128

quda_precision = {
    np.complex64: quda.enum_quda.QUDA_SINGLE_PRECISION,
    np.complex128: quda.enum_quda.QUDA_DOUBLE_PRECISION
}

def init(grid_size):
    """
    TODO: proper doc string...
    """
    global _is_init
    globals()["grid_size"] = grid_size # assign to the global variable grid_size
    if _is_init:
        raise RuntimeError("QudaPy is already initialized")
    if MPI.Is_initialized(): 
        raise RuntimeError("MPI is already initialized")
    quda.pyutils.init_comms(grid_size)
    _is_init = True

def finalize():
    """
    TODO: proper doc string...
    """
    global _is_init

    # Not initialized
    if not _is_init:
        return 
    quda.freeGaugeQuda()
    quda.freeCloverQuda()
    quda.endQuda()
    quda.finalize_comms()
    _is_init = False
    assert MPI.Is_finalized(), "MPI is not finalized by QUDA"

class Gauge_Field(object):
    """
    TODO: proper doc string...
    """
    def __init__(self, ndarray=None, gauge_param=None):
        #if quda_precision[ndarray.dtype] != gauge_param.cpu_prec:
        #    raise TypeError("ndarray and gauge_param must have the same precision on the CPU")
        self.ndarray = ndarray
        self.gauge_param = gauge_param

    def __str__(self):
        return "<GAUGED>"

def load_gauge(gauge_file, dims, dtype=_default_complex_type, **kwargs):
    """
    TODO: proper doc string...
    """
    if dtype != np.complex128 and dtype != np.complex64:
        raise TypeError("dtype must be np.complex64 or np.complex128")

    gauge_param = quda.newQudaGaugeParam() 
    gauge_param.type = quda.enum_quda.QUDA_SU3_LINKS
    gauge_param.X = np.copy(np.array(dims)/grid_size) # local lattice dimensions
    gauge_param.cpu_prec = quda_precision[dtype]
    gauge_param.cuda_prec = quda_precision[dtype]
    gauge_param.cuda_prec_precondition = quda_precision[dtype]
    gauge_param.gauge_order = quda.enum_quda.QUDA_QDP_GAUGE_ORDER
    gauge_param.t_boundary = quda.enum_quda.QUDA_PERIODIC_T 
    gauge_param.reconstruct = quda.enum_quda.QUDA_RECONSTRUCT_NO # store all 18 real numbers explicitly
    gauge_param.reconstruct_precondition = quda.enum_quda.QUDA_RECONSTRUCT_NO
    gauge_param.anisotropy = 1.0
    gauge_param.tadpole_coeff = 1.0
    gauge_param.ga_pad = 0
    gauge_param.mom_ga_pad = 0
    gauge_param.gauge_fix = quda.enum_quda.QUDA_GAUGE_FIXED_NO

    # For multi-GPU, ga_pad must be large enough to store a time-slice (in GPU, not in host)
    x_face_size = gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3] / 2
    y_face_size = gauge_param.X[0] * gauge_param.X[2] * gauge_param.X[3] / 2
    z_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[3] / 2
    t_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] / 2
    pad_size = max([x_face_size, y_face_size, z_face_size, t_face_size])
    gauge_param.ga_pad = int(pad_size)
    gauge_param.struct_size = quda.cfunc.sizeof(gauge_param)

    # Allocate memory and read
    gauge_site_size = 18 # storing all 3*3 complex numbers = 18 real numbers from SU(3) matrices
    gauge_field = np.full((4, np.prod(gauge_param.X), 3, 3, 2), fill_value=np.nan, dtype=np.double)
    quda.qio_field.read_gauge_field(gauge_file, gauge_field, quda_precision[dtype], 
                                    gauge_param.X, gauge_site_size)
    gauge_field = gauge_field.view(dtype)[..., 0] # eliminate the last complex dimension after view
    return Gauge_Field(ndarray=gauge_field, gauge_param=gauge_param)

# Print usage messages from QUDA and finalize the communications upon program exit
atexit.register(finalize)