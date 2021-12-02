# Disable automatic intialization/finalization by mpi4py
# Those will be handled by the Quda binding
# See https://bitbucket.org/mpi4py/mpi4py/issues/85/manual-finalizing-and-initializing-mpi
import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

import quda
import numpy as np

lattice_dim = (12, 12, 12, 24)
gauge_file = "/data/d10b/ensembles/isoClover/cl21_12_24_b6p1_m0p2800m0p2450/cl21_12_24_b6p1_m0p2800m0p2450/Lattices/cl21_12_24_b6p1_m0p2800m0p2450-1a/cfgs/cl21_12_24_b6p1_m0p2800m0p2450-1a_cfg_1000.lime"
clover_coeff = 1.0 # placeholder, must be equal to kappa*csw
cpu_prec = quda.enum_quda.QUDA_DOUBLE_PRECISION
cuda_prec = quda.enum_quda.QUDA_DOUBLE_PRECISION
gauge_site_size = 18 # storing all 9 complex numbers from SU(3)
grid_size = np.array([2, 1, 2, 1]) # grid_size (x, y, z, t) 

assert MPI.Is_initialized() == False, "MPI is initialized before QUDA initialization"
quda.init_comms(grid_size)
assert MPI.Is_initialized(), "MPI is not initialized by QUDA"

# Create param objects
# Don't use quda.QudaGaugeParam() and  QudaInvertParam() directly because it will not
# properly initialize the fields!
# But it is ok to use quda.ColorSpinorParam() - just remember to initialize it also!
gauge_param = quda.newQudaGaugeParam() 
inv_param = quda.newQudaInvertParam() 
cs_param = quda.ColorSpinorParam() 
#cs_param = quda.LatticeFieldParam() 

# Set gauge params
gauge_param.type = quda.enum_quda.QUDA_SU3_LINKS
gauge_param.X = np.copy(np.array(lattice_dim)/grid_size) # local lattice dimensions
gauge_param.cpu_prec = cpu_prec
gauge_param.cuda_prec = cuda_prec
gauge_param.cuda_prec_precondition = cuda_prec

gauge_param.gauge_order = quda.enum_quda.QUDA_QDP_GAUGE_ORDER
gauge_param.t_boundary = quda.enum_quda.QUDA_PERIODIC_T 
gauge_param.reconstruct = quda.enum_quda.QUDA_RECONSTRUCT_NO # store all 18 real numbers explicitly
gauge_param.reconstruct_precondition = quda.enum_quda.QUDA_RECONSTRUCT_NO
assert int(gauge_param.reconstruct_precondition) == gauge_site_size

gauge_param.anisotropy = 1.0
gauge_param.tadpole_coeff = 1.0

gauge_param.ga_pad = 0
gauge_param.mom_ga_pad = 0
gauge_param.gauge_fix = quda.enum_quda.QUDA_GAUGE_FIXED_NO

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

pad_size = 0
# For multi-GPU, ga_pad must be large enough to store a time-slice (in GPU, not in host)
if np.sum(grid_size) != 4:
    x_face_size = gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3] / 2
    y_face_size = gauge_param.X[0] * gauge_param.X[2] * gauge_param.X[3] / 2
    z_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[3] / 2
    t_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] / 2
    pad_size = max([x_face_size, y_face_size, z_face_size, t_face_size])
gauge_param.ga_pad = int(pad_size)
gauge_param.ga_pad = 2000

gauge_param.struct_size = quda.cfunc.sizeof(gauge_param)


# Set clover params based on tests/utils/set_params.cpp
inv_param.dslash_type = quda.enum_quda.QUDA_CLOVER_WILSON_DSLASH
inv_param.clover_cpu_prec = cpu_prec
inv_param.clover_cuda_prec = cuda_prec
inv_param.clover_cuda_prec_precondition = quda.enum_quda.QUDA_DOUBLE_PRECISION
inv_param.clover_order = quda.enum_quda.QUDA_PACKED_CLOVER_ORDER
inv_param.clover_coeff = clover_coeff
inv_param.compute_clover_trlog = True

# Set general inverter params based on tests/utils/set_params.cpp
inv_param.inv_type = quda.enum_quda.QUDA_CGNE_INVERTER
inv_param.solution_type = quda.enum_quda.QUDA_MAT_SOLUTION
inv_param.solve_type = quda.enum_quda.QUDA_NORMOP_SOLVE
inv_param.matpc_type = quda.enum_quda.QUDA_MATPC_EVEN_EVEN
inv_param.dagger = quda.enum_quda.QUDA_DAG_YES
inv_param.mass_normalization = quda.enum_quda.QUDA_MASS_NORMALIZATION
inv_param.solver_normalization = quda.enum_quda.QUDA_DEFAULT_NORMALIZATION
inv_param.pipeline = 0
inv_param.Nsteps = 2
inv_param.tol = 1e-12
inv_param.tol_restart = 1e-6


# Initialize the spinor params
cs_param.nVec = 1
cs_param.nColor = 3
cs_param.nSpin = 4
cs_param.nDim = 4
cs_param.x[:cs_param.nDim] = np.copy(gauge_param.X) # the extra [:cs_param.nDim] is needed here because 
                                                    # cs_param.x has a length of QUDA_MAX_DIM = 6 
cs_param.setPrecision(cpu_prec) 
cs_param.siteSubset = quda.enum_quda.QUDA_FULL_SITE_SUBSET
cs_param.print()

assert cs_param.Precision() == cpu_prec, "Precision is not set properly"

# Doing some checks on gauge_param to test setters and getters
original_X = np.copy(gauge_param.X) # important to copy!
gauge_param.X = [3, 2, 1, 0]
gauge_param.X[-1] = -100
assert list(gauge_param.X) == [3, 2, 1, -100]
gauge_param.X = original_X # restore original value after testings

# Create the gauge field buffer and load the file
gauge_field = np.full((4, np.prod(gauge_param.X), 3, 3, 2), fill_value=np.nan, dtype=np.double)
quda.qio_field.read_gauge_field(gauge_file, gauge_field, cpu_prec, 
                                gauge_param.X, gauge_site_size)
gauge_field = gauge_field.view(np.complex128)[..., 0] # eliminate the last complex dimension after view

assert np.nan not in gauge_field, "nan in gauge field"
assert gauge_field.flags.c_contiguous, "Fail to load gauge field. The data is not C contiguous"

quda.loadGaugeQuda(gauge_field, gauge_param) # load to device
plaq = np.full(3, np.nan, dtype=np.double) 
quda.plaqQuda(plaq)
quda.freeGaugeQuda()
print(f"total plaq = {plaq[0]}, spatial plaq = {plaq[1]}, temporal plaq = {plaq[2]}")

# Test unitarity
UUdagger = np.einsum("...ab,...cb -> ...ac", gauge_field, np.conj(gauge_field), optimize=True)
identity_shape = (4, np.prod(gauge_param.X), 3, 3) 
unit_gauge = np.zeros(identity_shape, dtype=np.complex128)
idx = np.arange(3)
unit_gauge[..., idx, idx] = 1.0   
assert np.allclose(UUdagger, unit_gauge), "gauge field is not unitary"
print("Test passed: gauge_field is unitary")

# Create some spinor fields             


quda.endQuda() # this will empty all CUDA resources and bunch of stuff

# This will end all communications -- you can't initialize again in the 
# same session! If you have more MPI stuff to do, do that first before
# calling this
quda.finalize_comms()
assert MPI.Is_finalized(), "MPI is not finalized by QUDA"
