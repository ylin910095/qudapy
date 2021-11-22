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
grid_size = np.array([1, 1, 1, 8]) # grid_size (x, y, z, t) 

assert MPI.Is_initialized() == False, "MPI is initialized before QUDA initialization"
quda.init_comms(grid_size)
assert MPI.Is_initialized(), "MPI is not initialized by QUDA"

# Create a new gauge parameter object.
# Don't use quda.QudaGaugeParam() directly because it will not
# properly initialize the fields!
gauge_param = quda.newQudaGaugeParam() 
inv_param = quda.newQudaInvertParam() 

# Set gauge params
gauge_param.type = quda.enum_quda.QUDA_SU3_LINKS
gauge_param.X = np.array(lattice_dim)/grid_size # local lattice dimensions
gauge_param.cpu_prec = cpu_prec
gauge_param.cuda_prec = cuda_prec
gauge_param.cuda_prec_precondition = cuda_prec

gauge_param.reconstruct = quda.enum_quda.QUDA_RECONSTRUCT_NO # store all 18 real numbers explicitly
gauge_param.reconstruct_precondition = quda.enum_quda.QUDA_RECONSTRUCT_NO
assert int(gauge_param.reconstruct_precondition) == gauge_site_size

gauge_param.anisotropy = 1.0
gauge_param.tadpole_coeff = 1.0

gauge_param.ga_pad = 0
gauge_param.mom_ga_pad = 0
gauge_param.gauge_fix = quda.enum_quda.QUDA_GAUGE_FIXED_NO
gauge_param.struct_size = quda.sizeof(gauge_param)

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


gauge_field = quda.qio_field.read_gauge_field(gauge_file, cpu_prec, 
                                              gauge_param.X, gauge_site_size)


"""
print(gauge_param.X)
gauge_param.X = [3,2,1,0]
gauge_param.X[-1] = 100
#gauge_param.X[5] = 1000 # KABOOM!
#gauge_param.X = [5,6,7,8,9] # KABOOM! Wrong length!
print(gauge_param.X.shape) # it is also a numpy array
print(type(gauge_param.X))
print(gauge_param.X)
"""

quda.endQuda() # this will empty all CUDA resources and bunch of stuff

# This will end all communications -- you can't initialize again in the 
# same session! If you have more MPI stuff to do, do that first before
# calling this
quda.finalize_comms()
assert MPI.Is_finalized(), "MPI is not finalized by QUDA"
