import quda
import numpy as np

# Data size
gauge_site_size = 18 # storing all 9 complex numbers from a SU(3) matrix

gamma_basis = quda.enum_quda.QUDA_DEGRAND_ROSSI_GAMMA_BASIS

# Default precisions
cpu_prec = quda.enum_quda.QUDA_DOUBLE_PRECISION
cpu_prec_sloppy = quda.enum_quda.QUDA_HALF_PRECISION
cuda_prec = quda.enum_quda.QUDA_DOUBLE_PRECISION
cuda_prec_sloppy = quda.enum_quda.QUDA_HALF_PRECISION
cuda_prec_precondition = cuda_prec

def create_quda_gauge_params(grid_size, lattice_dims):
    gauge_param = quda.newQudaGaugeParam() 
    # Set gauge params
    gauge_param.type = quda.enum_quda.QUDA_SU3_LINKS
    gauge_param.X = np.copy(np.array(lattice_dims)/grid_size) # local lattice dimensions
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

    pad_size = 0
    # For multi-GPU, ga_pad must be large enough to store a time-slice (in GPU, not in host)
    if np.sum(grid_size) != 4:
        x_face_size = gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3] / 2
        y_face_size = gauge_param.X[0] * gauge_param.X[2] * gauge_param.X[3] / 2
        z_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[3] / 2
        t_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] / 2
        pad_size = max([x_face_size, y_face_size, z_face_size, t_face_size])
    gauge_param.ga_pad = int(pad_size)
    gauge_param.ga_pad = 1728
    gauge_param.struct_size = quda.cfunc.sizeof(gauge_param)

    return gauge_param

def create_cloverinv_params(mass, clover_csw, solve_tol):
    inv_param = quda.newQudaInvertParam() 

    # Set clover params based on tests/utils/set_params.cpp
    inv_param.mass = mass
    inv_param.kappa = 1.0 / (2.0 * (4 + mass))
    inv_param.laplace3D = -1 # (idk what it means) omit this direction 
                            # from laplace operator: x,y,z,t -> 0,1,2,3 (-1 is full 4D) 
    inv_param.Ls = 1 # not domain wall
    inv_param.dslash_type = quda.enum_quda.QUDA_CLOVER_WILSON_DSLASH
    inv_param.cpu_prec = cpu_prec
    inv_param.cuda_prec = cuda_prec
    inv_param.clover_cpu_prec = cpu_prec
    inv_param.clover_cuda_prec = cuda_prec
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition
    inv_param.clover_cuda_prec_precondition = quda.enum_quda.QUDA_DOUBLE_PRECISION
    inv_param.clover_order = quda.enum_quda.QUDA_PACKED_CLOVER_ORDER
    inv_param.clover_coeff = clover_csw * inv_param.kappa
    inv_param.compute_clover_trlog = 0
    inv_param.compute_clover = 1 # compute the clover field on the device 
    inv_param.return_clover = 0 # copy the computed clover field to the host
    inv_param.compute_clover_inverse = 1 # compute the clover inverse field on the device 
    inv_param.return_clover_inverse = 0 # copy the computed clover field to the host
    inv_param.cl_pad = 0 # for clover
    inv_param.sp_pad = 0 # the padding to use for the fermion fields 
    inv_param.maxiter = 2000
    inv_param.reliable_delta = 0.1
    inv_param.preserve_source = quda.enum_quda.QUDA_PRESERVE_SOURCE_YES # keep the source intact
    inv_param.gamma_basis = gamma_basis
    inv_param.dirac_order = quda.enum_quda.QUDA_QDP_DIRAC_ORDER
    inv_param.verbosity = quda.enum_quda.QUDA_SUMMARIZE

    # Set general inverter params based on tests/utils/set_params.cpp
    inv_param.inv_type = quda.enum_quda.QUDA_CG_INVERTER
    inv_param.solution_type = quda.enum_quda.QUDA_MAT_SOLUTION
    inv_param.solve_type = quda.enum_quda.QUDA_NORMOP_SOLVE
    inv_param.matpc_type = quda.enum_quda.QUDA_MATPC_EVEN_EVEN
    inv_param.dagger = quda.enum_quda.QUDA_DAG_YES
    inv_param.mass_normalization = quda.enum_quda.QUDA_MASS_NORMALIZATION
    inv_param.solver_normalization = quda.enum_quda.QUDA_DEFAULT_NORMALIZATION
    inv_param.pipeline = 0
    inv_param.Nsteps = 2
    inv_param.tol = solve_tol
    inv_param.tol_restart = 1e-6

    return inv_param

def create_clover_params(grid_size, lattice_dims):
    cs_param = quda.ColorSpinorParam() 
    # Initialize the spinor params -- mimicing behavior of constructWilsonTestSpinorParam 
    # in test/utils/host_utils.cpp
    cs_param.nVec = 1
    cs_param.nColor = 3
    cs_param.nSpin = 4
    cs_param.nDim = 4
    cs_param.x[:cs_param.nDim] = np.copy(np.array(lattice_dims)/grid_size)  # the extra [:cs_param.nDim] is needed here because 
                                                                            # cs_param.x has a length of QUDA_MAX_DIM = 6 
    cs_param.setPrecision(cpu_prec) 
    cs_param.siteSubset = quda.enum_quda.QUDA_FULL_SITE_SUBSET
    cs_param.pad = 0
    cs_param.siteOrder = quda.enum_quda.QUDA_EVEN_ODD_SITE_ORDER
    cs_param.fieldOrder = quda.enum_quda.QUDA_SPACE_SPIN_COLOR_FIELD_ORDER
    cs_param.gammaBasis = gamma_basis
    cs_param.create = quda.enum_quda.QUDA_ZERO_FIELD_CREATE
    cs_param.location = quda.enum_quda.QUDA_CPU_FIELD_LOCATION

    return cs_param