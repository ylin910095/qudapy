# Disable automatic intialization/finalization by mpi4py
# Those will be handled by the Quda binding
# See https://bitbucket.org/mpi4py/mpi4py/issues/85/manual-finalizing-and-initializing-mpi
import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI
import test_params 

import quda
import numpy as np

lattice_dims = [12, 12, 12, 24]
gauge_file = "./test_config.lime"
clover_csw = 1.24930970916466 # is this correct?
solve_tol = 1e-12

# 72 = real numbers per block-diagonal clover matrix 
# From https://arxiv.org/abs/1109.2935 
# Each clover matrix has a Hermitian block diagonal, anti-
# Hermitian block off-diagonal structure, and can be fully described
# by 72 real numbers.
# Also see appendix A of https://arxiv.org/abs/2109.10687
# \sigma_{\mu\nu} gives 6 * 4 = 24 real numbers for all 6 directions of mu, nu
# F_{\mu\nu} gives 6 * (3**2 - 1) = 48 real numbers for all 6 directions of mu, nu
# 72 = 48 + 24 (is this the correct way of counting?)
clover_site_size = 72

mass = -0.2800
Nsrc = 3 # number of random source vectors for testings
gauge_site_size = 18 # storing all 9 complex numbers from a SU(3) matrix
grid_size = np.array([2, 1, 1, 2]) # grid_size (x, y, z, t) 
rho = 0.125 # for the stout smearing

quda.pyutils.setQudaVerbosityStdout(quda.enum_quda.QUDA_SUMMARIZE)

assert MPI.Is_initialized() == False, "MPI is initialized before QUDA initialization"
quda.pyutils.init_comms(grid_size)
assert MPI.Is_initialized(), "MPI is not initialized by QUDA"

# Find world size, rank, and grid_coor
world_size = quda.qmp.QMP_get_number_of_nodes()
rank = quda.qmp.QMP_get_node_number()
grid_corr = quda.qmp.QMP_get_logical_coordinates()
print(f"World size = {world_size}, rank = {rank}, grid coordinate = {grid_corr}")

# Create params
gauge_param = test_params.create_quda_gauge_params(grid_size, lattice_dims)
inv_param = test_params.create_cloverinv_params(mass, clover_csw, solve_tol)
cs_param = test_params.create_clover_params(grid_size, lattice_dims)

# Making sure the ColorSpinorParam is properly set up
check = quda.ColorSpinorField.Create(cs_param)

# Then we make bunch of them
in_quark_list = [quda.ColorSpinorField.Create(cs_param) for i in range(Nsrc)]
out_quark_list = [quda.ColorSpinorField.Create(cs_param) for i in range(Nsrc)]
for i in in_quark_list:
    i.Source(quda.enum_quda.QUDA_RANDOM_SOURCE)

    # Make sure this looks correct
    lattice_indx = np.zeros(4, dtype=np.int32)
    i.LatticeIndex(lattice_indx, 1526)

# Doing some checks on gauge_param to test setters and getters
original_X = np.copy(gauge_param.X) # important to copy!
gauge_param.X = [3, 2, 1, 0]
gauge_param.X[-1] = -100
assert list(gauge_param.X) == [3, 2, 1, -100]
gauge_param.X = original_X # restore original value after testings

# Create the gauge field buffer and load the file
gauge_field = np.full((4, np.prod(gauge_param.X), 3, 3, 2), fill_value=np.nan, dtype=np.double)
quda.qio_field.read_gauge_field(gauge_file, gauge_field, test_params.cpu_prec, 
                                gauge_param.X, gauge_site_size)
gauge_field = gauge_field.view(np.complex128)[..., 0] # eliminate the last complex dimension after view

assert np.nan not in gauge_field, "nan in gauge field"
assert gauge_field.flags.c_contiguous, "Fail to load gauge field. The data is not C contiguous"

# Gauge field stuff
quda.loadGaugeQuda(gauge_field, gauge_param) # load to gaugePrecise

quda.performSTOUTnStep(1, rho, -1) # smear one step (n=1), don't measure topological charge except for one (-1)
quda.pyutils.copySmearedToPrecise() # need to do this after smearing for inversion
plaq = np.full(3, np.nan, dtype=np.double) 
quda.plaqQuda(plaq)
print(f"total plaq = {plaq[0]}, spatial plaq = {plaq[1]}, temporal plaq = {plaq[2]}")

# Test unitarity
UUdagger = np.einsum("...ab,...cb -> ...ac", gauge_field, np.conj(gauge_field), optimize=True)
identity_shape = (4, np.prod(gauge_param.X), 3, 3) 
unit_gauge = np.zeros(identity_shape, dtype=np.complex128)
idx = np.arange(3)
unit_gauge[..., idx, idx] = 1.0   
assert np.allclose(UUdagger, unit_gauge), "gauge field is not unitary"
print("Test passed: gauge_field is unitary")

# Construct clover field
clover_field = np.full((np.prod(gauge_param.X) * clover_site_size), fill_value=np.nan, dtype=np.double)
clovinv_fied = np.full((np.prod(gauge_param.X) * clover_site_size), fill_value=np.nan, dtype=np.double)

# TIL: QUDA will segfault if the gauge field is not on thd device
quda.loadCloverQuda(clover_field, clovinv_fied, inv_param)

# Invert
for i in range(Nsrc):
    quda.invertQuda(out_quark_list[i].V(), in_quark_list[i].V(), inv_param)

# Try to delete memory created from the QUDA side to make sure no leaks there
del check 
for i, j in zip(in_quark_list, out_quark_list):
    del i
    del j
del in_quark_list
del out_quark_list # this is not suffice for some reasons?
quda.freeGaugeQuda() # free the gauge field on the device
quda.freeCloverQuda() # free the clover field on the device

quda.endQuda() # this will empty all CUDA resources and bunch of stuff

# This will end all communications -- you can't initialize again in the 
# same session! If you have more MPI stuff to do, do that first before
# calling this
quda.finalize_comms()
assert MPI.Is_finalized(), "MPI is not finalized by QUDA"