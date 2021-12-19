# Disable automatic intialization/finalization by mpi4py
# Those will be handled by the Quda binding
# See https://bitbucket.org/mpi4py/mpi4py/issues/85/manual-finalizing-and-initializing-mpi
import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

import quda
import numpy as np 

import test_params # for conveniences 

grid_size = [1, 2, 1, 2]
lattice_dims = [12, 12, 12, 24]
gauge_file = "./test_config.lime"

# Init 
quda.pyutils.setQudaVerbosityStdout(quda.enum_quda.QUDA_SUMMARIZE)
assert MPI.Is_initialized() == False, "MPI is initialized before QUDA initialization"
quda.pyutils.init_comms(grid_size)
assert MPI.Is_initialized(), "MPI is not initialized by QUDA"

# Find world size, rank, and grid_coor
world_size = quda.qmp.QMP_get_number_of_nodes()
rank = quda.qmp.QMP_get_node_number()
grid_corr = quda.qmp.QMP_get_logical_coordinates()
print(f"World size = {world_size}, rank = {rank}, grid coordinate = {grid_corr}")

# Create qudaGaugeParam
gauge_param = test_params.create_quda_gauge_params(grid_size, lattice_dims)

# Read
gauge_site_size = 18
gauge_field = np.full((4, np.prod(gauge_param.X), 3, 3, 2), fill_value=np.nan, dtype=np.double)
quda.qio_field.read_gauge_field(gauge_file, gauge_field, test_params.cpu_prec, 
                                gauge_param.X, gauge_site_size)
gauge_field = gauge_field.view(np.complex128)[..., 0] # eliminate the last complex dimension after view

# Create internal gauge params for ghost exchanges (not the same as QudaGaugeParam)
internal_gparam = quda.GaugeFieldParam(gauge_field, gauge_param)
cpu_gauge_field = quda.cpuGaugeField(internal_gparam)
ghost = cpu_gauge_field.Ghost()