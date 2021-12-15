import numpy as np
import quda

# Define constants
pi = np.pi
e = np.e
ncolors = 3
nspins = 4
ndims = 4
_is_init = False

# Initialize the QUDA library
def init(grid_size):
    quda.pyutils.init_comms(grid_size)
    init = True
