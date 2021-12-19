import numpy as np
# For sharing the global variable across different modules
grid_size = None # will be assigned by init
_is_init = False
_default_ftype = np.double
_default_ctype = np.complex128