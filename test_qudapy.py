import qudapy as qp
import numpy as np
grid_size = [1,1,2,2]
dims = [12, 12, 12, 24]
gfile = "./test_config.lime"
qp.init(grid_size) 
gf = qp.load_gauge(gfile, dims)
print(gf.ndarray)