import qudapy as qp
import quda
import numpy as np
import sys
grid_size = [1, 1, 2, 2]
dims = [12, 12, 12, 24]
gfile = "./test_config.lime"
qp.init(grid_size) 
gf = qp.load_gauge(gfile, dims)
plaq = qp.plaq(gf)
print(f"total plaq = {plaq[0]}, spatial plaq = {plaq[1]}, temporal plaq = {plaq[2]}")
