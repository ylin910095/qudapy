import numpy as np
import qudapy.config as config
import qudapy as qp

def plaq(gf: qp.Gauge_Field):
    """
    TODO: proper doc string...
    """
    if gf.loc != "host":
        raise RuntimeError("The gauge field is not on the host")

    # The plaqutte in six directions - 01, 02, 03, 12, 13, 23
    plaq = np.zeros(6, dtype=config._default_ftype)
    alldir = ["01", "02", "03", "12", "13", "23"]
    # This is wrong - we need to gather the neighbor...
    """
    for i, idir in enumerate(alldir):
        iU = gf.data[int(idir[0])]
        jU = gf.data[int(idir[1])]
        plaq[i] = np.einsum("iab, ibc, idc, iad",
                            iU, jU, np.conj(jU), np.conj(iU))
    norm = gf.data.shape[1] # local volume
    plaq = np.real(plaq)
    return plaq/norm
    """