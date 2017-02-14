import numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_gofs(double[:,:] pvs, double[:,:] coeff_priors, double ve, double vn):
    cdef int n_iters = coeff_priors.shape[0]
    cdef int n_plates = coeff_priors.shape[1]
    cdef double pe, pn
    cdef double[:] gofs = np.zeros((n_iters))
    
    for i in range(n_iters):
        
        pe = 0.
        pn = 0.
        
        for j in range(n_plates):
            pe += pvs[j,0] * coeff_priors[i,j]
            pn += pvs[j,1] * coeff_priors[i,j]
        
        gofs[i] = np.sqrt( (ve-pe)**2 + (vn-pn)**2 )
        
    return gofs
