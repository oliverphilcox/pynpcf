#!python
#cython: language_level=3

# cython functions for 4pcf_code

import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.float64
CTYPE = np.complex128
ctypedef np.float64_t DTYPE_t
ctypedef np.complex128_t CTYPE_t

# Define variables
cdef int numell
cdef int nbins
cdef bint warnings = True # include warnings?

def initialize(numell_input,nbins_input):
    """Write some global attributes here from the python script"""
    global numell
    numell = numell_input
    global nbins
    nbins = nbins_input

if warnings: print("should preload weight matrix also")

@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
cpdef void threepcf_sum(np.ndarray[CTYPE_t, ndim=2] y_all, np.ndarray[DTYPE_t, ndim=3] zeta3, np.ndarray[DTYPE_t, ndim=2] weights_3pcf):
    """Sum up multipole array into 3PCFs pythonically"""
    # sum up all ell (simply with weights)
    cdef int l1, m1
    cdef np.ndarray[CTYPE_t, ndim=1] a_l1m1
    cdef np.ndarray[CTYPE_t, ndim=1] a_l2m2
    for l1 in range(numell):
        for m1 in range(-l1,1):
            a_l1m1 = y_all[l1*(l1+1)//2+l1+m1]
            a_l2m2 = a_l1m1.conjugate()*(-1.)**m1
            # nb: using np.outer seems faster than doing it with for loops for 3PCF
            zeta3[l1] += (np.outer(a_l1m1,a_l2m2).real*weights_3pcf[l1**2+m1+l1,l1**2-m1+l1])*((m1!=0)+1) # can take real part since answer will be real!

@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
cpdef void fourpcf_sum(np.ndarray[CTYPE_t, ndim=2] y_all, np.ndarray[DTYPE_t, ndim=6] zeta4, np.ndarray[DTYPE_t, ndim=3] weights_4pcf):
    """Sum up multipole array into 4PCFs"""

    cdef int l1, l2, l3, m1, m2, m3
    cdef np.ndarray[CTYPE_t, ndim=1] a_l1m1
    cdef np.ndarray[CTYPE_t, ndim=1] a_l2m2
    cdef np.ndarray[CTYPE_t, ndim=1] a_l3m3
    cdef DTYPE_t this_weight
    cdef CTYPE_t a1, a2

    for l1 in range(numell):

        for l2 in range(numell):

            for l3 in range(abs(l1-l2),min(l1+l2+1,numell)):

                for m1 in range(-l1,l1+1):
                    # load a_l1m1
                    if m1<0:
                        a_l1m1 = y_all[l1*(l1+1)//2+l1+m1]
                    else:
                        a_l1m1 = y_all[l1*(l1+1)//2+l1-m1].conjugate()*(-1.)**m1

                    for m2 in range(-l2,l2+1):
                        # load a_l2m2
                        if m2<0:
                            a_l2m2 = y_all[l2*(l2+1)//2+l2+m2]
                        else:
                            a_l2m2 = y_all[l2*(l2+1)//2+l2-m2].conjugate()*(-1.)**m2

                        # set m3 from m1 + m2 + m3 = 0
                        m3 = -m1-m2
                        if abs(m3)>l3: continue

                        this_weight = weights_4pcf[l1**2+m1+l1,l2**2+m2+l2,l3**2+m3+l3]
                        if this_weight==0: continue

                        # load a_l3m3
                        if m3<0:
                            a_l3m3 = y_all[l3*(l3+1)//2+l3+m3]
                        else:
                            a_l3m3 = y_all[l3*(l3+1)//2+l3-m3].conjugate()*(-1.)**m3

                        # NB: the contribution from (-m1, -m2) is just the conjugate of that from (m1, m2)
                        # this can probably reduce the number of summations by ~ 2x
                        # todo: implement this
                        for a in range(nbins):
                            a1 = a_l1m1[a]
                            for b in range(nbins):
                                a2 = a_l2m2[b]
                                for c in range(nbins):
                                    zeta4[l1,l2,l3,a,b,c] += (a1*a2*a_l3m3[c]).real*this_weight

if warnings: print("could perhaps do better with a C++ NPCF class that is updated from python calls")
