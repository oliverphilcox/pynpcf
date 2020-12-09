#!python
#cython: language_level=3

# cython functions for 3/4pcf_code

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
cpdef void threepcf_sum(np.ndarray[CTYPE_t, ndim=2] y_all, np.ndarray[DTYPE_t, ndim=2] zeta3, np.ndarray[DTYPE_t, ndim=1] weights_3pcf):
    """Sum up multipole array into 3PCFs pythonically"""
    # sum up all ell (simply with weights)
    cdef int l1, m1, a, b, i
    cdef np.ndarray[CTYPE_t, ndim=1] a_l1m1
    cdef np.ndarray[CTYPE_t, ndim=1] a_l2m2
    cdef CTYPE_t a1
    cdef DTYPE_t this_weight

    # Fill up r2>=r1 elements of array
    for l1 in range(numell):
        for m1 in range(-l1,1):
            this_weight = weights_3pcf[l1**2+m1+l1]
            a_l1m1 = y_all[l1*(l1+1)//2+l1+m1]
            a_l2m2 = a_l1m1.conjugate()*(-1.)**m1
            i = 0
            for a in range(nbins):
                a1 = a_l1m1[a]
                for b in range(a,nbins):
                    zeta3[l1,i] += (a1*a_l2m2[b]).real*this_weight
                    i += 1

@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
cpdef void fourpcf_sum(np.ndarray[CTYPE_t, ndim=2] y_all, np.ndarray[DTYPE_t, ndim=4] zeta4, np.ndarray[DTYPE_t, ndim=3] weights_4pcf):
    """Sum up multipole array into 4PCFs"""

    cdef int l1, l2, l3, m1, m2, m3, i
    cdef np.ndarray[CTYPE_t, ndim=1] a_l1m1
    cdef np.ndarray[CTYPE_t, ndim=1] a_l2m2
    cdef np.ndarray[CTYPE_t, ndim=1] a_l3m3
    cdef DTYPE_t this_weight
    cdef CTYPE_t a1w, a12w

    cdef np.ndarray[CTYPE_t, ndim=2] y_all_conj
    cdef np.ndarray[CTYPE_t, ndim=2] local_y1
    cdef np.ndarray[CTYPE_t, ndim=2] local_y2
    cdef np.ndarray[CTYPE_t, ndim=2] local_y1_conj
    cdef np.ndarray[CTYPE_t, ndim=2] local_y2_conj
    cdef np.ndarray[CTYPE_t, ndim=2] local_y3

    ## might be able to preload which bins to sum over, i.e. a list of {L1,L2,L3,M1,M2} etc. and their weights
    ## this doens't actually improve things

    # do all complex conjugation first
    y_all_conj = y_all.conjugate()

    for l1 in range(numell):
        local_y1 = y_all[l1*(l1+1)//2:(l1+1)*(l1+2)//2]
        local_y1_conj = y_all_conj[l1*(l1+1)//2:(l1+1)*(l1+2)//2]

        for l2 in range(numell):
            local_y2 = y_all[l2*(l2+1)//2:(l2+1)*(l2+2)//2]
            local_y2_conj = y_all_conj[l2*(l2+1)//2:(l2+1)*(l2+2)//2]


            for l3 in range(abs(l1-l2),min(l1+l2+1,numell)):
                local_y3 = y_all[l3*(l3+1)//2:(l3+1)*(l3+2)//2]

                for m1 in range(-l1,l1+1):
                    # load a_l1m1
                    if m1<0:
                        a_l1m1 = local_y1[l1+m1]
                    else:
                        # nb: extra (-1)^m factor absorbed into weight
                        a_l1m1 = local_y1_conj[l1-m1]

                    # enforce m1+m2 >= 0 and |m3|<l3 here
                    for m2 in range(max(max(-m1,-l2),-m1-l3),min(l2+1,l3-m1+1)):

                        # set m3 from m1 + m2 + m3 = 0
                        m3 = -m1-m2

                        # load a_l2m2
                        if m2<0:
                            a_l2m2 = local_y2[l2+m2]
                        else:
                            # nb: extra (-1)^m factor absorbed into weight
                            a_l2m2 = local_y2_conj[l2-m2]

                        # NB: we combine (m1,m2,m3) and (-m1,-m2,-m3) terms together
                        # this allows us to skip m3>0, and multiply by 2 [encoded in m2>=-m1 condition above]
                        # if m1=m2=m3=0 we have only one possibiliy, so we multiply by 1
                        # this factor is included in the weights

                        this_weight = weights_4pcf[l1**2+m1+l1,l2**2+m2+l2,l3]
                        if this_weight==0: continue

                        # load a_l3m3 (m3<=0 always here)
                        a_l3m3 = local_y3[l3+m3]

                        # Sum up array, noting that we fill only r3>=r2>=r1
                        i = 0
                        for a in range(nbins):
                            a1w = a_l1m1[a]*this_weight
                            for b in range(a,nbins):
                                a12w = a1w*a_l2m2[b]
                                for c in range(b,nbins):
                                    zeta4[l1,l2,l3,i] += (a12w*a_l3m3[c]).real
                                    i += 1

if warnings: print("could perhaps do better with a C++ NPCF class that is updated from python calls? conversion numpy -> C arrays will be slow however")
