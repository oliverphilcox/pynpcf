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

from libc.stdlib cimport malloc

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


cdef class ThreePCF():

    # NB: not available in Python-space
    cdef double *weights_3pcf
    cdef int nbins, numell, n3pcf
    cdef double *zeta3
    cdef double[:,:] zeta3_view

    def __cinit__(self, numell, nbins, weights_3pcf):
        """Class to hold and accumulate the 3PCF"""
        # Load in attributes from Python
        self.numell = numell
        self.nbins = nbins

        # copy in weights array
        self.weights_3pcf = <double*>malloc(self.numell**2*sizeof(double))
        for i in range(self.numell**2):
            self.weights_3pcf[i] = weights_3pcf[i]

        # Set number of radial bins in 3PCF with r1<=r2
        self.n3pcf = self.nbins*(1+self.nbins)*(1+2*self.nbins)//6

        # Initialize 3PCF array of dimension (n_ell, n_3pcf)
        # This is where the matrix is physically stored
        self.zeta3 = <double*>malloc(self.numell*self.n3pcf*sizeof(double))

        # Initialize a memory view
        # This is how it is accessed and passed to Python
        self.zeta3_view = <double[:self.numell,:self.n3pcf]>self.zeta3

        self.reset()

    cdef void reset(self):
        """Reset the 3PCF array"""
        cdef int i
        for i in range(self.n3pcf*self.numell): self.zeta3[i] = 0.

    # define python outputs for 3PCF
    @property
    def zeta3(self):
        return np.array(self.zeta3_view)

    @cython.boundscheck(False)
    @cython.cdivision(False)
    @cython.wraparound(False)
    cpdef void add_to_sum(self, complex[:,:] y_all, complex[:,:] y_all_conj):
        """"Add multipoles in y_all to the 3PCF summation.
        This is stored in the C array and not returned here.

        All operations are done in C, besides the read-in of the a_lm arrays"""

        # sum up all ell (simply with weights)
        cdef int l1, m1, a, b, i
        cdef complex[:] a_l1m1
        cdef complex[:] a_l2m2
        cdef complex a1w
        cdef double this_weight

        # Fill up r2>=r1 elements of array
        for l1 in range(self.numell):
            for m1 in range(-l1,1):
                this_weight = self.weights_3pcf[l1**2+m1+l1]
                a_l1m1 = y_all[l1*(l1+1)//2+l1+m1]
                a_l2m2 = y_all_conj[l1*(l1+1)//2+l1+m1]
                i = 0
                for a in range(self.nbins):
                    a1w = a_l1m1[a]*this_weight
                    for b in range(a,self.nbins):
                        self.zeta3[self.n3pcf*l1+i] += (a1w*a_l2m2[b]).real
                        i += 1

cdef class FourPCF():

    # NB: not available in Python-space
    # Note all C arrays are stored in 1D
    cdef double *weights_4pcf
    cdef int nbins, numell, numell2, n4pcf
    cdef double *zeta4
    cdef double[:,:,:,::1] zeta4_view

    def __cinit__(self, numell, nbins, weights_4pcf):
        """Class to hold and accumulate the 4PCF"""
        # Load in attributes from Python
        self.numell = numell
        self.numell2 = self.numell**2
        self.nbins = nbins

        # copy in 4pcf Python array into C
        # create a view of the array and a location for holding it
        self.weights_4pcf = <double*>malloc(self.numell**5*sizeof(double))

        cdef int index = 0
        for i in range(self.numell2):
            for j in range(self.numell2):
                for k in range(self.numell):
                    self.weights_4pcf[index] = weights_4pcf[i,j,k]
                    index += 1

        # Set number of radial bins in 3PCF with r1<=r2
        self.n4pcf = self.nbins*(1+self.nbins)*(2+self.nbins)*(1+3*self.nbins)//24

        # Initialize 4PCF array of dimension (n_ell, n_ell, n_ell, n_4pcf)
        # This is where the matrix is physically stored
        self.zeta4 = <double*>malloc(self.numell*self.numell*self.numell*self.n4pcf*sizeof(double))

        # Initialize a memory view
        # This is how it is passed to Python
        self.zeta4_view = <double[:self.numell,:self.numell,:self.numell,:self.n4pcf]>self.zeta4

        # empty the array
        self.reset()

    cdef void reset(self):
        """Reset the 4PCF array"""
        cdef int i
        for i in range(self.numell*self.numell*self.numell*self.n4pcf): self.zeta4[i] = 0.

    # define python outputs for 4PCF
    @property
    def zeta4(self):
        return np.array(self.zeta4_view)

    @cython.boundscheck(False)
    @cython.cdivision(False)
    @cython.wraparound(False)
    cpdef void add_to_sum(self, complex[:,:] y_all, complex[:,:] y_all_conj):
        """"Add multipoles in y_all to the 4PCF summation.
        This is stored in the C array and not returned here.

        All operations are done in C, besides the read-in of the a_lm arrays"""

        cdef int l1, l2, l3, m1, m2, m3, i
        cdef complex[:] a_l1m1
        cdef complex[:] a_l2m2
        cdef complex[:] a_l3m3
        cdef double this_weight
        cdef complex a1w, a12w

        cdef complex[:,:] local_y1
        cdef complex[:,:] local_y2
        cdef complex[:,:] local_y1_conj
        cdef complex[:,:] local_y2_conj
        cdef complex[:,:] local_y3

        ## might be able to preload which bins to sum over, i.e. a list of {L1,L2,L3,M1,M2} etc. and their weights
        ## this doesn't actually improve things

        for l1 in range(self.numell):
            local_y1 = y_all[l1*(l1+1)//2:(l1+1)*(l1+2)//2]
            local_y1_conj = y_all_conj[l1*(l1+1)//2:(l1+1)*(l1+2)//2]

            for l2 in range(self.numell):
                local_y2 = y_all[l2*(l2+1)//2:(l2+1)*(l2+2)//2]
                local_y2_conj = y_all_conj[l2*(l2+1)//2:(l2+1)*(l2+2)//2]


                for l3 in range(abs(l1-l2),min(l1+l2+1,self.numell)):
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

                            this_weight = self.weights_4pcf[self.numell*(self.numell2*(l1**2+m1+l1)+l2**2+m2+l2)+l3]
                            if this_weight==0: continue

                            # load a_l3m3 (m3<=0 always here)
                            a_l3m3 = local_y3[l3+m3]

                            # Sum up array, noting that we fill only r3>=r2>=r1
                            i = 0
                            for a in range(self.nbins):
                                a1w = a_l1m1[a]*this_weight
                                for b in range(a,self.nbins):
                                    a12w = a1w*a_l2m2[b]
                                    for c in range(b,self.nbins):
                                        self.zeta4[self.n4pcf*(self.numell*(l1*self.numell+l2)+l3)+i] += (a12w*a_l3m3[c]).real
                                        i += 1

@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
cpdef void threepcf_sum(complex[:,:] y_all, complex[:,:] y_all_conj, double[:,:] zeta3, double[:] weights_3pcf):
    """Sum up multipole array into 3PCFs pythonically"""
    # sum up all ell (simply with weights)
    cdef int l1, m1, a, b, i
    cdef complex[:] a_l1m1
    cdef complex[:] a_l2m2
    cdef complex a1w
    cdef double this_weight

    # Fill up r2>=r1 elements of array
    for l1 in range(numell):
        for m1 in range(-l1,1):
            this_weight = weights_3pcf[l1**2+m1+l1]
            a_l1m1 = y_all[l1*(l1+1)//2+l1+m1]
            a_l2m2 = y_all_conj[l1*(l1+1)//2+l1+m1]
            i = 0
            for a in range(nbins):
                a1w = a_l1m1[a]*this_weight # a_lm(r)*weight
                for b in range(a,nbins):
                    zeta3[l1,i] += (a1w*a_l2m2[b]).real
                    i += 1

@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
cpdef void fourpcf_sum(complex[:,:] y_all, complex[:,:] y_all_conj, double[:,:,:,:] zeta4, double[:,:,:] weights_4pcf):
    """Sum up multipole array into 4PCFs. Note this uses full C typing to avoid accessing numpy arrays, loading only C memviews.
    This is possible since we don't do any operations on the arrays themselves
    The only numpy array to access is zeta4"""

    cdef int l1, l2, l3, m1, m2, m3, i
    cdef complex[:] a_l1m1
    cdef complex[:] a_l2m2
    cdef complex[:] a_l3m3
    cdef double this_weight
    cdef complex a1w, a12w

    cdef complex[:,:] local_y1
    cdef complex[:,:] local_y2
    cdef complex[:,:] local_y1_conj
    cdef complex[:,:] local_y2_conj
    cdef complex[:,:] local_y3

    ## might be able to preload which bins to sum over, i.e. a list of {L1,L2,L3,M1,M2} etc. and their weights
    ## this doesn't actually improve things

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
