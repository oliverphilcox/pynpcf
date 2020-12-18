#Code for NPCFs - Zack Slepian + Oliver Philcox
#Currently supports 3PCF + 4PCF inputs

######################## LOAD PYTHON MODULES ########################
#doing basic math
import numpy as np
#system operations
import os, sys
#for timing
import time
#3j manipulations
from sympy.physics.wigner import wigner_3j

######################## HISTOGRAM CODE (from numpy) ########################

def _search_sorted_inclusive(a, v):
    """
    Like `searchsorted`, but where the last item in `v` is placed on the right.
    In the context of a histogram, this makes the last bin edge inclusive.
    Taken from numpy.
    """
    return np.concatenate((
        a.searchsorted(v[:-1], 'left'),
        a.searchsorted(v[-1:], 'right')
    ))

def histogram_multi(a, bins=10, weight_matrix=None):
    r"""
    Compute the histogram of a set of data. Taken from numpy and modified to use multiple weights at once

    """
    # We set a block size, as this allows us to iterate over chunks when
    # computing histograms, to minimize memory usage.
    BLOCK = 65536

    # specializing to non-uniform bins version from numpy here

    # Compute via cumulative histogram
    cum_n = np.zeros((weight_matrix.shape[0],bins.shape[0]), weight_matrix.dtype)
    zero = np.zeros((weight_matrix.shape[0],1), dtype=weight_matrix.dtype)
    for i in range(0, len(a), BLOCK):
        tmp_a = a[i:i+BLOCK]
        tmp_w = weight_matrix[:,i:i+BLOCK]
        sorting_index = np.argsort(tmp_a)
        sa = tmp_a[sorting_index]
        sw = tmp_w[:,sorting_index]
        cw = np.concatenate((zero, sw.cumsum(axis=1)),axis=1)
        bin_index = _search_sorted_inclusive(sa, bins)
        cum_n += cw[:,bin_index]

    n = np.diff(cum_n)

    return n

######################## COUPLING MATRICES ########################

def compute_3pcf_coupling_matrix(numell):
    """Computing the 3PCF coupling matrix C_{ll'}^{mm'} between different (Lambda_i M_i) pairs.
    This is as defined in Slepian/Cahn 2020, with a few extra factors for later use.

    It is stored as a numpy array of length numell^2"""

    weights_3pcf = np.zeros(numell**2,dtype=np.float64)

    # Iterate over ell_1
    for ell_1 in range(numell):
        # Iterate over m_1, using only m_1<=0 (other terms computed by symmetry)
        for m_1 in range(-ell_1,1):
            # enforce (ell_i = ell_j) and m_i = m_j
            ell_2 = ell_1
            m_2 = -m_1
            # factor missed in spherical harmonic definition
            tmp_fac = (2.*ell_1+1.)/np.pi
            # add factor of 2x if m1 != 0 by symmmetry
            if m_1==0:
                tmp_fac *= 1.
            else:
                tmp_fac *= 2.
            # compute weights (note we absorb extra factor of (-1)^m from complex conjugate)
            weights_3pcf[ell_1**2+m_1+ell_1] = tmp_fac*pow(-1.,ell_1)/np.sqrt(2.*ell_1+1.)

    np.save('coupling_matrices/weights_3pcf_n%d'%numell,weights_3pcf)
    return weights_3pcf

def compute_4pcf_coupling_matrix(numell):
    """Computing the 4PCF coupling matrix C_{l1l2l3}^{m1m2m3} between different (Lambda_i M_i) pairs.
    This is as defined in Slepian/Cahn 2020, with a few extra factors for later use.

    It is stored as a numpy array of dimension [numell^2, numell^2, numell]"""

    # Note that the final index of this matrix only specifies ell_3 since m_3 is known
    weights_4pcf = np.zeros((numell**2,numell**2,numell),dtype=np.float64)

    print("Computing 4PCF coupling matrix: this may take some time.")
    for ell_1 in range(numell):
        for ell_2 in range(numell):
            # Enforce triangle condition on ell_1, ell_2, ell_3
            for ell_3 in range(np.abs(ell_1-ell_2),min(numell,ell_1+ell_2+1)):
                for m_1 in range(-ell_1,ell_1+1):
                    for m_2 in range(-ell_2,ell_2+1):
                        # m_3 is set from triangle condition
                        m_3 = -m_1-m_2
                        # factor missed in spherical harmonic definition
                        tmp_fac = np.sqrt((2.*ell_1+1.)*(2.*ell_2+1.)*(2.*ell_3+1.))/np.sqrt(np.pi**3.)
                        # we only count cases with m1+m2>=0. We add a factor of 2 if m1+m2>0 to account for this.
                        if m_1+m_2>0:
                            tmp_fac *= 2.
                        if m_1+m_2<0:
                            continue # shouldn't use these bins!
                        # add factors appearing in complex conjugates - we only store Y_lm with m<0
                        if m_1>0:
                            tmp_fac *= (-1.)**m_1
                        if m_2>0:
                            tmp_fac *= (-1.)**m_2
                        # compute weights
                        weights_4pcf[ell_1**2+m_1+ell_1, ell_2**2+m_2+ell_2, ell_3] = (-1.)**(ell_1+ell_2+ell_3)*wigner_3j(ell_1,ell_2,ell_3,m_1,m_2,m_3)*tmp_fac

    np.save('coupling_matrices/weights_4pcf_n%d'%numell,weights_4pcf)
    return weights_4pcf

#################### LEGENDRE WEIGHTS CODE ####################

def compute_weight_matrix(xmiydivr,zdivr,galwtr,numell):
    """Compute the matrix of spherical harmonics from input x,y,z,w matrices.
    NB: since this is just array multiplication, switching to Cython won't lead to a significant speed boost.
    Note that the inputs are arrays of (x-iy)/r, z/r and weight."""

    # Compute spherical harmonics on all bins for this galaxy; query_ball finds central itself too but this doesn't contribute to spherical harmonics because x, y, z are zero.

    # Accumulate weights (up to the number we actually use)
    n_mult = numell*(numell+1)//2
    all_weights = np.zeros((n_mult,len(galwtr)),np.complex128) # assign memory

    all_weights[0] = .5*np.ones_like(galwtr)
    if numell>1:
        # ell = 1
        all_weights[1] = .5*(1./2.)**.5*xmiydivr
        all_weights[2] = .5*zdivr
        if numell>2:
            # ell = 2
            zdivrsq=zdivr*zdivr # (z/r)^2
            xmiydivrsq=xmiydivr*xmiydivr # ((x-iy)/r)^2
            all_weights[3] = .25*(3./2.)**.5*xmiydivrsq
            all_weights[4] = .5*(3./2.)**.5*xmiydivr*zdivr
            all_weights[5] = .25*(3.*zdivrsq-1.)
            if numell>3:
                # ell = 3
                zdivrcu=zdivrsq*zdivr # (z/r)^3
                xmiydivrcu=xmiydivrsq*xmiydivr # ((x-iy)/r)^3
                all_weights[6] = .125*(5.)**.5*xmiydivrcu
                all_weights[7] = .25*(15./2.)**.5*xmiydivrsq*zdivr
                all_weights[8] = .125*(3.)**.5*xmiydivr*(5.*zdivrsq-1.)
                all_weights[9] = .25*zdivr*(5.*zdivrsq-3.)
                if numell>4:
                    # ell = 4
                    zdivrft=zdivrsq*zdivrsq # (z/r)^4
                    xmiydivrft=xmiydivrcu*xmiydivr # ((x-iy)/r)^5
                    all_weights[10] = .1875*np.sqrt(35./18.)*xmiydivrft
                    all_weights[11] = .375*np.sqrt(35./9.)*xmiydivrcu*zdivr
                    all_weights[12] = .375*np.sqrt(5./18.)*xmiydivrsq*(7.*zdivrsq-1)
                    all_weights[13] = .375*np.sqrt(5./9.)*xmiydivr*zdivr*(7.*zdivrsq-3.)
                    all_weights[14] = .1875*np.sqrt(1./9.)*(35.*zdivrft-30.*zdivrsq+3.)
                    if numell>5:
                        # ell = 5
                        zdivrfi=zdivrft*zdivr # (z/r)^5
                        xmiydivrfi=xmiydivrft*xmiydivr # ((x-iy)/r)^5
                        all_weights[15] = (3./32.)*np.sqrt(7.)*xmiydivrfi
                        all_weights[16] = (3./16.)*np.sqrt(35./2.)*xmiydivrft*zdivr
                        all_weights[17] = (1./32.)*np.sqrt(35.)*xmiydivrcu*(9.*zdivrsq-1.)
                        all_weights[18] = (1./8.)*np.sqrt(105./2.)*xmiydivrsq*(3.*zdivrcu-zdivr)
                        all_weights[19] = (1./16.)*np.sqrt(15./2.)*xmiydivr*(21.*zdivrft-14.*zdivrsq+1.)
                        all_weights[20] = (1./16.)*(63.*zdivrfi-70.*zdivrcu+15.*zdivr)
                        if numell>6:
                            # ell = 6
                            zdivrsi=zdivrfi*zdivr # (z/r)^6
                            xmiydivrsi=xmiydivrfi*xmiydivr # ((x-iy)/r)^6
                            all_weights[21] = (1./64.)*np.sqrt(231.)*xmiydivrsi
                            all_weights[22] = (3./32.)*np.sqrt(77.)*xmiydivrfi*zdivr
                            all_weights[23] = (3./32.)*np.sqrt(7./2.)*xmiydivrft*(11.*zdivrsq-1.)
                            all_weights[24] = (1./32.)*np.sqrt(105.)*xmiydivrcu*(11.*zdivrcu-3.*zdivr)
                            all_weights[25] = (1./64.)*np.sqrt(105.)*xmiydivrsq*(33.*zdivrft-18.*zdivrsq+1.)
                            all_weights[26] = (1./16.)*np.sqrt(21./2.)*xmiydivr*(33.*zdivrfi-30.*zdivrcu+5.*zdivr)
                            all_weights[27] = (1./32.)*(231.*zdivrsi-315.*zdivrft+105.*zdivrsq-5.)
                            # ell = 7
                            if numell>7:
                                zdivrse=zdivrsi*zdivr # (z/r)^7
                                xmiydivrse=xmiydivrsi*xmiydivr # ((x-iy)/r)^7
                                all_weights[28] = (3./64.)*np.sqrt(143./6.)*xmiydivrse
                                all_weights[29] = (3./64.)*np.sqrt(1001./3.)*xmiydivrsi*zdivr
                                all_weights[30] = (3./64.)*np.sqrt(77./6.)*xmiydivrfi*(13.*zdivrsq-1.)
                                all_weights[31] = (3./32.)*np.sqrt(77./6.)*xmiydivrft*(13.*zdivrcu-3.*zdivr)
                                all_weights[32] = (3./64.)*np.sqrt(7./6.)*xmiydivrcu*(143.*zdivrft-66.*zdivrsq+3.)
                                all_weights[33] = (3./64.)*np.sqrt(7./3.)*xmiydivrsq*(143.*zdivrfi-110.*zdivrcu+15.*zdivr)
                                all_weights[34] = (1./64.)*np.sqrt(7./2.)*xmiydivr*(429.*zdivrsi-495.*zdivrft+135.*zdivrsq-5.)
                                all_weights[35] = (1./32.)*(429.*zdivrse-693.*zdivrfi+315.*zdivrcu-35.*zdivr)
                                if numell>8:
                                    # ell = 8
                                    xmiydivret=xmiydivrse*xmiydivr # ((x-iy)/r)^7
                                    zdivret=zdivrse*zdivr # (z/r)^8
                                    all_weights[36] = (3./256.)*np.sqrt(715./2.)*xmiydivret
                                    all_weights[37] = (3./64.)*np.sqrt(715./2.)*xmiydivrse*zdivr
                                    all_weights[38] = (1./128.)*np.sqrt(429.)*xmiydivrsi*(15.*zdivrsq-1.)
                                    all_weights[39] = (3./64.)*np.sqrt(1001./2.)*xmiydivrfi*(5.*zdivrcu-zdivr)
                                    all_weights[40] = (3./128.)*np.sqrt(77./2.)*xmiydivrft*(65.*zdivrft-26.*zdivrsq+1.)
                                    all_weights[41] = (1./64.)*np.sqrt(1155./2.)*xmiydivrcu*(39.*zdivrfi-26.*zdivrcu+3.*zdivr)
                                    all_weights[42] = (3./128.)*np.sqrt(35.)*xmiydivrsq*(143.*zdivrsi-143.*zdivrft+33.*zdivrsq-1.)
                                    all_weights[43] = (3./64.)*np.sqrt(1./2.)*xmiydivr*(715.*zdivrse-1001.*zdivrfi+385.*zdivrcu-35.*zdivr)
                                    all_weights[44] = (1./256.)*(6435.*zdivret-12012.*zdivrsi+6930.*zdivrft-1260.*zdivrsq+35.)
                                    if numell>9:
                                        # ell = 9
                                        xmiydivrni=xmiydivret*xmiydivr # ((x-iy)/r)^9
                                        zdivrni=zdivret*zdivr # (z/r)^9
                                        all_weights[45] = (1./512.)*np.sqrt(12155.)*xmiydivrni
                                        all_weights[46] = (3./256.)*np.sqrt(12155./2.)*xmiydivret*zdivr
                                        all_weights[47] = (3./512.)*np.sqrt(715.)*xmiydivrse*(17.*zdivrsq-1.)
                                        all_weights[48] = (1./128.)*np.sqrt(2145.)*xmiydivrsi*(17.*zdivrcu-3.*zdivr)
                                        all_weights[49] = (3./256.)*np.sqrt(143.)*xmiydivrfi*(85.*zdivrft-30.*zdivrsq+1.)
                                        all_weights[50] = (3./128.)*np.sqrt(5005./2.)*xmiydivrft*(17.*zdivrfi-10.*zdivrcu+zdivr)
                                        all_weights[51] = (1./256.)*np.sqrt(1155.)*xmiydivrcu*(221.*zdivrsi-195.*zdivrft+39.*zdivrsq-1.)
                                        all_weights[52] = (3./128.)*np.sqrt(55.)*xmiydivrsq*(221.*zdivrse-273.*zdivrfi+91.*zdivrcu-7.*zdivr)
                                        all_weights[53] = (3./256.)*np.sqrt(5./2.)*xmiydivr*(2431.*zdivret-4004.*zdivrsi+2002.*zdivrft-308.*zdivrsq+7.)
                                        all_weights[54] = (1./256.)*(12155.*zdivrni-25740.*zdivrse+18018.*zdivrfi-4620.*zdivrcu+315.*zdivr)
                                        if numell>10:
                                            # ell = 10
                                            xmiydivrtn=xmiydivrni*xmiydivr # ((x-iy)/r)^10
                                            zdivrtn=zdivrni*zdivr # (z/r)^10
                                            all_weights[55] = (1./1024.)*np.sqrt(46189.)*xmiydivrtn
                                            all_weights[56] = (1./512.)*np.sqrt(230945.)*(xmiydivrni*zdivr)
                                            all_weights[57] = (1./512.)*np.sqrt(12155./2.)*(xmiydivret*(19.*zdivrsq-1.))
                                            all_weights[58] = (3./512.)*np.sqrt(12155./3.)*(xmiydivrse*(19.*zdivrcu-3.*zdivr))
                                            all_weights[59] = (3./1024.)*np.sqrt(715./3.)*(xmiydivrsi*(323.*zdivrft-102.*zdivrsq+3.))
                                            all_weights[60] = (3./256.)*np.sqrt(143./3.)*(xmiydivrfi*(323.*zdivrfi-170.*zdivrcu+15.*zdivr))
                                            all_weights[61] = (3./256.)*np.sqrt(715./6.)*(xmiydivrft*(323.*zdivrsi-255.*zdivrft+45.*zdivrsq-1.))
                                            all_weights[62] = (3./256.)*np.sqrt(715./3.)*(xmiydivrcu*(323.*zdivrse-357.*zdivrfi+105.*zdivrcu-7.*zdivr))
                                            all_weights[63] = (3./512.)*np.sqrt(55./6.)*(xmiydivrsq*(4199.*zdivret-6188.*zdivrsi+2730.*zdivrft-364.*zdivrsq+7.))
                                            all_weights[64] = (1./256.)*np.sqrt(55./2.)*(xmiydivr*(4199.*zdivrni-7956.*zdivrse+4914.*zdivrfi-1092.*zdivrcu+63.*zdivr))
                                            all_weights[65] = (1./512.)*(46189.*zdivrtn-109395.*zdivret+90090.*zdivrsi-30030.*zdivrft+3465.*zdivrsq-63.)

    # add in pair weights
    all_weights *= galwtr
    return all_weights
