#Code for NPCFs - Zack Slepian + Oliver Philcox
##########################################load needed modules for python#######################################################################
#doing basic math
import numpy as np
#3j manipulations
from sympy.physics.wigner import wigner_3j
#system operations
import os, sys
#for timing
import time
#kd tree
from scipy import spatial
#custom cython functions used here
import cython_utils

######################## HISTOGRAM CODE (from numpy) ########################

def _search_sorted_inclusive(a, v):
    """
    Like `searchsorted`, but where the last item in `v` is placed on the right.
    In the context of a histogram, this makes the last bin edge inclusive
    """
    return np.concatenate((
        a.searchsorted(v[:-1], 'left'),
        a.searchsorted(v[-1:], 'right')
    ))

def histogram_multi(a, bins=10, weight_matrix=None):
    r"""
    Compute the histogram of a set of data. Taken from numpy and modified to use multiple weihgts at once

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

######################## INPUT PARAMETERS ##############################

#infile = 'sample_feb27_unity_weights_rescale400_first500.dat'
infile = 'patchy_gal_pos.txt'
cut_number = 10000 # maximum number of galaxies to read in
boxsize = 2500. #size of box used in sample data file.
rescale = 1. #if you want to rescale the box (unity if not).

# Binning
rmax=np.sqrt(3.)*100. # *5
rmin=1e-5
nbins=10 # 3
numell = 11

# Other switches (mostly for testing)
no_weights = True # replace weights with unity
include_norm3 = True # if True, include factor of (-1)^ell / sqrt[2ell+1] for 3PCF for consistency with NPCF
use_cython = True # if True, bin using compiled Cython functions, if False, use Pythonic versions
compute_4PCF = True # else just compute 3PCF

### Verbosity / output options
verb=0 # if True, print useful(?) messages throughout
run_tests = 0 # if True, run some tests of the code
print_output = 0 # if True, print the output in the same format as the C++ code

assert numell<=11, "weights only computed up to ell=10!"

######################## LOAD IN DATA ##############################

if not os.path.exists(infile):
    raise Exception("Infile %s does not exist!"%infile)
boxsize *= rescale #this is important so the periodic duplicates are made correctly.
galx, galy, galz, galw = np.loadtxt(infile,unpack=True)[:,:cut_number]
galx, galy, galz = galx*rescale, galy*rescale, galz*rescale
if no_weights:
    galw  = np.ones_like(galw)

width_x = np.max(galx)-np.min(galx)
width_y = np.max(galy)-np.min(galy)
width_z = np.max(galz)-np.min(galz)
assert(max([width_x,width_y,width_z])<boxsize)

## Print some diagnostics
print("\nINPUT")
print("----------")
print("N_gal = %d"%len(galx))
print("Mean weight = %.2e"%np.mean(galw))
print("Particle Size: [%.2e, %.2e, %.2e]"%(width_x,width_y,width_z))
print("Boxsize: %.2e"%(boxsize))
print("Number density: %.2e"%(len(galx)/(width_x*width_y*width_z)))
print("Binning: [%.2e, %.2e] in %d bins"%(rmin,rmax,nbins))
print("L_max = %d"%(numell-1))
if compute_4PCF:
    print("Computing 3PCF + 4PCF")
else:
    print("Computing 3PCF only")
print("----------\n")

start_time=time.time()

eps=1e-8
ngal=len(galx)

# Define binning
deltr = float(rmax-rmin)/nbins
binarr=np.mgrid[0:nbins+1]*deltr
n_mult = numell*(numell+1)//2

# Output array for 3PCF
n3pcf = nbins*(1+nbins)*(1+2*nbins)//6 # number of 3PCF radial bins with r1<=r2
zeta3 = np.zeros((numell,n3pcf),dtype=np.float64)
# if compute_4PCF:
#     n4pcf = nbins*(1+nbins)*(2+nbins)*(1+3*nbins)//24 # number of 4PCF radial bins with r1<=r2<=r3
#     zeta4 = np.zeros((numell,numell,numell,n4pcf),dtype=np.float64)

histtime=0
bintime=0
transftime=0
end_time=time.time()

def shiftbox(galx,galy,galz,shiftbox,doxshift,doyshift,dozshift):
    galxsh=galx+doxshift*boxsize
    galysh=galy+doyshift*boxsize
    galzsh=galz+dozshift*boxsize

    galcoordssh=galxsh.ravel(),galysh.ravel(),galzsh.ravel()

    return galcoordssh

print("\ntime to set constants and bins=",end_time-start_time)

start_time = time.time()

#################### COMPUTE / LOAD WEIGHTING MATRIX ####################

if os.path.exists('weights_3pcf_n%d.npy'%numell):
    print("Loading 3PCF weights from file")
    weights_3pcf = np.asarray(np.load('weights_3pcf_n%d.npy'%numell),dtype=np.float64)
else:
    # compute weighting matrix for 3pcf (inefficiently coded, but not rate limiting)
    weights_3pcf = np.zeros(((numell+1)**2),dtype=np.float64)

    if verb: print("note matrix could be a lot less sparse!")
    for ell_1 in range(numell):
        for m_1 in range(-ell_1,1):
            # nb: don't use m_1>0 here
            ell_2 = ell_1
            m_2 = -m_1
            tmp_fac = (2.*ell_1+1.)/np.pi # factor missed in spherical harmonic definition
            # enforce (ell_i = ell_j) and m_i = m_j,
            # add factor of 2x if m1 != 0 by symmmetry
            if m_1==0:
                tmp_fac *= 1.
            else:
                tmp_fac *= 2.

            # compute weights (note we absorb extra factor of (-1)^m from complex conjugate)
            if not include_norm3:
                weights_3pcf[ell_1**2+m_1+ell_1] = tmp_fac
            else:
                weights_3pcf[ell_1**2+m_1+ell_1] = (-1.)**(ell_1)/np.sqrt(2.*ell_1+1.)*tmp_fac
    np.save('weights_3pcf_n%d'%numell,weights_3pcf)

if compute_4PCF:
    if os.path.exists('weights_4pcf_n%d.npy'%numell):
        print("Loading 4PCF weights from file")
        weights_4pcf = np.asarray(np.load('weights_4pcf_n%d.npy'%numell),dtype=np.float64)
    else:
        # compute weighting matrix for 4pcf (inefficiently coded, but not rate limiting)
        # note final index only specifies ell3 since m3 is known
        weights_4pcf = np.zeros(((numell+1)**2,(numell+1)**2,(numell+1)),dtype=np.float64)

        print("3j function computation takes ages...")
        for ell_1 in range(numell):
            for m_1 in range(-ell_1,ell_1+1):
                for ell_2 in range(numell):
                    for m_2 in range(-ell_2,ell_2+1):
                        for ell_3 in range(np.abs(ell_1-ell_2),min(numell,ell_1+ell_2+1)):
                            m_3 = -m_1-m_2 # from triangle condition
                            tmp_fac = np.sqrt((2.*ell_1+1.)*(2.*ell_2+1.)*(2.*ell_3+1.))/np.sqrt(np.pi**3.) # factor missed in spherical harmonic definition
                            if m_1==0 and m_2==0 and m_3==0:
                                tmp_fac *= 1.
                            else:
                                tmp_fac *= 2.
                            # add factors appearing in complex conjugates
                            if m_1>0:
                                tmp_fac *= (-1.)**m_1
                            if m_2>0:
                                tmp_fac *= (-1.)**m_2
                            if m_3>0:
                                continue # not used in code
                            weights_4pcf[ell_1**2+m_1+ell_1, ell_2**2+m_2+ell_2, ell_3] = (-1.)**(ell_1+ell_2+ell_3)*wigner_3j(ell_1,ell_2,ell_3,m_1,m_2,m_3)*tmp_fac
        np.save('weights_4pcf_n%d'%numell,weights_4pcf)

end_time = time.time()
print("\ntime to define 3PCF + 4PCF weighting matrices=",end_time-start_time)

#################### ASSIGN TO TREE ####################
start_time=time.time()

#now loop over shiftbox to append virtual boxes to coordinate list.
new_galx, new_galy, new_galz, new_galw = galx.ravel(), galy.ravel(), galz.ravel(), galw.ravel()
for doxshift in (-1,1,0):
    for doyshift in (-1,1,0):
        for dozshift in (-1,1,0):
            if verb: print(doxshift,doyshift,dozshift)
            if doxshift==0 and doyshift==0 and dozshift==0:
                if verb: print("no shift executed because in genuine box")
            else:
                shift_gal = shiftbox(galx,galy,galz,boxsize,doxshift,doyshift,dozshift)
                new_galx = np.append(new_galx,shift_gal[0])
                new_galy = np.append(new_galy,shift_gal[1])
                new_galz = np.append(new_galz,shift_gal[2])
                new_galw = np.append(new_galw,galw)

galcoords = np.asarray([new_galx.ravel(),new_galy.ravel(),new_galz.ravel(),new_galw.ravel()]).T
end_time=time.time()

print("\ntime to load",ngal,"galaxies, shift them, and place them in an array= (*)",end_time-start_time)

start_time=time.time()
#put galaxies in a tree
tree=spatial.cKDTree(galcoords,leafsize=3000)
end_time=time.time()
print("\ntime to put",ngal,"galaxies in tree=(*)",end_time-start_time)

#Choose to work on first nperit galaxies.
start_time=time.time()
nperit=ngal//10 #1 for just looking at one galaxy to see more granular timings. ngal/100 gives 20000/10=2000 per iteration.
totalits=ngal//nperit #5#ngal/nperit#5 for testing leaf size if I want to iterate over 1000 galaxies.
count=0
querytime=0.
complextime=0.
realtime=0.

if use_cython:
    cython_utils.initialize(numell,nbins)

    ThreePCF = cython_utils.ThreePCF(numell, nbins, weights_3pcf)
    if compute_4PCF:
        FourPCF = cython_utils.FourPCF(numell, nbins, weights_4pcf)
        FourPCF2 = cython_utils.FourPCF2(numell, nbins, weights_4pcf)

#################### LEGENDRE WEIGHTS CODE ####################

def compute_weight_matrix(galxtr,galytr,galztr,galwtr):
    """Compute the matrix of spherical harmonics from input x,y,z,w matrices.
    NB: since this is just array multiplication, switching to Cython won't significantly"""

    #xmiydivr,xdivr,ydivr,zdivr=(galxtr-1j*galytr)/rgals,galxtr/rgals,galytr/rgals,galztr/rgals #compute (x-iy)/r for galaxies in ball around central
                                                                                                                                         #zdivr=galztr/rgals
                                                                                                                                         #xdivr=galxtr/rgals
                                                                                                                                         #ydivr=galytr/rgals
    xmiydivr,xdivr,ydivr,zdivr=(galxtr-1j*galytr),galxtr,galytr,galztr#broke this to see how not dividing by r affects results. #compute (x-iy)/r for galaxies in ball around central

    #compute spherical harmonics on all bins for this galaxy; query_ball finds central itself too but this doesn't contribute to spherical harmonics because x, y, z are zero.

    if run_tests:
        y00=.5*(1./np.pi)**.5*histogram(rgals,bins=binarr,weights=galwtr)
        complex_test_start=time.time()
        y1m1=.5*(3./(2.*np.pi))**.5*histogram(rgals,bins=binarr,weights=galwtr*xmiydivr) #this just gives histogram y values; [1] would give bin edges with length of [0] + 1.
        complex_test_end=time.time()
        complextime=complex_test_end-complex_test_start+complextime
        real_test_start=time.time()
        y1m1test=.5*(3./(2.*np.pi))**.5*(histogram(rgals,bins=binarr,weights=galwtr*xdivr)-1j*histogram(rgals,bins=binarr,weights=galwtr*ydivr))
        real_test_end=time.time()
        realtime=real_test_end-real_test_start+realtime

    # Accumulate weights (up to the number we actually use)
    all_weights = np.zeros((n_mult,len(galwtr)),np.complex128) # assign memory

    all_weights[0] = .5*np.ones_like(rgals)
    if numell>1:
        # ell = 1
        all_weights[1] = .5*(1./2.)**.5*xmiydivr
        all_weights[2] = .5*zdivr
        if numell>2:
            # ell = 2
            xdivrsq=xdivr*xdivr # (x/r)^2
            ydivrsq=ydivr*ydivr # (y/r)^2
            zdivrsq=zdivr*zdivr # (z/r)^2
            xmiydivrsq=xmiydivr*xmiydivr # ((x-iy)/r)^2
            all_weights[3] = .25*(3./2.)**.5*xmiydivrsq
            all_weights[4] = .5*(3./2.)**.5*xmiydivr*zdivr
            all_weights[5] = .25*(2.*zdivrsq-xdivrsq-ydivrsq)
            if numell>3:
                # ell = 3
                zdivrcu=zdivrsq*zdivr # (z/r)^3
                xmiydivrcu=xmiydivrsq*xmiydivr # ((x-iy)/r)^3
                all_weights[6] = .125*(5.)**.5*xmiydivrcu
                all_weights[7] = .25*(15./2.)**.5*xmiydivrsq*zdivr
                all_weights[8] = .125*(3.)**.5*xmiydivr*(4.*zdivrsq-xdivrsq-ydivrsq)
                all_weights[9] = .25*zdivr*(2.*zdivrsq-3.*xdivrsq-3.*ydivrsq)
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
                                        all_weights[54] = (1./256.)*12155.*(zdivrni-25740.*zdivrse+18018.*zdivrfi-4620.*zdivrcu+315.*zdivr)
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

#################### MAIN LOOP ####################

first_it = 1 # if this is the first iteration
for i in range(0,totalits): #do nperit galaxies at a time for totalits total iterations
    print("group number = %d of %d"%(i+1,totalits))
    centralgals=galcoords[i*nperit:(i+1)*nperit] #select out central galaxies to use; note central galaxies must be first in list of coordinates including shift, which they will be by construction.
    print("len centralgals=", len(centralgals))
    print("using galaxies from", i*nperit, "to", (i+1)*nperit-1)
    print("i*nperit=",i*nperit)
    start_time_query=time.time()
    ball=tree.query_ball_point(centralgals,rmax+eps) #object with, for each index, an array with the ball of rmax about the central galaxy of that index; e.g. ball[0] gives indices of gals. within rmax of the 0th centralgal
    end_time_query=time.time()
    querytime=end_time_query-start_time_query+querytime

    for w in range(0,nperit):
        start_time_transf=time.time()
        ball[w].remove(i*nperit+w)
        #transform to reference frame of desired central galaxy
        ball_temp=ball[w]

        if len(ball_temp)==0:
            continue

        galxtr, galytr, galztr=np.asarray(galcoords[ball_temp][:,:3]-centralgals[w][:3],dtype=np.complex128).T
        galwtr = galcoords[ball_temp][:,3]

        rgalssq=galxtr*galxtr+galytr*galytr+galztr*galztr+eps
        rgals=np.sqrt(rgalssq)

        # COMPUTE WEIGHTS
        weighttime = time.time()
        all_weights = compute_weight_matrix(galxtr,galytr,galztr,galwtr)
        weighttime = time.time()-weighttime

        if verb: print("weights computation time: %.3e (Python)"%weighttime)

        end_time_transf=time.time()
        transftime=end_time_transf-start_time_transf+transftime
        start_time_hist=time.time()

        # histogram all these weights at once
        # this computes the Y_lm coefficients in a single list
        # the ordering is Y_00, Y_{1-1} Y_{10} Y_{2-2} etc.
        # dimension is (N_mult,N_bin) where N_mult is number of multipoles = (numell)*(numell+1)/2
        hist1 = time.time()
        y_all = histogram_multi(rgals, bins=binarr, weight_matrix=all_weights)
        if verb: print("Histogramming function time: %.3e"%(time.time()-hist1))

        end_time_hist=time.time()
        histtime=end_time_hist-start_time_hist+histtime
        count=count+1

        start_time_binning=time.time()

        # compute bin centers on the first iteration only
        if first_it:
            if verb: print('do we actually use these?')
            bin_val = np.ravel(np.outer(np.arange(nbins)+0.5,np.arange(nbins)+0.5))
            first_it = 0

        # Now perform summations into bins
        # Note that the code is in cython and defined in cython_utils.pyx
        # this gives a ~ 100x speed-boost for the 4PCF

        y_all_conj = y_all.conjugate() # compute complex conjugates only once

        if use_cython:
            ## NB: most things are in pure C in these functions, and arrays are converted to C memoryviews on input

            ### NPCF Classes
            t3pt = time.time()
            ThreePCF.add_to_sum(y_all, y_all_conj)
            t3pt = time.time()-t3pt

            if compute_4PCF:
                t4pt = time.time()
                FourPCF.add_to_sum(y_all, y_all_conj)
                t4pt = time.time()-t4pt

            # #### NPCF Functions
            # t3pt2 = time.time()
            # cython_utils.threepcf_sum(y_all, y_all_conj, zeta3, weights_3pcf)
            # t3pt2 = time.time()-t3pt2
            #
            # t4pt2 = time.time()
            # cython_utils.fourpcf_sum(y_all, y_all_conj, zeta4, weights_4pcf)
            # t4pt2 = time.time()-t4pt2

            if verb: print("Summation took %.2e s (3PCF) / %.2e s (4PCF) in class"%(t3pt, t4pt))
            #if verb: print("Summation took %.2e s (3PCF) / %.2e s (4PCF) in function"%(t3pt2, t4pt2))

        ### Alternatively use python:
        ### OLD + PROBABLY DOESNT WORK
        else:

            # Now perform summations (either python or cython)
            def threepcf_sum(y_all, zeta3):
                """Sum up multipole array into 3PCFs"""
                # sum up all ell (simply with weights)
                for l1 in range(numell):
                    for m1 in range(-l1,1):
                        a_l1m1 = y_all[l1*(l1+1)//2+l1+m1]
                        a_l2m2 = a_l1m1.conjugate()*(-1.)**m1
                        print("this is outdated with radial bins etc.")
                        zeta3[l1] += np.real(np.outer(a_l1m1,a_l2m2)*weights_3pcf[l1**2+m1+l1])*((m1!=0)+1) # can take real part since answer will be real!

            ## 3PCF Summation (using Python)
            t3pt = time.time()
            threepcf_sum(y_all, zeta3)
            t3pt = time.time()-t3pt

            ## 4PCF Summation
            def fourpcf_sum(y_all, zeta4):
                """Sum up multipole array into 4PCFs"""
                # sum up all ell (simply with weights)
                for l1 in range(numell):

                    for l2 in range(numell):

                        for l3 in range(np.abs(l1-l2),min(l1+l2+1,numell)):

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
                                    if np.abs(m3)>l3: continue

                                    this_weight = weights_4pcf[l1**2+m1+l1,l2**2+m2+l2,l3]
                                    if this_weight==0: continue

                                    # load a_l3m3
                                    if m3<0:
                                        a_l3m3 = y_all[l3*(l3+1)//2+l3+m3]
                                    else:
                                        a_l3m3 = y_all[l3*(l3+1)//2+l3-m3].conjugate()*(-1.)**m3

                                    # NB: the contribution from (-m1, -m2) is just the conjugate of that from (m1, m2)
                                    # this can probably reduce the number of summations by ~ 2x
                                    # todo: implement this
                                    print("this is outdated with radial bins etc.")
                                    zeta4[l1,l2,l3] += np.real(np.einsum('i,j,k->ijk',a_l1m1,a_l2m2,a_l3m3))*this_weight


            ## 3PCF Summation (using cython)
            t3pt1 = time.time()
            threepcf_sum(y_all, zeta3)
            t3pt1 = time.time()-t3pt1

            t4pt = time.time()
            fourpcf_sum(y_all, zeta4)
            t4pt = time.time()-t4pt

            #print("Summation took %.2e s (3PCF) / %.2e s (4PCF)"%(t3pt, t4pt))

        end_time_binning=time.time()
        bintime=end_time_binning-start_time_binning+bintime

#################### CREATE OUTPUTS ####################
zeta3 = ThreePCF.zeta3
if compute_4PCF:
    zeta4 = FourPCF.zeta4

print("NB: ordering has all radial bins in (a,b,c) ordering as a 1D array")

#2.*np.pi this latter factor was to agree with Eisenstein C++ code, which we now know also does not have right normalization factor, but that cancels out in edge correction.

if not include_norm3:
    zeta3 = [zz*1./(4.*np.pi) for zz in zeta3]

print("number of galaxies done=", count)

if print_output:
    for i in range(5):
        print("zeta%d="%i,zeta3[i])

postfix = 'DESI_Tutorial_results' #Use this to title arrays you save.
for i in range(numell):
    np.save('zeta%dtest_%s'%(i,postfix),zeta3[i])
print("results saved to %d numpy array files"%numell)

print("--------------------")
print("--------------------")
print("--------------------")

if print_output:
    #print to match Eisenstein C++ code output format
    for bin1 in np.arange(0,nbins):
        for bin2 in np.arange(0,nbins//2+1)[::-1]:
            if bin2<=bin1:
                print("#B1 B2 l=0 l=1            l=2          l=3                    l=4")
                print(bin1, bin2, np.real(zeta3[0][bin1,bin2]),np.real(zeta3[1][bin1,bin2]),np.real(zeta3[2][bin1,bin2]),np.real(zeta3[3][bin1,bin2]),np.real(zeta3[4][bin1,bin2]),np.real(zeta3[5][bin1,bin2]),np.real(zeta3[6][bin1,bin2]),np.real(zeta3[7][bin1,bin2]),np.real(zeta3[8][bin1,bin2]),np.real(zeta3[9][bin1,bin2]))
                #print bin1, bin2, np.real(zeta0[bin1,bin2])*2,np.real(zeta1[bin1,bin2])/np.real(zeta0[bin1,bin2]),np.real(zeta2[bin1,bin2])/ np.real(zeta0[bin1,bin2]),np.real(zeta3[bin1,bin2])/np.real(zeta0[bin1,bin2]),np.real(zeta4[bin1,bin2])/np.real(zeta0[bin1,bin2]),np.real(zeta5[bin1,bin2])/np.real(zeta0[bin1,bin2]),np.real(zeta6[bin1,bin2])/np.real(zeta0[bin1,bin2]),np.real(zeta7[bin1,bin2])/np.real(zeta0[bin1,bin2]),np.real(zeta8[bin1,bin2])/np.real(zeta0[bin1,bin2]),np.real(zeta9[bin1,bin2])/np.real(zeta0[bin1,bin2]),np.real(zeta10[bin1,bin2])/np.real(zeta0[bin1,bin2])

end_time=time.time()
timecost=end_time-start_time
ballfrac=querytime/timecost
print("ngal=", ngal, "time for computation=",timecost)
print("fraction of time for query_ball=",ballfrac)
print("histtime=", histtime)
print("bintime=",bintime)
print("transformation time", transftime)
print("ball query time=", querytime)
print("transftime+bintime+histtime+balltime=",transftime+histtime+bintime+querytime)
print("complextesttime=",complextime)
print("realtesttime=",realtime)

print("\nTOTAL TIME: %.4f seconds for ngal = %d"%(timecost,ngal))

exit()

#----
#Load back in results, if you like.
postfix = 'DESI_Tutorial_results.npy'

zs = []
for i in range(numell):
    zs.append(np.load('zeta%dtest_%s'%(i,postfix)))

exit()

###########direct counting to compare
docheck=0.
if docheck:
    start_time=time.time()
    zetacts = [np.zeros((nbins,nbins))+0j for _ in range(numell)]

    count2=0
    for m in range(0,ngal):
        print("m=", m)
        centralgal=galcoords[m]
        ballct=tree.query_ball_point(centralgal,rmax+eps)
        #print "ballct=",ballct
        lenballct=len(ballct)
        #print "ballct.remove(m)"
        ballct.remove(m) # this step is absolutely essential or counting method will be incorrect! this removes the self term.
        galxtrct=galcoords[ballct][:,0]-centralgal[0]
        galytrct=galcoords[ballct][:,1]-centralgal[1]
        galztrct=galcoords[ballct][:,2]-centralgal[2]
        print("ballct=", ballct)
        print("galcoords", galcoords)
        #exit()

        #galxtrct=galx[ballct]-centralgal[0]
        #galytrct=galy[ballct]-centralgal[1]
        #galztrct=galz[ballct]-centralgal[2]
        galrtrct=(galxtrct**2+galytrct**2+galztrct**2+eps)**.5
        print("galrtrct=", galrtrct)
        for k in range(lenballct-1):
            #print "m, k", m, ballct[k]
            for h in range(lenballct-1):
                #calculate relative angle using a dot product: note that central galaxy gets included but contributes nothing.
                costheta=(galxtrct[k]*galxtrct[h]+galytrct[k]*galytrct[h]+galztrct[k]*galztrct[h])/(galrtrct[k]*galrtrct[h])
                #print "costheta=",costheta
                p0=1.
                p1=costheta
                p2=.5*(3.*costheta**2-1.)
                p3=.5*(5.*costheta**3-3.*costheta)
                p4=.125*(35.*costheta**4-30.*costheta**2+3.)
                p5=.125*(63.*costheta**5-70.*costheta**3+15.*costheta)
                p6=.0625*(231.*costheta**6-315.*costheta**4+105.*costheta**2-5.)
                p7=.0625*(-35.*costheta + 315.*costheta**3 - 693.*costheta**5 + 429.*costheta**7)
                p8=(1./128.)*(35.-1260.*costheta**2+6930.*costheta**4-12012.*costheta**6+6435.*costheta**8)
                p9=(1./128.)*(315.*costheta-4620.*costheta**3+18018.*costheta**5-25740.*costheta**7+12155.*costheta**9)
                p10=(1./256.)*(-63.+3465*costheta**2-30030.*costheta**4+90090.*costheta**6-109395.*costheta**8+46189.*costheta**10)
                ps = [p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10]
                #print "p1,p2,p3=",p1,p2,p3
                bin1=int(galrtrct[k]/deltr)
                bin2=int(galrtrct[h]/deltr)
                #print "bin1, bin2=", bin1, bin2
                for i in range(numell):
                    zetacts[i][bin1,bin2] += ps[i]
                #print "k,h",k,h, "zeta2ct=",zeta2ct




        count2=count2+1

    #if m==2:
    #   print "galcoords ct=", galcoords[m]
    #   zeta1ctsave=zeta1ct/2.
    #   print zeta1ctsave

#p2 is not zero for costheta=0, so it is actually counting the self term, i.e. when the satellite galaxy is the central, you are done for! note this was not an issue in older (noncyclic) version because in that version, we weren't using kdtrees and hence we did not pick up the self term when we were finding neighbors.


#what is correct factor here? may 27 2015.
    zetacts = [zetacts[i]*(2.*i+1.)/2./(8.*np.pi**2) for i in range(numell)]
    end_time=time.time()
    timecostct=end_time-start_time
    print("timecostct=", timecostct)

#check symmetry
#print "zeta1 check", sum(abs(zeta1-np.transpose(zeta1)))
#print "zeta2 check", sum(abs(zeta2-np.transpose(zeta2)))
#print "zeta3 check", sum(abs(zeta3-np.transpose(zeta3)))

#print "zeta1ct check", sum(abs(zeta1ct-np.transpose(zeta1ct)))
#print "zeta2ct check", sum(abs(zeta2ct-np.transpose(zeta2ct)))
#print "zeta3ct check", sum(abs(zeta3ct-np.transpose(zeta3ct)))

    for i in range(numell):
        print("zeta%d comparison check"%i, zeta3[i]-zetacts[i])

#can I use same structure idea to add spherical harmonics one at a time to relevant bin? i.e. avoid using histogram---but on the other hand, using histogram, if it works how stephen says, would not be a problem.
