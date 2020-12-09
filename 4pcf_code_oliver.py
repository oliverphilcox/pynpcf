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

################## Histogram code from numpy ##########

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

#######################################################setup definitions#######################################################################

### PARAMETERS
#infile = 'sample_feb27_unity_weights_rescale400_first500.dat'
infile = 'patchy_gal_pos.txt'
cut_number = 1000 # maximum number of galaxies to read in
boxsize = 2500. #size of box used in sample data file.
rescale = 1. #if you want to rescale the box (unity if not).
no_weights = True # replace weights with unity
include_norm3 = True # if True, include factor of (-1)^ell / sqrt[2ell+1] for 3PCF for consistency with NPCF
use_cython = True # if True, bin using compiled Cython functions, if False, use Pythonic versions

# Binning
rmax=np.sqrt(3.)*100. # *5
rmin=1e-5
nbins=10 # 3
numell = 11

### LOAD IN DATA
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
print("read %d galaxies from file with average weight %.2f"%(len(galx),np.mean(galw)))
print("galaxy widths: [%.2e, %.2e, %.2e] vs boxsize %.2e"%(width_x,width_y,width_z,boxsize))
assert(max([width_x,width_y,width_z])<boxsize)
print("number in file=",len(galx))

#exit()

### Verbosity / output options
verb=0 # if True, print useful(?) messages throughout
run_tests = 0 # if True, run some tests of the code
print_output = 0 # if True, print the output in the same format as the C++ code

start_time=time.time()
linspace=0.

eightpi_sq=8.*np.pi**2
eps=1e-8
ngal=len(galx)

#### SPECIFY BINNING
deltr = float(rmax-rmin)/nbins
binarr=np.mgrid[0:nbins+1]*deltr
n_mult = numell*(numell+1)//2

# Output array for 3PCF
zeta3 = np.zeros((numell,nbins,nbins),dtype=np.float64)
zeta4 = np.zeros((numell,numell,numell,nbins,nbins,nbins),dtype=np.float64)

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

### WEIGHTING MATRIX #############

if os.path.exists('weights_3pcf_n%d.npy'%numell):
    print("Loading 3PCF weights from file")
    weights_3pcf = np.asarray(np.load('weights_3pcf_n%d.npy'%numell),dtype=np.float64)
else:
    # compute weighting matrix for 3pcf (inefficiently coded, but not rate limiting)
    weights_3pcf = np.zeros(((numell+1)**2,(numell+1)**2),dtype=np.float64)

    if verb: print("note matrix could be a lot less sparse!")
    for ell_1 in range(numell):
        for m_1 in range(-ell_1,ell_1+1):
            for ell_2 in range(numell):
                for m_2 in range(-ell_2,ell_2+1):
                    # enforce (ell_i = ell_j) and m_i = m_j,
                    if not include_norm3:
                        weights_3pcf[ell_1**2+m_1+ell_1, ell_2**2+m_2+ell_2] = (ell_1==ell_2)*(m_1+m_2==0)*(-1.)**(-m_1)
                    else:
                        weights_3pcf[ell_1**2+m_1+ell_1, ell_2**2+m_2+ell_2] = (ell_1==ell_2)*(m_1+m_2==0)*(-1.)**(ell_1-m_1)/np.sqrt(2.*ell_1+1.)
    np.save('weights_3pcf_n%d'%numell,weights_3pcf)

if os.path.exists('weights_4pcf_n%d.npy'%numell):
    print("Loading 4PCF weights from file")
    weights_4pcf = np.asarray(np.load('weights_4pcf_n%d.npy'%numell),dtype=np.float64)
else:
    # compute weighting matrix for 4pcf (inefficiently coded, but not rate limiting)
    weights_4pcf = np.zeros(((numell+1)**2,(numell+1)**2,(numell+1)**2),dtype=np.float64)

    print("3j function computation takes ages...")
    for ell_1 in range(numell):
        for m_1 in range(-ell_1,ell_1+1):
            for ell_2 in range(numell):
                for m_2 in range(-ell_2,ell_2+1):
                    for ell_3 in range(np.abs(ell_1-ell_2),min(numell,ell_1+ell_2+1)):
                        m_3 = -m_1-m_2 # from triangle condition
                        weights_4pcf[ell_1**2+m_1+ell_1, ell_2**2+m_2+ell_2, ell_3**2+m_3+ell_3] = (-1.)**(ell_1+ell_2+ell_3)*wigner_3j(ell_1,ell_2,ell_3,m_1,m_2,m_3)
    np.save('weights_4pcf_n%d'%numell,weights_4pcf)

end_time = time.time()
print("\ntime to define 3PCF + 4PCF weighting matrices=",end_time-start_time)

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
print("\ntime to put",ngal,"random galaxies in tree=(*)",end_time-start_time)

#Choose to work on first nperit galaxies.
start_time=time.time()
nperit=ngal//1 #1 for just looking at one galaxy to see more granular timings. ngal/100 gives 20000/10=2000 per iteration.
totalits=ngal//nperit #5#ngal/nperit#5 for testing leaf size if I want to iterate over 1000 galaxies.
count=0
querytime=0.
complextime=0.
realtime=0.

if use_cython:
    cython_utils.initialize(numell,nbins)

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

        galxtr, galytr, galztr=(galcoords[ball_temp][:,:3]-centralgals[w][:3]).T
        galwtr = galcoords[ball_temp][:,3]
        rgalssq=galxtr*galxtr+galytr*galytr+galztr*galztr+eps
        rgals=np.sqrt(rgalssq)
        #xmiydivr,xdivr,ydivr,zdivr=(galxtr-1j*galytr)/rgals,galxtr/rgals,galytr/rgals,galztr/rgals #compute (x-iy)/r for galaxies in ball around central
                                                                                                                                             #zdivr=galztr/rgals
                                                                                                                                             #xdivr=galxtr/rgals
                                                                                                                                             #ydivr=galytr/rgals
        xmiydivr,xdivr,ydivr,zdivr=(galxtr-1j*galytr),galxtr,galytr,galztr#broke this to see how not dividing by r affects results. #compute (x-iy)/r for galaxies in ball around central


        #compute squares: need to replace inline computations using these below.

        xdivrsq=xdivr*xdivr # (x/r)^2
        ydivrsq=ydivr*ydivr # (y/r)^2
        zdivrsq=zdivr*zdivr # (z/r)^2
        xmiydivrsq=xmiydivr*xmiydivr # ((x-iy)/r)^2
        xmiydivrcu=xmiydivrsq*xmiydivr # ((x-iy)/r)^3
        xmiydivrft=xmiydivrcu*xmiydivr # ((x-iy)/r)^4
        xmiydivrfi=xmiydivrft*xmiydivr # ((x-iy)/r)^5
        xmiydivrsi=xmiydivrfi*xmiydivr # ((x-iy)/r)^6
        xmiydivrse=xmiydivrsi*xmiydivr # ((x-iy)/r)^7
        xmiydivret=xmiydivrse*xmiydivr # ((x-iy)/r)^8
        xmiydivrni=xmiydivret*xmiydivr # ((x-iy)/r)^9
        xmiydivrtn=xmiydivrni*xmiydivr # ((x-iy)/r)^10

        zdivrcu=zdivrsq*zdivr # (z/r)^3
        zdivrft=zdivrsq*zdivrsq # (z/r)^4
        zdivrfi=zdivrft*zdivr # (z/r)^5
        zdivrsi=zdivrfi*zdivr # (z/r)^6
        zdivrse=zdivrsi*zdivr # (z/r)^7
        zdivret=zdivrse*zdivr # (z/r)^8
        zdivrni=zdivret*zdivr # (z/r)^9
        zdivrtn=zdivrni*zdivr # (z/r)^10
        #compute spherical harmonics on all bins for this galaxy; query_ball finds central itself too but this doesn't contribute to spherical harmonics because x, y, z are zero.
        #newly added sept 2.
        end_time_transf=time.time()
        transftime=end_time_transf-start_time_transf+transftime
        start_time_hist=time.time()

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

        if verb: print("isn't there a way to automate these?")

        # accumulate all weighted spherical harmonics, including particle weights
        # these are just the values of a_lm(r) in each bin
        all_weights = galwtr*np.vstack([# ell = 0
                                 .5*(1./np.pi)**.5*np.ones_like(rgals),
                                 # ell = 1
                                 .5*(3./(2.*np.pi))**.5*xmiydivr,
                                 .5*(3./np.pi)**.5*zdivr,
                                 # ell = 2
                                 .25*(15./(2.*np.pi))**.5*xmiydivrsq,
                                 .5*(15./(2.*np.pi))**.5*xmiydivr*zdivr,
                                 .25*(5./np.pi)**.5*(2.*zdivrsq-xdivrsq-ydivrsq),
                                 # ell = 3
                                 .125*(35./np.pi)**.5*xmiydivrcu,
                                 .25*(105./(2.*np.pi))**.5*xmiydivrsq*zdivr,
                                 .125*(21./np.pi)**.5*xmiydivr*(4.*zdivrsq-xdivrsq-ydivrsq),
                                 .25*(7./np.pi)**.5*zdivr*(2.*zdivrsq-3.*xdivrsq-3.*ydivrsq),
                                 # ell = 4
                                 .1875*np.sqrt(35./(2.*np.pi))*xmiydivrft,
                                 .375*np.sqrt(35./np.pi)*xmiydivrcu*zdivr,
                                 .375*np.sqrt(5./(2.*np.pi))*xmiydivrsq*(7.*zdivrsq-1),
                                 .375*np.sqrt(5./np.pi)*xmiydivr*zdivr*(7.*zdivrsq-3.),
                                 .1875*np.sqrt(1./np.pi)*(35.*zdivrft-30.*zdivrsq+3.),
                                 # ell = 5
                                 (3./32.)*np.sqrt(77./np.pi)*xmiydivrfi,
                                 (3./16.)*np.sqrt(385./(2.*np.pi))*xmiydivrft*zdivr,
                                 (1./32.)*np.sqrt(385./np.pi)*xmiydivrcu*(9.*zdivrsq-1.),
                                 (1./8.)*np.sqrt(1155./(2.*np.pi))*xmiydivrsq*(3.*zdivrcu-zdivr),
                                 (1./16.)*np.sqrt(165./(2.*np.pi))*xmiydivr*(21.*zdivrft-14.*zdivrsq+1.),
                                 (1./16.)*np.sqrt(11./np.pi)*(63.*zdivrfi-70.*zdivrcu+15.*zdivr),
                                 # ell = 6
                                 (1./64.)*np.sqrt(3003./np.pi)*xmiydivrsi,
                                 (3./32.)*np.sqrt(1001./np.pi)*xmiydivrfi*zdivr,
                                 (3./32.)*np.sqrt(91./(2.*np.pi))*xmiydivrft*(11.*zdivrsq-1.),
                                 (1./32.)*np.sqrt(1365./np.pi)*xmiydivrcu*(11.*zdivrcu-3.*zdivr),
                                 (1./64.)*np.sqrt(1365./np.pi)*xmiydivrsq*(33.*zdivrft-18.*zdivrsq+1.),
                                 (1./16.)*np.sqrt(273./(2.*np.pi))*xmiydivr*(33.*zdivrfi-30.*zdivrcu+5.*zdivr),
                                 (1./32.)*np.sqrt(13./np.pi)*(231.*zdivrsi-315.*zdivrft+105.*zdivrsq-5.),
                                 # ell = 7
                                 (3./64.)*np.sqrt(715./(2.*np.pi))*xmiydivrse,
                                 (3./64.)*np.sqrt(5005./np.pi)*xmiydivrsi*zdivr,
                                 (3./64.)*np.sqrt(385./(2.*np.pi))*xmiydivrfi*(13.*zdivrsq-1.),
                                 (3./32.)*np.sqrt(385./(2.*np.pi))*xmiydivrft*(13.*zdivrcu-3.*zdivr),
                                 (3./64.)*np.sqrt(35./(2.*np.pi))*xmiydivrcu*(143.*zdivrft-66.*zdivrsq+3.),
                                 (3./64.)*np.sqrt(35./np.pi)*xmiydivrsq*(143.*zdivrfi-110.*zdivrcu+15.*zdivr),
                                 (1./64.)*np.sqrt(105./(2.*np.pi))*xmiydivr*(429.*zdivrsi-495.*zdivrft+135.*zdivrsq-5.),
                                 (1./32.)*np.sqrt(15./np.pi)*(429.*zdivrse-693.*zdivrfi+315.*zdivrcu-35.*zdivr),
                                 # ell = 8
                                 (3./256.)*np.sqrt(12155./(2.*np.pi))*xmiydivret,
                                 (3./64.)*np.sqrt(12155./(2.*np.pi))*xmiydivrse*zdivr,
                                 (1./128.)*np.sqrt(7293./np.pi)*xmiydivrsi*(15.*zdivrsq-1.),
                                 (3./64.)*np.sqrt(17017./(2.*np.pi))*xmiydivrfi*(5.*zdivrcu-zdivr),
                                 (3./128.)*np.sqrt(1309./(2.*np.pi))*xmiydivrft*(65.*zdivrft-26.*zdivrsq+1.),
                                 (1./64.)*np.sqrt(19635./(2.*np.pi))*xmiydivrcu*(39.*zdivrfi-26.*zdivrcu+3.*zdivr),
                                 (3./128.)*np.sqrt(595./np.pi)*xmiydivrsq*(143.*zdivrsi-143.*zdivrft+33.*zdivrsq-1.),
                                 (3./64.)*np.sqrt(17./(2.*np.pi))*xmiydivr*(715.*zdivrse-1001.*zdivrfi+385.*zdivrcu-35.*zdivr),
                                 (1./256.)*np.sqrt(17./np.pi)*(6435.*zdivret-12012.*zdivrsi+6930.*zdivrft-1260.*zdivrsq+35.),
                                 # ell = 9
                                 (1./512.)*np.sqrt(230945./np.pi)*xmiydivrni,
                                 (3./256.)*np.sqrt(230945./(2.*np.pi))*xmiydivret*zdivr,
                                 (3./512.)*np.sqrt(13585./np.pi)*xmiydivrse*(17.*zdivrsq-1.),
                                 (1./128.)*np.sqrt(40755./np.pi)*xmiydivrsi*(17.*zdivrcu-3.*zdivr),
                                 (3./256.)*np.sqrt(2717./np.pi)*xmiydivrfi*(85.*zdivrft-30.*zdivrsq+1.),
                                 (3./128.)*np.sqrt(95095./(2.*np.pi))*xmiydivrft*(17.*zdivrfi-10.*zdivrcu+zdivr),
                                 (1./256.)*np.sqrt(21945./np.pi)*xmiydivrcu*(221.*zdivrsi-195.*zdivrft+39.*zdivrsq-1.),
                                 (3./128.)*np.sqrt(1045./np.pi)*xmiydivrsq*(221.*zdivrse-273.*zdivrfi+91.*zdivrcu-7.*zdivr),
                                 (3./256.)*np.sqrt(95./(2.*np.pi))*xmiydivr*(2431.*zdivret-4004.*zdivrsi+2002.*zdivrft-308.*zdivrsq+7.),
                                 (1./256.)*np.sqrt(19./np.pi)*12155.*(zdivrni-25740.*zdivrse+18018.*zdivrfi-4620.*zdivrcu+315.*zdivr),
                                 # ell = 10
                                 (1./1024.)*np.sqrt(969969./np.pi)*xmiydivrtn,
                                 (1./512.)*np.sqrt(4849845./np.pi)*(xmiydivrni*zdivr),
                                 (1./512.)*np.sqrt(255255./(2.*np.pi))*(xmiydivret*(19.*zdivrsq-1.)),
                                 (3./512.)*np.sqrt(85085./np.pi)*(xmiydivrse*(19.*zdivrcu-3.*zdivr)),
                                 (3./1024.)*np.sqrt(5005./np.pi)*(xmiydivrsi*(323.*zdivrft-102.*zdivrsq+3.)),
                                 (3./256.)*np.sqrt(1001./np.pi)*(xmiydivrfi*(323.*zdivrfi-170.*zdivrcu+15.*zdivr)),
                                 (3./256.)*np.sqrt(5005./(2.*np.pi))*(xmiydivrft*(323.*zdivrsi-255.*zdivrft+45.*zdivrsq-1.)),
                                 (3./256.)*np.sqrt(5005./np.pi)*(xmiydivrcu*(323.*zdivrse-357.*zdivrfi+105.*zdivrcu-7.*zdivr)),
                                 (3./512.)*np.sqrt(385./(2.*np.pi))*(xmiydivrsq*(4199.*zdivret-6188.*zdivrsi+2730.*zdivrft-364.*zdivrsq+7.)),
                                 (1./256.)*np.sqrt(1155./(2.*np.pi))*(xmiydivr*(4199.*zdivrni-7956.*zdivrse+4914.*zdivrfi-1092.*zdivrcu+63.*zdivr)),
                                 (1./512.)*np.sqrt(21./np.pi)*(46189.*zdivrtn-109395.*zdivret+90090.*zdivrsi-30030.*zdivrft+3465.*zdivrsq-63.),
                                 ])
        all_weights = np.asarray(all_weights,dtype=np.complex128)

        # histogram all these weights at once
        # this computes the Y_lm coefficients in a single list
        # the ordering is Y_00, Y_{1-1} Y_{10} Y_{2-2} etc.
        # dimension is (N_mult,N_bin) where N_mult is number of multipoles = (numell)*(numell+1)/2
        y_all = histogram_multi(rgals, bins=binarr, weight_matrix=all_weights)

        end_time_hist=time.time()
        histtime=end_time_hist-start_time_hist+histtime
        count=count+1

        start_time_binning=time.time()

        # compute bin centers on the first iteration only
        if first_it:
            print('do we actually use these?')
            bin_val = np.ravel(np.outer(np.arange(nbins)+0.5,np.arange(nbins)+0.5))
            first_it = 0

        # Now perform summations into bins
        # Note that the code is in cython and defined in cython_utils.pyx
        # this gives a ~ 5x speed-boost for the 4PCF

        if use_cython:
            t3pt = time.time()
            cython_utils.threepcf_sum(y_all, zeta3, weights_3pcf)
            t3pt = time.time()-t3pt

            t4pt = time.time()
            cython_utils.fourpcf_sum(y_all, zeta4, weights_4pcf)
            t4pt = time.time()-t4pt

            print("Summation took %.2e s (3PCF) / %.2e s (4PCF)"%(t3pt, t4pt))

        ### Alternatively use python:
        else:

            # Now perform summations (either python or cython)
            def threepcf_sum(y_all, zeta3):
                """Sum up multipole array into 3PCFs"""
                # sum up all ell (simply with weights)
                for l1 in range(numell):
                    for m1 in range(-l1,1):
                        a_l1m1 = y_all[l1*(l1+1)//2+l1+m1]
                        a_l2m2 = a_l1m1.conjugate()*(-1.)**m1
                        zeta3[l1] += np.real(np.outer(a_l1m1,a_l2m2)*weights_3pcf[l1**2+m1+l1,l1**2-m1+l1])*((m1!=0)+1) # can take real part since answer will be real!

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
                                    zeta4[l1,l2,l3] += np.real(np.einsum('i,j,k->ijk',a_l1m1,a_l2m2,a_l3m3))*this_weight


            ## 3PCF Summation (using cython)
            t3pt1 = time.time()
            threepcf_sum(y_all, zeta3)
            t3pt1 = time.time()-t3pt1

            t4pt = time.time()
            fourpcf_sum(y_all, zeta4)
            t4pt = time.time()-t4pt

            print("Summation took %.2e s (3PCF) / %.2e s (4PCF)"%(t3pt, t4pt))

        end_time_binning=time.time()
        bintime=end_time_binning-start_time_binning+bintime

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
    zetacts = [zetacts[i]*(2.*i+1.)/2./eightpi_sq for i in range(numell)]
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
