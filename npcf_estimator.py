#Code for NPCFs - Zack Slepian + Oliver Philcox
#Currently supports 3PCF + 4PCF inputs

######################## LOAD PYTHON MODULES ########################
#doing basic math
import numpy as np
#system operations
import os, sys
#for timing
import time
#kd tree
from scipy import spatial
#custom cython functions used here
import cython_utils
#custom python functions used here
from python_utils import *

######################## INPUT PARAMETERS ##############################

#infile = 'sample_feb27_unity_weights_rescale400_first500.dat'
infile = 'patchy_gal_pos.txt'
cut_number = 1001 # maximum number of galaxies to read in
boxsize = 2500. #size of box used in sample data file.
rescale = .2 # if you want to rescale the box (unity if not).
outstr = 'test' # string to prepend to output files

# Binning
rmax = np.sqrt(3.)*100.
rmin = 1e-5
nbins= 10
numell = 11

# Other switches (mostly for testing)
no_weights = True # replace weights with unity
compute_4PCF = True # else just compute 3PCF
n_its = 1 # number of iterations to split the computation into (each analyzes N_gal/n_it central galaxies)
verb = 0 # if True, print useful(?) messages throughout

assert numell<=11, "# Spherical harmonic weights are configured only up to ell=10!"

######################## LOAD IN DATA ##############################

# Load data-file
if not os.path.exists(infile):
    raise Exception("# Infile %s does not exist!"%infile)

infile = np.loadtxt(infile,unpack=True)[:,:cut_number]

if len(infile!=4):
    assert len(infile==3), "# Input file has incorrect format!"
    if not no_weights:
        print("# No weights are supplied. They will be set to unity.")
        no_weights = True
        assert len(infile==3)
galx, galy, galz, galw = infile
del infile
ngal=len(galx)

# Add in rescaling factor
boxsize *= rescale #this is important so the periodic duplicates are made correctly.
galx, galy, galz = galx*rescale, galy*rescale, galz*rescale

if no_weights:
    galw  = np.ones_like(galw)

# Check edges
width_x = np.max(galx)-np.min(galx)
width_y = np.max(galy)-np.min(galy)
width_z = np.max(galz)-np.min(galz)
assert max([width_x,width_y,width_z])<boxsize, "#"

##Print some diagnostics
print("\nINPUT")
print("----------")
print("N_gal = %d"%ngal)
print("Mean weight = %.2e"%np.mean(galw))
print("Particle Size: [%.2e, %.2e, %.2e]"%(width_x,width_y,width_z))
print("Boxsize: %.2e"%(boxsize))
print("Number density: %.2e"%(ngal/(width_x*width_y*width_z)))
print("Binning: [%.2e, %.2e] in %d bins"%(rmin,rmax,nbins))
print("L_max = %d"%(numell-1))
if compute_4PCF:
    print("Compute 4PCF: True")
else:
    print("Compute 4PCF: False")
print("----------\n")

######################## PRE-PROCESSING ##############################
start_time=time.time()

# Small quantity to avoid zero-errors
eps=1e-8

# Define binning
deltr = float(rmax-rmin)/nbins
binarr=np.mgrid[0:nbins+1]*deltr
assert rmin<boxsize, "# Maximum radius must be less than boxsize!"

# Timers for diagnostics
histtime=0
bintime=0
transftime=0

def shiftbox(galx,galy,galz,shiftbox,doxshift,doyshift,dozshift):
    """Shift the primary volume by the dimensions of the box, to allow for tree computation"""
    galxsh=galx+doxshift*boxsize
    galysh=galy+doyshift*boxsize
    galzsh=galz+dozshift*boxsize

    galcoordssh=galxsh.ravel(),galysh.ravel(),galzsh.ravel()

    return galcoordssh

start_time = time.time()

#################### COMPUTE / LOAD WEIGHTING MATRIX ####################

infile_3pcf = 'coupling_matrices/weights_3pcf_n%d.npy'%numell
if os.path.exists(infile_3pcf):
    print("Loading precomputed 3PCF coupling weights from file")
    weights_3pcf = np.asarray(np.load(infile_3pcf),dtype=np.float64)
else:
    weights_3pcf = compute_3pcf_coupling_matrix(numell)

if compute_4PCF:

    infile_4pcf = 'coupling_matrices/weights_4pcf_n%d.npy'%numell
    if os.path.exists(infile_4pcf):
        print("Loading precomputed 4PCF coupling weights from file")
        weights_4pcf = np.asarray(np.load(infile_4pcf),dtype=np.float64)
    else:
        weights_4pcf = compute_4pcf_coupling_matrix(numell)

# Load Cython classes to hold NPCFs
ThreePCF = cython_utils.ThreePCF(numell, nbins, weights_3pcf)
if compute_4PCF:
    FourPCF = cython_utils.FourPCF(numell, nbins, weights_4pcf)

end_time = time.time()
print("\n# Time to define 3PCF + 4PCF weighting matrices = %.2e s"%(end_time-start_time))

#################### ASSIGN PARTICLES TO TREE ####################

start_time=time.time()

# Loop over shiftbox to append virtual boxes to coordinate list.
# This adds periodic wrapping needed for the KDTree
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
del new_galx, new_galy, new_galz, new_galw, galx, galy, galz, galw
end_time=time.time()

print("# Time to shift and place galaxies in an array = %.2e s"%(end_time-start_time))

# Now put galaxies in a tree
start_time=time.time()
tree=spatial.cKDTree(galcoords,leafsize=3000)
end_time=time.time()
print("# Time to put %d galaxies in tree = %.2e s"%(ngal,end_time-start_time))

#################### MAIN LOOP ####################

start_time = time.time()
nperit=ngal//n_its # number of galaxies per iteration.
if ngal%nperit==0:
    totalits = ngal//nperit
else:
    totalits=ngal//nperit+1 # total number of iterations

# Diagnostic functions
count = 0 # number of centrals analyzed
querytime = 0.

print("\nStarting main computation...\n")

# Now loop over the iterations
for i in range(0,totalits):

    # Print some diagonstics
    print("# Analyzing group number = %d of %d"%(i+1,totalits))
    # Select out central galaxies to use; note central galaxies must be first in list of coordinates including shift, which they will be by construction.
    centralgals=galcoords[i*nperit:min((i+1)*nperit,ngal)]
    print("This contains %d central galaxies, starting from galaxy %d."%(len(centralgals),i*nperit+1))

    # Query the tree to pick out the galaxies within rmax of each central.
    # e.g. ball[0] gives indices of gals. within rmax of the 0th centralgal
    start_time_query=time.time()
    ball=tree.query_ball_point(centralgals,rmax+eps)
    end_time_query=time.time()
    querytime=end_time_query-start_time_query+querytime

    # Iterate over central galaxies in the list
    for w in range(len(centralgals)):

        start_time_transf=time.time()

        # Remove this galaxy from the ball
        ball[w].remove(i*nperit+w)
        ball_temp=ball[w]

        # Skip if no galaxies nearby!
        if len(ball_temp)==0:
            continue

        # Transform to reference frame of desired central galaxy
        galxtr, galytr, galztr=np.asarray(galcoords[ball_temp][:,:3]-centralgals[w][:3]).T
        galwtr = galcoords[ball_temp][:,3] # galaxy weight

        # Compute squared distance from center
        rgalssq=galxtr*galxtr+galytr*galytr+galztr*galztr+eps
        rgals=np.sqrt(rgalssq)

        ## COMPUTE SPHERICAL HARMONIC WEIGHTS
        # These are the spherical harmonics for each galaxy, without radial binning
        weighttime = time.time()
        all_weights = compute_weight_matrix((galxtr-1.0j*galytr)/rgals,galztr/rgals,galwtr,numell)
        weighttime = time.time()-weighttime
        del galxtr, galytr, galztr, galwtr, rgalssq

        if verb: print("Computation time for spherical harmonic weights (single galaxy): %.3e s"%weighttime)

        end_time_transf=time.time()
        transftime=end_time_transf-start_time_transf+transftime

        ## HISTOGRAM GALAXIES
        # This computes the binned Y_lm coefficients in a single list, histogramming all weights simultaneously
        # The ordering is Y_00, Y_{1-1} Y_{10} Y_{2-2} etc.
        # Dimension is (N_mult,N_bin) where N_mult is number of multipoles = (numell)*(numell+1)/2
        start_time_hist=time.time()

        y_all = histogram_multi(rgals, bins=binarr, weight_matrix=all_weights)
        y_all_conj = y_all.conjugate() # compute complex conjugates only once

        end_time_hist=time.time()
        histtime=end_time_hist-start_time_hist+histtime
        if verb: print("Histogramming function time: %.3e"%(end_time_hist-start_time_hist))

        # Count up number of centrals analyzed
        count=count+1

        ## BIN WEIGHTS INTO MULTIPOLES
        # This is mainly done using custom cython code for speed.
        # The y_all and y_all_conj arrays are converted to C memoryviews on input
        start_time_binning=time.time()

        # 3PCF summations
        t3pt = time.time()
        ThreePCF.add_to_sum(y_all, y_all_conj)
        t3pt = time.time()-t3pt

        # 4PCF summations
        if compute_4PCF:
            t4pt = time.time()
            FourPCF.add_to_sum(y_all, y_all_conj)
            t4pt = time.time()-t4pt

            if verb: print("Summation took %.2e s (3PCF) / %.2e s (4PCF)"%(t3pt, t4pt))

        end_time_binning=time.time()
        bintime=end_time_binning-start_time_binning+bintime

print("\nComputation complete!\n")

end_time=time.time()
timecost=end_time-start_time
ballfrac=querytime/timecost

#################### CREATE OUTPUTS ####################

out_time = time.time()

zeta3 = ThreePCF.zeta3
if compute_4PCF:
    zeta4 = FourPCF.zeta4
if verb: print("Note that 3PCF output differs from Slepian/Eisenstein expansion by a factor (-1)^l_1 / sqrt(2l_1+1) due to different choice of basis functions")

# Compute bin centers
all_centers = 0.5*(binarr[1:]+binarr[:-1])

bin_centers_3pcf = np.zeros((len(zeta3[0]),2), np.float64)
index = 0
for a in range(nbins):
    for b in range(a,nbins):
        bin_centers_3pcf[index,0] = all_centers[a]
        bin_centers_3pcf[index,1] = all_centers[b]
        index += 1

if compute_4PCF:
    bin_centers_4pcf = np.zeros((len(zeta4[0,0,0]),3), np.float64)
    index = 0
    for a in range(nbins):
        for b in range(a,nbins):
            for c in range(b,nbins):
                bin_centers_4pcf[index,0] = all_centers[a]
                bin_centers_4pcf[index,1] = all_centers[b]
                bin_centers_4pcf[index,2] = all_centers[c]
                index += 1

# Save bin-centers and CFs to .npz file
outfile_3pcf = 'outputs/%s_3PCF_n%d.npz'%(outstr,numell)
np.savez(outfile_3pcf,
         bin_centers = bin_centers_3pcf,
         zeta = zeta3)
print("3PCF saved to %s"%outfile_3pcf)

if compute_4PCF:
    outfile_4pcf = 'outputs/%s_4PCF_n%d.npz'%(outstr,numell)
    np.savez(outfile_4pcf,
             bin_centers = bin_centers_4pcf,
             zeta = zeta4)
    print("4PCF saved to %s"%outfile_4pcf)

out_time = time.time() - out_time

#################### PRINT DIAGNOSTICS ####################

print("\nDIAGNOSTICS")
print("----------")
print("Number of galaxies analyzed: %d"%count)
print("Total number of galaxies: %d"%ngal)
print("Transformation time: %.3e s"%transftime)
print("Histogram time: %.3e s"%histtime)
print("Binning time: %.3e s"%bintime)
print("Ball query time: %.3e s"%querytime)
print("Total Time: %.3e s"%timecost)
print("")
print("Transformation speed: %.3e s / galaxy"%(transftime/ngal))
print("Histogram speed: %.3e s / galaxy"%(histtime/ngal))
print("Binning speed: %.3e s / galaxy"%(bintime/ngal))
print("----------\n")
