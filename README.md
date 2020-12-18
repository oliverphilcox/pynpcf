# pynpcf

Code for O(N^2) estimation of three- and four-point isotropic correlation function multipoles, as in Slepian et al. 2016-2021, written in Python/Cython. This is a sister code to the [C++ code](https://github.com/oliverphilcox/NPCF-Estimator), which is more interpretable, though somewhat slower. Whilst the main code is written in pure Python, the most CPU-intensive sections are written in Cython for speed. All conventions follow that of the NPCF paper of Slepian et al. (in prep.).

To run the code, first compile the Cython using ```python setup.py build_ext --inplace```. The main code can then be run using ```python npcf_estimator.py``` and input options are specified within the ```npcf_estimator.py``` script.

Requirements:
- python3 (also compatible with python2)
- cython
- scipy
- sympy (for Wigner 3j manipulations)

Authors:
- Oliver Philcox (Princeton / IAS)
- Zachary Slepian (Florida)
- Daniel Eisenstein (Harvard)

Usage:
- To run the code, first compile the Cython using ```python setup.py build_ext --inplace```. This only needs to be done once, unless you modify the code.
- The main code can then be run using ```python npcf_estimator.py [INFILE] [OUT_STRING]``` where ```INFILE``` specifies the input CSV file of positions and (optionally) particle weights and ```OUT_STRING``` is a string that will be prepended to the file names. Other options can be specified within the Python code (see below)
- The output products are stored in the ```outputs/``` directory as compressed ```.npz``` files. In general, the NPCF arrays have the dimension (N_l x N_l x ... ) x N_bins, where N_bins is a flattened array of the radial bins, specified by the ```bin_centers``` attribute.
- Generally, one will run the code on ~ 30 (data-random) files, the corresponding (random-random) files, then combine to obtain the NPCF estimates. For (data-random) inputs, the randoms should have negative weights, such that the total summed weight is zero.

Main Options:
- ```rmax```: Maximum binning radius in Mpc/h (default: 170).
- ```rmin```: Minimum binning radius in Mpc/h (default: 1e-5).
- ```nbins```: Number of radial bins (default: 5).
- ```numell```: Total number of multipoles, equal to 1+ell_max (default: 6).
- ```compute_4PCF```: If set, compute the 4PCF in addition to the 3PCF (default: True).
- ```n_its```: Number of iterations to split the computation into (each analyzes N_gal/n_it central galaxies) (default: 10).

Other Options:
- ```cut_number```: If set, read in only this number of galaxies (default: -1).
- ```boxsize```: Size of the cubic box in which to place the particles in Mpc/h (default 3300).
- ```rescale```: If set, rescale the particle co-ordinates by this factor (default: 1).
- ```no_weights```: If set, replace the particle weights with unity, if present (default: False).
- ```verb:```: If set, print useful(?) messages throughout runtime (default: False).
