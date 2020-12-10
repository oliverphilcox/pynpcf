# pynpcf

Code for O(N^2) estimation of three- and four-point isotropic correlation function multipoles, as in Slepian et al. 2016-2021. Loosely based on the Slepian/Eisenstein C++ code, and written in Python/Cython.

To run the code, first compile the Cython using ```python setup.py build_ext --inplace```. The main code can then be run using ```python npcf_estimator.py``` and input options are specified within the ```npcf_estimator.py``` script.

Requirements:
- python3 (also compatible with python2)
- cython
- scipy
- sympy (for Wigner 3j manipulations)
