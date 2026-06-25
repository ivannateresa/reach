"""
Script to run pndrs calibration bootstrapping routine.

Required software:
 - numpy, scipy, matplotlib, astropy, pandas
 - pndrs, PIONIER data reduction pipeline
 - extinction, https://github.com/kbarbary/extinction
 - bolometric-corrections, https://github.com/casaluca/bolometric-corrections
 - PyPDF2, https://pypi.org/project/PyPDF2/ (Only for pndrs pdf inspection)
"""
from __future__ import division, print_function

import os
import time
import glob
import numpy as np
import pandas as pd
import reach.diameters as rdiam
import reach.diagnostics as rdiag
import reach.plotting as rplt
import reach.photometry as rphot
import reach.pndrs as rpndrs
import reach.utils as rutils
import reach.parameters as rparam
import platform
from sys import exit as sys_exit

tgt_info = pd.read_csv("data/tgt_info.csv", header=0, sep=',')


targets = np.unique(tgt_info[tgt_info["Science"] == "TRUE"]["HD_ID"])

# ============================================================
# Default offsets
# These are applied to all targets first
# ============================================================

teff_default = (0, 150)
vw3_default = (-0.5, 0.1)
vw4_default = (-0.6, 0.2)
bv_feh_default = (-0.6, 0.2)
lit_diam_default = (-0.6, 0.2)



teff = dict((star, teff_default) for star in targets)

vw3 = dict((star, vw3_default) for star in targets)

vw4 = dict((star, vw4_default) for star in targets)

bv_feh = dict((star, bv_feh_default) for star in targets)

lit_diam = dict((star, lit_diam_default) for star in targets)

# ============================================================
# Manual corrections
# If one star needs a different offset, change it here
# ============================================================

teff["alfcmi"] = (0, 150)
teff["iotpsc"] = (0, 150)