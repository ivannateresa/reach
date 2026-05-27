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
# -----------------------------------------------------------------------------
# Llamar a reducir los datos pero no calibrados aun. 
# -----------------------------------------------------------------------------

base_path = "/home2/ihernand/Desktop/reach/complete_sequences/"


rpndrs.reduce_all_observations(base_path)

