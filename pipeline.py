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
# Define Bootstrapping Parameters
# -----------------------------------------------------------------------------
# Run specific parameters
#lb_pc = 70                          # The size of the local bubble in pc
#use_plx_systematic = True           # Use Stassun & Torres 18 plx offset
#do_random_ifg_sampling = True       # Sample interferograms with repeats
#do_gaussian_diam_sampling = True    # Sample diameters from normal distribution
#assign_default_uncertainties = True # Assign conservative placeholder errors
#force_claret_params = False         # Force Claret & Bloemen 2011 u_lambda
#n_bootstraps = 5000                 # Number of bootstrapping iterations
#pred_ldd_col = "LDD_pred"           # tgt_info column with LDD colour relation
#e_pred_ldd_col = "e_LDD_pred"       # tgt_info column with LDD relation errors
#n_calib_runs = 8                    # N calibration runs to split nights among
#calib_run_i = 0                     # ith calibration run to perform, 0 indexed



lb_pc = 70                          # The size of the local bubble in pc
use_plx_systematic = False           # Use Stassun & Torres 18 plx offset
do_random_ifg_sampling = True       # Sample interferograms with repeats
do_gaussian_diam_sampling = True    # Sample diameters from normal distribution
assign_default_uncertainties = True # Assign conservative placeholder errors
force_claret_params = False         # Force Claret & Bloemen 2011 u_lambda
n_bootstraps = 10                 # Number of bootstrapping iterations
pred_ldd_col = "LDD_pred"           # tgt_info column with LDD colour relation
e_pred_ldd_col = "e_LDD_pred"       # tgt_info column with LDD relation errors
n_calib_runs = 1                   # N calibration runs to split nights among, correr en paralelo n times, por cada noche 
calib_run_i = 1                     # ith calibration run to perform, 0 indexed


# Folder mask where the reduced files are stored
base_path = "/home2/ihernand/Desktop/reach/complete_sequences/%s_v3.94_abcd/"
# Day and N bootstrap specific results folder details
str_date = time.strftime("%y-%m-%d")  
results_folder = "%s_i%i" % (str_date, n_bootstraps)
if not os.path.exists("/home2/ihernand/Desktop/reach/results/"):
    os.mkdir("/home2/ihernand/Desktop/reach/results/")

results_path = "/home2/ihernand/Desktop/reach/results/%s/" % results_folder

if not os.path.exists(results_path):
    os.mkdir(results_path)

# Path to Casagrande & VandenBerg 2014/2018a/2018b bolometric correction code
# and filters to use when calculating fbol_final from [Hp, Bt, Vt, Bp, Rp]
bc_path =  "/home2/ihernand/Desktop/reach/bolometric-corrections"
band_mask = [1, 1, 1, 0, 0]

# Set these if investigating the quality of calibrators
calibrate_calibrators = False
test_all_cals = False

# Set if writing files locally (i.e. not on the server) to check files, or if
# running without writing new files (e.g. to check what pndrs does by default)
run_local = False
already_calibrated = False

print("\nBeginning calibration and fitting run. Parameters set as follow:")
print(" - n_bootstraps\t\t\t=\t%i" % n_bootstraps)
print(" - do_random_ifg_sampling\t=\t%s" % do_random_ifg_sampling)
print(" - do_gaussian_diam_sampling\t=\t%s" % do_gaussian_diam_sampling)

# -----------------------------------------------------------------------------
# Import target details and sample parameters
# -----------------------------------------------------------------------------
# Targets information is loaded into a pandas dataframe, with column labels for
# each of the stored parameters (e.g. VTmag) and row indices of HD ID

tgt_info = rutils.initialise_tgt_info(assign_default_uncertainties, lb_pc,
                                      use_plx_systematic)
print(list(tgt_info.columns))

print("\n", "-"*79, "\n", "\tSampling\n", "-"*79)  

# If already created, load sampled diameters
if rutils.sampling_already_done(results_folder, force_claret_params):
    print("Sampling already done, loading...")
    n_pred_ldd, e_pred_ldd = rutils.load_sampled_ldd(results_folder)

# Sample diameters for bootstrapping (if n_bootstraps < 1, actual predictions)
# and initialise the sampled stellar parameters (though we only require this
# later when doing the fits)    
else:
    print("Sampling not yet done, doing now...")
    n_pred_ldd, e_pred_ldd = rdiam.sample_n_pred_ldd(tgt_info, n_bootstraps, 
                                                 pred_ldd_col, e_pred_ldd_col,
                                                 do_gaussian_diam_sampling)
    
    rutils.save_sampled_ldd(n_pred_ldd, e_pred_ldd, results_folder)
                                                 
    # Sample stellar parameters
    sampled_sci_params = rparam.sample_all(tgt_info, n_bootstraps, bc_path,
                                           force_claret_params, band_mask)
    rutils.save_sampled_params(sampled_sci_params, results_folder)

# -----------------------------------------------------------------------------
# Import observing logs, remove unwanted sequences/stars
# -----------------------------------------------------------------------------
# Load in the summarising data structures created in organise_obs.py

complete_sequences, sequences = rutils.load_sequence_logs()

# Currently no proxima cen or gam pav data, so pop
#sequences.pop((102, 'gamPav', 'faint'))
#sequences.pop((102, 'gamPav', 'bright'))
#sequences.pop((102, 'ProximaCen', 'bright'))

# Don't care about distant RGB, pop
#sequences.pop((99, "HD187289", "faint"))
#sequences.pop((99, "HD187289", "bright"))
#complete_sequences.pop((99, 'HD187289', 'faint'))
#complete_sequences.pop((99, 'HD187289', 'bright'))

# -----------------------------------------------------------------------------
# [Optional] Calibrate calibrators against each other
# -----------------------------------------------------------------------------
if calibrate_calibrators:
    print("-"*79, "\nCalibrating Calibrators\n", "-"*79)
    rdiag.calibrate_calibrators(sequences, complete_sequences, base_path, 
                                tgt_info, n_pred_ldd, e_pred_ldd, 
                                test_all_cals)
    
    # Finished calibrating calibrators, exit
    print("Finished calibrating calibrators")
    sys_exit(0)


# -----------------------------------------------------------------------------
# Write nightly pndrs scripts as YYYY-MM-DD_pndrsScript.i
# -----------------------------------------------------------------------------
# Do the following:
#  i)  Exclude bad calibrators
#  ii) Split nights between sequences
# **ONLY** for the first calib run (i.e. only do this once, but for all seq
if calib_run_i == 0:
    if not run_local and not already_calibrated:
        rpndrs.save_nightly_pndrs_script(complete_sequences, tgt_info, base_path)
    elif not already_calibrated:
        rpndrs.save_nightly_pndrs_script(complete_sequences, tgt_info, base_path,
                                         run_local=run_local)

# -----------------------------------------------------------------------------
# Split into multiple bootstrapping runs if required
# -----------------------------------------------------------------------------
# We have the option to run the calibration runs separately - i.e. calibrate
# certain sets of sequences in parallel. Set this up here

# Easiest to parallelise at the night level. Get the list of (unique) nights
# and sort of consistency
nights = [complete_sequences[seq][0] for seq in complete_sequences.keys()]
nights = list(set(nights))
nights.sort()

n_init_seq = len(complete_sequences)
valid_nights = nights
    
if n_calib_runs != 1:
    # This is setup such that we can run n_calib_runs separate runs of this 
    # script, with equal amounts of nights between them (rounding up for the 
    # last such run)
    n_nights = np.round(len(nights) / n_calib_runs).astype(int)
    
    min_night_i = n_nights * calib_run_i  
    max_night_i = n_nights * (calib_run_i + 1)
    
    if max_night_i > len(nights) : max_night_i = len(nights)
    
    # Run only on the sequences associated with these nights
    valid_nights = nights[min_night_i:max_night_i]
    valid_seqs = [seq for seq in complete_sequences.keys()
                  if complete_sequences[seq][0] in valid_nights]                

    complete_sequences = {seq:complete_sequences[seq] for seq in valid_seqs}
    
    sequences = {seq:sequences[seq] for seq in complete_sequences}

# -----------------------------------------------------------------------------
# Run bootstrapping
# -----------------------------------------------------------------------------
print("\n", "-"*79, "\n", "\tBootstrapping\n", "-"*79)
print("Bootstrapping run %i/%i" % (calib_run_i + 1, n_calib_runs))
print("Running on %i/%i sequences over %i/%i nights" 
      % (len(complete_sequences), n_init_seq, len(valid_nights), len(nights)))
rpndrs.run_n_bootstraps(sequences, complete_sequences, base_path, tgt_info,
                        n_pred_ldd, e_pred_ldd, n_bootstraps, results_path,
                        run_local=run_local, 
                        already_calibrated=already_calibrated,
                        do_random_ifg_sampling=do_random_ifg_sampling)