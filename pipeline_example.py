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
import diagnostics as dig
from colte import colte

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
use_plx_systematic = False           # Use Stassun & Torres 18 plx offset --> don't use, was a Gaia DR1 thing. We should probably use Bailer-Jones distances

do_random_ifg_sampling = True       # Sample interferograms with repeats
do_gaussian_diam_sampling = True    # Sample diameters from normal distribution
assign_default_uncertainties = True # Assign conservative placeholder errors
force_claret_params = True         # Force Claret & Bloemen 2011 u_lambda
n_bootstraps = 500              # Number of bootstrapping iterations
pred_ldd_col = "LDD_pred"           # tgt_info column with LDD colour relation
e_pred_ldd_col = "e_LDD_pred"       # tgt_info column with LDD relation errors
n_calib_runs = 1                   # N calibration runs to split nights among, correr en paralelo n times, por cada noche 
calib_run_i = 0                     # ith calibration run to perform, 0 indexed


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


save_data_path = "/home2/ihernand/Desktop/reach/data/outputs"

if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)

# Path to Casagrande & VandenBerg 2014/2018a/2018b bolometric correction code
# and filters to use when calculating fbol_final from [Hp, Bt, Vt, Bp, Rp]
bc_path =  "/home2/ihernand/Desktop/reach/bolometric-corrections"
band_mask = [1, 1, 1, 0, 0]

#lo que hace colte es a apartir de Gaia DR3 y 2MASS, utiliza este codigo para calcular la temperatura fotometrica 

colte_path = "/home2/ihernand/Desktop/reach/colte"


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

print("\n", "-"*79, "\n", "\tSampling\n", "-"*79)  


#Pondre un plot donde me entregue los angular diameter predicted de cada uno, para ver como funciona, la relacion entre color y magnitud

print("\n", "-"*79, "\n", "\tSave tgt_info in Data\n", "-"*79) 
tgt_info.to_csv("data/tgt_info.csv")


### Diagnostico para ver que estrella no tienen algunos datos para ver que podemos hacer. 

print("\n", "-"*79, "\n", "\tDiagnostic\n", "-"*79)

cols_check = [
    "Primary", "Science", "SpT_simple",
    "BTmag", "VTmag",
    "Bmag", "Vmag",
    "Plx", "e_Plx", "Plx_alt", "e_Plx_alt",
    "Dist", "e_Dist",
    "FeH_rel",
    "BTmag_dr", "VTmag_dr",
    "LDD_pred", "teff_casagrande"
]

nan_rows = tgt_info[
    tgt_info[["Dist", "e_Dist", "Bmag", "Vmag", "LDD_pred", "teff_casagrande"]]
    .isnull()
    .any(axis=1)
]

for star, row in nan_rows.iterrows():
    print("\n", star, row["Primary"])

    if pd.isnull(row["BTmag"]) or pd.isnull(row["VTmag"]):
        print("  Falta BTmag o VTmag -> no se puede calcular Bmag/Vmag")

    if pd.isnull(row["Plx"]) and pd.isnull(row["Plx_alt"]):
        print("  Falta Plx y Plx_alt -> no se puede calcular Dist")

    if pd.isnull(row["e_Plx"]) and pd.isnull(row["e_Plx_alt"]):
        print("  Falta e_Plx/e_Plx_alt -> no se puede calcular e_Dist")

    if pd.isnull(row["FeH_rel"]):
        print("  Falta FeH_rel -> teff_casagrande queda NaN")



diagnostic = dig.save_initialise_diagnostics(
    tgt_info,
    outdir="/home2/ihernand/Desktop/reach/data/diagnostics",
    prefix="nan_diagnostic"
)


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
    #s)
    rutils.save_sampled_ldd(n_pred_ldd, e_pred_ldd, results_folder)
                                                 
    # Sample stellar parameters
    sampled_sci_params = rparam.sample_all(tgt_info, n_bootstraps, bc_path,
                                           force_claret_params, band_mask)

    rutils.save_sampled_params(sampled_sci_params, results_folder)

    # Save sampled predicted diameters also as CSV
    n_pred_ldd_csv = os.path.join(save_data_path, "n_pred_ldd.csv")
    e_pred_ldd_csv = os.path.join(save_data_path, "e_pred_ldd.csv")
    pd.DataFrame(n_pred_ldd).to_csv(n_pred_ldd_csv, index=True)
    pd.DataFrame(e_pred_ldd).to_csv(e_pred_ldd_csv, index=True)

    print("Saved sampled LDD:")
    print(n_pred_ldd_csv)
    print(e_pred_ldd_csv)



# -----------------------------------------------------------------------------
# Import observing logs, remove unwanted sequences/stars
# -----------------------------------------------------------------------------
# Load in the summarising data structures created in organise_obs.py

complete_sequences, sequences = rutils.load_sequence_logs() 


sequences.pop((106, "bet_Hyi", "bright"))
sequences.pop((105, "HD142860", "bright"))
complete_sequences.pop((106, "bet_Hyi", "bright"))
complete_sequences.pop((105, "HD142860", "bright"))


sequences_df = dig.save_sequence_logs_diagnostics(
    complete_sequences,
    sequences,
    outdir="/home2/ihernand/Desktop/reach/data/diagnostics",
    prefix="pionier"
)


# -----------------------------------------------------------------------------
# [Optional] Calibrate calibrators against each other
# -----------------------------------------------------------------------------
if calibrate_calibrators:
    print("-"*79, "\nCalibrating Calibrators\n", "-"*79)
    rdiag.calibrate_calibrators(sequences, complete_sequences, base_path, 
                                tgt_info, n_pred_ldd, e_pred_ldd, 
                                test_all_cals)
    
    # Finished calibrating calibrators, exit
    # -----------------------------------------------------------------------------
    cal_quality = dig.diagnose_calibrator_quality(
    tgt_info,
    cal_folder="/home2/ihernand/Desktop/reach/diagnostics/",
    outdir="/home2/ihernand/Desktop/reach/diagnostics/",
    prefix="calibrator_quality",
    update_tgt_info=False)
    cal_plot_summary = dig.plot_final_calibrators(
    tgt_info,
    cal_folder="/home2/ihernand/Desktop/reach/diagnostics/",
    outdir="/home2/ihernand/Desktop/reach/diagnostics/",
    prefix="final_calibrators",
    quality_csv="/home2/ihernand/Desktop/reach/diagnostics/calibrator_quality.csv")


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


print("\n", "-"*79, "\n", "\tBootstrapping\n", "-"*79)
print("Bootstrapping run %i/%i" % (calib_run_i + 1, n_calib_runs))
print("Running on %i/%i sequences over %i/%i nights" 
      % (len(complete_sequences), n_init_seq, len(valid_nights), len(nights)))

run_bootstrap_diagnostics = True

if run_bootstrap_diagnostics:

    bootstrap_diag_path = os.path.join(
        results_path,
        "bootstrap_diagnostics_run_%03i" % calib_run_i
    )

    bootstrap_summary = dig.run_n_bootstraps_diagnostic_with_visibility_plots(
        sequences,
        complete_sequences,
        base_path,
        tgt_info,
        n_pred_ldd,
        e_pred_ldd,
        n_bootstraps,
        results_path,
        diag_path=bootstrap_diag_path,
        run_local=run_local,
        already_calibrated=already_calibrated,
        do_random_ifg_sampling=do_random_ifg_sampling,
        stop_on_error=False
    )

    print("\nBootstrap diagnostic summary:")
    print(bootstrap_summary)
