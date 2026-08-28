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

do_random_ifg_sampling = False       # Sample interferograms with repeats
do_gaussian_diam_sampling = True    # Sample diameters from normal distribution
assign_default_uncertainties = True # Assign conservative placeholder errors
force_claret_params = False         # Force Claret & Bloemen 2011 u_lambda
n_bootstraps = 3             # Number of bootstrapping iterations
pred_ldd_col = "LDD_pred"           # tgt_info column with LDD colour relation
e_pred_ldd_col = "e_LDD_pred"       # tgt_info column with LDD relation errors
n_calib_runs = 1                   # N calibration runs to split nights among, correr en paralelo n times, por cada noche 
calib_run_i = 0                     # ith calibration run to perform, 0 indexed
# ============================================================
# BASELINE MODE
# ============================================================

use_bad_baselines = True

# Folder mask where the reduced files are stored
base_path = "/home2/ihernand/Desktop/reach/complete_sequences/%s_v3.94_abcd/"
# Day and N bootstrap specific results folder details
str_date = time.strftime("%y-%m-%d")  
if use_bad_baselines:

    baseline_mode = "BAD_BL_REMOVED"

else:

    baseline_mode = "ALL_BASELINES"


save_data_path = "/home2/ihernand/Desktop/reach/data/outputs"

if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)

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

print("\n", "-"*79, "\n", "\tSampling\n", "-"*79)  

# ============================================================
# CALIBRATORS EXCLUDED FROM CALIBRATION
# ============================================================

bad_calibrators = []

for target_id, row in tgt_info.iterrows():

    if row["Quality"] == "BAD":

        if "Primary" in tgt_info.columns:
            name = str(row["Primary"])
        else:
            name = str(target_id)

        bad_calibrators.append(name)


print("")
print("=" * 70)
print("CALIBRATORS EXCLUDED")
print("=" * 70)

if len(bad_calibrators) == 0:

    print("None")

else:

    for cal in bad_calibrators:
        print("  %s" % cal)

print("=" * 70)

def clean_for_filename(name):

    name = str(name)

    name = name.replace(" ", "")
    name = name.replace("_", "")
    name = name.replace("/", "")
    name = name.replace("\\", "")

    return name


if len(bad_calibrators) == 0:

    calibrator_mode = "ALL_CALS"

else:

    clean_bad_cals = [
        clean_for_filename(cal)
        for cal in bad_calibrators
    ]

    calibrator_mode = (
        "NO_" + "_".join(clean_bad_cals)
    )

# ============================================================

results_folder = "%s_i%i_%s_%s" % (
    str_date,
    n_bootstraps,
    baseline_mode,
    calibrator_mode
)


results_root = "/home2/ihernand/Desktop/reach/results/"

if not os.path.exists(results_root):
    os.mkdir(results_root)


results_path = os.path.join(
    results_root,
    results_folder
) + "/"


if not os.path.exists(results_path):
    os.mkdir(results_path)


print("")
print("=" * 70)
print("RUN CONFIGURATION")
print("=" * 70)
print("results_folder       :", results_folder)
print("baseline_mode        :", baseline_mode)
print("bad_calibrators      :", bad_calibrators)
print("random IFG sampling  :", do_random_ifg_sampling)
print("=" * 70)
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

# ============================================================
# gam_Lep
# Keep only: HR2090 -> gam_Lep -> HD1947
# ============================================================

key = (106, "gam_Lep", "faint")

wanted_sequence = ["HR_2090", "gam_Lep", "HD_42747"]

def clean_name(name):
    return str(name).replace("_", "").replace(" ", "").lower()


if key in sequences and key in complete_sequences:

    print("\n" + "=" * 70)
    print("MODIFYING gam_Lep SEQUENCE")
    print("=" * 70)

    print("Original sequence:")
    print(sequences[key])

    # --------------------------------------------------------
    # 1. Find HR2090 -> gam_Lep -> HD1947 in sequences
    # --------------------------------------------------------

    seq_clean = [clean_name(x) for x in sequences[key]]
    wanted_clean = [clean_name(x) for x in wanted_sequence]

    start_seq = None

    for i in range(len(seq_clean) - 2):

        if seq_clean[i:i+3] == wanted_clean:
            start_seq = i
            break

    if start_seq is None:
        raise RuntimeError(
            "Could not find HR2090 -> gam_Lep -> HD1947 "
            "in sequence %s" % str(sequences[key])
        )

    # Keep ONLY these 3 blocks
    sequences[key] = sequences[key][start_seq:start_seq+3]

    print("New sequence:")
    print(sequences[key])

    # --------------------------------------------------------
    # 2. Modify complete_sequences
    # --------------------------------------------------------

    observations = complete_sequences[key][2]

    print("Number of observations before:",
          len(observations))

    # --------------------------------------------------------
    # Identify contiguous observing blocks
    # --------------------------------------------------------

    blocks = []

    block_start = 0
    previous_target = clean_name(observations[0][2])

    for i in range(1, len(observations)):

        current_target = clean_name(observations[i][2])

        if current_target != previous_target:

            blocks.append(
                (
                    previous_target,
                    block_start,
                    i
                )
            )

            block_start = i
            previous_target = current_target

    # Last block
    blocks.append(
        (
            previous_target,
            block_start,
            len(observations)
        )
    )

    print("\nObserving blocks found:")

    for i, block in enumerate(blocks):
        print(
            i,
            observations[block[1]][2],
            block[1],
            block[2]
        )

    # --------------------------------------------------------
    # Find consecutive:
    #
    # HR2090 -> gam_Lep -> HD1947
    # --------------------------------------------------------

    wanted_clean = [
        clean_name("HR_2090"),
        clean_name("gam_Lep"),
        clean_name("HD_42747")
    ]

    first_good_block = None

    for i in range(len(blocks) - 2):

        names = [
            blocks[i][0],
            blocks[i+1][0],
            blocks[i+2][0]
        ]

        if names == wanted_clean:
            first_good_block = i
            break

    if first_good_block is None:

        raise RuntimeError(
            "Could not find observing blocks "
            "HR2090 -> gam_Lep -> HD1947"
        )

    # Indices in the observation array
    first_obs = blocks[first_good_block][1]

    # End of HD1947 block
    last_obs = blocks[first_good_block + 2][2]

    # --------------------------------------------------------
    # complete_sequences[key] is a tuple
    # Convert to list -> modify -> tuple
    # --------------------------------------------------------

    tmp = list(complete_sequences[key])

    tmp[2] = observations[first_obs:last_obs]

    complete_sequences[key] = tuple(tmp)

    print("\nNumber of observations after:",
          len(complete_sequences[key][2]))

    # --------------------------------------------------------
    # Check final blocks
    # --------------------------------------------------------

    print("\nFINAL gam_Lep OBSERVING BLOCKS")

    previous_target = None

    for obs in complete_sequences[key][2]:

        target = obs[2]

        if clean_name(target) != clean_name(previous_target):

            print("   ", target)
            previous_target = target

    print("=" * 70)
sequences.pop((105, "HD142860", "bright"))
complete_sequences.pop((105, "HD142860", "bright"))


print("\n" + "=" * 79)
print("ALL COMPLETE SEQUENCES TO TEST")
print("=" * 79)

for seq in sorted(complete_sequences.keys()):

    print(
        "%s  night=%s"
        % (
            seq,
            complete_sequences[seq][0]
        )
    )


print("\nTOTAL COMPLETE SEQUENCES: %i"
      % len(complete_sequences))


all_nights = sorted(
    set(
        complete_sequences[seq][0]
        for seq in complete_sequences
    )
)

print("TOTAL NIGHTS: %i"
      % len(all_nights))

print("=" * 79)

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
        rpndrs.save_nightly_pndrs_script(complete_sequences, tgt_info, base_path, use_bad_baselines=use_bad_baselines)
    elif not already_calibrated:
        rpndrs.save_nightly_pndrs_script(complete_sequences, tgt_info, base_path,
                                         run_local=run_local, use_bad_baselines=use_bad_baselines)

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

run_bootstrap_diagnostics = False

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
