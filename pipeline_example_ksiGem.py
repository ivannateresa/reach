from __future__ import division, print_function
 
import os
import glob
import shutil
import numpy as np
 
import reach.utils as rutils
import reach.diameters as rdiam
import reach.pndrs as rpndrs
 
 
# =============================================================================
# 1. SETTINGS
# =============================================================================
 
# ------------------------------------------------------------
# Sequence that we want to calibrate
# ------------------------------------------------------------
 
seq_test = (106, "ksi_Gem", "faint")
 
 
# ------------------------------------------------------------
# First test with ONE bootstrap
# When everything works, change to 500
# ------------------------------------------------------------
 
n_bootstraps = 1
 
 
# ------------------------------------------------------------
# Bootstrap settings
# ------------------------------------------------------------
 
do_random_ifg_sampling = True
do_gaussian_diam_sampling = True
 
assign_default_uncertainties = True
use_plx_systematic = False
 
lb_pc = 70
 
pred_ldd_col = "LDD_pred"
e_pred_ldd_col = "e_LDD_pred"
 
 
run_local = False
already_calibrated = False
 
 
# =============================================================================
# 2. PATHS
# =============================================================================
 
# Change ONLY this root if your Chapman installation is in /home2 instead.
ROOT = "/home/ihernand/Desktop/reach"
 
 
# Reduced data are directly inside:
#
# complete_sequences/
#     2022-03-01_v3.94_abcd/
#         PIONI...._oidata.fits
#
base_path = os.path.join(
    ROOT,
    "complete_sequences",
    "%s_v3.94_abcd"
) + "/"
 
 
# Separate results folder for this experiment
results_path = os.path.join(
    ROOT,
    "results",
    "test_ksi_Gem_bad_baseline"
) + "/"
 
 
if not os.path.exists(results_path):
    os.makedirs(results_path)
 
 
# =============================================================================
# 3. LOAD TARGET INFORMATION
# =============================================================================
 
print("\n" + "=" * 79)
print("INITIALISE TARGET INFORMATION")
print("=" * 79)
 
 
tgt_info = rutils.initialise_tgt_info(
    assign_default_uncertainties,
    lb_pc,
    use_plx_systematic
)
 
 
# =============================================================================
# 4. LOAD OBSERVING SEQUENCES
# =============================================================================
 
print("\n" + "=" * 79)
print("LOAD OBSERVING SEQUENCES")
print("=" * 79)
 
 
complete_sequences, sequences = rutils.load_sequence_logs()
 
 
# Check that ksi_Gem exists
if seq_test not in complete_sequences:
 
    print("\nERROR:")
    print("Sequence not found in complete_sequences:")
    print(seq_test)
 
    raise SystemExit
 
 
if seq_test not in sequences:
 
    print("\nERROR:")
    print("Sequence not found in sequences:")
    print(seq_test)
 
    raise SystemExit
 
 
# =============================================================================
# 5. KEEP ONLY ksi_Gem
# =============================================================================
 
complete_sequences = {
    seq_test: complete_sequences[seq_test]
}
 
sequences = {
    seq_test: sequences[seq_test]
}
 
 
night = complete_sequences[seq_test][0]
 
 
print("\nSelected sequence:")
print(seq_test)
 
print("\nNight:")
print(night)
 
print("\nCAL-SCI sequence:")
print(sequences[seq_test])
 
 
# =============================================================================
# 6. DEFINE REDUCED AND CALIBRATION FOLDERS
# =============================================================================
 
# ------------------------------------------------------------
# Reduced data folder:
#
# 2022-03-01_v3.94_abcd/
#
# ------------------------------------------------------------
 
reduced_folder = base_path % night
 
 
# ------------------------------------------------------------
# Calibration / bootstrap folder:
#
# 2022-03-01_v3.94_abcd/2022-03-01/
#
# ------------------------------------------------------------
 
calibration_folder = os.path.join(
    reduced_folder,
    night
)
 
 
print("\n" + "=" * 79)
print("FOLDERS")
print("=" * 79)
 
print("\nReduced folder:")
print(reduced_folder)
 
print("\nCalibration folder:")
print(calibration_folder)
 
 
# =============================================================================
# 7. CHECK THAT THE NIGHT IS REDUCED
# =============================================================================
 
print("\n" + "=" * 79)
print("CHECK REDUCED DATA")
print("=" * 79)
 
 
reduced_files = sorted(
    glob.glob(
        os.path.join(
            reduced_folder,
            "PIONI*_oidata.fits"
        )
    )
)
 
 
print(
    "\nReduced OIFITS found:",
    len(reduced_files)
)
 
 
if len(reduced_files) == 0:
 
    print("\nERROR:")
    print("No reduced PIONI*_oidata.fits found in:")
    print(reduced_folder)
 
    raise SystemExit
 
 
print("\nReduced data found -> OK")
 
 
# =============================================================================
# 8. CHECK THAT IT HAS NOT BEEN CALIBRATED
# =============================================================================
 
print("\n" + "=" * 79)
print("CHECK PREVIOUS CALIBRATION")
print("=" * 79)
 
 
if os.path.exists(calibration_folder):
 
    old_calibrated = sorted(
        glob.glob(
            os.path.join(
                calibration_folder,
                "*oidataCalibrated.fits"
            )
        )
    )
 
else:
 
    old_calibrated = []
 
 
print(
    "\nPreviously calibrated OIFITS:",
    len(old_calibrated)
)
 
 
if len(old_calibrated) > 0:
 
    print("\nWARNING:")
    print("This folder already contains calibrated files:")
 
    for filename in old_calibrated:
        print(
            "  %s"
            % os.path.basename(filename)
        )
 
    print("\nStopping because we want a clean test.")
 
    raise SystemExit
 
 
print(
    "\nSTATUS: REDUCED BUT NOT CALIBRATED -> PERFECT"
)
 
 
# =============================================================================
# 9. SHOW SCIENCE / CALIBRATOR CLASSIFICATION
# =============================================================================
 
print("\n" + "=" * 79)
print("SCIENCE / CALIBRATOR CLASSIFICATION")
print("=" * 79)
 
 
for star in sequences[seq_test]:
 
    matched_id = rpndrs.match_target_name(
        tgt_info,
        star,
        verbose=False
    )
 
 
    if matched_id is None:
 
        print(
            "%-20s  NOT FOUND"
            % star
        )
 
        continue
 
 
    science = bool(
        tgt_info.loc[matched_id]["Science"]
    )
 
 
    if science:
        role = "SCI"
    else:
        role = "CAL"
 
 
    print(
        "%-20s matched=%-20s role=%s"
        % (
            star,
            matched_id,
            role
        )
    )
 
 
# =============================================================================
# 10. SAMPLE CALIBRATOR DIAMETERS
# =============================================================================
 
print("\n" + "=" * 79)
print("SAMPLE PREDICTED DIAMETERS")
print("=" * 79)
 
 
n_pred_ldd, e_pred_ldd = rdiam.sample_n_pred_ldd(
    tgt_info,
    n_bootstraps,
    pred_ldd_col,
    e_pred_ldd_col,
    do_gaussian_diam_sampling
)
 
 
# =============================================================================
# 11. GET THE OBSERVING TIME OF EACH BLOCK
# =============================================================================
 
print("\n" + "=" * 79)
print("OBSERVING BLOCK DURATIONS")
print("=" * 79)
 
 
durations = rpndrs.calculate_target_durations(
    complete_sequences
)
 
 
ksi_intervals = []
 
 
for this_night in durations:
 
    print("\nNight:", this_night)
 
    for target, start, end in durations[this_night]:
 
        print(
            "%-20s  %.10f  %.10f"
            % (
                target,
                start,
                end
            )
        )
 
 
        # ----------------------------------------------------
        # Keep only the observing blocks corresponding to
        # ksi_Gem
        # ----------------------------------------------------
 
        if (
            rpndrs.clean_target_id(target)
            ==
            rpndrs.clean_target_id("ksi_Gem")
        ):
 
            ksi_intervals.append(
                [
                    float(start),
                    float(end)
                ]
            )
 
 
print("\n" + "-" * 79)
print("ksi_Gem intervals")
print("-" * 79)
 
 
for i, interval in enumerate(ksi_intervals):
 
    print(
        "Block %i: %.10f -> %.10f"
        % (
            i + 1,
            interval[0],
            interval[1]
        )
    )
 
 
if len(ksi_intervals) == 0:
 
    print("\nERROR:")
    print("No ksi_Gem observing intervals were found.")
 
    raise SystemExit
 
 
# =============================================================================
# 12. CREATE NORMAL PNDRS SCRIPT
# =============================================================================
 
print("\n" + "=" * 79)
print("CREATE NORMAL PNDRS SCRIPT")
print("=" * 79)
 
 
rpndrs.save_nightly_pndrs_script(
    complete_sequences,
    tgt_info,
    base_path,
    run_local=False
)
 
 
# =============================================================================
# 13. LOCATION OF PNDRS SCRIPT
# =============================================================================
 
pndrs_script = os.path.join(
    calibration_folder,
    "%s_pndrsScript.i" % night
)
 
 
print("\nPNDRS script:")
print(pndrs_script)
 
 
# save_nightly_pndrs_script may decide that no script is required.
# In our case we DO need one because we want to remove a baseline.
if not os.path.exists(calibration_folder):
 
    os.makedirs(calibration_folder)
 
 
if not os.path.exists(pndrs_script):
 
    print("\nNo pndrsScript existed.")
    print("Creating an empty one.")
 
    with open(pndrs_script, "w"):
        pass
 
 
# =============================================================================
# 14. BAD BASELINE
# =============================================================================
 
# ============================================================================
#
# IMPORTANT
#
# Put here the baseline identifier that PNDRS uses.
#
# If PNDRS really identifies it as AT2-AT3:
#
#     bad_baseline = "AT2-AT3"
#
# If your OI_ARRAY shows that AT2-AT3 corresponds, for example, to G1-J2:
#
#     bad_baseline = "G1-J2"
#
# ============================================================================
 
bad_baseline = "AT2-AT3"
 
 
print("\n" + "=" * 79)
print("BAD BASELINE")
print("=" * 79)
 
print(
    "\nBaseline to remove:",
    bad_baseline
)
 
 
# =============================================================================
# 15. ADD BAD BASELINE ONLY DURING ksi_Gem
# =============================================================================
 
print("\n" + "=" * 79)
print("ADD BAD BASELINE TO PNDRS SCRIPT")
print("=" * 79)
 
 
with open(pndrs_script, "a") as f:
 
    for block_i, interval in enumerate(ksi_intervals):
 
        start = interval[0]
        end = interval[1]
 
 
        f.write("\n")
 
        f.write(
            'yocoLogInfo, '
            '"Ignore bad baseline %s for ksi_Gem block %i";\n'
            % (
                bad_baseline,
                block_i + 1
            )
        )
 
 
        f.write(
            "startend = [%.10f, %.10f];\n"
            % (
                start,
                end
            )
        )
 
 
        f.write(
            'station = "*%s*";\n'
            % bad_baseline
        )
 
 
        f.write(
            "oiFitsFlagOiData, "
            "oiWave, oiArray, oiVis2, oiT3, oiVis, "
            "base=station, "
            "tlimit=startend;\n"
        )
 
 
print(
    "\nAdded %i ksi_Gem intervals"
    % len(ksi_intervals)
)
 
 
# =============================================================================
# 16. PRINT FINAL PNDRS SCRIPT
# =============================================================================
 
print("\n" + "=" * 79)
print("FINAL PNDRS SCRIPT")
print("=" * 79)
 
 
with open(pndrs_script, "r") as f:
 
    print(
        f.read()
    )
 
 
# =============================================================================
# 17. RUN CALIBRATION
# =============================================================================
 
print("\n" + "=" * 79)
print("RUN CALIBRATION")
print("=" * 79)
 
 
rpndrs.run_n_bootstraps(
    sequences,
    complete_sequences,
    base_path,
    tgt_info,
    n_pred_ldd,
    e_pred_ldd,
    n_bootstraps,
    results_path,
    run_local=run_local,
    already_calibrated=already_calibrated,
    do_random_ifg_sampling=do_random_ifg_sampling
)
 
 
# =============================================================================
# 18. SHOW CALIBRATED FILES
# =============================================================================
 
print("\n" + "=" * 79)
print("CALIBRATED FILES")
print("=" * 79)
 
 
calibrated_files = sorted(
    glob.glob(
        os.path.join(
            calibration_folder,
            "*oidataCalibrated.fits"
        )
    )
)
 
 
print(
    "\nCalibrated files generated:",
    len(calibrated_files)
)
 
 
for filename in calibrated_files:
 
    print(
        os.path.basename(filename)
    )
 
 
# =============================================================================
# 19. SHOW FILES COPIED TO RESULTS
# =============================================================================
 
print("\n" + "=" * 79)
print("RESULTS")
print("=" * 79)
 
 
result_files = sorted(
    glob.glob(
        os.path.join(
            results_path,
            "*.fits"
        )
    )
)
 
 
print(
    "\nResult files:",
    len(result_files)
)
 
 
for filename in result_files:
 
    print(
        os.path.basename(filename)
    )
 
 
print("\n" + "=" * 79)
print("FINISHED")
print("=" * 79)
 
print("\nReduced data remain in:")
print(reduced_folder)
 
print("\nCalibration performed in:")
print(calibration_folder)
 
print("\nFinal copied files are in:")
print(results_path)


