from __future__ import division, print_function
 
import os
import glob
import numpy as np
 
from astropy.io import fits
 
import reach.utils as rutils
import reach.pndrs as rpndrs
import reach.diameters as rdiam
 
 
# =============================================================================
# PARAMETERS
# =============================================================================
 
lb_pc = 70
use_plx_systematic = False
assign_default_uncertainties = True
 
# Only one bootstrap for this test
n_bootstraps = 5
 
pred_ldd_col = "LDD_pred"
e_pred_ldd_col = "e_LDD_pred"
 
do_gaussian_diam_sampling = True
do_random_ifg_sampling = True
 
 
# =============================================================================
# PATHS
# =============================================================================
 
# This is the path you are using in Chapman
base_path = (
    "/home/ihernand/Desktop/reach/"
    "complete_sequences/%s_v3.94_abcd/"
)
 
 
# Separate folder only for this test
results_path = (
    "/home/ihernand/Desktop/reach/"
    "results/test_first_calibration_HR2998/"
)
 
 
if not os.path.exists(results_path):
    os.makedirs(results_path)
 
 
# Remove ONLY previous files from this test-results folder
# This does NOT touch the reduced data
old_test_files = glob.glob(results_path + "*")
 
for f in old_test_files:
 
    if os.path.isfile(f):
        os.remove(f)
 
 
# =============================================================================
# TARGET INFO
# =============================================================================
 
tgt_info = rutils.initialise_tgt_info(
    assign_default_uncertainties,
    lb_pc,
    use_plx_systematic
)
 
 
# =============================================================================
# LOAD SEQUENCES
# =============================================================================
 
complete_sequences, sequences = rutils.load_sequence_logs()
 
 
# =============================================================================
# KEEP ONLY HR2998 FAINT
# =============================================================================
 
seq_test = (104, "HR_2998", "faint")
 
 
print("\n" + "=" * 79)
print("SEQUENCE TO TEST")
print("=" * 79)
 
print(seq_test)
 
 
# Make sure sequence exists
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
 
 
# Keep only this sequence
complete_sequences = {
    seq_test: complete_sequences[seq_test]
}
 
sequences = {
    seq_test: sequences[seq_test]
}
 
 
print("\nSequence:")
print(sequences[seq_test])
 
print("\nNight:")
print(complete_sequences[seq_test][0])
 
 
# =============================================================================
# 1. CHECK SCIENCE/CALIBRATOR STATUS IN tgt_info
# =============================================================================
 
print("\n" + "=" * 79)
print("SCIENCE VALUES IN tgt_info")
print("=" * 79)
 
 
for star in sequences[seq_test]:
 
    matched_id = rpndrs.match_target_name(
        tgt_info,
        star,
        verbose=False
    )
 
    if matched_id is None:
 
        print(
            "%-20s NOT FOUND"
            % star
        )
 
        continue
 
 
    science = tgt_info.loc[matched_id]["Science"]
 
    if bool(science):
        expected_role = "SCI"
    else:
        expected_role = "CAL"
 
 
    print(
        "%-20s matched=%-20s Science=%-5s expected=%s"
        % (
            star,
            matched_id,
            str(science),
            expected_role
        )
    )
 
 
# =============================================================================
# 2. SAMPLE PREDICTED DIAMETERS
# =============================================================================
 
n_pred_ldd, e_pred_ldd = rdiam.sample_n_pred_ldd(
    tgt_info,
    n_bootstraps,
    pred_ldd_col,
    e_pred_ldd_col,
    do_gaussian_diam_sampling
)
 
 
# =============================================================================
# FIRST BOOTSTRAP ONLY
# =============================================================================
 
bs_i = 0
 
 
print("\n" + "=" * 79)
print("BOOTSTRAP %i" % bs_i)
print("=" * 79)
 
 
# =============================================================================
# 3. INITIALISE INTERFEROGRAMS
# =============================================================================
 
rpndrs.initialise_interferograms(
    complete_sequences,
    base_path,
    n_ifg=5,
    do_random_ifg_sampling=do_random_ifg_sampling
)
 
 
# =============================================================================
# 4. WRITE oiDiam
# =============================================================================
 
nights = rpndrs.save_nightly_ldd(
    sequences,
    complete_sequences,
    tgt_info,
    n_pred_ldd.iloc[bs_i],
    e_pred_ldd,
    base_path
)
 
 
# =============================================================================
# 5. INSPECT oiDiam BEFORE CALIBRATION
# =============================================================================
 
print("\n" + "=" * 79)
print("OI_DIAM BEFORE PNDRS CALIBRATION")
print("=" * 79)
 
 
for night in nights.keys():
 
    oiDiam = (
        (base_path % night)
        + "%s/%s_oiDiam.fits"
        % (
            night,
            night
        )
    )
 
 
    print("\nNight:", night)
    print("oiDiam:", oiDiam)
 
 
    with fits.open(oiDiam) as hdul:
 
        data = hdul["OIU_DIAM"].data
 
 
        for row in data:
 
            target = row["TARGET"]
            iscal = row["ISCAL"]
 
 
            if isinstance(target, bytes):
 
                try:
                    target = target.decode("utf-8")
 
                except Exception:
                    pass
 
 
            if iscal == 1:
                role = "CAL"
 
            elif iscal == 0:
                role = "SCI"
 
            else:
                role = "???"
 
 
            print(
                "%-20s ISCAL = %s   --> %s"
                % (
                    target,
                    str(iscal),
                    role
                )
            )
 
 
# =============================================================================
# 6. CHECK THAT THIS REALLY IS A CLEAN FIRST PNDRS RUN
# =============================================================================
 
print("\n" + "=" * 79)
print("CALIBRATED FILES BEFORE FIRST PNDRS RUN")
print("=" * 79)
 
 
for night in nights.keys():
 
    obs_folder = (
        (base_path % night)
        + "%s/"
        % night
    )
 
 
    old_calibrated = glob.glob(
        obs_folder + "*oidataCalibrated.fits"
    )
 
    old_calibrated.sort()
 
 
    if len(old_calibrated) == 0:
 
        print(
            "No calibrated files found -> CLEAN FIRST RUN"
        )
 
 
    else:
 
        print("\nWARNING:")
        print(
            "Calibrated files already exist."
        )
 
        print(
            "This is NOT a clean first calibration."
        )
 
 
        for f in old_calibrated:
 
            print(
                "  %s"
                % os.path.basename(f)
            )
 
 
        print("\nSTOPPING TEST.")
        print(
            "Reduce the night again before running this test."
        )
 
        raise SystemExit
 
 
# =============================================================================
# 7. INSPECT INPUT FITS BEFORE PNDRS
# =============================================================================
 
print("\n" + "=" * 79)
print("INPUT FITS BEFORE PNDRS")
print("=" * 79)
 
 
for night in nights.keys():
 
    obs_folder = (
        (base_path % night)
        + "%s/"
        % night
    )
 
 
    input_files = glob.glob(
        obs_folder + "PIONI*_oidata.fits"
    )
 
    input_files.sort()
 
 
    print(
        "\nFound %i input files"
        % len(input_files)
    )
 
 
    for f in input_files:
 
        try:
 
            with fits.open(f) as hdul:
 
                hdr = hdul[0].header
 
 
                obj = hdr.get(
                    "OBJECT",
                    "UNKNOWN"
                )
 
 
                targ_name = hdr.get(
                    "HIERARCH ESO OCS TARG NAME",
                    "UNKNOWN"
                )
 
 
                dpr_catg = hdr.get(
                    "HIERARCH ESO DPR CATG",
                    "UNKNOWN"
                )
 
 
                dpr_type = hdr.get(
                    "HIERARCH ESO DPR TYPE",
                    "UNKNOWN"
                )
 
 
                print(
                    "\n%s"
                    % os.path.basename(f)
                )
 
                print(
                    "   OBJECT       = %s"
                    % obj
                )
 
                print(
                    "   TARG NAME    = %s"
                    % targ_name
                )
 
                print(
                    "   DPR CATG     = %s"
                    % dpr_catg
                )
 
                print(
                    "   DPR TYPE     = %s"
                    % dpr_type
                )
 
 
        except Exception as e:
 
            print(
                "\nCould not inspect:"
            )
 
            print(
                "   %s"
                % f
            )
 
            print(
                "   %s"
                % str(e)
            )
 
 
# =============================================================================
# 8. RUN PNDRS CALIBRATE
# =============================================================================
 
print("\n" + "=" * 79)
print("RUNNING PNDRS CALIBRATE")
print("=" * 79)
 
 
for night in nights.keys():
 
    obs_folder = (
        (base_path % night)
        + "%s/"
        % night
    )
 
 
    print(
        "\nCalibrating: %s"
        % night
    )
 
 
    log_file = os.path.join(
        obs_folder,
        "cal_test_first_run.txt"
    )
 
 
    command = (
        "(cd %s; pndrsCalibrate > cal_test_first_run.txt 2>&1)"
        % obs_folder
    )
 
 
    return_code = os.system(
        command
    )
 
 
    print(
        "\npndrsCalibrate return code = %s"
        % str(return_code)
    )
 
 
    # =========================================================================
    # 9. SHOW RAW PNDRS OUTPUT
    # =========================================================================
 
    print("\n" + "-" * 79)
    print("RAW PNDRS OUTPUT")
    print("-" * 79)
 
 
    calibrated_files = glob.glob(
        obs_folder + "*oidataCalibrated.fits"
    )
 
    calibrated_files.sort()
 
 
    if len(calibrated_files) == 0:
 
        print(
            "No calibrated files generated."
        )
 
 
    for f in calibrated_files:
 
        fname = os.path.basename(f)
 
 
        if "_SCI_" in fname:
 
            role = "SCI"
 
 
        elif "_CAL_" in fname:
 
            role = "CAL"
 
 
        else:
 
            role = "???"
 
 
        print(
            "%-5s %s"
            % (
                role,
                fname
            )
        )
 
 
    # =========================================================================
    # 10. LOOK FOR TARGETS IN PNDRS LOG
    # =========================================================================
 
    print("\n" + "-" * 79)
    print("PNDRS LOG: HR2998 AND CALIBRATORS")
    print("-" * 79)
 
 
    keywords = [
        "HR2998",
        "HR_2998",
        "HD62644",
        "HD61574",
        "HR3069",
        "HR_3069",
        "HD64181",
        "HD65187"
    ]
 
 
    if os.path.exists(log_file):
 
        with open(log_file, "r") as f:
 
            log_lines = f.readlines()
 
 
        found_lines = set()
 
 
        for i, line in enumerate(log_lines):
 
            if any(
                key.lower() in line.lower()
                for key in keywords
            ):
 
                start = max(
                    0,
                    i - 2
                )
 
                end = min(
                    len(log_lines),
                    i + 3
                )
 
 
                for j in range(start, end):
 
                    found_lines.add(j)
 
 
        if len(found_lines) == 0:
 
            print(
                "No references found in PNDRS log."
            )
 
 
        else:
 
            for j in sorted(found_lines):
 
                print(
                    log_lines[j].rstrip()
                )
 
 
    else:
 
        print(
            "Log file not found:"
        )
 
        print(
            log_file
        )
 
 
    # =========================================================================
    # 11. RUN REACH move_sci_oifits
    # =========================================================================
 
    print("\n" + "=" * 79)
    print("RUNNING move_sci_oifits")
    print("=" * 79)
 
 
    rpndrs.move_sci_oifits(
        obs_folder,
        results_path,
        bs_i,
        tgt_info=tgt_info
    )
 
 
# =============================================================================
# 12. SHOW FINAL REACH OUTPUT
# =============================================================================
 
print("\n" + "=" * 79)
print("FINAL REACH OUTPUT")
print("=" * 79)
 
 
final_files = glob.glob(
    results_path + "*oidataCalibrated*.fits"
)
 
final_files.sort()
 
 
if len(final_files) == 0:
 
    print(
        "No files found in results_path."
    )
 
 
for f in final_files:
 
    fname = os.path.basename(f)
 
 
    if "_SCI_" in fname:
 
        role = "SCI"
 
 
    elif "_CAL_" in fname:
 
        role = "CAL"
 
 
    else:
 
        role = "???"
 
 
    print(
        "%-5s %s"
        % (
            role,
            fname
        )
    )
 
 
# =============================================================================
# 13. COMPARE RAW PNDRS CLASSIFICATION WITH tgt_info
# =============================================================================
 
print("\n" + "=" * 79)
print("PNDRS vs EXPECTED FROM tgt_info")
print("=" * 79)
 
 
for night in nights.keys():
 
    obs_folder = (
        (base_path % night)
        + "%s/"
        % night
    )
 
 
    calibrated_files = glob.glob(
        obs_folder + "*oidataCalibrated.fits"
    )
 
    calibrated_files.sort()
 
 
    for f in calibrated_files:
 
        fname = os.path.basename(f)
 
 
        # Ignore ALL and individual PIONI files
        if "_SCI_" not in fname and "_CAL_" not in fname:
            continue
 
 
        target_raw = (
            rpndrs.get_target_from_calibrated_filename(f)
        )
 
 
        matched_id = rpndrs.match_target_name(
            tgt_info,
            target_raw,
            verbose=False
        )
 
 
        if "_SCI_" in fname:
            pndrs_role = "SCI"
 
        else:
            pndrs_role = "CAL"
 
 
        if matched_id is None:
 
            print(
                "%-20s PNDRS=%-4s expected=UNKNOWN"
                % (
                    target_raw,
                    pndrs_role
                )
            )
 
            continue
 
 
        is_science = bool(
            tgt_info.loc[matched_id]["Science"]
        )
 
 
        if is_science:
            expected_role = "SCI"
 
        else:
            expected_role = "CAL"
 
 
        if pndrs_role == expected_role:
            status = "OK"
        else:
            status = "DISCORDANCE"
 
 
        print(
            "%-20s PNDRS=%-4s expected=%-4s  %s"
            % (
                target_raw,
                pndrs_role,
                expected_role,
                status
            )
        )
 
 
print("\n" + "=" * 79)
print("TEST FINISHED")
print("=" * 79)
 
print("\nRaw PNDRS data:")
print(
    "/home/ihernand/Desktop/reach/"
    "complete_sequences/2019-11-25_v3.94_abcd/2019-11-25/"
)
 
print("\nCorrected/copied REACH results:")
print(results_path)
