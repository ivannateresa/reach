from __future__ import division, print_function

import os
import time
import glob
import sys

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


# ============================================================
# BOOTSTRAPPING PARAMETERS
# ============================================================

lb_pc = 70

use_plx_systematic = False

do_random_ifg_sampling = True

do_gaussian_diam_sampling = True

assign_default_uncertainties = True

force_claret_params = False

# The shell runner exports N_BOOTSTRAPS so that the pipeline and analysis
# always use the same value.  Direct runs default to 100.
n_bootstraps = int(
    os.environ.get(
        "N_BOOTSTRAPS",
        "1"
    )
)

pred_ldd_col = "LDD_pred"

e_pred_ldd_col = "e_LDD_pred"

n_calib_runs = 1

calib_run_i = 0


# ============================================================
# GAMLEP EXPERIMENT CONFIGURATION
# ============================================================
#
# gam Lep has TWO faint sequences:
#
#   P104 faint: HD38579, HD39913, HR2090
#   P106 faint: HD38295, HD42747
#
# Both sequences are treated independently.  The command-line
# calibrator arguments are therefore P104 and P106, NOT
# bright and faint.
#
# Usage:
#
#   python pipeline_gamlep_tests.py ALL
#
#   python pipeline_gamlep_tests.py NO_BL
#
#   python pipeline_gamlep_tests.py NO_CAL CAL_P104 CAL_P106
#
#   python pipeline_gamlep_tests.py NO_BL_NO_CAL CAL_P104 CAL_P106
#
# Examples:
#
#   python pipeline_gamlep_tests.py NO_CAL HD38579 HD38295
#   python pipeline_gamlep_tests.py NO_BL_NO_CAL HD39913 HD42747
#   python pipeline_gamlep_tests.py NO_CAL NONE HD38295
#
# NONE means that no calibrator is removed from that period.
# ============================================================


# ------------------------------------------------------------
# Experiment mode
# ------------------------------------------------------------

if len(sys.argv) > 1:

    experiment_mode = str(
        sys.argv[1]
    ).upper()

else:

    experiment_mode = "ALL"


valid_modes = [
    "ALL",
    "NO_BL",
    "NO_CAL",
    "NO_BL_NO_CAL"
]


if experiment_mode not in valid_modes:

    raise ValueError(
        "Mode must be one of: %s"
        % ", ".join(valid_modes)
    )


# ------------------------------------------------------------
# Are we removing baselines?
# ------------------------------------------------------------

remove_bad_baselines = (
    experiment_mode
    in [
        "NO_BL",
        "NO_BL_NO_CAL"
    ]
)


# In the local modified REACH version:
#
#   use_bad_baselines = True
#       -> read data/bad_baselines.txt and flag its entries
#
#   use_bad_baselines = False
#       -> do not apply the bad-baseline log

use_bad_baselines = (
    remove_bad_baselines
)


# ------------------------------------------------------------
# Are we removing calibrators?
# ------------------------------------------------------------

remove_calibrators_by_sequence = (
    experiment_mode
    in [
        "NO_CAL",
        "NO_BL_NO_CAL"
    ]
)


# ------------------------------------------------------------
# P104 calibrator to remove
# ------------------------------------------------------------

if (
    remove_calibrators_by_sequence
    and len(sys.argv) > 2
):

    P104_CAL_TO_REMOVE = str(
        sys.argv[2]
    )

else:

    P104_CAL_TO_REMOVE = None


# ------------------------------------------------------------
# P106 calibrator to remove
# ------------------------------------------------------------

if (
    remove_calibrators_by_sequence
    and len(sys.argv) > 3
):

    P106_CAL_TO_REMOVE = str(
        sys.argv[3]
    )

else:

    P106_CAL_TO_REMOVE = None


# ------------------------------------------------------------
# Convert NONE to Python None
# ------------------------------------------------------------

if P104_CAL_TO_REMOVE is not None:

    if (
        P104_CAL_TO_REMOVE
        .strip()
        .upper()
        == "NONE"
    ):

        P104_CAL_TO_REMOVE = None


if P106_CAL_TO_REMOVE is not None:

    if (
        P106_CAL_TO_REMOVE
        .strip()
        .upper()
        == "NONE"
    ):

        P106_CAL_TO_REMOVE = None


# ============================================================
# CALIBRATOR REMOVAL CONFIGURATION
# ============================================================

CAL_TO_REMOVE_BY_PERIOD = {

    "p104":
        P104_CAL_TO_REMOVE,

    "p106":
        P106_CAL_TO_REMOVE

}


P104_CALIBRATORS = [
    "HD38579",
    "HD39913",
    "HR2090"
]


P106_CALIBRATORS = [
    "HD38295",
    "HD42747"
]


# ============================================================
# Reproducible bootstraps
# ============================================================

np.random.seed(
    12345
)


# ============================================================
# Print experiment
# ============================================================

print("")

print("=" * 79)

print("GAMLEP EXPERIMENT")

print("=" * 79)

print(
    "experiment_mode       :",
    experiment_mode
)

print(
    "n_bootstraps          :",
    n_bootstraps
)

print(
    "remove bad baselines  :",
    remove_bad_baselines
)

print(
    "use bad baselines     :",
    use_bad_baselines
)

print(
    "remove calibrators    :",
    remove_calibrators_by_sequence
)

print(
    "P104 cal removed      :",
    P104_CAL_TO_REMOVE
)

print(
    "P106 cal removed      :",
    P106_CAL_TO_REMOVE
)

print("=" * 79)


# ============================================================
# Paths
# ============================================================

base_path = (
    "/home2/ihernand/Desktop/reach/"
    "complete_sequences/%s_v3.94_abcd/"
)


str_date = time.strftime(
    "%y-%m-%d"
)


if remove_bad_baselines:

    baseline_mode = (
        "BAD_BL_REMOVED"
    )

else:

    baseline_mode = (
        "ALL_BASELINES"
    )


save_data_path = (
    "/home2/ihernand/Desktop/reach/"
    "data/outputs"
)


if not os.path.exists(
        save_data_path):

    os.makedirs(
        save_data_path
    )


# ============================================================
# Bolometric corrections
# ============================================================

bc_path = (
    "/home2/ihernand/Desktop/reach/"
    "bolometric-corrections"
)


band_mask = [
    1,
    1,
    1,
    0,
    0
]


# ============================================================
# Calibrator diagnostics
# ============================================================

calibrate_calibrators = False

test_all_cals = False


# ============================================================
# Pipeline execution options
# ============================================================

run_local = False

already_calibrated = False


print(
    "\nBeginning calibration and fitting run."
)


print(
    " - n_bootstraps\t\t\t=\t%i"
    % n_bootstraps
)


print(
    " - do_random_ifg_sampling\t=\t%s"
    % do_random_ifg_sampling
)


print(
    " - do_gaussian_diam_sampling\t=\t%s"
    % do_gaussian_diam_sampling
)


# ============================================================
# Helper functions
# ============================================================

def normalise_name(name):

    return (
        str(name)
        .replace("_", "")
        .replace(" ", "")
        .lower()
    )


def clean_for_filename(name):

    if name is None:

        return "NONE"

    name = str(name)

    name = name.replace(
        " ",
        ""
    )

    name = name.replace(
        "_",
        ""
    )

    name = name.replace(
        "/",
        ""
    )

    name = name.replace(
        "\\",
        ""
    )

    return name


def same_name(name1, name2):

    return (
        normalise_name(name1)
        ==
        normalise_name(name2)
    )


def identify_gamlep_period(seq_key, sequence_targets, sequence_info):

    clean_targets = set(
        normalise_name(target)
        for target in sequence_targets
    )

    # Prefer unique calibrators because both gam Lep sequences
    # are labelled faint.
    p104_unique = [
        "HD38579",
        "HD39913"
    ]

    p106_unique = [
        "HD38295",
        "HD42747"
    ]

    if any(
            normalise_name(cal) in clean_targets
            for cal in p104_unique):

        return "p104"


    if any(
            normalise_name(cal) in clean_targets
            for cal in p106_unique):

        return "p106"


    # Fallback to the night stored by REACH.
    try:

        night = str(
            sequence_info[0]
        )

        if night.startswith("2020-"):
            return "p104"

        if night.startswith("2021-"):
            return "p106"

    except Exception:

        pass


    raise RuntimeError(
        "Could not identify gam Lep period for %s; sequence=%s"
        % (
            str(seq_key),
            str(sequence_targets)
        )
    )


# ============================================================
# Import target information
# ============================================================

tgt_info = rutils.initialise_tgt_info(
    assign_default_uncertainties,
    lb_pc,
    use_plx_systematic
)


# ============================================================
# CONTROLLED GAMLEP EXPERIMENT
#
# Start every experiment with all calibrators enabled.
# Otherwise, calibrators marked Quality="BAD" in tgt_info are
# removed globally by REACH, even when experiment_mode == "ALL".
#
# NO_CAL and NO_BL_NO_CAL will still remove only the calibrators
# explicitly requested on the command line.
# ============================================================

quality_bad_mask = (
    tgt_info["Quality"]
    .astype(str)
    .str.strip()
    .str.upper()
    == "BAD"
)


globally_restored = []


for target_id, row in tgt_info[quality_bad_mask].iterrows():

    if "Primary" in tgt_info.columns:

        restored_name = str(
            row["Primary"]
        )

    else:

        restored_name = str(
            target_id
        )


    globally_restored.append(
        restored_name
    )


tgt_info.loc[
    quality_bad_mask,
    "Quality"
] = "GOOD"


print("")

print("=" * 79)

print(
    "GLOBAL QUALITY EXCLUSIONS DISABLED "
    "FOR GAMLEP TESTS"
)

print("=" * 79)


if len(globally_restored) == 0:

    print("No Quality=BAD entries found.")

else:

    for restored_name in globally_restored:

        print(
            "  restored as GOOD: %s"
            % restored_name
        )


print("=" * 79)


# ============================================================
# IMPORTANT
#
# Quality exclusions were disabled above so that ALL really
# means all calibrators. Sequence-specific exclusions are made
# only in NO_CAL and NO_BL_NO_CAL.
# ============================================================


# ============================================================
# CALIBRATORS ALREADY BAD GLOBALLY
# ============================================================

bad_calibrators = []


for target_id, row in tgt_info.iterrows():

    if row["Quality"] == "BAD":

        if "Primary" in tgt_info.columns:

            name = str(
                row["Primary"]
            )

        else:

            name = str(
                target_id
            )


        bad_calibrators.append(
            name
        )


print("")

print("=" * 79)

print(
    "GLOBAL CALIBRATORS ALREADY MARKED BAD"
)

print("=" * 79)


if len(bad_calibrators) == 0:

    print("None")

else:

    for cal in bad_calibrators:

        print(
            "  %s"
            % cal
        )


print("=" * 79)


# ============================================================
# Result-folder calibrator label
# ============================================================

calibrator_parts = []


if remove_calibrators_by_sequence:

    if P104_CAL_TO_REMOVE is not None:

        calibrator_parts.append(
            "P104_NO_%s"
            % clean_for_filename(
                P104_CAL_TO_REMOVE
            )
        )


    if P106_CAL_TO_REMOVE is not None:

        calibrator_parts.append(
            "P106_NO_%s"
            % clean_for_filename(
                P106_CAL_TO_REMOVE
            )
        )


if len(bad_calibrators) > 0:

    for cal in bad_calibrators:

        calibrator_parts.append(
            "GLOBAL_NO_%s"
            % clean_for_filename(
                cal
            )
        )


if len(calibrator_parts) == 0:

    calibrator_mode = (
        "ALL_CALS"
    )

else:

    calibrator_mode = "_".join(
        calibrator_parts
    )


# ============================================================
# Results folder
# ============================================================

results_folder = (
    "%s_GAMLEP_i%i_%s_%s_%s"
    % (
        str_date,
        n_bootstraps,
        experiment_mode,
        baseline_mode,
        calibrator_mode
    )
)


results_root = (
    "/home2/ihernand/Desktop/reach/results/"
)


if not os.path.exists(
        results_root):

    os.mkdir(
        results_root
    )


results_path = os.path.join(
    results_root,
    results_folder
) + "/"


if not os.path.exists(
        results_path):

    os.mkdir(
        results_path
    )


# ============================================================
# Configuration summary
# ============================================================

print("")

print("=" * 79)

print("RUN CONFIGURATION")

print("=" * 79)

print(
    "results_folder        :",
    results_folder
)

print(
    "baseline_mode         :",
    baseline_mode
)

print(
    "global bad cals       :",
    bad_calibrators
)

print(
    "P104 cal removed      :",
    P104_CAL_TO_REMOVE
)

print(
    "P106 cal removed      :",
    P106_CAL_TO_REMOVE
)

print(
    "random IFG sampling   :",
    do_random_ifg_sampling
)

print(
    "experiment_mode       :",
    experiment_mode
)

print("=" * 79)


# ============================================================
# Save tgt_info
# ============================================================

print(
    "\n",
    "-" * 79,
    "\n",
    "\tSave tgt_info in Data\n",
    "-" * 79
)


tgt_info.to_csv(
    os.path.join(
        results_path,
        "tgt_info_used.csv"
    )
)


# ============================================================
# Initial target diagnostic
# ============================================================

print(
    "\n",
    "-" * 79,
    "\n",
    "\tDiagnostic\n",
    "-" * 79
)


cols_check = [

    "Primary",
    "Science",
    "SpT_simple",

    "BTmag",
    "VTmag",

    "Bmag",
    "Vmag",

    "Plx",
    "e_Plx",

    "Plx_alt",
    "e_Plx_alt",

    "Dist",
    "e_Dist",

    "FeH_rel",

    "BTmag_dr",
    "VTmag_dr",

    "LDD_pred",
    "teff_casagrande"

]


nan_rows = tgt_info[

    tgt_info[
        [
            "Dist",
            "e_Dist",
            "Bmag",
            "Vmag",
            "LDD_pred",
            "teff_casagrande"
        ]
    ]

    .isnull()

    .any(
        axis=1
    )

]


for star, row in nan_rows.iterrows():

    print(
        "\n",
        star,
        row["Primary"]
    )


    if (
        pd.isnull(
            row["BTmag"]
        )
        or
        pd.isnull(
            row["VTmag"]
        )
    ):

        print(
            "  Falta BTmag o VTmag "
            "-> no se puede calcular Bmag/Vmag"
        )


    if (
        pd.isnull(
            row["Plx"]
        )
        and
        pd.isnull(
            row["Plx_alt"]
        )
    ):

        print(
            "  Falta Plx y Plx_alt "
            "-> no se puede calcular Dist"
        )


    if (
        pd.isnull(
            row["e_Plx"]
        )
        and
        pd.isnull(
            row["e_Plx_alt"]
        )
    ):

        print(
            "  Falta e_Plx/e_Plx_alt "
            "-> no se puede calcular e_Dist"
        )


    if pd.isnull(
            row["FeH_rel"]):

        print(
            "  Falta FeH_rel "
            "-> teff_casagrande queda NaN"
        )


diagnostic = dig.save_initialise_diagnostics(

    tgt_info,

    outdir=(
        "/home2/ihernand/Desktop/reach/"
        "data/diagnostics"
    ),

    prefix="nan_diagnostic"

)


# ============================================================
# Sample predicted diameters
# ============================================================

if rutils.sampling_already_done(
        results_folder,
        force_claret_params):


    print(
        "Sampling already done, loading..."
    )


    n_pred_ldd, e_pred_ldd = (
        rutils.load_sampled_ldd(
            results_folder
        )
    )


else:


    print(
        "Sampling not yet done, doing now..."
    )


    n_pred_ldd, e_pred_ldd = (
        rdiam.sample_n_pred_ldd(

            tgt_info,

            n_bootstraps,

            pred_ldd_col,

            e_pred_ldd_col,

            do_gaussian_diam_sampling
        )
    )


    rutils.save_sampled_ldd(

        n_pred_ldd,

        e_pred_ldd,

        results_folder
    )


    sampled_sci_params = (
        rparam.sample_all(

            tgt_info,

            n_bootstraps,

            bc_path,

            force_claret_params,

            band_mask
        )
    )


    rutils.save_sampled_params(

        sampled_sci_params,

        results_folder
    )


    # --------------------------------------------------------
    # Save sampled predicted diameters as CSV
    # --------------------------------------------------------

    n_pred_ldd_csv = os.path.join(
        save_data_path,
        "n_pred_ldd.csv"
    )


    e_pred_ldd_csv = os.path.join(
        save_data_path,
        "e_pred_ldd.csv"
    )


    pd.DataFrame(
        n_pred_ldd
    ).to_csv(
        n_pred_ldd_csv,
        index=True
    )


    pd.DataFrame(
        e_pred_ldd
    ).to_csv(
        e_pred_ldd_csv,
        index=True
    )


    print(
        "Saved sampled LDD:"
    )

    print(
        n_pred_ldd_csv
    )

    print(
        e_pred_ldd_csv
    )


# ============================================================
# Load observing logs
# ============================================================

complete_sequences, sequences = (
    rutils.load_sequence_logs()
)


# ============================================================
# KEEP ONLY GAMLEP
# ============================================================

gamlep_keys = [

    key

    for key
    in complete_sequences.keys()

    if (
        normalise_name(
            key[1]
        )
        ==
        normalise_name(
            "gam_Lep"
        )
    )

]


print("")

print("=" * 79)

print("GAMLEP SEQUENCES FOUND")

print("=" * 79)


for key in sorted(
        gamlep_keys):

    print(
        "%s  night=%s"
        % (
            str(key),
            complete_sequences[key][0]
        )
    )


print("=" * 79)


if len(gamlep_keys) == 0:

    raise RuntimeError(
        "No gam_Lep sequences found "
        "in sequence logs"
    )


complete_sequences = {

    key:
        complete_sequences[key]

    for key
    in gamlep_keys

}


sequences = {

    key:
        sequences[key]

    for key
    in gamlep_keys

}


# ============================================================
# SHOW ORIGINAL SEQUENCES
# ============================================================

print("")

print("=" * 79)

print(
    "ORIGINAL GAMLEP SEQUENCES"
)

print("=" * 79)


for seq_key in sorted(
        sequences.keys()):

    print("")

    print(
        "KEY:",
        seq_key
    )

    print(
        "NIGHT:",
        complete_sequences[
            seq_key
        ][0]
    )

    print(
        "SEQUENCE BEFORE:",
        sequences[
            seq_key
        ]
    )


print("=" * 79)


# ============================================================
# REMOVE DIFFERENT CALIBRATOR FROM P104 AND P106
# ============================================================

removed_calibrators = []


if remove_calibrators_by_sequence:


    print("")

    print("=" * 79)

    print(
        "SEQUENCE-SPECIFIC CALIBRATOR REMOVAL"
    )

    print("=" * 79)


    for seq_key in sorted(
            sequences.keys()):


        sequence_before = list(
            sequences[
                seq_key
            ]
        )


        sequence_info = list(
            complete_sequences[
                seq_key
            ]
        )


        period_name = identify_gamlep_period(
            seq_key,
            sequence_before,
            sequence_info
        )


        cal_to_remove = (
            CAL_TO_REMOVE_BY_PERIOD.get(
                period_name,
                None
            )
        )


        print("")

        print(
            "Sequence:",
            seq_key
        )

        print(
            "Night:",
            sequence_info[0]
        )

        print(
            "gam Lep period:",
            period_name.upper()
        )

        print(
            "Calibrator to remove:",
            cal_to_remove
        )


        if cal_to_remove is None:

            print(
                "Nothing removed."
            )

            continue


        # Validate that the requested calibrator belongs to this
        # particular sequence.  This prevents accidentally removing
        # the wrong epoch.
        if not any(
                same_name(target, cal_to_remove)
                for target in sequence_before):

            raise RuntimeError(
                "Requested %s calibrator %s is not present in %s"
                % (
                    period_name.upper(),
                    cal_to_remove,
                    str(sequence_before)
                )
            )


        sequence_after = []


        for target in sequence_before:

            if same_name(
                    target,
                    cal_to_remove):

                print(
                    "Removing from sequence list:",
                    target
                )

                continue


            sequence_after.append(
                target
            )


        sequences[
            seq_key
        ] = sequence_after


        print(
            "Sequence BEFORE:",
            sequence_before
        )

        print(
            "Sequence AFTER :",
            sequence_after
        )


        if len(sequence_info) < 3:

            raise RuntimeError(
                "Unexpected complete_sequences structure for %s"
                % str(seq_key)
            )


        obs_blocks = sequence_info[2]

        obs_before = len(
            obs_blocks
        )

        filtered_obs_blocks = []


        for obs in obs_blocks:

            try:

                obs_target = obs[2]

            except Exception:

                filtered_obs_blocks.append(
                    obs
                )

                continue


            if same_name(
                    obs_target,
                    cal_to_remove):

                print(
                    "Removing observing block:",
                    obs_target
                )

                continue



            filtered_obs_blocks.append(
                obs
            )


        sequence_info[2] = filtered_obs_blocks

        complete_sequences[
            seq_key
        ] = sequence_info


        obs_after = len(
            filtered_obs_blocks
        )


        print(
            "Observing blocks:",
            obs_before,
            "->",
            obs_after
        )


        removed_calibrators.append(
            (
                seq_key,
                period_name,
                cal_to_remove
            )
        )


    print("=" * 79)


# ============================================================
# FINAL SEQUENCES AFTER CALIBRATOR REMOVAL
# ============================================================

print("")

print("=" * 79)

print(
    "FINAL GAMLEP SEQUENCES"
)

print("=" * 79)


for seq_key in sorted(
        sequences.keys()):


    print("")

    print(
        "KEY:",
        seq_key
    )

    print(
        "NIGHT:",
        complete_sequences[
            seq_key
        ][0]
    )

    print(
        "SEQUENCE:",
        sequences[
            seq_key
        ]
    )


print("=" * 79)


# ============================================================
# All complete sequences
# ============================================================

print(
    "\n" + "=" * 79
)

print(
    "ALL COMPLETE SEQUENCES TO TEST"
)

print("=" * 79)


for seq in sorted(
        complete_sequences.keys()):


    print(
        "%s  night=%s"
        % (
            seq,
            complete_sequences[
                seq
            ][0]
        )
    )


print(
    "\nTOTAL COMPLETE SEQUENCES: %i"
    % len(
        complete_sequences
    )
)


all_nights = sorted(

    set(

        complete_sequences[
            seq
        ][0]

        for seq
        in complete_sequences

    )

)


print(
    "TOTAL NIGHTS: %i"
    % len(
        all_nights
    )
)


print("=" * 79)


# ============================================================
# Save sequence diagnostic AFTER removal
# ============================================================

sequences_df = (
    dig.save_sequence_logs_diagnostics(

        complete_sequences,

        sequences,

        outdir=(
            "/home2/ihernand/Desktop/reach/"
            "data/diagnostics"
        ),

        prefix="pionier"

    )
)


# ============================================================
# Optional calibrator quality tests
# ============================================================

if calibrate_calibrators:


    print(
        "-" * 79,
        "\nCalibrating Calibrators\n",
        "-" * 79
    )


    rdiag.calibrate_calibrators(

        sequences,

        complete_sequences,

        base_path,

        tgt_info,

        n_pred_ldd,

        e_pred_ldd,

        test_all_cals

    )


    cal_quality = (
        dig.diagnose_calibrator_quality(

            tgt_info,

            cal_folder=(
                "/home2/ihernand/Desktop/"
                "reach/diagnostics/"
            ),

            outdir=(
                "/home2/ihernand/Desktop/"
                "reach/diagnostics/"
            ),

            prefix="calibrator_quality",

            update_tgt_info=False
        )
    )


    cal_plot_summary = (
        dig.plot_final_calibrators(

            tgt_info,

            cal_folder=(
                "/home2/ihernand/Desktop/"
                "reach/diagnostics/"
            ),

            outdir=(
                "/home2/ihernand/Desktop/"
                "reach/diagnostics/"
            ),

            prefix="final_calibrators",

            quality_csv=(
                "/home2/ihernand/Desktop/"
                "reach/diagnostics/"
                "calibrator_quality.csv"
            )

        )
    )


    print(
        "Finished calibrating calibrators"
    )


    sys_exit(
        0
    )


# ============================================================
# Write nightly PNDRS scripts
# ============================================================

#
# IMPORTANT:
#
# complete_sequences has ALREADY been modified above.
#
# Therefore:
#
# bright and faint can now contain different calibrators.
#
# Existing globally BAD calibrators are still handled by
# tgt_info.
#
# ============================================================

if calib_run_i == 0:


    if (
        not run_local
        and
        not already_calibrated
    ):


        rpndrs.save_nightly_pndrs_script(

            complete_sequences,

            tgt_info,

            base_path,

            use_bad_baselines=
                use_bad_baselines

        )


    elif not already_calibrated:


        rpndrs.save_nightly_pndrs_script(

            complete_sequences,

            tgt_info,

            base_path,

            run_local=
                run_local,

            use_bad_baselines=
                use_bad_baselines

        )


# ============================================================
# Split bootstrap runs
# ============================================================

nights = [

    complete_sequences[
        seq
    ][0]

    for seq
    in complete_sequences.keys()

]


nights = list(
    set(
        nights
    )
)


nights.sort()


n_init_seq = len(
    complete_sequences
)


valid_nights = nights


if n_calib_runs != 1:


    n_nights = np.round(

        len(nights)
        /
        n_calib_runs

    ).astype(
        int
    )


    min_night_i = (
        n_nights
        *
        calib_run_i
    )


    max_night_i = (
        n_nights
        *
        (
            calib_run_i
            + 1
        )
    )


    if max_night_i > len(
            nights):

        max_night_i = len(
            nights
        )


    valid_nights = nights[
        min_night_i:
        max_night_i
    ]


    valid_seqs = [

        seq

        for seq
        in complete_sequences.keys()

        if (
            complete_sequences[
                seq
            ][0]
            in valid_nights
        )

    ]


    complete_sequences = {

        seq:
            complete_sequences[
                seq
            ]

        for seq
        in valid_seqs

    }


    sequences = {

        seq:
            sequences[
                seq
            ]

        for seq
        in complete_sequences

    }


# ============================================================
# Run bootstrapping
# ============================================================

print(
    "\n",
    "-" * 79,
    "\n",
    "\tBootstrapping\n",
    "-" * 79
)


print(
    "Bootstrapping run %i/%i"
    % (
        calib_run_i + 1,
        n_calib_runs
    )
)


print(
    "Running on %i/%i sequences "
    "over %i/%i nights"

    % (
        len(
            complete_sequences
        ),

        n_init_seq,

        len(
            valid_nights
        ),

        len(
            nights
        )
    )
)


rpndrs.run_n_bootstraps(

    sequences,

    complete_sequences,

    base_path,

    tgt_info,

    n_pred_ldd,

    e_pred_ldd,

    n_bootstraps,

    results_path,

    run_local=
        run_local,

    already_calibrated=
        already_calibrated,

    do_random_ifg_sampling=
        do_random_ifg_sampling

)


# ============================================================
# Optional bootstrap diagnostics
# ============================================================

run_bootstrap_diagnostics = False


if run_bootstrap_diagnostics:


    bootstrap_diag_path = os.path.join(

        results_path,

        "bootstrap_diagnostics_run_%03i"
        % calib_run_i

    )


    bootstrap_summary = (
        dig.run_n_bootstraps_diagnostic_with_visibility_plots(

            sequences,

            complete_sequences,

            base_path,

            tgt_info,

            n_pred_ldd,

            e_pred_ldd,

            n_bootstraps,

            results_path,

            diag_path=
                bootstrap_diag_path,

            run_local=
                run_local,

            already_calibrated=
                already_calibrated,

            do_random_ifg_sampling=
                do_random_ifg_sampling,

            stop_on_error=False

        )
    )


    print(
        "\nBootstrap diagnostic summary:"
    )


    print(
        bootstrap_summary
    )
