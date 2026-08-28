"""
Focused analysis of the ksi_Gem bootstrap experiments.

This script analyses only ksi_Gem and reuses the sampled parameters
already written by pipeline_ksigem_tests.py.

It does NOT call rparam.sample_all(), so the bolometric-correction
code is not run again.


Usage
-----

ALL:

    python analyse_ksigem_tests.py ALL


NO_BL:

    python analyse_ksigem_tests.py NO_BL


NO_CAL:

    python analyse_ksigem_tests.py NO_CAL HR2426 HR2610

meaning:

    bright -> remove HR2426
    faint  -> remove HR2610


NO_BL_NO_CAL:

    python analyse_ksigem_tests.py NO_BL_NO_CAL HR2426 HR2610


Remove calibrator only from faint:

    python analyse_ksigem_tests.py NO_CAL NONE HR2610


Remove calibrator only from bright:

    python analyse_ksigem_tests.py NO_CAL HR2426 NONE


Optional explicit results folder:

    python analyse_ksigem_tests.py ALL \
        --folder 26-08-28_KSIGEM_i2_ALL_ALL_BASELINES_ALL_CALS
"""


from __future__ import division, print_function


import os
import sys
import glob
import traceback

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


import reach.diameters as rdiam
import reach.plotting as rplt
import reach.utils as rutils


# =============================================================================
# BASIC CONFIGURATION
# =============================================================================

lb_pc = 70

use_plx_systematic = False

assign_default_uncertainties = True

force_claret_params = False


# IMPORTANT:
#
# This must match the number used in pipeline_ksigem_tests.py
#
# For tests:
# n_bootstraps = 2
#
# For final run:
# n_bootstraps = 1000

n_bootstraps = 2


combined_fit = False

fitting_method = "odr"

e_wl_frac = 0.0035


RESULTS_ROOT = (
    "/home2/ihernand/Desktop/reach/results"
)

ANALYSIS_ROOT = (
    "/home2/ihernand/Desktop/reach/analysis_runs"
)


TARGET_NAME = "ksi_Gem"


# =============================================================================
# HELPERS
# =============================================================================

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


def convert_none(value):

    if value is None:

        return None


    value = str(value)


    if value.strip().upper() == "NONE":

        return None


    return value


# =============================================================================
# COMMAND-LINE CONFIGURATION
# =============================================================================

valid_modes = [

    "ALL",

    "NO_BL",

    "NO_CAL",

    "NO_BL_NO_CAL",

]


# -----------------------------------------------------------------------------
# Experiment mode
# -----------------------------------------------------------------------------

if len(sys.argv) < 2:

    experiment_mode = "ALL"

else:

    experiment_mode = str(
        sys.argv[1]
    ).upper()


if experiment_mode not in valid_modes:

    raise ValueError(

        "Mode must be one of: %s"

        % ", ".join(
            valid_modes
        )

    )


# -----------------------------------------------------------------------------
# Separate positional arguments from --folder
# -----------------------------------------------------------------------------

remaining_args = list(
    sys.argv[2:]
)


explicit_results_folder = None


if "--folder" in remaining_args:

    folder_i = remaining_args.index(
        "--folder"
    )


    if folder_i + 1 >= len(
            remaining_args):

        raise ValueError(
            "--folder requires a results-folder name"
        )


    explicit_results_folder = str(
        remaining_args[
            folder_i + 1
        ]
    )


    calibrator_args = remaining_args[
        :folder_i
    ]


else:

    calibrator_args = remaining_args


# =============================================================================
# CALIBRATORS REMOVED PER SEQUENCE
# =============================================================================

BRIGHT_CAL_TO_REMOVE = None

FAINT_CAL_TO_REMOVE = None


if experiment_mode in [
        "NO_CAL",
        "NO_BL_NO_CAL"]:


    if len(calibrator_args) > 0:

        BRIGHT_CAL_TO_REMOVE = (
            convert_none(
                calibrator_args[0]
            )
        )


    if len(calibrator_args) > 1:

        FAINT_CAL_TO_REMOVE = (
            convert_none(
                calibrator_args[1]
            )
        )


CAL_TO_REMOVE_BY_SEQUENCE = {

    "bright":
        BRIGHT_CAL_TO_REMOVE,

    "faint":
        FAINT_CAL_TO_REMOVE,

}


# =============================================================================
# FIND RESULTS FOLDER
# =============================================================================

def find_results_folder():

    # -------------------------------------------------------------------------
    # Explicit folder supplied by user
    # -------------------------------------------------------------------------

    if explicit_results_folder is not None:


        candidate = os.path.join(

            RESULTS_ROOT,

            explicit_results_folder

        )


        if not os.path.isdir(
                candidate):


            raise RuntimeError(

                "Results folder does not exist: %s"

                % candidate

            )


        return explicit_results_folder


    # -------------------------------------------------------------------------
    # Reconstruct naming convention used by pipeline_ksigem_tests.py
    # -------------------------------------------------------------------------

    if experiment_mode in [
            "NO_BL",
            "NO_BL_NO_CAL"]:


        baseline_token = (
            "BAD_BL_REMOVED"
        )


    else:


        baseline_token = (
            "ALL_BASELINES"
        )


    pattern = os.path.join(

        RESULTS_ROOT,

        "*_KSIGEM_i%i_%s_%s_*"

        % (

            n_bootstraps,

            experiment_mode,

            baseline_token

        )

    )


    candidates = [

        path

        for path in glob.glob(
            pattern
        )

        if os.path.isdir(
            path
        )

    ]


    # -------------------------------------------------------------------------
    # Filter using bright calibrator
    # -------------------------------------------------------------------------

    if experiment_mode in [
            "NO_CAL",
            "NO_BL_NO_CAL"]:


        if BRIGHT_CAL_TO_REMOVE is not None:


            bright_token = (
                normalise_name(

                    "BRIGHT_NO_%s"

                    % clean_for_filename(
                        BRIGHT_CAL_TO_REMOVE
                    )

                )
            )


            candidates = [

                path

                for path in candidates

                if bright_token
                in normalise_name(
                    os.path.basename(
                        path
                    )
                )

            ]


        # ---------------------------------------------------------------------
        # Filter using faint calibrator
        # ---------------------------------------------------------------------

        if FAINT_CAL_TO_REMOVE is not None:


            faint_token = (
                normalise_name(

                    "FAINT_NO_%s"

                    % clean_for_filename(
                        FAINT_CAL_TO_REMOVE
                    )

                )
            )


            candidates = [

                path

                for path in candidates

                if faint_token
                in normalise_name(
                    os.path.basename(
                        path
                    )
                )

            ]


    # -------------------------------------------------------------------------
    # No folder found
    # -------------------------------------------------------------------------

    if len(candidates) == 0:


        raise RuntimeError(

            "No matching results folder found.\n"
            "Pattern: %s\n"
            "Mode: %s\n"
            "Bright calibrator removed: %s\n"
            "Faint calibrator removed: %s"

            % (

                pattern,

                experiment_mode,

                BRIGHT_CAL_TO_REMOVE,

                FAINT_CAL_TO_REMOVE

            )

        )


    # -------------------------------------------------------------------------
    # Use newest matching folder
    # -------------------------------------------------------------------------

    candidates.sort(

        key=os.path.getmtime,

        reverse=True

    )


    if len(candidates) > 1:


        print("")

        print(
            "Multiple matching folders found."
        )

        print(
            "Using newest:"
        )


        for path in candidates:


            print(

                "  %s"

                % os.path.basename(
                    path
                )

            )


    return os.path.basename(
        candidates[0]
    )


# =============================================================================
# FILTER KSI GEM SEQUENCES
# =============================================================================

def filter_ksi_sequences(
        complete_sequences,
        sequences):


    ksi_keys = [

        key

        for key
        in complete_sequences.keys()

        if (
            normalise_name(
                key[1]
            )
            ==
            normalise_name(
                TARGET_NAME
            )
        )

    ]


    if len(ksi_keys) == 0:


        raise RuntimeError(

            "No ksi_Gem sequences found "
            "in sequence logs"

        )


    new_complete = {

        key:
            complete_sequences[key]

        for key
        in ksi_keys

    }


    new_sequences = {

        key:
            sequences[key]

        for key
        in ksi_keys

    }


    return (
        new_complete,
        new_sequences
    )


# =============================================================================
# APPLY SAME SEQUENCE-SPECIFIC CALIBRATOR REMOVAL AS PIPELINE
# =============================================================================

def apply_sequence_calibrator_removal(
        complete_sequences,
        sequences):


    if experiment_mode not in [
            "NO_CAL",
            "NO_BL_NO_CAL"]:


        return (
            complete_sequences,
            sequences
        )


    print("")

    print("=" * 79)

    print(
        "APPLYING SEQUENCE-SPECIFIC "
        "CALIBRATOR REMOVAL TO ANALYSIS"
    )

    print("=" * 79)


    for seq_key in sorted(
            sequences.keys()):


        # ---------------------------------------------------------------------
        # bright / faint
        # ---------------------------------------------------------------------

        try:

            sequence_name = str(
                seq_key[2]
            ).lower()


        except Exception:


            print(

                "WARNING: cannot determine "
                "sequence type for %s"

                % str(
                    seq_key
                )

            )


            continue


        cal_to_remove = (
            CAL_TO_REMOVE_BY_SEQUENCE.get(
                sequence_name,
                None
            )
        )


        print("")

        print(
            "Sequence:",
            seq_key
        )

        print(
            "type:",
            sequence_name
        )

        print(
            "calibrator removed:",
            cal_to_remove
        )


        if cal_to_remove is None:

            continue


        # ---------------------------------------------------------------------
        # sequences dictionary
        # ---------------------------------------------------------------------

        sequence_before = list(
            sequences[
                seq_key
            ]
        )


        sequence_after = [

            target

            for target
            in sequence_before

            if not same_name(
                target,
                cal_to_remove
            )

        ]


        sequences[
            seq_key
        ] = sequence_after


        print(
            "sequence before:",
            sequence_before
        )

        print(
            "sequence after :",
            sequence_after
        )


        # ---------------------------------------------------------------------
        # complete_sequences
        # ---------------------------------------------------------------------

        try:

            sequence_info = list(

                complete_sequences[
                    seq_key
                ]

            )


            if len(sequence_info) >= 3:


                obs_blocks = sequence_info[
                    2
                ]


                filtered_obs_blocks = []


                for obs in obs_blocks:


                    try:

                        obs_target = obs[
                            2
                        ]


                    except Exception:


                        filtered_obs_blocks.append(
                            obs
                        )

                        continue


                    if same_name(
                            obs_target,
                            cal_to_remove):


                        print(

                            "removing observing block:",
                            obs_target

                        )


                        continue


                    filtered_obs_blocks.append(
                        obs
                    )


                sequence_info[
                    2
                ] = filtered_obs_blocks


                complete_sequences[
                    seq_key
                ] = sequence_info


        except Exception as error:


            print(

                "WARNING: could not update "
                "complete_sequences for %s: %s"

                % (
                    str(
                        seq_key
                    ),

                    str(
                        error
                    )
                )

            )


    print("")

    print("=" * 79)


    return (
        complete_sequences,
        sequences
    )


# =============================================================================
# FILTER TARGET INFORMATION
# =============================================================================

def filter_ksi_tgt_info(
        tgt_info):


    if "Primary" not in tgt_info.columns:


        raise RuntimeError(

            "tgt_info has no Primary column"

        )


    mask = tgt_info[
        "Primary"
    ].apply(

        lambda value:

        normalise_name(
            value
        )

        ==

        normalise_name(
            TARGET_NAME
        )

    )


    # -------------------------------------------------------------------------
    # Alternative names
    # -------------------------------------------------------------------------

    if np.sum(mask) == 0:


        alternatives = [

            "ksiGem",

            "xiGem",

            "ksi Gem",

        ]


        mask = tgt_info[
            "Primary"
        ].apply(

            lambda value:

            normalise_name(
                value
            )

            in [

                normalise_name(
                    x
                )

                for x
                in alternatives

            ]

        )


    if np.sum(mask) == 0:


        print("")

        print(
            "Available science target names:"
        )


        if "Science" in tgt_info.columns:


            sci_mask = (
                tgt_info[
                    "Science"
                ]
                == True
            )


            print(

                tgt_info.loc[

                    sci_mask,

                    [
                        "Primary"
                    ]

                ]

            )


        raise RuntimeError(

            "Could not identify "
            "ksi_Gem in tgt_info"

        )


    return tgt_info.loc[
        mask
    ].copy()


# =============================================================================
# RUN PLOT SAFELY
# =============================================================================

def run_plot(
        label,
        function,
        *args,
        **kwargs):


    print("")

    print("=" * 79)

    print(
        "Generating plot: %s"
        % label
    )

    print("=" * 79)


    try:


        function(
            *args,
            **kwargs
        )


        print(
            "SUCCESS: %s"
            % label
        )


        return True


    except Exception as error:


        print(
            "FAILED: %s"
            % label
        )


        print(
            "Error: %s"
            % str(
                error
            )
        )


        with open(
                plot_failure_log,
                "a") as handle:


            handle.write(

                "Plot: %s\n"

                % label

            )


            handle.write(

                "Error: %s\n"

                % str(
                    error
                )

            )


            handle.write(
                traceback.format_exc()
            )


            handle.write(

                "\n"
                + "-" * 79
                + "\n\n"

            )


        traceback.print_exc()


        return False


    finally:


        plt.close(
            "all"
        )


# =============================================================================
# LOCATE BOOTSTRAP RESULTS
# =============================================================================

results_folder = (
    find_results_folder()
)


results_path = os.path.join(

    RESULTS_ROOT,

    results_folder

) + "/"


# =============================================================================
# ANALYSIS NAME
# =============================================================================

analysis_name = (
    "KSIGEM_%s"
    % experiment_mode
)


if experiment_mode in [
        "NO_CAL",
        "NO_BL_NO_CAL"]:


    if BRIGHT_CAL_TO_REMOVE is not None:


        analysis_name += (

            "_BRIGHT_NO_%s"

            % normalise_name(
                BRIGHT_CAL_TO_REMOVE
            ).upper()

        )


    if FAINT_CAL_TO_REMOVE is not None:


        analysis_name += (

            "_FAINT_NO_%s"

            % normalise_name(
                FAINT_CAL_TO_REMOVE
            ).upper()

        )


analysis_root = os.path.join(

    ANALYSIS_ROOT,

    results_folder,

    analysis_name

)


plots_output = os.path.join(

    analysis_root,

    "plots"

)


diagnostics_folder = os.path.join(

    analysis_root,

    "diagnostics"

)


for directory in [

        analysis_root,

        plots_output,

        diagnostics_folder]:


    if not os.path.exists(
            directory):


        os.makedirs(
            directory
        )


# Some legacy REACH plotting routines write directly to paper/
if not os.path.exists(
        "paper"):

    os.makedirs(
        "paper"
    )


plot_failure_log = os.path.join(

    diagnostics_folder,

    "plot_failures.txt"

)


with open(
        plot_failure_log,
        "w") as handle:


    handle.write(

        "Failures while generating "
        "ksi_Gem plots\n"

    )


    handle.write(

        "=" * 79
        + "\n\n"

    )


# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

print("")

print("=" * 79)

print(
    "KSI GEM ANALYSIS"
)

print("=" * 79)


print(
    "experiment mode      : %s"
    % experiment_mode
)


print(
    "results folder       : %s"
    % results_folder
)


print(
    "results path         : %s"
    % results_path
)


print(
    "n_bootstraps         : %i"
    % n_bootstraps
)


print(
    "fitting method       : %s"
    % fitting_method
)


print(
    "combined fit         : %s"
    % str(
        combined_fit
    )
)


print(
    "bright cal removed   : %s"

    % (
        BRIGHT_CAL_TO_REMOVE
        if BRIGHT_CAL_TO_REMOVE
        is not None
        else "None"
    )
)


print(
    "faint cal removed    : %s"

    % (
        FAINT_CAL_TO_REMOVE
        if FAINT_CAL_TO_REMOVE
        is not None
        else "None"
    )
)


print(
    "analysis output      : %s"
    % analysis_root
)


print("=" * 79)


# =============================================================================
# LOAD TARGET AND SEQUENCE INFORMATION
# =============================================================================

print("")

print(
    "Loading target information..."
)


tgt_info_all = (
    rutils.initialise_tgt_info(

        assign_default_uncertainties,

        lb_pc,

        use_plx_systematic

    )
)


complete_sequences_all, sequences_all = (
    rutils.load_sequence_logs()
)


complete_sequences, sequences = (
    filter_ksi_sequences(

        complete_sequences_all,

        sequences_all

    )
)


# =============================================================================
# APPLY SAME CALIBRATOR REMOVAL USED BY PIPELINE
# =============================================================================

complete_sequences, sequences = (
    apply_sequence_calibrator_removal(

        complete_sequences,

        sequences

    )
)


tgt_info = (
    filter_ksi_tgt_info(
        tgt_info_all
    )
)


print("")

print(
    "ksi_Gem target row:"
)

print(
    tgt_info
)


print("")

print(
    "ksi_Gem sequences used in analysis:"
)


for key in sorted(
        complete_sequences.keys()):


    print(

        "  %s  night=%s  sequence=%s"

        % (

            str(
                key
            ),

            complete_sequences[
                key
            ][0],

            str(
                sequences[
                    key
                ]
            )

        )

    )


# =============================================================================
# LOAD SAMPLED PARAMETERS CREATED BY PIPELINE
# =============================================================================

print("")

print(
    "Loading sampled stellar parameters..."
)


sampled_sci_params = (
    rutils.load_sampled_params(

        results_folder,

        force_claret_params

    )
)


# =============================================================================
# FIT LDD FOR ALL BOOTSTRAPS
# =============================================================================

print("")

print("=" * 79)

print(
    "FITTING KSI GEM LDD"
)

print("=" * 79)


bs_results = (
    rdiam.fit_ldd_for_all_bootstraps(

        tgt_info,

        n_bootstraps,

        results_path,

        sampled_sci_params,

        method=
            fitting_method,

        e_wl_frac=
            e_wl_frac,

        prune_errant_baselines=
            True,

        combined_fit=
            combined_fit

    )
)


results = (
    rdiam.summarise_results(

        bs_results,

        tgt_info,

        e_wl_frac=
            e_wl_frac,

        add_e_wl_to_ldd_in_quad=
            False
        

    )
)

# ============================================================
# DEBUG VIS2: EACH BOOTSTRAP VS FINAL MEAN
# ============================================================
# ============================================================
# DEBUG VIS2: EACH BOOTSTRAP VS FINAL MEAN
# ============================================================

print("")
print("=" * 100)
print("VIS2 BOOTSTRAP COMPARISON")
print("=" * 100)


for star in bs_results.keys():

    print("")
    print("STAR:", star)

    # --------------------------------------------------------
    # Individual bootstraps
    # --------------------------------------------------------

    for bs_i in range(
            len(bs_results[star])):

        vis2_bs = np.asarray(
            bs_results[
                star
            ].iloc[
                bs_i
            ][
                "VIS2"
            ],
            dtype=float
        )

        print(
            "bootstrap %i: min=%.6f  max=%.6f"
            % (
                bs_i,
                np.nanmin(vis2_bs),
                np.nanmax(vis2_bs)
            )
        )


    # --------------------------------------------------------
    # Identify correct result row
    # --------------------------------------------------------

    if isinstance(
            star,
            tuple):

        star_name = str(
            star[0]
        )

        sequence_name = str(
            star[1]
        )

        period = int(
            star[2]
        )

    else:

        star_name = str(
            star
        )

        sequence_name = "combined"

        period = None


    # --------------------------------------------------------
    # Match STAR + SEQUENCE
    # --------------------------------------------------------

    mask = (
        results[
            "STAR"
        ].astype(str)
        ==
        star_name
    )

    mask = (
        mask
        &
        (
            results[
                "SEQUENCE"
            ].astype(str)
            ==
            sequence_name
        )
    )


    matching_rows = results.loc[
        mask
    ]


    if len(
            matching_rows) == 0:

        print(
            "WARNING: no result row found for %s %s"
            % (
                star_name,
                sequence_name
            )
        )

        continue


    result_row = matching_rows.iloc[
        0
    ]


    vis2_final = np.asarray(
        result_row[
            "VIS2"
        ],
        dtype=float
    )


    print("")
    print(
        "FINAL %s VIS2: min=%.6f  max=%.6f"
        % (
            sequence_name,
            np.nanmin(
                vis2_final
            ),
            np.nanmax(
                vis2_final
            )
        )
    )


print("=" * 100)

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("")

print(
    "Saving REACH result objects..."
)


rutils.save_results(

    bs_results,

    results,

    results_folder

)


summary_csv = os.path.join(

    analysis_root,

    "ksi_gem_results_summary.csv"

)


results.to_csv(

    summary_csv,

    index=False

)


print("")

print("=" * 79)

print(
    "KSI GEM RESULT"
)

print("=" * 79)


print(
    results
)


print("")

print(
    "Saved summary:"
)

print(
    summary_csv
)


print("=" * 79)


# =============================================================================
# PRINT C SCALE
# =============================================================================

if "C_SCALE" in results.columns:


    print("")

    print("=" * 79)

    print(
        "C_SCALE VALUES"
    )

    print("=" * 79)


    for row_i in range(
            len(results)):


        row = results.iloc[
            row_i
        ]


        c_values = np.asarray(

            row[
                "C_SCALE"
            ],

            dtype=float

        ).ravel()


        print(

            "%s C_SCALE = %s"

            % (

                row[
                    "STAR"
                ]
                if "STAR"
                in results.columns
                else TARGET_NAME,

                str(
                    c_values
                )

            )

        )


        if len(c_values) == 2:


            print(

                "  bright = %.8f"
                % c_values[0]

            )


            print(

                "  faint  = %.8f"
                % c_values[1]

            )


    print("=" * 79)


# =============================================================================
# SAVE BOOTSTRAP LDD VALUES
# =============================================================================

bootstrap_rows = []


for star_id in bs_results.keys():


    try:


        star_data = bs_results[
            star_id
        ]


        ldd_values = np.asarray(

            star_data[
                "LDD_FIT"
            ].values,

            dtype=float

        ).ravel()


        finite = np.isfinite(
            ldd_values
        )


        for i, value in enumerate(
                ldd_values):


            bootstrap_rows.append(

                {

                    "star":
                        str(
                            star_id
                        ),

                    "bootstrap":
                        i,

                    "LDD_FIT":
                        value,

                    "finite":
                        bool(
                            finite[i]
                        ),

                }

            )


    except Exception as error:


        print(

            "Could not export bootstrap LDDs "
            "for %s: %s"

            % (

                str(
                    star_id
                ),

                str(
                    error
                )

            )

        )


bootstrap_csv = os.path.join(

    analysis_root,

    "ksi_gem_bootstrap_ldd.csv"

)


pd.DataFrame(

    bootstrap_rows

).to_csv(

    bootstrap_csv,

    index=False

)


print(
    "Saved bootstrap values:"
)

print(
    bootstrap_csv
)


# =============================================================================
# LDD HISTOGRAM
# =============================================================================

def save_ldd_histogram():


    all_ldd = []


    for star_id in bs_results.keys():


        values = np.asarray(

            bs_results[
                star_id
            ][
                "LDD_FIT"
            ].values,

            dtype=float

        ).ravel()


        values = values[
            np.isfinite(
                values
            )
        ]


        all_ldd.extend(
            values.tolist()
        )


    all_ldd = np.asarray(

        all_ldd,

        dtype=float

    )


    if len(all_ldd) == 0:


        raise RuntimeError(

            "No finite LDD bootstrap values"

        )


    plt.figure(
        figsize=(7, 5)
    )


    plt.hist(

        all_ldd,

        bins=30,

        histtype="step"

    )


    median = np.nanmedian(
        all_ldd
    )


    p16 = np.nanpercentile(
        all_ldd,
        16
    )


    p84 = np.nanpercentile(
        all_ldd,
        84
    )


    plt.axvline(

        median,

        linestyle="--",

        label=(
            "median = %.5f mas"
            % median
        )

    )


    plt.axvline(

        p16,

        linestyle=":"

    )


    plt.axvline(

        p84,

        linestyle=":"

    )


    plt.xlabel(
        r"$\theta_{\rm LD}$ [mas]"
    )


    plt.ylabel(
        "Number of bootstrap samples"
    )


    plt.title(

        "ksi Gem - %s"
        % experiment_mode

    )


    plt.legend()


    plt.tight_layout()


    output_pdf = os.path.join(

        plots_output,

        "ksi_gem_ldd_bootstrap_histogram.pdf"

    )


    output_png = os.path.join(

        plots_output,

        "ksi_gem_ldd_bootstrap_histogram.png"

    )


    plt.savefig(
        output_pdf
    )


    plt.savefig(

        output_png,

        dpi=200

    )


    print(
        "Saved:"
    )


    print(
        "  %s"
        % output_pdf
    )


    print(
        "  %s"
        % output_png
    )


run_plot(

    "ksi_Gem LDD bootstrap histogram",

    save_ldd_histogram

)


# =============================================================================
# REACH BOOTSTRAP SUMMARY
# =============================================================================

run_plot(

    "ksi_Gem bootstrap summary",

    rplt.plot_bootstrapping_summary,

    results,

    bs_results,

    n_bins=30,

    plot_cal_info=True,

    sequences=
        sequences,

    complete_sequences=
        complete_sequences,

    tgt_info=
        tgt_info,

    e_wl_frac=
        e_wl_frac

)


# =============================================================================
# SINGLE VISIBILITY PLOTS
# =============================================================================
#
# IMPORTANT:
#
# DO NOT average C_SCALE here.
#
# For combined ksi Gem fits we may have:
#
# C_SCALE = [C_bright, C_faint]
#
# These must remain separate.
#
# The old analysis code did:
#
#     np.nanmean(values)
#
# which is scientifically incorrect for this experiment.
#
# =============================================================================

run_plot(

    "ksi_Gem single visibility plots",

    rplt.plot_single_vis2,

    results,

    e_wl_frac=
        e_wl_frac

)


# =============================================================================
# JOINT-SEQUENCE VISIBILITY PLOT
# =============================================================================

run_plot(

    "ksi_Gem joint sequence visibility plot",

    rplt.plot_joint_seq_paper_vis2_fits,

    tgt_info,

    results,

    n_rows=4,

    n_cols=2,

    rasterize=False

)


# =============================================================================
# POINT-BY-POINT VISIBILITY DIAGNOSTIC
# =============================================================================

visibility_diagnostic_output = (
    os.path.join(

        diagnostics_folder,

        "ksi_gem_visibility_diagnostics.pdf"

    )
)


run_plot(

    "ksi_Gem visibility diagnostics",

    rplt.plot_visibility_diagnostic_summary,

    results,

    bs_results,

    tgt_info,

    output_file=
        visibility_diagnostic_output,

    bootstrap_index=
        0,

    sigma_threshold=
        3.0,

    raw_residual_threshold=
        0.05,

    e_wl_frac=
        e_wl_frac,

    star_filter=
        TARGET_NAME,

    highlight_night=
        None,

    highlight_pair=
        None,

    highlight_baseline_range=
        None,

    highlight_wavelength_index=
        None,

    max_annotations=
        15,

    use_predicted_if_missing=
        True

)


# =============================================================================
# RUN INFORMATION
# =============================================================================

run_info_file = os.path.join(

    analysis_root,

    "run_info.txt"

)


with open(
        run_info_file,
        "w") as handle:


    handle.write(

        "ksi_Gem REACH analysis\n"

    )


    handle.write(

        "=" * 60
        + "\n"

    )


    handle.write(

        "experiment_mode = %s\n"

        % experiment_mode

    )


    handle.write(

        "results_folder = %s\n"

        % results_folder

    )


    handle.write(

        "n_bootstraps = %i\n"

        % n_bootstraps

    )


    handle.write(

        "fitting_method = %s\n"

        % fitting_method

    )


    handle.write(

        "combined_fit = %s\n"

        % str(
            combined_fit
        )

    )


    handle.write(

        "e_wl_frac = %.6f\n"

        % e_wl_frac

    )


    handle.write(

        "bright_removed_calibrator = %s\n"

        % (

            BRIGHT_CAL_TO_REMOVE

            if BRIGHT_CAL_TO_REMOVE
            is not None

            else "None"

        )

    )


    handle.write(

        "faint_removed_calibrator = %s\n"

        % (

            FAINT_CAL_TO_REMOVE

            if FAINT_CAL_TO_REMOVE
            is not None

            else "None"

        )

    )

# =============================================================================
# COMPLETE-SEQUENCE VISIBILITY FROM REACH
# =============================================================================

print("")
print("=" * 79)
print("Generating REACH complete-sequence visibility plots")
print("=" * 79)


ksi_nights = [

    (
        "bright",
        "2022-02-26"
    ),

    (
        "faint",
        "2022-03-01"
    ),

]


for sequence_name, night in ksi_nights:


    night_directory = (

        "/home2/ihernand/Desktop/reach/"
        "complete_sequences/"
        "%s_v3.94_abcd/%s/"
        % (
            night,
            night
        )

    )


    output_file = os.path.join(

        plots_output,

        "ksi_gem_%s_complete_sequence_vis2.pdf"
        % sequence_name

    )


    try:

        rplt.plot_complete_sequence_vis2(

            night_directory=
                night_directory,

            science_target=
                TARGET_NAME,

            output_file=
                output_file,

            y_min=
                0.0,

            y_max=
                1.3,

            low_v2_threshold=
                0.70

        )


    except Exception as error:

        print(
            "FAILED complete sequence %s: %s"
            % (
                sequence_name,
                str(error)
            )
        )

        traceback.print_exc()
# =============================================================================
# FINISHED
# =============================================================================

print("")

print("=" * 79)

print(
    "KSI GEM ANALYSIS FINISHED"
)

print("=" * 79)


print(
    "Results folder:"
)

print(
    "  %s"
    % results_folder
)


print(
    "Analysis output:"
)

print(
    "  %s"
    % analysis_root
)


print(
    "Summary:"
)

print(
    "  %s"
    % summary_csv
)


print(
    "Bootstrap LDD values:"
)

print(
    "  %s"
    % bootstrap_csv
)


print(
    "Plot failure log:"
)

print(
    "  %s"
    % plot_failure_log
)


print(
    "Run information:"
)

print(
    "  %s"
    % run_info_file
)


print("=" * 79)