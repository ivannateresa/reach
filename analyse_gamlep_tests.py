"""
Focused analysis of the gam_Lep bootstrap experiments.

This script analyses only gam_Lep and reuses the sampled parameters
already written by pipeline_gamlep_tests.py.

It does NOT call rparam.sample_all(), so the bolometric-correction
code is not run again.

Usage
-----

ALL:

    python analyse_gamlep_tests.py ALL

NO_BL:

    python analyse_gamlep_tests.py NO_BL

NO_CAL:

    python analyse_gamlep_tests.py NO_CAL CAL_P104 CAL_P106

meaning:

    P104 faint -> remove CAL_P104
    P106 faint -> remove CAL_P106

NO_BL_NO_CAL:

    python analyse_gamlep_tests.py NO_BL_NO_CAL CAL_P104 CAL_P106

Remove calibrator only from P106:

    python analyse_gamlep_tests.py NO_CAL NONE CAL_P106

Remove calibrator only from P104:

    python analyse_gamlep_tests.py NO_CAL CAL_P104 NONE

Optional explicit results folder:

    python analyse_gamlep_tests.py ALL \
        --folder 26-09-03_GAMLEP_i1_ALL_ALL_BASELINES_ALL_CALS
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
# This must match the number used in pipeline_gamlep_tests.py
#
# For tests:
# n_bootstraps = 2
#
# For final run:
# n_bootstraps = 1000

# The shell runner exports N_BOOTSTRAPS so that the pipeline and analysis
# always use the same value.  Direct runs default to 100.
n_bootstraps = int(
    os.environ.get(
        "N_BOOTSTRAPS",
        "1"
    )
)


combined_fit = False

fitting_method = "odr"

e_wl_frac = 0.0035


RESULTS_ROOT = (
    "/home2/ihernand/Desktop/reach/results"
)

ANALYSIS_ROOT = (
    "/home2/ihernand/Desktop/reach/analysis_runs"
)


TARGET_NAME = "gam_Lep"


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
# IDENTIFY GAM LEP PERIOD (P104 / P106)
# =============================================================================

def identify_gamlep_period(
        seq_key,
        sequence_targets,
        sequence_info):

    clean_targets = set(
        normalise_name(target)
        for target in sequence_targets
    )

    # Prefer calibrators unique to each observing period.
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


    # Fallback to the REACH night.
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

P104_CAL_TO_REMOVE = None

P106_CAL_TO_REMOVE = None


if experiment_mode in [
        "NO_CAL",
        "NO_BL_NO_CAL"]:


    if len(calibrator_args) > 0:

        P104_CAL_TO_REMOVE = (
            convert_none(
                calibrator_args[0]
            )
        )


    if len(calibrator_args) > 1:

        P106_CAL_TO_REMOVE = (
            convert_none(
                calibrator_args[1]
            )
        )


CAL_TO_REMOVE_BY_PERIOD = {

    "p104":
        P104_CAL_TO_REMOVE,

    "p106":
        P106_CAL_TO_REMOVE,

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
    # Reconstruct naming convention used by pipeline_gamlep_tests.py
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

        "*_GAMLEP_i%i_%s_%s_*"

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
    # Filter using P104 calibrator
    # -------------------------------------------------------------------------

    if experiment_mode in [
            "NO_CAL",
            "NO_BL_NO_CAL"]:


        if P104_CAL_TO_REMOVE is not None:


            p104_token = (
                normalise_name(

                    "P104_NO_%s"

                    % clean_for_filename(
                        P104_CAL_TO_REMOVE
                    )

                )
            )


            candidates = [

                path

                for path in candidates

                if p104_token
                in normalise_name(
                    os.path.basename(
                        path
                    )
                )

            ]


        # ---------------------------------------------------------------------
        # Filter using P106 calibrator
        # ---------------------------------------------------------------------

        if P106_CAL_TO_REMOVE is not None:


            p106_token = (
                normalise_name(

                    "P106_NO_%s"

                    % clean_for_filename(
                        P106_CAL_TO_REMOVE
                    )

                )
            )


            candidates = [

                path

                for path in candidates

                if p106_token
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
            "P104 calibrator removed: %s\n"
            "P106 calibrator removed: %s"

            % (

                pattern,

                experiment_mode,

                P104_CAL_TO_REMOVE,

                P106_CAL_TO_REMOVE

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
# FILTER GAM LEP SEQUENCES
# =============================================================================

def filter_gamlep_sequences(
        complete_sequences,
        sequences):


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
                TARGET_NAME
            )
        )

    ]


    if len(gamlep_keys) == 0:


        raise RuntimeError(

            "No gam_Lep sequences found "
            "in sequence logs"

        )


    new_complete = {

        key:
            complete_sequences[key]

        for key
        in gamlep_keys

    }


    new_sequences = {

        key:
            sequences[key]

        for key
        in gamlep_keys

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
        "APPLYING P104/P106 CALIBRATOR "
        "REMOVAL TO ANALYSIS"
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
            "night:",
            sequence_info[0]
        )
        print(
            "gam Lep period:",
            period_name.upper()
        )
        print(
            "calibrator removed:",
            cal_to_remove
        )

        if cal_to_remove is None:
            continue

        sequence_after = [
            target
            for target
            in sequence_before
            if not same_name(
                target,
                cal_to_remove
            )
        ]

        if len(sequence_after) == len(sequence_before):
            raise RuntimeError(
                "Requested calibrator %s was not found in %s (%s): %s"
                % (
                    cal_to_remove,
                    str(seq_key),
                    period_name.upper(),
                    str(sequence_before)
                )
            )

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

        if len(sequence_info) >= 3:

            obs_blocks = sequence_info[
                2
            ]

            filtered_obs_blocks = []

            for obs in obs_blocks:

                try:
                    obs_target = obs[2]
                except Exception:
                    filtered_obs_blocks.append(obs)
                    continue

                if same_name(
                        obs_target,
                        cal_to_remove):

                    print(
                        "removing observing block:",
                        obs_target
                    )
                    continue

                filtered_obs_blocks.append(obs)

            sequence_info[
                2
            ] = filtered_obs_blocks

            complete_sequences[
                seq_key
            ] = sequence_info


    print("")
    print("=" * 79)


    return (
        complete_sequences,
        sequences
    )


# =============================================================================
# FILTER TARGET INFORMATION
# =============================================================================

def filter_gamlep_tgt_info(
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

            "gam_Lep",

            "gam Lep",

            "gamLep",

            "GAM_LEP",
            "gamma Lep",
            "gamma_Lep",

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
            "gam_Lep in tgt_info"

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
    "GAMLEP_%s"
    % experiment_mode
)


if experiment_mode in [
        "NO_CAL",
        "NO_BL_NO_CAL"]:


    if P104_CAL_TO_REMOVE is not None:


        analysis_name += (

            "_P104_NO_%s"

            % normalise_name(
                P104_CAL_TO_REMOVE
            ).upper()

        )


    if P106_CAL_TO_REMOVE is not None:


        analysis_name += (

            "_P106_NO_%s"

            % normalise_name(
                P106_CAL_TO_REMOVE
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
        "gam_Lep plots\n"

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
    "GAM LEP ANALYSIS"
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
    "P104 cal removed   : %s"

    % (
        P104_CAL_TO_REMOVE
        if P104_CAL_TO_REMOVE
        is not None
        else "None"
    )
)


print(
    "P106 cal removed    : %s"

    % (
        P106_CAL_TO_REMOVE
        if P106_CAL_TO_REMOVE
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
    filter_gamlep_sequences(

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
    filter_gamlep_tgt_info(
        tgt_info_all
    )
)


print("")

print(
    "gam_Lep target row:"
)

print(
    tgt_info
)


print("")

print(
    "gam_Lep sequences used in analysis:"
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
    "FITTING GAM LEP LDD"
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


    # gam Lep has two faint sequences. If summarise_results exposes
    # PERIOD, include it so P104 and P106 are not mixed.
    if (
        period is not None
        and "PERIOD" in results.columns
    ):

        mask = (
            mask
            &
            (
                pd.to_numeric(
                    results[
                        "PERIOD"
                    ],
                    errors="coerce"
                )
                ==
                period
            )
        )

    matching_rows = results.loc[
        mask
    ]

    if len(matching_rows) > 1:
        print(
            "WARNING: multiple result rows found for %s %s; "
            "using the first row"
            % (
                star_name,
                sequence_name
            )
        )


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

    "gamlep_results_summary.csv"

)


results.to_csv(

    summary_csv,

    index=False

)


print("")

print("=" * 79)

print(
    "GAM LEP RESULT"
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

                "  P104 = %.8f"
                % c_values[0]

            )


            print(

                "  P106 = %.8f"
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

    "gamlep_bootstrap_ldd.csv"

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

        "gam Lep - %s"
        % experiment_mode

    )


    plt.legend()


    plt.tight_layout()


    output_pdf = os.path.join(

        plots_output,

        "gamlep_ldd_bootstrap_histogram.pdf"

    )


    output_png = os.path.join(

        plots_output,

        "gamlep_ldd_bootstrap_histogram.png"

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

    "gam_Lep LDD bootstrap histogram",

    save_ldd_histogram

)


# =============================================================================
# REACH BOOTSTRAP SUMMARY
# =============================================================================


bootstrap_summary_output = os.path.join(

    plots_output,

    "bootstrapped_summary.pdf"

)


def save_reach_bootstrap_summary():

    # Older REACH versions save this PDF to the fixed relative path
    # plots/bootstrapped_summary.pdf. Run only this plot from analysis_root
    # so that both old and new plotting.py versions save it in this analysis.

    original_working_directory = os.getcwd()


    try:

        os.chdir(
            analysis_root
        )


        plot_result = rplt.plot_bootstrapping_summary(

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


    finally:

        os.chdir(
            original_working_directory
        )


    if not os.path.isfile(
            bootstrap_summary_output):

        raise RuntimeError(

            "Bootstrap summary was not created: %s"

            % bootstrap_summary_output

        )


    print(

        "Saved bootstrap summary: %s"

        % bootstrap_summary_output

    )


    return plot_result


run_plot(

    "gam_Lep bootstrap summary",

    save_reach_bootstrap_summary

)


# =============================================================================
# REACH BOOTSTRAP SUMMARY COLOURED BY BASELINE
# =============================================================================

if hasattr(
        rplt,
        "plot_bootstrapping_summary_by_baseline"):


    baseline_summary_output = os.path.join(

        plots_output,

        "bootstrapped_summary_by_baseline.pdf"

    )


    run_plot(

        "gam_Lep bootstrap summary by baseline",

        rplt.plot_bootstrapping_summary_by_baseline,

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
            e_wl_frac,

        output_file=
            baseline_summary_output

    )


else:

    print("")
    print(
        "Skipping baseline-coloured bootstrap summary: "
        "plot_bootstrapping_summary_by_baseline() "
        "is not present in reach.plotting."
    )



# =============================================================================
# SINGLE VISIBILITY PLOTS
# =============================================================================
#
# IMPORTANT:
#
# DO NOT average C_SCALE here.
#
# For combined GAM LEP fits we may have:
#
# C_SCALE = [C_P104, C_P106]
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

    "gam_Lep single visibility plots",

    rplt.plot_single_vis2,

    results,

    e_wl_frac=
        e_wl_frac

)


# =============================================================================
# JOINT-SEQUENCE VISIBILITY PLOT
# =============================================================================

run_plot(

    "gam_Lep joint sequence visibility plot",

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

        "gamlep_visibility_diagnostics.pdf"

    )
)


run_plot(

    "gam_Lep visibility diagnostics",

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

        "gam_Lep REACH analysis\n"

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

        "p104_removed_calibrator = %s\n"

        % (

            P104_CAL_TO_REMOVE

            if P104_CAL_TO_REMOVE
            is not None

            else "None"

        )

    )


    handle.write(

        "p106_removed_calibrator = %s\n"

        % (

            P106_CAL_TO_REMOVE

            if P106_CAL_TO_REMOVE
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


# Build the gam_Lep P104/P106 night list directly from the filtered
# sequence logs. Both sequences may be labelled "faint", so the period
# identifier is used instead of the sequence label.
gamlep_nights = []


for seq_key in sorted(
        complete_sequences.keys()):

    sequence_targets = list(
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
        sequence_targets,
        sequence_info
    )

    night = str(
        sequence_info[0]
    )

    period_night = (
        period_name,
        night
    )

    if period_night not in gamlep_nights:
        gamlep_nights.append(
            period_night
        )


print("")
print("gam_Lep period/night combinations:")


for period_name, night in gamlep_nights:

    print(
        "  %s  %s"
        % (
            period_name.upper(),
            night
        )
    )


for period_name, night in gamlep_nights:

    # Read the calibrated OIFITS belonging to this exact experiment.
    night_directory = results_path

    safe_night = (
        night
        .replace("/", "-")
        .replace(" ", "_")
    )

    output_file = os.path.join(
        plots_output,
        "gamlep_%s_%s_complete_sequence_vis2.pdf"
        % (
            period_name,
            safe_night
        )
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
                0.70,
            bootstrap_index=
                0,
            night=
                night,
            case_label=
                analysis_name
        )

    except Exception as error:

        print(
            "FAILED complete sequence %s %s: %s"
            % (
                period_name.upper(),
                night,
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
    "GAM LEP ANALYSIS FINISHED"
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
