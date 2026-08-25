"""
Focused analysis of the ksi_Gem bootstrap experiments.

This script is intentionally smaller than the general analyse_example.py:
it analyses only ksi_Gem and reuses the sampled parameters already written
by pipeline_ksigem_tests.py. It does NOT call rparam.sample_all(), so the
bolometric-correction code is not run again.

Usage
-----
python analyse_ksigem_tests.py ALL
python analyse_ksigem_tests.py NO_BL
python analyse_ksigem_tests.py NO_CAL HR2426
python analyse_ksigem_tests.py NO_BL_NO_CAL HR2426

Optional explicit results folder:
python analyse_ksigem_tests.py ALL --folder 26-08-23_KSIGEM_i1000_ALL_ALL_BASELINES_ALL_CALS
"""

from __future__ import division, print_function

import os
import sys
import glob
import traceback
import shutil

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

n_bootstraps = 1000
combined_fit = True
fitting_method = "odr"
e_wl_frac = 0.0035

RESULTS_ROOT = "/home2/ihernand/Desktop/reach/results"
ANALYSIS_ROOT = "/home2/ihernand/Desktop/reach/analysis_runs"

TARGET_NAME = "ksi_Gem"


# =============================================================================
# COMMAND-LINE CONFIGURATION
# =============================================================================

valid_modes = [
    "ALL",
    "NO_BL",
    "NO_CAL",
    "NO_BL_NO_CAL",
]

if len(sys.argv) < 2:
    experiment_mode = "ALL"
else:
    experiment_mode = str(sys.argv[1]).upper()

if experiment_mode not in valid_modes:
    raise ValueError(
        "Mode must be one of: %s" % ", ".join(valid_modes)
    )


# Default calibrator used by the corresponding pipeline tests.
CAL_TO_REMOVE = "HR2426"

if experiment_mode in ["NO_CAL", "NO_BL_NO_CAL"]:
    if len(sys.argv) > 2 and sys.argv[2] != "--folder":
        CAL_TO_REMOVE = str(sys.argv[2])


# Optional explicit folder.
explicit_results_folder = "26-08-24_KSIGEM_i1000_NO_BL_BAD_BL_REMOVED_ALL_CALS"

if "--folder" in sys.argv:
    folder_i = sys.argv.index("--folder")

    if folder_i + 1 >= len(sys.argv):
        raise ValueError("--folder requires a results-folder name")

    explicit_results_folder = str(sys.argv[folder_i + 1])


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


def find_results_folder():

    if explicit_results_folder is not None:

        candidate = os.path.join(
            RESULTS_ROOT,
            explicit_results_folder
        )

        if not os.path.isdir(candidate):
            raise RuntimeError(
                "Results folder does not exist: %s"
                % candidate
            )

        return explicit_results_folder

    # ---------------------------------------------------------
    # Reconstruct the naming convention used by
    # pipeline_ksigem_tests.py.
    # ---------------------------------------------------------

    if experiment_mode in ["NO_BL", "NO_BL_NO_CAL"]:
        baseline_token = "BAD_BL_REMOVED"
    else:
        baseline_token = "ALL_BASELINES"

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
        path for path in glob.glob(pattern)
        if os.path.isdir(path)
    ]

    if experiment_mode in ["NO_CAL", "NO_BL_NO_CAL"]:

        cal_token = normalise_name(CAL_TO_REMOVE)

        candidates = [
            path for path in candidates
            if cal_token in normalise_name(
                os.path.basename(path)
            )
        ]

    if len(candidates) == 0:

        raise RuntimeError(
            "No matching results folder found.\n"
            "Pattern: %s\n"
            "Mode: %s\n"
            "Calibrator: %s"
            % (
                pattern,
                experiment_mode,
                CAL_TO_REMOVE
            )
        )

    # Use newest matching folder.
    candidates.sort(
        key=os.path.getmtime,
        reverse=True
    )

    if len(candidates) > 1:

        print("")
        print("Multiple matching folders found.")
        print("Using newest:")

        for path in candidates:
            print(
                "  %s"
                % os.path.basename(path)
            )

    return os.path.basename(candidates[0])


def filter_ksi_sequences(
        complete_sequences,
        sequences):

    ksi_keys = [
        key
        for key in complete_sequences.keys()
        if normalise_name(key[1])
        == normalise_name(TARGET_NAME)
    ]

    if len(ksi_keys) == 0:
        raise RuntimeError(
            "No ksi_Gem sequences found "
            "in sequence logs"
        )

    new_complete = {
        key: complete_sequences[key]
        for key in ksi_keys
    }

    new_sequences = {
        key: sequences[key]
        for key in ksi_keys
    }

    return new_complete, new_sequences


def filter_ksi_tgt_info(tgt_info):

    if "Primary" not in tgt_info.columns:
        raise RuntimeError(
            "tgt_info has no Primary column"
        )

    mask = tgt_info["Primary"].apply(
        lambda value:
        normalise_name(value)
        == normalise_name(TARGET_NAME)
    )

    # Some REACH tables may use a compact alias.
    if np.sum(mask) == 0:

        alternatives = [
            "ksiGem",
            "xiGem",
            "ksi Gem",
        ]

        mask = tgt_info["Primary"].apply(
            lambda value:
            normalise_name(value)
            in [
                normalise_name(x)
                for x in alternatives
            ]
        )

    if np.sum(mask) == 0:

        print("")
        print("Available science target names:")

        if "Science" in tgt_info.columns:
            sci_mask = tgt_info["Science"] == True
            print(
                tgt_info.loc[
                    sci_mask,
                    ["Primary"]
                ]
            )

        raise RuntimeError(
            "Could not identify ksi_Gem in tgt_info"
        )

    return tgt_info.loc[mask].copy()


def run_plot(
        label,
        function,
        *args,
        **kwargs):

    print("")
    print("=" * 79)
    print("Generating plot: %s" % label)
    print("=" * 79)

    try:

        function(*args, **kwargs)

        print("SUCCESS: %s" % label)
        return True

    except Exception as error:

        print("FAILED: %s" % label)
        print("Error: %s" % str(error))

        with open(
                plot_failure_log,
                "a") as handle:

            handle.write(
                "Plot: %s\n" % label
            )

            handle.write(
                "Error: %s\n" % str(error)
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
        plt.close("all")


# =============================================================================
# LOCATE BOOTSTRAP RESULTS
# =============================================================================

results_folder = find_results_folder()

results_path = os.path.join(
    RESULTS_ROOT,
    results_folder
) + "/"

analysis_name = "KSIGEM_%s" % experiment_mode

if experiment_mode in [
        "NO_CAL",
        "NO_BL_NO_CAL"]:

    analysis_name += "_NO_%s" % (
        normalise_name(CAL_TO_REMOVE).upper()
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

    if not os.path.exists(directory):
        os.makedirs(directory)


plot_failure_log = os.path.join(
    diagnostics_folder,
    "plot_failures.txt"
)

with open(plot_failure_log, "w") as handle:
    handle.write(
        "Failures while generating ksi_Gem plots\n"
    )
    handle.write("=" * 79 + "\n\n")


print("")
print("=" * 79)
print("KSI GEM ANALYSIS")
print("=" * 79)
print("experiment mode  : %s" % experiment_mode)
print("results folder   : %s" % results_folder)
print("results path     : %s" % results_path)
print("n_bootstraps     : %i" % n_bootstraps)
print("fitting method   : %s" % fitting_method)
print("combined fit     : %s" % str(combined_fit))
print(
    "removed cal      : %s"
    % (
        CAL_TO_REMOVE
        if experiment_mode
        in ["NO_CAL", "NO_BL_NO_CAL"]
        else "None"
    )
)
print("analysis output  : %s" % analysis_root)
print("=" * 79)


# =============================================================================
# LOAD TARGET AND SEQUENCE INFORMATION
# =============================================================================

print("")
print("Loading target information...")

tgt_info_all = rutils.initialise_tgt_info(
    assign_default_uncertainties,
    lb_pc,
    use_plx_systematic
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

tgt_info = filter_ksi_tgt_info(
    tgt_info_all
)


print("")
print("ksi_Gem target row:")
print(tgt_info)

print("")
print("ksi_Gem sequences:")

for key in sorted(complete_sequences.keys()):

    print(
        "  %s  night=%s  sequence=%s"
        % (
            str(key),
            complete_sequences[key][0],
            str(sequences[key])
        )
    )


# =============================================================================
# LOAD THE SAMPLED PARAMETERS CREATED BY THE PIPELINE
# =============================================================================
#
# IMPORTANT:
# We do not call rparam.sample_all() here.
# The analysis must reuse the same bootstrap parameter samples written by
# pipeline_ksigem_tests.py.
# =============================================================================

print("")
print("Loading sampled stellar parameters...")

sampled_sci_params = rutils.load_sampled_params(
    results_folder,
    force_claret_params
)


# =============================================================================
# FIT LDD FOR ALL 1000 BOOTSTRAPS
# =============================================================================

print("")
print("=" * 79)
print("FITTING KSI GEM LDD")
print("=" * 79)

bs_results = rdiam.fit_ldd_for_all_bootstraps(
    tgt_info,
    n_bootstraps,
    results_path,
    sampled_sci_params,
    method=fitting_method,
    e_wl_frac=e_wl_frac,
    prune_errant_baselines=False,
    combined_fit=combined_fit
)


results = rdiam.summarise_results(
    bs_results,
    tgt_info,
    e_wl_frac=e_wl_frac,
    add_e_wl_to_ldd_in_quad=False
)


# =============================================================================
# SAVE RESULTS
# =============================================================================

print("")
print("Saving REACH result objects...")

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
print("KSI GEM RESULT")
print("=" * 79)

print(results)

print("")
print("Saved summary:")
print(summary_csv)

print("=" * 79)


# =============================================================================
# SAVE BOOTSTRAP LDD VALUES
# =============================================================================

bootstrap_rows = []

for star_id in bs_results.keys():

    try:

        star_data = bs_results[star_id]

        ldd_values = np.asarray(
            star_data["LDD_FIT"].values,
            dtype=float
        ).ravel()

        finite = np.isfinite(ldd_values)

        for i, value in enumerate(ldd_values):

            bootstrap_rows.append(
                {
                    "star": str(star_id),
                    "bootstrap": i,
                    "LDD_FIT": value,
                    "finite": bool(finite[i]),
                }
            )

    except Exception as error:

        print(
            "Could not export bootstrap LDDs "
            "for %s: %s"
            % (
                str(star_id),
                str(error)
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

print("Saved bootstrap values:")
print(bootstrap_csv)


# =============================================================================
# LDD HISTOGRAM
# =============================================================================

def save_ldd_histogram():

    all_ldd = []

    for star_id in bs_results.keys():

        values = np.asarray(
            bs_results[star_id]["LDD_FIT"].values,
            dtype=float
        ).ravel()

        values = values[
            np.isfinite(values)
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

    plt.figure(figsize=(7, 5))

    plt.hist(
        all_ldd,
        bins=30,
        histtype="step"
    )

    median = np.nanmedian(all_ldd)
    p16 = np.nanpercentile(all_ldd, 16)
    p84 = np.nanpercentile(all_ldd, 84)

    plt.axvline(
        median,
        linestyle="--",
        label="median = %.5f mas" % median
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
        "ksi Gem - %s" % experiment_mode
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

    plt.savefig(output_pdf)
    plt.savefig(output_png, dpi=200)

    print("Saved:")
    print("  %s" % output_pdf)
    print("  %s" % output_png)


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
    sequences=sequences,
    complete_sequences=complete_sequences,
    tgt_info=tgt_info,
    e_wl_frac=e_wl_frac
)


# =============================================================================
# SINGLE VISIBILITY PLOTS
# =============================================================================

try:

    legacy_results = results.copy(
        deep=True
    )

    if "C_SCALE" in legacy_results.columns:

        legacy_c = []

        for value in legacy_results["C_SCALE"]:

            values = np.asarray(
                value,
                dtype=float
            ).ravel()

            values = values[
                np.isfinite(values)
            ]

            if len(values) > 0:
                legacy_c.append(
                    float(np.nanmean(values))
                )
            else:
                legacy_c.append(1.0)

        legacy_results[
            "C_SCALE"
        ] = legacy_c


    run_plot(
        "ksi_Gem single visibility plots",
        rplt.plot_single_vis2,
        legacy_results,
        e_wl_frac=e_wl_frac
    )

except Exception as error:

    print(
        "Skipping single visibility plots: %s"
        % str(error)
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

visibility_diagnostic_output = os.path.join(
    diagnostics_folder,
    "ksi_gem_visibility_diagnostics.pdf"
)


run_plot(
    "ksi_Gem visibility diagnostics",
    rplt.plot_visibility_diagnostic_summary,
    results,
    bs_results,
    tgt_info,
    output_file=visibility_diagnostic_output,
    bootstrap_index=0,
    sigma_threshold=3.0,
    raw_residual_threshold=0.05,
    e_wl_frac=e_wl_frac,
    star_filter=TARGET_NAME,
    highlight_night=None,
    highlight_pair=None,
    highlight_baseline_range=None,
    highlight_wavelength_index=None,
    max_annotations=15,
    use_predicted_if_missing=True
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
        "=" * 60 + "\n"
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
        % str(combined_fit)
    )

    handle.write(
        "e_wl_frac = %.6f\n"
        % e_wl_frac
    )

    handle.write(
        "removed_calibrator = %s\n"
        % (
            CAL_TO_REMOVE
            if experiment_mode
            in ["NO_CAL", "NO_BL_NO_CAL"]
            else "None"
        )
    )


print("")
print("=" * 79)
print("KSI GEM ANALYSIS FINISHED")
print("=" * 79)
print("Results folder:")
print("  %s" % results_folder)
print("Analysis output:")
print("  %s" % analysis_root)
print("Summary:")
print("  %s" % summary_csv)
print("Bootstrap LDD values:")
print("  %s" % bootstrap_csv)
print("Plot failure log:")
print("  %s" % plot_failure_log)
print("=" * 79)
