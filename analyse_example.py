"""Script to take bootstrapped oifits files and combine for final results
"""
from __future__ import division, print_function

import numpy as np
import pandas as pd
import reach.diameters as rdiam
import reach.diagnostics as rdiag
import reach.parameters as rparam
import reach.paper as rpaper
import reach.plotting as rplt
import reach.photometry as rphot
import reach.pndrs as rpndrs
import reach.utils as rutils
import os
import pickle
import diagnostics as dig
# Import plotting xy offsets map
import xy_map_file
import matplotlib.pyplot as plt
import traceback
import shutil
import itertools

import matplotlib.cm as cm

from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------------------------------------------------------
# Setup & Loading
# -----------------------------------------------------------------------------
lb_pc = 70                          # The size of the local bubble in pc
use_plx_systematic =  True          # Use Stassun & Torres 18 plx offset
combined_fit =True                # Fit for LDD for multiple seq at once
load_saved_results = True         # Load or do fitting fresh
assign_default_uncertainties = True # Give default errors to stars without
force_claret_params = False         # Force use of Claret+11 limb d. params
n_bootstraps = 500
fitting_method = "ls"               # Fitting method to use: ls or odr
e_wl_frac = 0.0035                  # Fractional error on wl scale

# If using least squares fitting, the wavelength uncertainty is added in 
# quadrature to the LDD uncertainty at the end. If using orthogonal distance
# regression, it is incorporated into the fit itself
if fitting_method == "ls":
    add_e_wl_to_ldd_in_quad = True
else:
    add_e_wl_to_ldd_in_quad = False

#results_folder = "19-06-27_i2000"       # Parallel!
#results_folder = "19-07-05_i3000"       # Long run with all bad cals removed
results_folder = "26-07-10_i500"       # Final run for 1st draft
results_path = "/home2/ihernand/Desktop/reach/results/%s/" % results_folder

# Path to Casagrande & VandenBerg 2014/2018a/2018b bolometric correction code
# and filters to use when calculating fbol_final from [Hp, Bt, Vt, Bp, Rp]
bc_path =  "/home2/ihernand/Desktop/reach/bolometric-corrections"
#bc_path =  "/home/arains/code/bolometric-corrections"
band_mask = [1, 1, 1, 0, 0]


# Load in files
print("Loading in files...")
tgt_info = rutils.initialise_tgt_info(assign_default_uncertainties, lb_pc, 
                                      use_plx_systematic)

complete_sequences, sequences = rutils.load_sequence_logs()


diagnostics_folder = os.path.join(results_folder, "analysis_diagnostics")

if not os.path.exists(diagnostics_folder):
    os.makedirs(diagnostics_folder)

# -----------------------------------------------------------------------------
# Loading Existing Results
# -----------------------------------------------------------------------------
# Collate bootstrapped results
if load_saved_results:
    print("Loading saved results...")
    sampled_sci_params = rutils.load_sampled_params(results_folder, 
                                                    force_claret_params,
                                                    final_teff_sample=True)
    
    bs_results, results = rutils.load_results(results_folder)

    rparam.calc_sample_and_final_params(tgt_info, sampled_sci_params, 
                                        bs_results, results)
# -----------------------------------------------------------------------------
# Calculating Results For First Time
# -----------------------------------------------------------------------------
# Do two iterations of the fitting, one with literature teffs, and one with
# interferometric teffs
    
else:
    # 1111111111111111111111111111111111111111111111111111111111111111111111111
    # Run through initially using **literature** teffs
    # 1111111111111111111111111111111111111111111111111111111111111111111111111
    print("-"*79, "\n", "\tInitial Analysis (Literature Teff)\n", "-"*79)
    sampled_sci_params = rutils.load_sampled_params(results_folder, 
                                                    force_claret_params)
    
    print("Getting results of bootstrapping for %s bootstraps..." 
          % n_bootstraps)

    bs_results = rdiam.fit_ldd_for_all_bootstraps(tgt_info, n_bootstraps, 
                                            results_path, sampled_sci_params,
                                            method=fitting_method, 
                                            e_wl_frac=e_wl_frac,
                                            combined_fit=combined_fit) 

    # Summarise results
    results = rdiam.summarise_results(bs_results, tgt_info,
                                      e_wl_frac=e_wl_frac,
                                      add_e_wl_to_ldd_in_quad=\
                                          add_e_wl_to_ldd_in_quad)
    
    # -------------------------------------------------------------------------
# Diagnostic for initial analysis
# -----------------------------------------------------------------------
    initial_diag = dig.diagnose_ldd_analysis_results(
    label="initial_literature_teff",
    tgt_info=tgt_info,
    sampled_sci_params=sampled_sci_params,
    bs_results=bs_results,
    results=results,
    outdir=diagnostics_folder)
    
    # Calculate **initial** fundamental parameters using literature values
    print("Determining **initial** fundamental parameters...")
    rparam.calc_sample_and_final_params(tgt_info, sampled_sci_params, 
                                        bs_results, results)
    
    # 2222222222222222222222222222222222222222222222222222222222222222222222222
    # Now resample, and run through again using **interferometric** teffs
    # 2222222222222222222222222222222222222222222222222222222222222222222222222
    print("-"*79, "\n", "\tFinal Analysis (Interferometric Teff)\n", "-"*79)
    sampled_sci_params = rparam.sample_all(tgt_info, n_bootstraps, bc_path,
                                           force_claret_params, band_mask,
                                           use_literature_teffs=False)
                                          
    rutils.save_sampled_params(sampled_sci_params, results_folder, 
                               force_claret_params=force_claret_params,
                               final_teff_sample=True)
    
    bs_results = rdiam.fit_ldd_for_all_bootstraps(tgt_info, n_bootstraps, 
                                            results_path, sampled_sci_params,
                                            method=fitting_method,  
                                            e_wl_frac=e_wl_frac,
                                            combined_fit=combined_fit) 
    # Summarise results
    results = rdiam.summarise_results(bs_results, tgt_info, 
                                      e_wl_frac=e_wl_frac,
                                      add_e_wl_to_ldd_in_quad=\
                                          add_e_wl_to_ldd_in_quad)
    # -------------------------------------------------------------------------
# Diagnostic for final analysis
# -------------------------------------------------------------------------
    final_diag = dig.diagnose_ldd_analysis_results(
    label="final_interferometric_teff",
    tgt_info=tgt_info,
    sampled_sci_params=sampled_sci_params,
    bs_results=bs_results,
    results=results,
    outdir=diagnostics_folder)

    
    # Save results
    rutils.save_results(bs_results, results, results_folder)
    
    # Calculate **final** fundamental parameters using interferometric values
    print("Determining **final** fundamental parameters...")
    rparam.calc_sample_and_final_params(tgt_info, sampled_sci_params, 
                                        bs_results, results)

# Summarise C param fits
rutils.summarise_cs(results)
rutils.get_mean_delta_h(tgt_info, complete_sequences, sequences)
                                        
# -----------------------------------------------------------------------------
# Table generation and plotting
# -----------------------------------------------------------------------------
print("-"*79, "\n", "\tTables and Plots (Literature Teff)\n", "-"*79)
# Generate tables
print("Generating tables...")
rpaper.make_table_targets(tgt_info)
rpaper.make_table_calibrators(tgt_info, sequences)
rpaper.make_table_observation_log(tgt_info, complete_sequences, sequences)
rpaper.make_table_fbol(tgt_info)
rpaper.make_table_seq_results(results)
rpaper.make_table_final_results(tgt_info)
rpaper.make_table_limb_darkening(tgt_info)

#print("Generating plots...")
#rplt.plot_fbol_comp(tgt_info)
#rplt.plot_hr_diagram(tgt_info, plot_isochrones_basti=True)
#rplt.plot_casagrande_teff_comp(tgt_info, xy_map_file.teff)
#rplt.plot_lit_diam_comp(tgt_info, xy_map.lit_diam)
#rplt.plot_all_sidelobe_vis2_fits(
#    tgt_info,
#    results,
#    output_dir="paper/sidelobes"
#)

#rplt.plot_joint_seq_paper_vis2_fits(tgt_info, results, n_rows=4, n_cols=2)
#rplt.plot_colour_rel_diam_comp(tgt_info, 
#                               xy_maps=(xy_map_file.vw3, xy_map_file.vw4, xy_map_file.bv_feh))
#rplt.plot_bootstrapping_summary(results, bs_results, plot_cal_info=True, 
#                                sequences=sequences, 
#                                complete_sequences=complete_sequences, 
#                                tgt_info=tgt_info)

# =============================================================================
# GENERATE ALL AVAILABLE PLOTS
# =============================================================================

print("Generating all plots...")

# -------------------------------------------------------------------------
# Create output directories
# -------------------------------------------------------------------------
plot_directories = [
    "plots",
    "plots/single_vis2",
    "plots/raw_vis2",
    "plots/diameter_comparisons",
    "paper",
    "paper/sidelobes",
]

for directory in plot_directories:
    if not os.path.exists(directory):
        os.makedirs(directory)


# -------------------------------------------------------------------------
# Plot failure log
# -------------------------------------------------------------------------
plot_failure_log = os.path.join(
    diagnostics_folder,
    "plot_failures.txt"
)

with open(plot_failure_log, "w") as handle:
    handle.write("Failures while generating plots\n")
    handle.write("=" * 79 + "\n\n")


def run_plot(label, function, *args, **kwargs):
    """
    Execute one plotting function.

    Returns
    -------
    success : bool
        True if the function finished without an exception.
    """

    print("\n" + "=" * 79)
    print("Generating plot: %s" % label)
    print("=" * 79)

    try:

        function(*args, **kwargs)

        print("SUCCESS: %s" % label)

        return True

    except Exception as error:

        print("FAILED: %s" % label)
        print("Error: %s" % str(error))

        with open(plot_failure_log, "a") as handle:

            handle.write("Plot: %s\n" % label)
            handle.write("Error: %s\n" % str(error))
            handle.write(traceback.format_exc())
            handle.write("\n" + "-" * 79 + "\n\n")

        traceback.print_exc()

        return False

    finally:

        plt.close("all")

def copy_plot(source, destination):
    """
    Copy a plot when two plotting functions use the same output filename.
    """

    if os.path.exists(source):
        shutil.copyfile(source, destination)
        print("Copied:")
        print("  %s" % destination)
    else:
        print("Could not copy missing plot:")
        print("  %s" % source)


# =============================================================================
# 1. Fundamental parameters
# =============================================================================

run_plot(
    "Bolometric flux comparison",
    rplt.plot_fbol_comp,
    tgt_info
)


run_plot(
    "HR diagram with BaSTI isochrones",
    rplt.plot_hr_diagram,
    tgt_info,
    plot_isochrones_basti=True,
    plot_isochrones_padova=False,
    feh=0.062,
    basti_folder="data/basti",
    basti_ages_myr=[
        60,
        100,
        500,
        1000,
        2000,
        5000,
        10000,
        14000
    ])

copy_plot(
    "paper/hr_diagram.pdf",
    "paper/hr_diagram_basti.pdf"
)



padova_file = "data/padova_isochrone_age.dat"

if os.path.exists(padova_file):

    source_plot = "paper/hr_diagram.pdf"

    if os.path.exists(source_plot):
        os.remove(source_plot)

    success = run_plot(
        "HR diagram with Padova isochrones",
        rplt.plot_hr_diagram,
        tgt_info,
        plot_isochrones_basti=False,
        plot_isochrones_padova=True
    )

    if success:
        copy_plot(
            source_plot,
            "paper/hr_diagram_padova.pdf"
        )

else:

    print("Skipping Padova HR diagram")
    print("Missing file: %s" % padova_file)

run_plot(
    "HR diagram without isochrones",
    rplt.plot_hr_diagram,
    tgt_info,
    plot_isochrones_basti=False,
    plot_isochrones_padova=False
)

copy_plot(
    "paper/hr_diagram.pdf",
    "paper/hr_diagram_no_isochrones.pdf"
)


run_plot(
    "PIONIER versus Casagrande temperatures",
    rplt.plot_casagrande_teff_comp,
    tgt_info,
    xy_map_file.teff
)


run_plot(
    "Temperature comparison with literature",
    rplt.plot_lit_teff_comp,
    tgt_info
)


# =============================================================================
# 2. Angular-diameter comparisons
# =============================================================================

run_plot(
    "Colour-relation diameters coloured by metallicity",
    rplt.plot_colour_rel_diam_comp,
    tgt_info,
    colour_rels=["V-W3", "V-W4", "B-V_feh"],
    cbar="feh",
    xy_maps=(
        xy_map_file.vw3,
        xy_map_file.vw4,
        xy_map_file.bv_feh
    )
)


run_plot(
    "Colour-relation diameters coloured by Teff",
    rplt.plot_colour_rel_diam_comp,
    tgt_info,
    colour_rels=["V-W3", "V-W4", "B-V_feh"],
    cbar="teff",
    xy_maps=(
        xy_map_file.vw3,
        xy_map_file.vw4,
        xy_map_file.bv_feh
    )
)


run_plot(
    "Predicted versus JSDC diameters",
    rplt.plot_jsdc_ldd_comp,
    tgt_info
)


# -------------------------------------------------------------------------
# Literature diameter comparison
# -------------------------------------------------------------------------
# Start with zero offsets for every target. If xy_map_file.lit_diam exists,
# replace the default offsets with those defined by the user.

literature_xy_map = {}

if "Primary" in tgt_info.columns:

    for primary_name in tgt_info["Primary"].values:

        if pd.notnull(primary_name):
            literature_xy_map[str(primary_name)] = (0.0, 0.0)

if hasattr(xy_map_file, "lit_diam"):
    literature_xy_map.update(xy_map_file.lit_diam)


run_plot(
    "PIONIER versus literature diameters",
    rplt.plot_lit_diam_comp,
    tgt_info,
    xy_map=literature_xy_map
)
# =============================================================================
# Compatibility copy for legacy plotting functions
# =============================================================================

legacy_results = results.copy(deep=True)

legacy_c_scale = []
legacy_u_lld = []

u_lambda_columns = [
    "u_lambda_0",
    "u_lambda_1",
    "u_lambda_2",
    "u_lambda_3",
    "u_lambda_4",
    "u_lambda_5",
]


for row_i in range(len(legacy_results)):

    # -------------------------------------------------------------------------
    # Convert C_SCALE array to one representative scalar.
    # This is only for the old plotting functions.
    # -------------------------------------------------------------------------
    c_values = np.asarray(
        legacy_results.iloc[row_i]["C_SCALE"],
        dtype=float
    ).ravel()

    c_values = c_values[np.isfinite(c_values)]

    if len(c_values) > 0:
        legacy_c_scale.append(float(np.nanmean(c_values)))
    else:
        legacy_c_scale.append(1.0)

    # -------------------------------------------------------------------------
    # Calculate a scalar mean limb-darkening coefficient.
    # -------------------------------------------------------------------------
    sci = str(legacy_results.iloc[row_i]["STAR"])

    pid = rplt.match_target_for_plot(
        tgt_info,
        sci,
        verbose=False
    )

    if pid is None:

        print("Could not determine u_LLD for %s" % sci)

        # Used only for old diagnostic plots.
        legacy_u_lld.append(0.3)

        continue

    available_u_columns = [
        column
        for column in u_lambda_columns
        if column in tgt_info.columns
    ]

    u_values = np.asarray(
        tgt_info.loc[pid, available_u_columns].values,
        dtype=float
    )

    u_values = u_values[np.isfinite(u_values)]

    if len(u_values) > 0:
        legacy_u_lld.append(float(np.nanmean(u_values)))
    else:
        print("No finite limb-darkening coefficients for %s" % sci)
        legacy_u_lld.append(0.3)


legacy_results["C_SCALE"] = legacy_c_scale
legacy_results["u_LLD"] = legacy_u_lld

# -------------------------------------------------------------------------
# All available reddened/dereddened diameter comparisons
# -------------------------------------------------------------------------
# Search automatically for columns such as:
#
# LDD_VW3      and LDD_VW3_dr
# LDD_VW4      and LDD_VW4_dr
#
# Then compare every available pair.

diameter_relations = []

for column in tgt_info.columns:

    if not str(column).startswith("LDD_"):
        continue

    if str(column).endswith("_dr"):
        continue

    dereddened_column = str(column) + "_dr"

    if dereddened_column in tgt_info.columns:
        diameter_relations.append(str(column))


print("\nDiameter relations with reddened and corrected values:")
print(diameter_relations)


for relation_1, relation_2 in itertools.combinations(
        diameter_relations, 2):

    corrected_1 = relation_1 + "_dr"
    corrected_2 = relation_2 + "_dr"

    label_1 = relation_1.replace("LDD_", "")
    label_2 = relation_2.replace("LDD_", "")

    plot_label = (
        "Diameter comparison: %s versus %s"
        % (label_1, label_2)
    )

    run_plot(
        plot_label,
        rplt.plot_diameter_comparison,
        np.asarray(tgt_info[relation_1], dtype=float),
        np.asarray(tgt_info[relation_2], dtype=float),
        np.asarray(tgt_info[corrected_1], dtype=float),
        np.asarray(tgt_info[corrected_2], dtype=float),
        label_1,
        label_2
    )

    safe_label_1 = label_1.replace("/", "_").replace(" ", "_")
    safe_label_2 = label_2.replace("/", "_").replace(" ", "_")

    copy_plot(
        "plots/angular_diameter_comp.pdf",
        os.path.join(
            "plots/diameter_comparisons",
            "angular_diameter_%s_vs_%s.pdf"
            % (safe_label_1, safe_label_2)
        )
    )


# =============================================================================
# 3. Visibility plots
# =============================================================================

# -------------------------------------------------------------------------
# Multi-page PDF using the older visibility plotting function
# -------------------------------------------------------------------------
run_plot(
    "All visibility fits",
    rplt.plot_all_vis2_fits,
    legacy_results,
    tgt_info
)


# -------------------------------------------------------------------------
# One file per target/sequence
# -------------------------------------------------------------------------
run_plot(
    "Single visibility plots",
    rplt.plot_single_vis2,
    legacy_results,
    e_wl_frac=e_wl_frac
)


# -------------------------------------------------------------------------
# Old paper visibility grid
# -------------------------------------------------------------------------
run_plot(
    "Paper visibility grid",
    rplt.plot_paper_vis2_fits,
    legacy_results,
    n_rows=8,
    n_cols=2
)


# -------------------------------------------------------------------------
# New joint-sequence visibility plot
# -------------------------------------------------------------------------
run_plot(
    "Joint sequence visibility plots new",
    rplt.plot_joint_seq_paper_vis2_fits,
    tgt_info,
    results,
    n_rows=4,
    n_cols=2,
    rasterize=False
)

copy_plot(
    "paper/joint_seq_vis2_plots.pdf",
    "paper/joint_seq_vis2_plots_new.pdf"
)


# -------------------------------------------------------------------------
# Old joint-sequence function
# -------------------------------------------------------------------------
# This is redundant but is intentionally included.

run_plot(
    "Joint sequence visibility plots old",
    rplt.plot_joint_seq_paper_vis2_fits_old,
    tgt_info,
    results,
    n_rows=4,
    n_cols=2,
    rasterize=False
)

copy_plot(
    "paper/joint_seq_vis2_plots.pdf",
    "paper/joint_seq_vis2_plots_old.pdf"
)


# Restore the new version as the default joint-sequence PDF.
if os.path.exists("paper/joint_seq_vis2_plots_new.pdf"):

    shutil.copyfile(
        "paper/joint_seq_vis2_plots_new.pdf",
        "paper/joint_seq_vis2_plots.pdf"
    )


# -------------------------------------------------------------------------
# Sidelobe plots for every results row
# -------------------------------------------------------------------------
run_plot(
    "All sidelobe visibility plots",
    rplt.plot_all_sidelobe_vis2_fits,
    tgt_info,
    results,
    output_dir="paper/sidelobes"
)


# =============================================================================
# 4. Fourier transforms of science targets
# =============================================================================

run_plot(
    "Fourier transforms for science stars",
    rplt.plot_science_fourier_transforms,
    tgt_info,
    results,
    output_file="plots/science_fourier_transforms.pdf",
    q_max=2.5E8,
    n_model_points=20000,
    use_predicted_if_missing=True
)

run_plot(
    "RA-Dec intensity maps for science stars",
    rplt.plot_science_intensity_maps,
    tgt_info,
    results,
    output_file="plots/science_intensity_maps.pdf",
    wavelength_index=2,
    n_pixels=400,
    field_factor=1.4,
    use_predicted_if_missing=True
)



# =============================================================================
# 5. Bootstrap plots
# =============================================================================

run_plot(
    "Bootstrap summary",
    rplt.plot_bootstrapping_summary,
    results,
    bs_results,
    n_bins=20,
    plot_cal_info=False,
    sequences=sequences,
    complete_sequences=complete_sequences,
    tgt_info=tgt_info,
    e_wl_frac=e_wl_frac
)

# =============================================================================
# 4. Bootstrap plots
# =============================================================================

run_plot(
    "Bootstrap summary",
    rplt.plot_bootstrapping_summary,
    results,
    bs_results,
    n_bins=20,
    plot_cal_info=True,
    sequences=sequences,
    complete_sequences=complete_sequences,
    tgt_info=tgt_info,
    e_wl_frac=e_wl_frac
)


# -------------------------------------------------------------------------
# Build the dictionary expected by plot_ldd_hists
# -------------------------------------------------------------------------
n_ldd_fit = {}

for star_id in bs_results.keys():

    try:
        values = np.asarray(
            bs_results[star_id]["LDD_FIT"].values,
            dtype=float
        )

        values = values[np.isfinite(values)]

        if len(values) > 0:
            n_ldd_fit[str(star_id)] = values

    except Exception as error:
        print(
            "Could not extract LDD bootstrap values for %s: %s"
            % (str(star_id), str(error))
        )


if len(n_ldd_fit) > 0:

    run_plot(
        "LDD bootstrap histograms",
        rplt.plot_ldd_hists,
        n_ldd_fit,
        n_bins=20
    )

else:
    print("Skipping plot_ldd_hists: no valid LDD bootstrap values")


# =============================================================================
# 5. C scale distribution
# =============================================================================

run_plot(
    "C scale histogram",
    rplt.plot_c_hist,
    results,
    n_bins=10
)


# =============================================================================
# 6. Distance distribution
# =============================================================================

def plot_and_save_distance_histogram():

    rplt.plot_distance_hists(tgt_info)

    plt.tight_layout()

    plt.savefig(
        "plots/distance_histogram.pdf"
    )

    plt.savefig(
        "plots/distance_histogram.png",
        dpi=200
    )


run_plot(
    "Distance histogram",
    plot_and_save_distance_histogram
)


# =============================================================================
# 7. Intrinsic-colour grid
# =============================================================================

# plot_bv_intrinsic requires a variable called grid.
# It is only run when that grid has already been loaded or calculated.

if "grid" in globals():

    run_plot(
        "Intrinsic B-V colour grid",
        rplt.plot_bv_intrinsic,
        grid
    )

else:

    print(
        "Skipping plot_bv_intrinsic: "
        "the variable 'grid' has not been defined"
    )


# =============================================================================
# 8. Extinction plots
# =============================================================================

# plot_extinction_hists requires a two-dimensional array called a_mags
# containing extinction for B, V, J, H, K, W1, W2, W3 and W4.

if "a_mags" in globals():

    run_plot(
        "Extinction histograms and extinction versus distance",
        rplt.plot_extinction_hists,
        a_mags,
        tgt_info
    )

else:

    print(
        "Skipping plot_extinction_hists: "
        "the variable 'a_mags' has not been defined"
    )


# =============================================================================
# 9. Claret versus STAGGER comparison
# =============================================================================

claret_file = "results/paper_results/diams_claret.csv"
stagger_file = "results/paper_results/diams_stagger.csv"


def plot_and_save_claret_stagger():

    rplt.plot_claret_vs_stagger_diam_comp()

    plt.tight_layout()

    plt.savefig(
        "paper/claret_vs_stagger_diam_comp.pdf"
    )

    plt.savefig(
        "paper/claret_vs_stagger_diam_comp.png",
        dpi=200
    )


if os.path.exists(claret_file) and os.path.exists(stagger_file):

    run_plot(
        "Claret versus STAGGER diameters",
        plot_and_save_claret_stagger
    )

else:

    print("Skipping Claret versus STAGGER comparison")
    print("Missing one or both files:")
    print("  %s" % claret_file)
    print("  %s" % stagger_file)


# =============================================================================
# 10. Raw visibility plots from the last bootstrap
# =============================================================================

def plot_last_bootstrap_oifits():
    """
    Plot the OIFITS files associated with the last bootstrap only.

    This avoids producing plots for all 500 copies of every observation.
    """

    output_dir = "plots/raw_vis2"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    final_bootstrap_index = n_bootstraps - 1
    expected_suffix = "_%i.fits" % final_bootstrap_index

    n_created = 0

    for root, directories, filenames in os.walk(results_path):

        for filename in filenames:

            if not filename.endswith(expected_suffix):
                continue

            if "oidataCalibrated" not in filename:
                continue

            input_file = os.path.join(root, filename)

            star_label = os.path.splitext(filename)[0]

            try:
                rplt.plot_vis2(
                    input_file,
                    star_label
                )

                plt.tight_layout()

                output_name = os.path.join(
                    output_dir,
                    "%s.pdf" % star_label
                )

                plt.savefig(output_name)

                print("Saved raw visibility plot:")
                print(output_name)

                n_created += 1

            except Exception as error:

                print(
                    "Failed raw visibility plot for %s: %s"
                    % (input_file, str(error))
                )

            finally:
                plt.close("all")
    if n_created == 0:
        raise RuntimeError(
        "No raw OIFITS plots were successfully created")
    print(
        "Created %i raw OIFITS visibility plots"
        % n_created
    )


run_plot(
    "Raw visibility plots from final bootstrap",
    plot_last_bootstrap_oifits
)


# =============================================================================
# 11. Presentation visibility curves
# =============================================================================

run_plot(
    "Presentation PIONIER and PAVO visibility curves",
    rplt.presentation_vis2_plot
)


# =============================================================================
# Final plotting summary
# =============================================================================

print("\n" + "=" * 79)
print("Finished generating plots")
print("Plot failure log:")
print(plot_failure_log)
print("=" * 79)

# =============================================================================
# Point-by-point visibility diagnostics
# =============================================================================
visibility_diagnostic_output = os.path.join(
    results_path,
    "analysis_diagnostics",
    "visibility_diagnostic_summary.pdf"
)

run_plot(
    "Visibility diagnostics by night and baseline",
    rplt.plot_visibility_diagnostic_summary,
    results,
    bs_results,
    tgt_info,
    output_file=visibility_diagnostic_output,
    bootstrap_index=0,
    sigma_threshold=3.0,
    raw_residual_threshold=0.05,
    e_wl_frac=e_wl_frac,
    star_filter=None,
    highlight_night=None,
    highlight_pair=None,
    highlight_baseline_range=None,
    highlight_wavelength_index=None,
    max_annotations=10,
    use_predicted_if_missing=True
)

