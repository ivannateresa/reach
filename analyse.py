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
import pickle

# Import plotting xy offsets map
import xy_map

# -----------------------------------------------------------------------------
# Setup & Loading
# -----------------------------------------------------------------------------
lb_pc = 70                          # The size of the local bubble in pc
use_plx_systematic =  False          # Use Stassun & Torres 18 plx offset
combined_fit = True                 # Fit for LDD for multiple seq at once
load_saved_results = False          # Load or do fitting fresh
assign_default_uncertainties = True # Give default errors to stars without
force_claret_params = False         # Force use of Claret+11 limb d. params
n_bootstraps = 10
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
results_folder = "26-05-20_i10"       # Final run for 1st draft
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
    print(tgt_info)
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

# Generate plots
print("Generating plots...")
rplt.plot_fbol_comp(tgt_info)
rplt.plot_hr_diagram(tgt_info, plot_isochrones_basti=True)
rplt.plot_casagrande_teff_comp(tgt_info, xy_map.teff)
#rplt.plot_lit_diam_comp(tgt_info, xy_map.lit_diam)
#rplt.plot_sidelobe_vis2_fit(tgt_info, results)  
#rplt.plot_joint_seq_paper_vis2_fits(tgt_info, results, n_rows=4, n_cols=2)
rplt.plot_colour_rel_diam_comp(tgt_info, 
                               xy_maps=(xy_map.vw3, xy_map.vw4, xy_map.bv_feh))
#rplt.plot_bootstrapping_summary(results, bs_results, plot_cal_info=False, 
#                                sequences=sequences, 
#                                complete_sequences=complete_sequences, 
#                                tgt_info=tgt_info)