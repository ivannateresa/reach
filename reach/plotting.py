"""File to contain various plotting functions of reach.
"""
from __future__ import division, print_function
import glob
import numpy as np
import pandas as pd
import itertools
import reach.diameters as rdiam
import reach.utils as rutils
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as plticker
import matplotlib.cm as cm
import matplotlib.transforms as transforms
import os
from scipy.special import jv
import matplotlib.cm as cm
import traceback

from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages


def clean_target_name_for_plot(name):
    """
    Clean target names for robust matching.

    Examples
    --------
    HR_2998    -> hr2998
    HR2998     -> hr2998
    alf_CMi    -> alfcmi
    psi Vel A  -> psivela
    HD  63734  -> hd63734
    """

    import pandas as pd

    if pd.isnull(name):
        return ""

    name = str(name)
    name = name.replace("_", "")
    name = name.replace(" ", "")
    name = name.replace(".", "")
    name = name.replace("-", "")
    name = name.replace("\t", "")
    name = name.lower()

    return name


def match_target_for_plot(tgt_info, sci, verbose=True):
    """
    Robustly match a science target name from results to the corresponding
    index in tgt_info.

    It searches in:
    - tgt_info index
    - Primary
    - Bayer_ID
    - Ref_ID_1
    - Ref_ID_2
    - Ref_ID_3
    - HD_ID
    - HP
    """

    sci_clean = clean_target_name_for_plot(sci)

    search_cols = [
        "Primary",
        "Bayer_ID",
        "Ref_ID_1",
        "Ref_ID_2",
        "Ref_ID_3",
        "HD_ID",
        "HP"
    ]

    search_cols = [col for col in search_cols if col in tgt_info.columns]

    matches = []

    # Match columns
    for col in search_cols:

        col_clean = tgt_info[col].apply(clean_target_name_for_plot)
        this_match = tgt_info.index[col_clean == sci_clean]

        if len(this_match) > 0:
            matches.extend(list(this_match))

    # Match index
    index_clean = []

    for idx in tgt_info.index:
        index_clean.append(clean_target_name_for_plot(idx))

    for i, idx_clean in enumerate(index_clean):
        if idx_clean == sci_clean:
            matches.append(tgt_info.index[i])

    # Remove duplicates
    matches_unique = []

    for m in matches:
        if m not in matches_unique:
            matches_unique.append(m)

    if len(matches_unique) == 0:

        if verbose:
            print("Could not match target in tgt_info:")
            print("  sci: %s" % sci)
            print("  cleaned: %s" % sci_clean)

        return None

    if len(matches_unique) > 1:

        if verbose:
            print("WARNING: multiple matches for %s:" % sci)
            print(matches_unique)
            print("Using first match: %s" % matches_unique[0])

    if verbose:
        print("Matched %s -> %s" % (sci, matches_unique[0]))

    return matches_unique[0]
#
def plot_diameter_comparison(diam_rel_1, diam_rel_2, diam_rel_1_dr, 
                            diam_rel_2_dr, diam_rel_1_label, diam_rel_2_label):
    """Function to compare two different measures of angular diameter (e.g. two
    different colour relations) before and after extinction correction.
    
    Parameters
    ----------
    diam_rel_1: float array
        Diameters from the first relation *before* extinction correction (mas)
    
    diam_rel_2: float array
        Diameters from the second relation *before* extinction correction (mas)
        
    diam_rel_1_dr: float array
        Diameters from the first relation *after* extinction correction (mas)
    
    diam_rel_2_dr: float array
        Diameters from the second relation *after* extinction correction (mas)
        
    diam_rel_1_label: string
        Name/label of the first relation (for legend)
        
    diam_rel_2_label: string
        Name/label of the second relation (for legend)
    """
    plt.close("all")
    plt.figure()
    plt.plot(diam_rel_1, diam_rel_2, "*", label="Reddened", alpha=0.5)
    plt.plot(diam_rel_1_dr, diam_rel_2_dr, "+", label="Corrected", alpha=0.5)
    plt.title("Angular diameter comparison for reddened/corrected photometry")
    plt.xlabel(diam_rel_1_label + "(mas)")
    plt.ylabel(diam_rel_2_label + "(mas)")
    plt.legend(loc="best")
    plt.xlim([0,5])
    plt.ylim([0,5])
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("plots/angular_diameter_comp.pdf")
    

def plot_bv_intrinsic(grid):
    """Function to plot the grid of (B-V) colours for visualisation/comparison
    purposes.
    
    Parameters
    ----------
    grid: pandas dataframe
        The grid mapping Teff to SpT and (B-V)_0
    """
    plt.close("all")
    plt.figure()
    plt.plot(grid["Teff"], grid["V"], "*-", label="V (Mamajek)")
    plt.plot(grid["Teff"], grid["skV"], "o", label="V (Schmidt-Kaler)")
    plt.plot(grid["Teff"], grid["IV"], "1-", label="IV (Mean V-III)")
    plt.plot(grid["Teff"], grid["III"], "x-", label="III (Schmidt-Kaler)")
    plt.plot(grid["Teff"], grid["II"], "+-", label="II (Schmidt-Kaler)")
    plt.plot(grid["Teff"], grid["Ib"], "v-", label="Ib (Schmidt-Kaler)")
    plt.plot(grid["Teff"], grid["Iab"], "s-", label="Iab (Schmidt-Kaler)")
    plt.plot(grid["Teff"], grid["Ia"], "d-", label="Ia (Schmidt-Kaler)")
    
    flip = True
    for row_i, row in grid.iterrows():
        if flip and row["Teff"] > 2400:
            plt.text(row["Teff"], 0, row.name, fontsize=7, rotation="vertical",
                     horizontalalignment="center")
            plt.axvline(row["Teff"], alpha=0.5, color="grey", linestyle="--")
            
        flip = not flip
    plt.xlabel(r"T$_{\rm eff}$")
    plt.ylabel(r"(B-V)$_0$")
    plt.legend(loc="best")
    plt.xlim([46000,2400])
    plt.xscale("log")
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("plots/intrinsic_colours.pdf")
    
    
def plot_extinction_hists(a_mags, tgt_info):
    """Function for plotting diagnostic extinction related plots.
    """
    plt.close("all")
    plt.figure()
    
    mag_labels = ["B", "V", "J", "H", "K", "W1", "W2", "W3", "W4"]
    
    for mag_i, mag in enumerate(a_mags.T):
        plt.hist(mag[~np.isnan(mag)], bins=25, label=mag_labels[mag_i], 
                 alpha=0.25)
    
    plt.title("Distribution of stellar extinction, B through W4 filters")
    plt.xlabel("Extinction (mags)")
    plt.ylabel("# Stars")
    plt.legend(loc="best")
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("plots/extinction_hists.pdf")
    
    plt.figure()
    dists = 1000/tgt_info["Plx"]
    
    for mag_i, mag in enumerate(a_mags.T): 
        plt.plot(dists, mag, "+", label=mag_labels[mag_i])
        
    ids = tgt_info.index.values
    
    for star_i, star in enumerate(ids):
        plt.text(dists[star_i], -0.25, star, fontsize=6, rotation="vertical",
                     horizontalalignment="center")
          
    plt.xlabel("Dist (pc)")
    plt.ylabel("Extinction (mags)")
    plt.legend(loc="best")
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("plots/extinction_vs_distance.pdf")
    
    
def plot_distance_hists(tgt_info):
    """Function to plot a distance histogram of all targets.
    """
    plt.close("all")
    plt.figure()
    
    plt.hist(1000/tgt_info["Plx"][~np.isnan(tgt_info["Plx"])], bins=25, 
             alpha=0.75)
    plt.xlabel("Distance (pc)")
    plt.ylabel("# Stars")
    
    
def plot_vis2_fit(sfreq, vis2, e_vis2, ldd_fit, e_ldd_fit, ldd_pred, 
                  e_ldd_pred, u_lld, target):
    """Function to plot squared calibrated visibilities, with curves for
    predicted diameter and fitted diameter.
    """
    x = np.arange(1*10**6, 25*10**7, 10000)
    y_fit = rdiam.calculate_vis2(x, ldd_fit, u_lld)
    y_fit_low = rdiam.calculate_vis2(x, ldd_fit - e_ldd_fit, u_lld)
    y_fit_high = rdiam.calculate_vis2(x, ldd_fit + e_ldd_fit, u_lld)    
    
    y_pred = rdiam.calculate_vis2(x, ldd_pred, u_lld)
    y_pred_low = rdiam.calculate_vis2(x, ldd_pred - e_ldd_pred, u_lld)
    y_pred_high = rdiam.calculate_vis2(x, ldd_pred + e_ldd_pred, u_lld)
    
    #plt.close("all")
    plt.figure()
    
    # Plot the data points and best fit curve
    plt.errorbar(sfreq, vis2, yerr=e_vis2, fmt=".", label="Data")
    
    plt.plot(x, y_fit, "--", 
             label=r"Fit ($\theta_{\rm LDD}$=%f $\pm$ %f, %0.2f%%)" 
                   % (ldd_fit, e_ldd_fit, e_ldd_fit/ldd_fit*100))
    plt.fill_between(x, y_fit_low, y_fit_high, alpha=0.25)
    
    # Plot the predicted diameter with error
    plt.plot(x, y_pred, "--", 
             label=r"Predicted ($\theta_{\rm LDD}$=%f $\pm$ %f, %0.2f%%)" 
                   % (ldd_pred, e_ldd_pred, e_ldd_pred/ldd_pred*100))
    plt.fill_between(x, y_pred_low, y_pred_high, alpha=0.25)
    
    plt.xlabel(r"Spatial Frequency (rad$^{-1})$")
    plt.ylabel(r"Visibility$^2$")
    plt.title(target + r" (%i vis$^2$ points)" % len(vis2))
    plt.legend(loc="best")
    plt.xlim([0.0,25E7])
    plt.ylim([0.0,1.0])
    plt.grid()
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("plots/vis2_fit.pdf")
    
    
def plot_all_vis2_fits(results, tgt_info):
    """Plot a single multi-page pdf of all fits using plot_vis2_fit
    """
    plt.close("all")
    with PdfPages("plots/bootstrapped_fits.pdf") as pdf:
        for star_i in np.arange(0, len(results)):
            #try:
            sci = results.iloc[star_i]["STAR"]
        
            #pid = tgt_info[tgt_info["Primary"]==sci].index.values[0]
            pid = match_target_for_plot(tgt_info,sci,verbose=True)
            if pid is None:
                print("Skipping plot_all_vis2_fits for %s" % sci)
                continue

            n_bl = len(results.iloc[star_i]["BASELINE"])
            n_wl = len(results.iloc[star_i]["WAVELENGTH"])
            bl_grid = np.tile(results.iloc[star_i]["BASELINE"], n_wl).reshape([n_wl, n_bl]).T
            wl_grid = np.tile(results.iloc[star_i]["WAVELENGTH"], n_bl).reshape([n_bl, n_wl])
    
            sfreq = (bl_grid / wl_grid).flatten()
            plot_vis2_fit(sfreq, results.iloc[star_i]["VIS2"].flatten(), 
                          results.iloc[star_i]["e_VIS2"].flatten(),  
                          results.iloc[star_i]["LDD_FIT"], 
                          results.iloc[star_i]["e_LDD_FIT"], 
                          tgt_info.loc[pid, "LDD_pred"],
                          tgt_info.loc[pid, "e_LDD_pred"],
                          np.nanmean(np.asarray(tgt_info.loc[pid, [
            "u_lambda_0",
            "u_lambda_1",
            "u_lambda_2",
            "u_lambda_3",
            "u_lambda_4",
            "u_lambda_5",
        ]].values,
        dtype=float
)),
                          sci)
            pdf.savefig()
            plt.close()
            #except:
                #print("Failed on star #%i, %s" % (star_i, sci))
            

def plot_ldd_hists(n_ldd_fit, n_bins=10):
    """Function to plot a grid of histogram for LDD realisations from each 
    bootstrapping run.
    """
    plt.close("all")
    fig, axes = plt.subplots(4, 5)
    axes = axes.flatten()
    
    # For each science target, plot a histogram of N LDD realisations
    for sci_i, sci in enumerate(n_ldd_fit.keys()):
        ldd_percentiles = np.percentile(n_ldd_fit[sci], [50, 84.1, 15.9]) 
        err_ldd = np.abs(ldd_percentiles[1:] - ldd_percentiles[0])
    
        axes[sci_i].hist(n_ldd_fit[sci], n_bins)
        
        text_x = ldd_percentiles[0]
        text_y = axes[sci_i].get_ylim()[1] * 0.95
        
        axes[sci_i].set_title(sci)
        y_height = axes[sci_i].get_ylim()[1]
        axes[sci_i].vlines(ldd_percentiles[0], 0, y_height, 
                           linestyles="dashed")
        axes[sci_i].vlines(ldd_percentiles[1], 0, y_height, 
                           colors="red", linestyles="dotted")
        axes[sci_i].vlines(ldd_percentiles[2], 0, y_height, 
                           colors="red", linestyles="dotted")
        axes[sci_i].text(text_x, text_y, 
                         r"$\theta_{\rm LDD}=$%0.4f +%0.4f / -%0.4f" 
                         % (ldd_percentiles[0], err_ldd[0], err_ldd[1]),
                         horizontalalignment="center", fontsize=7)
        #axes[sci_i].set_xlabel("LDD (mas)")
    
    n_bs = len(n_ldd_fit[sci])
    fig.suptitle("LDD histograms for %i bootstrapping iterations" % n_bs)
    fig.tight_layout()
    plt.gcf().set_size_inches(16, 9)
    plt.savefig("plots/ldd_hists.pdf")
    

def plot_bootstrapping_summary_old_old(results, bs_results, n_bins=20, 
                               plot_cal_info=True, sequences=None, 
                               complete_sequences=None, tgt_info=None, 
                               e_wl_frac=0.03):
    """Plot side by side vis^2 points and fit, with histogram of LDD dist.
    """
    plt.close("all")
    with PdfPages("plots/bootstrapped_summary.pdf") as pdf:
        for star_i in np.arange(0, len(results)):
            # Get the science target name
            sci = results.iloc[star_i]["STAR"]
            hd_id = results.iloc[star_i]["HD"]
            period = results.iloc[star_i]["PERIOD"]
            sequence = results.iloc[star_i]["SEQUENCE"]
            
            print(sci)
            
            if sequence == "combined":
                stitle = sci
                star_id = sci
            else:
                stitle = "%s (%s, %s)" % (sci, sequence, period)
                star_id = (sci, sequence, period)
            
            # -----------------------------------------------------------------
            # Plot vis^2 fits
            # -----------------------------------------------------------------
            n_bl = len(results.iloc[star_i]["BASELINE"])
            n_wl = len(results.iloc[star_i]["WAVELENGTH"])
            bl_grid = np.tile(results.iloc[star_i]["BASELINE"], 
                              n_wl).reshape([n_wl, n_bl]).T
            wl_grid = np.tile(results.iloc[star_i]["WAVELENGTH"], 
                              n_bl).reshape([n_bl, n_wl])
            
            sfreq = (bl_grid / wl_grid).flatten()
            
            vis2 = results.iloc[star_i]["VIS2"].flatten()
            e_vis2 = results.iloc[star_i]["e_VIS2"].flatten()
            ldd_fit = results.iloc[star_i]["LDD_FIT"]
            e_ldd_fit = results.iloc[star_i]["e_LDD_FIT"]
            
            ldd_pred = tgt_info.loc[hd_id]["LDD_pred"]
            e_ldd_pred = tgt_info.loc[hd_id]["e_LDD_pred"]
            
            u_lambdas = ['u_lambda_0', 'u_lambda_1', 'u_lambda_2', 'u_lambda_3',
                         'u_lambda_4', 'u_lambda_5']
            
            u_lld = np.mean(tgt_info.loc[hd_id][u_lambdas])
            
            #c_scale = results.iloc[star_i]["C_SCALE"]
            x = np.arange(1*10**6, 25*10**7, 10000)

            c_scale = 1
            n_points = (len(x),)
            s_lambda = 1
            
            y_fit = rdiam.calc_vis2(x, ldd_fit, c_scale, n_points, u_lld,
                                       s_lambda)
            y_fit_low = rdiam.calc_vis2(x, ldd_fit - e_ldd_fit, c_scale, 
                                           n_points, u_lld, s_lambda)
            y_fit_high = rdiam.calc_vis2(x, ldd_fit + e_ldd_fit, c_scale, 
                                            n_points, u_lld, s_lambda)    
    
            y_pred = rdiam.calc_vis2(x, ldd_pred, 1, n_points, u_lld,
                                        s_lambda)
            y_pred_low = rdiam.calc_vis2(x, ldd_pred - e_ldd_pred, 1, 
                                            n_points, u_lld, s_lambda)
            y_pred_high = rdiam.calc_vis2(x, ldd_pred + e_ldd_pred, 1, 
                                             n_points, u_lld, s_lambda)
    
            fig, axes = plt.subplots(1, 2)
            axes = axes.flatten()
            
            # Setup lower panel for residuals
            divider = make_axes_locatable(axes[0])
            res_ax = divider.append_axes("bottom", size="20%", pad=0)
            axes[0].figure.add_axes(res_ax)
    
            # Plot the data points and best fit curve
            axes[0].errorbar(sfreq, vis2, xerr=sfreq*e_wl_frac,
                            yerr=e_vis2, fmt=".", 
                            label="Data", elinewidth=0.1, capsize=0.2, 
                            capthick=0.1)
    
            axes[0].plot(x, y_fit, "--", 
                     label=r"Fit ($\theta_{\rm LDD}$=%f $\pm$ %f, %0.2f%%)" 
                           % (ldd_fit, e_ldd_fit, e_ldd_fit/ldd_fit*100))
            #axes[0].fill_between(x, y_fit_low, y_fit_high, alpha=0.25,
                                 #color="C1")
    
            # Plot the predicted diameter with error
            label=(r"Predicted ($\theta_{\rm LDD}$=%f $\pm$ %f, %0.2f%%)" 
                   % (ldd_pred, e_ldd_pred, e_ldd_pred/ldd_pred*100))
            axes[0].plot(x, y_pred, "--", label=label)
            #axes[0].fill_between(x, y_pred_low, y_pred_high, alpha=0.25, 
                                 #color="C2")
    
            #axes[0].set_xlabel(r"Spatial Frequency (rad$^{-1})$")
            axes[0].set_ylabel(r"Visibility$^2$")
            axes[0].set_title(stitle + r" (%i vis$^2$ points)" % len(vis2))
            axes[0].legend(loc="best")
            axes[0].set_xlim([0.0,25E7])
            axes[0].set_ylim([0.0,1.1])
            axes[0].grid()
            
            # Plot residuals below the vis2 plot
            axes[0].set_xticks([])

            n_points = (len(sfreq),)
            s_lambda = 1

            residuals = vis2 - rdiam.calc_vis2(sfreq, ldd_fit, c_scale,
                                               n_points, u_lld, s_lambda)
            
            res_ax.errorbar(sfreq, residuals, xerr=sfreq*e_wl_frac,
                            yerr=e_vis2, fmt=".", 
                            label="Residuals", elinewidth=0.1, capsize=0.2, 
                            capthick=0.1)
            res_ax.set_xlim([0.0,25E7])
            res_ax.hlines(0, 0, 25E7, linestyles="dotted")
            res_ax.set_ylabel("Residuals")
            res_ax.set_xlabel(r"Spatial Frequency (rad$^{-1})$")
            
            # -----------------------------------------------------------------
            # Plot calibrator angular diameters and magnitudes for diagnostic
            # purposes
            # -----------------------------------------------------------------
            if plot_cal_info:
                sci_h = tgt_info[tgt_info["Primary"]==sci]["Hmag"].values[0]
                sci_e_h = tgt_info[tgt_info["Primary"]==sci]["e_Hmag"].values[0]
                sci_jsdc = tgt_info[tgt_info["Primary"]==sci]["JSDC_LDD"].values[0]
                
                if sequence == "combined":
                    stars = set(sequences[(period, sci, "bright")][::2]
                                + sequences[(period, sci, "faint")][::2])
                else:
                    stars = sequences[(period, sci, sequence)][::2]
                    
                stars = [star.replace("_", "").replace(".","").replace(" ", "") 
                            for star in stars]
                            
                stars = rutils.get_unique_key(tgt_info, stars)
                
                # Print science details
                text_x = axes[0].get_xlim()[1] * 5/8 
                text_y = 3/4 + 0.125
                text = "C = %0.2f" % c_scale
                axes[0].text(text_x, text_y, text, fontsize="x-small",
                             horizontalalignment="center")
                
                text_x = axes[0].get_xlim()[1] * 5/8 
                text_y = 3/4 + 0.1
                
                text = (r"%s, $\theta_{\rm LDD}=%0.3f$, "
                        r"$\theta_{\rm JSDC}$=%0.3f, Hmag=%0.2f$\pm$ %0.2f" 
                        % (sci, ldd_fit, sci_jsdc, sci_h, sci_e_h))
                
                axes[0].text(text_x, text_y, text, fontsize="xx-small",
                             horizontalalignment="center")
                
                cal_ldd = []
                cal_h = []
                 
                # Print ESO's quality information about the observations
                if sequence == "bright" or sequence == "combined":
                    # Bright
                    text_x = axes[0].get_xlim()[1] * 5/8 
                    text_y = 3/4 + 0.075
                    
                    text = "Bright Quality: %s" % complete_sequences[(period, sci, "bright")][1]
                    
                    axes[0].text(text_x, text_y, text, fontsize="xx-small",
                                 horizontalalignment="center")
                                 
                # Faint
                if sequence == "faint" or sequence == "combined":
                    text_x = axes[0].get_xlim()[1] * 5/8 
                    text_y = 3/4 + 0.05
                    
                    text = "Faint Quality: %s" % complete_sequences[(period, sci, "faint")][1]
                    
                    axes[0].text(text_x, text_y, text, fontsize="xx-small",
                                 horizontalalignment="center")
                 
                             
                # Print calibrator details
                for star_i, star in enumerate(stars):
                    star_info = tgt_info.loc[star]
                    
                    text_x = axes[0].get_xlim()[1] * 5/8 
                    text_y = 3/4 - (0.025 * star_i)
                    
                    # Cross out stars we have ignored
                    if star_info["Quality"] == "BAD":
                        st = u"\u0336"
                        star = st.join(star) + st
                    
                    ldd_diff = star_info["LDD_pred"] - star_info["JSDC_LDD"]
                    
                    text = (r"%s, Hmag=%0.2f$\pm$ %0.2f, $\theta_{\rm LDD}=%0.3f\pm %0.3f$ (%s), "
                            r"$\theta_{\rm JSDC}=%0.3f\pm %0.3f$,   "
                            r"[$\theta_{\rm diff}=%0.3f$]" 
                            % (star, star_info["Hmag"], star_info["e_Hmag"],
                               star_info["LDD_pred"], 
                               star_info["e_LDD_pred"], star_info["LDD_rel"],
                               star_info["JSDC_LDD"], star_info["e_JSDC_LDD"],
                               ldd_diff))
                    
                    cal_ldd.append(star_info["LDD_pred"])
                    cal_h.append(star_info["Hmag"])
                    
                    axes[0].text(text_x, text_y, text, fontsize="xx-small",
                                 horizontalalignment="center")
            
                                 
                # Print average 
                text = r"$\theta_{\rm LDD (AVG)}=%0.3f$" % np.nanmean(cal_ldd)
                axes[0].text(text_x, text_y-0.05, text, fontsize="x-small",
                                 horizontalalignment="center")
                                 
                text = r"Hmag$_{\rm AVG}=%0.3f$" % np.nanmean(cal_h)
                axes[0].text(text_x, text_y-0.075, text, fontsize="x-small",
                                 horizontalalignment="center")
                
            
            # -----------------------------------------------------------------
            # Plot histograms
            # -----------------------------------------------------------------
                        # -----------------------------------------------------------------
            # Plot histograms
            # -----------------------------------------------------------------

            ldd_samples = np.asarray(bs_results[star_id]["LDD_FIT"].values, dtype=float)

            # Quitar NaN e infinitos
            ldd_samples = ldd_samples[np.isfinite(ldd_samples)]

            print("DEBUG hist:", star_id)
            print("N valid LDD_FIT =", len(ldd_samples))

            if len(ldd_samples) == 0:
                print("WARNING: no valid LDD_FIT values for", star_id)

                axes[1].text(0.5, 0.5, "No valid bootstrap values",
                            transform=axes[1].transAxes,
                            horizontalalignment="center",
                            verticalalignment="center")
                axes[1].set_title(stitle)

            else:
                ldd_min = np.nanmin(ldd_samples)
                ldd_max = np.nanmax(ldd_samples)

                print("min =", ldd_min, "max =", ldd_max)

                # Si todos los valores son iguales, hist() falla.
                # Por eso le damos un rango artificial.
                if ldd_min == ldd_max:
                    print("WARNING: all LDD_FIT values are identical for", star_id)

                    if np.isfinite(ldd_fit) and np.isfinite(e_ldd_fit) and e_ldd_fit > 0:
                        hist_range = (ldd_fit - 5.0 * e_ldd_fit,
                                    ldd_fit + 5.0 * e_ldd_fit)
                    elif np.isfinite(ldd_fit):
                        hist_range = (ldd_fit - 0.01,
                                    ldd_fit + 0.01)
                    else:
                        hist_range = (ldd_min - 0.01,
                                    ldd_max + 0.01)

                    axes[1].hist(ldd_samples, n_bins, range=hist_range)

                else:
                    axes[1].hist(ldd_samples, n_bins)

                text_y = axes[1].get_ylim()[1]

                axes[1].set_title(stitle + r" (${\rm N}_{\rm bootstraps} = $%i)"
                                % len(ldd_samples))

                y_height = axes[1].get_ylim()[1]

                axes[1].vlines(ldd_fit, 0, y_height, linestyles="dashed")
                axes[1].vlines(ldd_fit - e_ldd_fit, 0, y_height,
                            colors="red", linestyles="dotted")
                axes[1].vlines(ldd_fit + e_ldd_fit, 0, y_height,
                            colors="red", linestyles="dotted")

                axes[1].text(ldd_fit, text_y,
                            r"$\theta_{\rm LDD}=%0.4f \pm%0.4f$"
                            % (ldd_fit, e_ldd_fit),
                            horizontalalignment="center")
                
            #axes[1].hist(bs_results[star_id]["LDD_FIT"].values.tolist(), n_bins)
        
            #text_y = axes[1].get_ylim()[1]
        
            #axes[1].set_title(stitle + r" (${\rm N}_{\rm bootstraps} = $%i)" 
             #                % len(bs_results[star_id]["LDD_FIT"].values.tolist()))
            #y_height = axes[1].get_ylim()[1]
            #axes[1].vlines(ldd_fit, 0, y_height, linestyles="dashed")
            #axes[1].vlines(ldd_fit-e_ldd_fit, 0, y_height, colors="red", 
            #               linestyles="dotted")
            #axes[1].vlines(ldd_fit+e_ldd_fit, 0, y_height, colors="red", 
              #             linestyles="dotted")
            #axes[1].text(ldd_fit, text_y, r"$\theta_{\rm LDD}=%0.4f \pm%0.4f$" 
            #            % (ldd_fit, e_ldd_fit), horizontalalignment="center") 
            
            plt.gcf().set_size_inches(16, 9)
            pdf.savefig()
            plt.close()

def plot_bootstrapping_summary(
        results,
        bs_results,
        n_bins=20,
        plot_cal_info=True,
        sequences=None,
        complete_sequences=None,
        tgt_info=None,
        e_wl_frac=0.03):
    """
    Plot the corrected VIS2 fit and the bootstrap LDD distribution.

    For combined fits, each observing sequence can have its own
    fitted C_SCALE. Each VIS2 sequence and its uncertainty are divided
    by the corresponding C value. The normalized LDD model is then
    evaluated using C=1.

    One page is produced for each science target.

    Parameters
    ----------
    results : pandas.DataFrame
        Final summarized fitting results.

    bs_results : dict
        Bootstrap results for every science target.

    n_bins : int
        Number of histogram bins.

    plot_cal_info : bool
        Include science and calibrator information.

    sequences : dict or None
        Dictionary containing CAL-SCI sequences.

    complete_sequences : dict or None
        Dictionary containing sequence-quality information.

    tgt_info : pandas.DataFrame
        Target information table.

    e_wl_frac : float
        Fractional wavelength uncertainty.

    Returns
    -------
    output_file : str
        Path to the multipage PDF.
    """

    # ========================================================================
    # Small helper functions
    # ========================================================================

    def safe_float(value):
        """
        Convert a value to float. Return NaN when conversion fails.
        """

        try:
            return float(value)
        except Exception:
            return np.nan


    def format_number(value, fmt="%.3f"):
        """
        Format a finite number safely.
        """

        value = safe_float(value)

        if np.isfinite(value):
            return fmt % value

        return "unavailable"


    def format_target(target):
        """
        Apply REACH target formatting without stopping the plot.
        """

        try:
            return rutils.format_id(target)
        except Exception:
            return str(target)


    # ========================================================================
    # Basic checks
    # ========================================================================

    if tgt_info is None:
        raise ValueError(
            "tgt_info must be provided to plot_bootstrapping_summary"
        )

    if results is None or len(results) == 0:
        raise ValueError(
            "results is empty in plot_bootstrapping_summary"
        )

    if bs_results is None:
        bs_results = {}

    plt.close("all")

    output_dir = "plots"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(
        output_dir,
        "bootstrapped_summary.pdf"
    )

    print("")
    print("=" * 79)
    print("Creating bootstrap summary")
    print("Output file:")
    print(output_file)
    print("=" * 79)

    n_successful_pages = 0
    n_failed_pages = 0

    # ========================================================================
    # Multipage output
    # ========================================================================

    with PdfPages(output_file) as pdf:

        for result_i in xrange(len(results)):

            row = results.iloc[result_i]

            sci = str(row["STAR"])

            print("")
            print("=" * 79)
            print(
                "Bootstrapping summary for %s"
                % sci
            )
            print("=" * 79)

            try:

                # ============================================================
                # Basic target information
                # ============================================================

                hd_id = row["HD"]
                period = row["PERIOD"]
                sequence = str(row["SEQUENCE"])

                if sequence == "combined":

                    stitle = sci
                    star_id = sci

                else:

                    stitle = "%s (%s, %s)" % (
                        sci,
                        sequence,
                        period
                    )

                    star_id = (
                        sci,
                        sequence,
                        period
                    )

                # ------------------------------------------------------------
                # Match the target against tgt_info
                # ------------------------------------------------------------

                if hd_id not in tgt_info.index:

                    matched_id = match_target_for_plot(
                        tgt_info,
                        sci,
                        verbose=True
                    )

                    if matched_id is None:
                        raise ValueError(
                            "Could not match %s in tgt_info"
                            % sci
                        )

                    hd_id = matched_id

                # ============================================================
                # Visibility arrays
                # ============================================================

                baselines = np.asarray(
                    row["BASELINE"],
                    dtype=float
                ).ravel()

                wavelengths = np.asarray(
                    row["WAVELENGTH"],
                    dtype=float
                ).ravel()

                vis2_matrix = np.asarray(
                    row["VIS2"],
                    dtype=float
                )

                e_vis2_matrix = np.asarray(
                    row["e_VIS2"],
                    dtype=float
                )
                                # ============================================================
                # Remove completely invalid baseline rows
                # ============================================================

                invalid_baseline_rows = ~np.isfinite(baselines)

                print("Invalid baseline rows:", np.where(invalid_baseline_rows)[0])

                if np.any(invalid_baseline_rows):

                    print(
                        "Removing %i invalid baseline row(s) for %s"
                        % (
                            np.sum(invalid_baseline_rows),
                            sci
                        )
                    )

                    baselines = baselines[
                        ~invalid_baseline_rows
                    ]

                    vis2_matrix = vis2_matrix[
                        ~invalid_baseline_rows,
                        :
                    ]

                    e_vis2_matrix = e_vis2_matrix[
                        ~invalid_baseline_rows,
                        :
                    ]

                n_bl = len(baselines)
                n_wl = len(wavelengths)

                if n_bl == 0:
                    raise ValueError(
                        "No baselines available for %s"
                        % sci
                    )

                if n_wl == 0:
                    raise ValueError(
                        "No wavelengths available for %s"
                        % sci
                    )

                # ------------------------------------------------------------
                # Ensure VIS2 is n_baselines x n_wavelengths
                # ------------------------------------------------------------

                expected_size = n_bl * n_wl

                if vis2_matrix.ndim == 1:

                    if vis2_matrix.size != expected_size:
                        raise ValueError(
                            "Cannot reshape VIS2 for %s: "
                            "size=%i, expected=%i"
                            % (
                                sci,
                                vis2_matrix.size,
                                expected_size
                            )
                        )

                    vis2_matrix = vis2_matrix.reshape(
                        n_bl,
                        n_wl
                    )

                if e_vis2_matrix.ndim == 1:

                    if e_vis2_matrix.size != expected_size:
                        raise ValueError(
                            "Cannot reshape e_VIS2 for %s: "
                            "size=%i, expected=%i"
                            % (
                                sci,
                                e_vis2_matrix.size,
                                expected_size
                            )
                        )

                    e_vis2_matrix = e_vis2_matrix.reshape(
                        n_bl,
                        n_wl
                    )

                if vis2_matrix.shape != (n_bl, n_wl):
                    raise ValueError(
                        "Unexpected VIS2 shape for %s: %s; "
                        "expected (%i, %i)"
                        % (
                            sci,
                            str(vis2_matrix.shape),
                            n_bl,
                            n_wl
                        )
                    )

                if vis2_matrix.shape != e_vis2_matrix.shape:
                    raise ValueError(
                        "VIS2 and e_VIS2 have different shapes "
                        "for %s: %s versus %s"
                        % (
                            sci,
                            str(vis2_matrix.shape),
                            str(e_vis2_matrix.shape)
                        )
                    )

                # ============================================================
                # Spatial-frequency matrix
                # ============================================================

                baseline_grid = np.repeat(
                    baselines[:, np.newaxis],
                    n_wl,
                    axis=1
                )

                wavelength_grid = np.repeat(
                    wavelengths[np.newaxis, :],
                    n_bl,
                    axis=0
                )

                sfreq_matrix = (
                    baseline_grid
                    / wavelength_grid
                )

                # ============================================================
                # Fitted and predicted diameters
                # ============================================================

                ldd_fit = safe_float(
                    row["LDD_FIT"]
                )

                e_ldd_fit = safe_float(
                    row["e_LDD_FIT"]
                )

                valid_ldd_fit = (
                    np.isfinite(ldd_fit)
                    and np.isfinite(e_ldd_fit)
                    and ldd_fit > 0
                    and e_ldd_fit >= 0
                )

                if not valid_ldd_fit:

                    print(
                        "WARNING: invalid fitted LDD for %s: "
                        "LDD_FIT=%s, e_LDD_FIT=%s"
                        % (
                            sci,
                            str(ldd_fit),
                            str(e_ldd_fit)
                        )
                    )

                # IMPORTANT:
                # Define the predicted values before validating them.
                ldd_pred = safe_float(
                    tgt_info.loc[
                        hd_id,
                        "LDD_pred"
                    ]
                )

                e_ldd_pred = safe_float(
                    tgt_info.loc[
                        hd_id,
                        "e_LDD_pred"
                    ]
                )

                valid_ldd_pred = (
                    np.isfinite(ldd_pred)
                    and np.isfinite(e_ldd_pred)
                    and ldd_pred > 0
                    and e_ldd_pred >= 0
                )

                if not valid_ldd_pred:

                    print(
                        "WARNING: invalid predicted LDD for %s: "
                        "LDD_pred=%s, e_LDD_pred=%s"
                        % (
                            sci,
                            str(ldd_pred),
                            str(e_ldd_pred)
                        )
                    )

                # ============================================================
                # Limb-darkening coefficient
                # ============================================================

                u_lambda_columns = [
                    "u_lambda_0",
                    "u_lambda_1",
                    "u_lambda_2",
                    "u_lambda_3",
                    "u_lambda_4",
                    "u_lambda_5",
                ]

                available_u_columns = [
                    column
                    for column in u_lambda_columns
                    if column in tgt_info.columns
                ]

                if len(available_u_columns) > 0:

                    u_values = np.asarray(
                        tgt_info.loc[
                            hd_id,
                            available_u_columns
                        ].values,
                        dtype=float
                    ).ravel()

                    u_values = u_values[
                        np.isfinite(u_values)
                    ]

                else:

                    u_values = np.array(
                        [],
                        dtype=float
                    )

                if len(u_values) == 0:

                    print(
                        "WARNING: no finite limb-darkening "
                        "coefficients for %s; using u=0.3"
                        % sci
                    )

                    u_lld = 0.3

                else:

                    u_lld = float(
                        np.nanmean(u_values)
                    )

                # ============================================================
                # Read C_SCALE
                # ============================================================

                c_values = np.asarray(
                    row["C_SCALE"],
                    dtype=float
                ).ravel()
                

          

                if len(c_values) == 0:

                    print(
                        "WARNING: empty C_SCALE for %s; using C=1"
                        % sci
                    )

                    c_values = np.array(
                        [1.0],
                        dtype=float
                    )

                invalid_c = (
                    ~np.isfinite(c_values)
                    | (c_values <= 0)
                )

                if np.any(invalid_c):

                    print(
                        "WARNING: invalid C_SCALE for %s:"
                        % sci
                    )

                    print(c_values)

                    c_values[invalid_c] = 1.0

                n_c = len(c_values)
                print(n_c, n_bl)
                print("")
                print("DEBUG C MAPPING")
                print("STAR:", sci)
                print("SEQUENCE:", sequence)
                print("SEQ_ORDER:", row.get("SEQ_ORDER", None))
                print("C_SCALE:", c_values)
                print("n_C:", len(c_values))
                print("n_baselines:", n_bl)
                print("BASELINE:", baselines)
                print("VIS2 shape:", vis2_matrix.shape)
                print("")
                if n_bl % n_c != 0:
                    raise ValueError(
                        "Cannot associate C_SCALE with VIS2 rows "
                        "for %s: n_bl=%i, n_C=%i, C=%s"
                        % (
                            sci,
                            n_bl,
                            n_c,
                            str(c_values)
                        )
                    )

                n_bl_per_c = int(
                    n_bl / n_c
                )

                # One C for every baseline row.
                c_per_baseline = np.repeat(
                    c_values,
                    n_bl_per_c
                )

                # Repeat each baseline C over all wavelength channels.
                c_matrix = np.repeat(
                    c_per_baseline[:, np.newaxis],
                    n_wl,
                    axis=1
                )

                if c_matrix.shape != vis2_matrix.shape:
                    raise ValueError(
                        "C matrix and VIS2 have different shapes "
                        "for %s: %s versus %s"
                        % (
                            sci,
                            str(c_matrix.shape),
                            str(vis2_matrix.shape)
                        )
                    )

                # ============================================================
                # Apply C correction
                # ============================================================

                vis2_corrected_matrix = (
                    vis2_matrix
                    / c_matrix
                )

                e_vis2_corrected_matrix = (
                    e_vis2_matrix
                    / c_matrix
                )

                sfreq = sfreq_matrix.flatten()

                vis2_corrected = (
                    vis2_corrected_matrix.flatten()
                )

                e_vis2_corrected = (
                    e_vis2_corrected_matrix.flatten()
                )

                valid_points = (
                    np.isfinite(sfreq)
                    & np.isfinite(vis2_corrected)
                    & np.isfinite(e_vis2_corrected)
                    & (e_vis2_corrected > 0)
                )

                sfreq_plot = sfreq[
                    valid_points
                ]

                vis2_plot = vis2_corrected[
                    valid_points
                ]

                e_vis2_plot = e_vis2_corrected[
                    valid_points
                ]

                print(
                    "C_SCALE values: %s"
                    % str(c_values)
                )

                print(
                    "Baseline rows per C: %i"
                    % n_bl_per_c
                )

                print(
                    "Valid VIS2 points: %i / %i"
                    % (
                        len(vis2_plot),
                        len(vis2_corrected)
                    )
                )

                # ============================================================
                # C labels
                # ============================================================

                try:
                    seq_order = row["SEQ_ORDER"]
                except Exception:
                    seq_order = []

                c_label_parts = []

                for c_i, c_value in enumerate(c_values):

                    sequence_name = None

                    try:

                        sequence_name = str(
                            seq_order[c_i][1]
                        )

                    except Exception:
                        sequence_name = None

                    if sequence_name is None:

                        c_label_parts.append(
                            "C%i=%.3f"
                            % (
                                c_i + 1,
                                c_value
                            )
                        )

                    else:

                        c_label_parts.append(
                            "%s=%.3f"
                            % (
                                sequence_name,
                                c_value
                            )
                        )

                c_text = ", ".join(
                    c_label_parts
                )

                # ============================================================
                # Model curves
                # ============================================================

                x_model = np.arange(
                    1.0E6,
                    25.0E7,
                    10000.0
                )

                n_points_model = (
                    len(x_model),
                )

                c_model = 1.0
                s_lambda = 1.0

                if valid_ldd_fit:

                    y_fit = rdiam.calc_vis2(
                        x_model,
                        ldd_fit,
                        c_model,
                        n_points_model,
                        u_lld,
                        s_lambda
                    )

                    y_fit_low = rdiam.calc_vis2(
                        x_model,
                        ldd_fit - e_ldd_fit,
                        c_model,
                        n_points_model,
                        u_lld,
                        s_lambda
                    )

                    y_fit_high = rdiam.calc_vis2(
                        x_model,
                        ldd_fit + e_ldd_fit,
                        c_model,
                        n_points_model,
                        u_lld,
                        s_lambda
                    )

                else:

                    y_fit = None
                    y_fit_low = None
                    y_fit_high = None

                if valid_ldd_pred:

                    y_pred = rdiam.calc_vis2(
                        x_model,
                        ldd_pred,
                        1.0,
                        n_points_model,
                        u_lld,
                        s_lambda
                    )

                    y_pred_low = rdiam.calc_vis2(
                        x_model,
                        ldd_pred - e_ldd_pred,
                        1.0,
                        n_points_model,
                        u_lld,
                        s_lambda
                    )

                    y_pred_high = rdiam.calc_vis2(
                        x_model,
                        ldd_pred + e_ldd_pred,
                        1.0,
                        n_points_model,
                        u_lld,
                        s_lambda
                    )

                else:

                    y_pred = None
                    y_pred_low = None
                    y_pred_high = None

                # ============================================================
                # Create figure
                # ============================================================

                fig, axes = plt.subplots(
                    1,
                    2
                )

                axes = np.atleast_1d(
                    axes
                ).flatten()

                fig.set_size_inches(
                    16,
                    9
                )

                divider = make_axes_locatable(
                    axes[0]
                )

                res_ax = divider.append_axes(
                    "bottom",
                    size="22%",
                    pad=0.05
                )

                # ============================================================
                # Corrected visibility data
                # ============================================================

                if len(vis2_plot) > 0:

                    axes[0].errorbar(
                        sfreq_plot,
                        vis2_plot,
                        xerr=sfreq_plot * e_wl_frac,
                        yerr=e_vis2_plot,
                        fmt=".",
                        label=r"Data corrected by fitted $C$",
                        elinewidth=0.1,
                        capsize=0.2,
                        capthick=0.1
                    )

                else:

                    axes[0].text(
                        0.5,
                        0.5,
                        "No valid visibility points",
                        transform=axes[0].transAxes,
                        horizontalalignment="center",
                        verticalalignment="center"
                    )

                # ------------------------------------------------------------
                # Fitted diameter curve
                # ------------------------------------------------------------

                if valid_ldd_fit:

                    fit_label = (
                        r"Fit "
                        r"($\theta_{\rm LDD}=%.4f\pm%.4f$ mas, %.2f%%)"
                        % (
                            ldd_fit,
                            e_ldd_fit,
                            100.0 * e_ldd_fit / ldd_fit
                        )
                    )

                    axes[0].plot(
                        x_model,
                        y_fit,
                        "--",
                        label=fit_label
                    )

                    axes[0].fill_between(
                        x_model,
                        y_fit_low,
                        y_fit_high,
                        alpha=0.15
                    )

                else:

                    axes[0].text(
                        0.5,
                        0.87,
                        "No valid fitted LDD",
                        transform=axes[0].transAxes,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize="large"
                    )

                # ------------------------------------------------------------
                # Predicted diameter curve
                # ------------------------------------------------------------

                if valid_ldd_pred:

                    predicted_label = (
                        r"Predicted "
                        r"($\theta_{\rm LDD}=%.4f\pm%.4f$ mas, %.2f%%)"
                        % (
                            ldd_pred,
                            e_ldd_pred,
                            100.0 * e_ldd_pred / ldd_pred
                        )
                    )

                    axes[0].plot(
                        x_model,
                        y_pred,
                        "--",
                        label=predicted_label
                    )

                    axes[0].fill_between(
                        x_model,
                        y_pred_low,
                        y_pred_high,
                        alpha=0.15
                    )

                # ------------------------------------------------------------
                # Main panel formatting
                # ------------------------------------------------------------

                axes[0].set_ylabel(
                    r"Corrected visibility$^2$"
                )

                axes[0].set_title(
                    stitle
                    + r" (%i valid vis$^2$ points)"
                    % len(vis2_plot)
                )

                axes[0].set_xlim(
                    [0.0, 25.0E7]
                )

                if len(vis2_plot) > 0:

                    vis2_upper = np.nanpercentile(
                        vis2_plot,
                        99
                    )

                    y_max = max(
                        1.1,
                        vis2_upper + 0.05
                    )

                    y_max = min(
                        y_max,
                        1.5
                    )

                else:

                    y_max = 1.1

                axes[0].set_ylim(
                    [0.0, y_max]
                )

                axes[0].grid()

                axes[0].set_xticklabels(
                    []
                )

                handles, labels = (
                    axes[0].get_legend_handles_labels()
                )

                if len(handles) > 0:

                    axes[0].legend(
                        loc="best",
                        fontsize="small"
                    )

                axes[0].text(
                    0.03,
                    0.05,
                    r"$C$: " + c_text,
                    transform=axes[0].transAxes,
                    fontsize="small",
                    horizontalalignment="left",
                    verticalalignment="bottom",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="none"
                    )
                )

                # ============================================================
                # Residuals
                # ============================================================

                if valid_ldd_fit and len(sfreq_plot) > 0:

                    n_points_residuals = (
                        len(sfreq_plot),
                    )

                    model_at_data = rdiam.calc_vis2(
                        sfreq_plot,
                        ldd_fit,
                        1.0,
                        n_points_residuals,
                        u_lld,
                        s_lambda
                    )

                    residuals = (
                        vis2_plot
                        - model_at_data
                    )

                    res_ax.errorbar(
                        sfreq_plot,
                        residuals,
                        xerr=sfreq_plot * e_wl_frac,
                        yerr=e_vis2_plot,
                        fmt=".",
                        elinewidth=0.1,
                        capsize=0.2,
                        capthick=0.1
                    )

                    res_ax.hlines(
                        0.0,
                        0.0,
                        25.0E7,
                        linestyles="dotted"
                    )

                else:

                    res_ax.text(
                        0.5,
                        0.5,
                        "Residuals unavailable",
                        transform=res_ax.transAxes,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize="small"
                    )

                res_ax.set_xlim(
                    [0.0, 25.0E7]
                )

                res_ax.set_ylabel(
                    "Residuals"
                )

                res_ax.set_xlabel(
                    r"Spatial frequency (rad$^{-1}$)"
                )

                # ============================================================
                # Science and calibrator information
                # ============================================================

                if plot_cal_info:

                    info_lines = []

                    info_lines.append(
                        "C: " + c_text
                    )

                    science_info = tgt_info.loc[
                        hd_id
                    ]

                    sci_h = science_info.get(
                        "Hmag",
                        np.nan
                    )

                    sci_e_h = science_info.get(
                        "e_Hmag",
                        np.nan
                    )

                    sci_jsdc = science_info.get(
                        "JSDC_LDD",
                        np.nan
                    )

                    info_lines.append(
                        "%s: fit=%s mas, JSDC=%s mas, H=%s +/- %s"
                        % (
                            format_target(sci),
                            format_number(ldd_fit),
                            format_number(sci_jsdc),
                            format_number(sci_h, "%.2f"),
                            format_number(sci_e_h, "%.2f")
                        )
                    )

                    # --------------------------------------------------------
                    # Sequence quality
                    # --------------------------------------------------------

                    if complete_sequences is not None:

                        if sequence == "combined":

                            sequence_names = [
                                "bright",
                                "faint"
                            ]

                        else:

                            sequence_names = [
                                sequence
                            ]

                        for sequence_name in sequence_names:

                            quality_key = (
                                period,
                                sci,
                                sequence_name
                            )

                            if quality_key in complete_sequences:

                                try:

                                    quality = complete_sequences[
                                        quality_key
                                    ][1]

                                    info_lines.append(
                                        "%s quality: %s"
                                        % (
                                            sequence_name.capitalize(),
                                            str(quality)
                                        )
                                    )

                                except Exception:
                                    pass

                    # --------------------------------------------------------
                    # Calibrator list
                    # --------------------------------------------------------

                    calibrators = []

                    if sequences is not None:

                        if sequence == "combined":

                            for sequence_name in [
                                    "bright",
                                    "faint"]:

                                sequence_key = (
                                    period,
                                    sci,
                                    sequence_name
                                )

                                if sequence_key in sequences:

                                    calibrators.extend(
                                        sequences[
                                            sequence_key
                                        ][::2]
                                    )

                        else:

                            sequence_key = (
                                period,
                                sci,
                                sequence
                            )

                            if sequence_key in sequences:

                                calibrators.extend(
                                    sequences[
                                        sequence_key
                                    ][::2]
                                )

                    calibrators_unique = []

                    for calibrator in calibrators:

                        if calibrator not in calibrators_unique:

                            calibrators_unique.append(
                                calibrator
                            )

                    for calibrator in calibrators_unique:

                        calibrator_id = match_target_for_plot(
                            tgt_info,
                            calibrator,
                            verbose=False
                        )

                        if calibrator_id is None:
                            continue

                        calibrator_info = tgt_info.loc[
                            calibrator_id
                        ]

                        cal_hmag = calibrator_info.get(
                            "Hmag",
                            np.nan
                        )

                        cal_ldd = calibrator_info.get(
                            "LDD_pred",
                            np.nan
                        )

                        cal_e_ldd = calibrator_info.get(
                            "e_LDD_pred",
                            np.nan
                        )

                        cal_jsdc = calibrator_info.get(
                            "JSDC_LDD",
                            np.nan
                        )

                        cal_quality = calibrator_info.get(
                            "Quality",
                            ""
                        )

                        quality_marker = ""

                        if str(cal_quality).upper() == "BAD":
                            quality_marker = " [BAD]"

                        info_lines.append(
                            "%s%s: H=%s, pred=%s +/- %s mas, JSDC=%s mas"
                            % (
                                format_target(calibrator),
                                quality_marker,
                                format_number(
                                    cal_hmag,
                                    "%.2f"
                                ),
                                format_number(cal_ldd),
                                format_number(cal_e_ldd),
                                format_number(cal_jsdc)
                            )
                        )

                    axes[0].text(
                        0.98,
                        0.98,
                        "\n".join(info_lines),
                        transform=axes[0].transAxes,
                        fontsize=5.5,
                        horizontalalignment="right",
                        verticalalignment="top",
                        bbox=dict(
                            facecolor="white",
                            alpha=0.75,
                            edgecolor="none"
                        )
                    )

                # ============================================================
                # Bootstrap histogram
                # ============================================================

                bootstrap_key = star_id

                # Fallback for saved results whose key is only the star name.
                if (
                    bootstrap_key not in bs_results
                    and sci in bs_results
                ):

                    bootstrap_key = sci

                if bootstrap_key not in bs_results:

                    print(
                        "WARNING: %s not found in bs_results"
                        % str(star_id)
                    )

                    axes[1].text(
                        0.5,
                        0.5,
                        "No bootstrap results",
                        transform=axes[1].transAxes,
                        horizontalalignment="center",
                        verticalalignment="center"
                    )

                    axes[1].set_title(
                        stitle
                    )

                else:

                    bootstrap_table = bs_results[
                        bootstrap_key
                    ]

                    ldd_samples = np.asarray(
                        bootstrap_table["LDD_FIT"],
                        dtype=float
                    ).ravel()

                    ldd_samples = ldd_samples[
                        np.isfinite(ldd_samples)
                        & (ldd_samples > 0)
                    ]

                    print(
                        "N valid bootstrap LDD values: %i"
                        % len(ldd_samples)
                    )

                    if len(ldd_samples) == 0:

                        axes[1].text(
                            0.5,
                            0.5,
                            "No valid bootstrap LDD values",
                            transform=axes[1].transAxes,
                            horizontalalignment="center",
                            verticalalignment="center"
                        )

                        axes[1].set_title(
                            stitle
                        )

                    else:

                        ldd_min = np.nanmin(
                            ldd_samples
                        )

                        ldd_max = np.nanmax(
                            ldd_samples
                        )

                        if ldd_min == ldd_max:

                            if (
                                valid_ldd_fit
                                and e_ldd_fit > 0
                            ):

                                hist_range = (
                                    ldd_fit
                                    - 5.0 * e_ldd_fit,
                                    ldd_fit
                                    + 5.0 * e_ldd_fit
                                )

                            else:

                                hist_range = (
                                    ldd_min - 0.01,
                                    ldd_max + 0.01
                                )

                            axes[1].hist(
                                ldd_samples,
                                n_bins,
                                range=hist_range
                            )

                        else:

                            axes[1].hist(
                                ldd_samples,
                                n_bins
                            )

                        axes[1].set_title(
                            stitle
                            + r" ($N_{\rm bootstrap}=%i$)"
                            % len(ldd_samples)
                        )

                        axes[1].set_xlabel(
                            r"$\theta_{\rm LDD}$ (mas)"
                        )

                        axes[1].set_ylabel(
                            "Number of realisations"
                        )

                        # Fitted value and uncertainty only when valid.
                        if valid_ldd_fit:

                            axes[1].axvline(
                                ldd_fit,
                                linestyle="--",
                                label="Final fit"
                            )

                            axes[1].axvline(
                                ldd_fit - e_ldd_fit,
                                linestyle=":"
                            )

                            axes[1].axvline(
                                ldd_fit + e_ldd_fit,
                                linestyle=":"
                            )

                            axes[1].text(
                                0.5,
                                0.95,
                                (
                                    r"$\theta_{\rm LDD}"
                                    r"=%.4f\pm%.4f$ mas"
                                    % (
                                        ldd_fit,
                                        e_ldd_fit
                                    )
                                ),
                                transform=axes[1].transAxes,
                                horizontalalignment="center",
                                verticalalignment="top"
                            )

                        else:

                            bootstrap_median = np.nanmedian(
                                ldd_samples
                            )

                            axes[1].axvline(
                                bootstrap_median,
                                linestyle="--",
                                label="Bootstrap median"
                            )

                            axes[1].text(
                                0.5,
                                0.95,
                                "Final LDD fit unavailable",
                                transform=axes[1].transAxes,
                                horizontalalignment="center",
                                verticalalignment="top"
                            )

                # ============================================================
                # Save this page
                # ============================================================

                fig.tight_layout()

                pdf.savefig(
                    fig
                )

                plt.close(
                    fig
                )

                n_successful_pages += 1

                print(
                    "Saved bootstrap page for %s"
                    % sci
                )

            except Exception as error:

                n_failed_pages += 1

                print(
                    "FAILED bootstrap page for %s"
                    % sci
                )

                print(
                    "Error: %s"
                    % str(error)
                )

                plt.close(
                    "all"
                )

                # Add a diagnostic page instead of stopping the whole PDF.
                error_fig, error_ax = plt.subplots()

                error_ax.axis(
                    "off"
                )

                error_ax.text(
                    0.5,
                    0.60,
                    "Could not generate bootstrap summary",
                    transform=error_ax.transAxes,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize="large"
                )

                error_ax.text(
                    0.5,
                    0.50,
                    "Target: %s" % sci,
                    transform=error_ax.transAxes,
                    horizontalalignment="center",
                    verticalalignment="center"
                )

                error_ax.text(
                    0.5,
                    0.40,
                    "Error: %s" % str(error),
                    transform=error_ax.transAxes,
                    horizontalalignment="center",
                    verticalalignment="center",
                    wrap=True
                )

                error_fig.set_size_inches(
                    16,
                    9
                )

                pdf.savefig(
                    error_fig
                )

                plt.close(
                    error_fig
                )

    # ========================================================================
    # Final report
    # ========================================================================

    print("")
    print("=" * 79)
    print("Bootstrap summary finished")
    print(
        "Successful pages: %i"
        % n_successful_pages
    )
    print(
        "Failed pages: %i"
        % n_failed_pages
    )
    print("Saved:")
    print(output_file)
    print("=" * 79)

    if n_successful_pages == 0:

        raise RuntimeError(
            "No valid bootstrap-summary pages were generated"
        )

    return output_file

def plot_single_vis2(results, e_wl_frac=0.03):
    """Plot side by side vis^2 points and fit, with histogram of LDD dist.
    """
    for star_i in np.arange(0, len(results)):
        # Get the science target name
        sci = results.iloc[star_i]["STAR"]
        period = results.iloc[star_i]["PERIOD"]
        sequence = results.iloc[star_i]["SEQUENCE"]
        
        stitle = "%s (%s, P%s)" % (sci, sequence, period)
        
        # -----------------------------------------------------------------
        # Plot vis^2 fits
        # -----------------------------------------------------------------
        n_bl = len(results.iloc[star_i]["BASELINE"])
        n_wl = len(results.iloc[star_i]["WAVELENGTH"])
        bl_grid = np.tile(results.iloc[star_i]["BASELINE"], 
                          n_wl).reshape([n_wl, n_bl]).T
        wl_grid = np.tile(results.iloc[star_i]["WAVELENGTH"], 
                          n_bl).reshape([n_bl, n_wl])
        
        sfreq = (bl_grid / wl_grid).flatten()
        
        vis2 = results.iloc[star_i]["VIS2"].flatten()
        e_vis2 = results.iloc[star_i]["e_VIS2"].flatten()
        ldd_fit = results.iloc[star_i]["LDD_FIT"]
        e_ldd_fit = results.iloc[star_i]["e_LDD_FIT"]
        
        u_lld = results.iloc[star_i]["u_LLD"]
        
        c_scale = results.iloc[star_i]["C_SCALE"]
        
        x = np.arange(1*10**6, 25*10**7, 10000)

        n_points = (len(x),)
        s_lambda = 1

        y_fit = rdiam.calc_vis2(x, ldd_fit, c_scale, n_points, u_lld, s_lambda)

        plt.close("all")
        fig, ax = plt.subplots()
        
        # Setup lower panel for residuals
        divider = make_axes_locatable(ax)
        res_ax = divider.append_axes("bottom", size="20%", pad=0)
        ax.figure.add_axes(res_ax)

        # Plot the data points and best fit curve
        ax.errorbar(sfreq, vis2, xerr=sfreq*e_wl_frac,
                        yerr=e_vis2, fmt=".", 
                        label="Data", elinewidth=0.1, capsize=0.2, 
                        capthick=0.1)

        ax.plot(x, y_fit, "--", 
                 label=r"Fit ($\theta_{\rm LDD}$=%0.3f mas)"# $\pm$ %f mas)" 
                       % (ldd_fit))#, e_ldd_fit))

        #axes[0].set_xlabel(r"Spatial Frequency (rad$^{-1})$")
        ax.set_ylabel(r"Visibility$^2$")
        ax.set_title(stitle)
        ax.legend(loc="best")
        ax.set_xlim([0.0,10E7])
        ax.set_ylim([0.0, c_scale+0.1])
        ax.grid()
        
        # Plot residuals below the vis2 plot
        ax.set_xticks([])
        n_points = (len(sfreq),)

        residuals = vis2 - rdiam.calc_vis2(sfreq, ldd_fit, c_scale, n_points,
                                           u_lld, s_lambda)
        
        res_ax.errorbar(sfreq, residuals, xerr=sfreq*e_wl_frac,
                        yerr=e_vis2, fmt=".", 
                        label="Residuals", elinewidth=0.1, capsize=0.2, 
                        capthick=0.1)
        res_ax.set_xlim([0.0,10E7])
        res_ax.hlines(0, 0, 25E7, linestyles="dotted")
        res_ax.set_ylabel("Residuals")
        res_ax.set_xlabel(r"Spatial Frequency (rad$^{-1})$")
        
        # -----------------------------------------------------------------
        # Save figs
        # -----------------------------------------------------------------
        #plt.gcf().set_size_inches(16, 9)
        plt.savefig("plots/single_vis2/vis2_%s_%s_%s.pdf" % (sci, period, sequence))
        plt.close()


def plot_paper_vis2_fits(results, n_rows=8, n_cols=2):
    """Plot side by side vis^2 points and fit, with histogram of LDD dist.
    """
    plt.close("all")
    with PdfPages("paper/seq_vis2_plots.pdf") as pdf:
        # Figure out how many sets of plots are needed
        num_sets = int(np.ceil(len(results) / n_rows / n_cols))
        n_rows_init = n_rows
        
        # For every set, save a page
        for set_i in np.arange(0, num_sets):
            # Ensure we don't have an incomplete set of subplots
            if set_i + 1 == num_sets:
                n_rows = int((len(results) - set_i*n_rows*n_cols) / n_cols)
            
            # Setup the axes
            fig, axes = plt.subplots(n_rows, n_cols)#), sharex=True, sharey=True)
            plt.subplots_adjust(wspace=0.3, hspace=0.4)
            axes = axes.flatten()
    
            for star_i in np.arange(set_i*n_rows*n_cols, (set_i+1)*n_rows*n_cols):
                # Subplot index < n_rows
                plt_i = star_i % (n_rows * n_cols)
               
                # Might not be able to finish
                if star_i >= len(results):
                    axes[plt_i].axis("off")
                    continue
            
                # Get the science target name
                sci = results.iloc[star_i]["STAR"]
                period = results.iloc[star_i]["PERIOD"]
                sequence = results.iloc[star_i]["SEQUENCE"]
            
                stitle = "%s (%s, %s)" % (sci, sequence, period)
            
                print("%i, %i, %s %s %s" % (set_i, plt_i, sci, period, sequence))
            
                # -------------------------------------------------------------
                # Plot vis^2 fits
                # -------------------------------------------------------------
                n_bl = len(results.iloc[star_i]["BASELINE"])
                n_wl = len(results.iloc[star_i]["WAVELENGTH"])
                bl_grid = np.tile(results.iloc[star_i]["BASELINE"], 
                                  n_wl).reshape([n_wl, n_bl]).T
                wl_grid = np.tile(results.iloc[star_i]["WAVELENGTH"], 
                                  n_bl).reshape([n_bl, n_wl])
            
                sfreq = (bl_grid / wl_grid).flatten()
            
                vis2 = results.iloc[star_i]["VIS2"].flatten()
                e_vis2 = results.iloc[star_i]["e_VIS2"].flatten()
                ldd_fit = results.iloc[star_i]["LDD_FIT"]
                e_ldd_fit = results.iloc[star_i]["e_LDD_FIT"]
            
                u_lld = results.iloc[star_i]["u_LLD"]
            
                c_scale = results.iloc[star_i]["C_SCALE"]
                x = np.arange(1*10**6, 25*10**7, 10000)
                n_points = (len(x),)
                s_lambda = 1

                
                y_fit = rdiam.calc_vis2(x, ldd_fit, c_scale, n_points, 
                                        u_lld, s_lambda)
                y_fit_low = rdiam.calc_vis2(x, ldd_fit - e_ldd_fit, c_scale, 
                                            n_points, u_lld, s_lambda)
                y_fit_high = rdiam.calc_vis2(x, ldd_fit + e_ldd_fit, c_scale, 
                                             n_points, u_lld, s_lambda)    
            
                # Setup lower panel for residuals
                divider = make_axes_locatable(axes[plt_i])
                res_ax = divider.append_axes("bottom", size="35%", pad=0.1)
                axes[plt_i].figure.add_axes(res_ax, sharex=axes[plt_i])
    
                # Plot the data points and best fit curve
                axes[plt_i].errorbar(sfreq, vis2, yerr=e_vis2, fmt=".", 
                                label="Data", elinewidth=0.1, capsize=0.2, 
                                capthick=0.1, markersize=0.5)
    
                axes[plt_i].plot(x, y_fit, "--", linewidth=0.25,
                         label=r"Fit ($\theta_{\rm LDD}$=%f $\pm$ %f, %0.2f%%)" 
                               % (ldd_fit, e_ldd_fit, e_ldd_fit/ldd_fit*100))
                
                # Annotate the sequence name
                xx = (axes[plt_i].get_xlim()[1] - axes[plt_i].get_xlim()[0]) * 0.05
                yy = (axes[plt_i].get_ylim()[1] - axes[plt_i].get_ylim()[0]) * 0.05
                axes[plt_i].text(xx, yy, stitle, fontsize="xx-small")
                
                # Set up ticks
                axes[plt_i].set_xlim([0.0,10E7])
                axes[plt_i].set_ylim([0.0,1.1])
                
                axes[plt_i].set_xticklabels([])
                
                axes[plt_i].tick_params(axis="both", top=True, right=True)
                res_ax.tick_params(axis="y", right=True)
                
                maj_loc = plticker.MultipleLocator(base=0.2)
                min_loc = plticker.MultipleLocator(base=0.1)
                
                axes[plt_i].yaxis.set_major_locator(maj_loc)
                axes[plt_i].yaxis.set_minor_locator(min_loc)
                
                n_points = (len(sfreq),)

                # Plot residuals below the vis2 plot
                residuals = vis2 - rdiam.calc_vis2(sfreq, ldd_fit, c_scale,
                                                   n_points, u_lld, s_lambda)
            
                res_ax.errorbar(sfreq, residuals, yerr=e_vis2, fmt=".", 
                                label="Residuals", elinewidth=0.1, capsize=0.2, 
                                capthick=0.1, markersize=0.5)
                res_ax.set_xlim([0.0,10E7])
                res_ax.hlines(0, 0, 25E7, linestyles="dotted", linewidth=0.25)
                #res_ax.set_ylabel("Residuals")
                #res_ax.set_xlabel(r"Spatial Frequency (rad$^{-1})$")
                
                plt.setp(axes[plt_i].get_xticklabels(), fontsize="xx-small")
                plt.setp(axes[plt_i].get_yticklabels(), fontsize="xx-small")
                plt.setp(res_ax.get_xticklabels(), fontsize="xx-small")
                plt.setp(res_ax.get_yticklabels(), fontsize="xx-small")
                res_ax.xaxis.offsetText.set_fontsize("xx-small")
                res_ax.yaxis.offsetText.set_fontsize("xx-small")
                
                
                # Only show res_ax x labels on the bottom row
                #if not (plt_i >= (n_rows*n_cols - n_cols)):
                    #res_ax.set_xticklabels([])
                
                # Only show res_ax y labels if on left
                #if not (plt_i % n_cols == 0):
                    #res_ax.set_yticklabels([])
            # -----------------------------------------------------------------
            # Finalise
            # -----------------------------------------------------------------
            fig.text(0.5, 0.005, r"Spatial Frequency (rad$^{-1})$", ha='center')
            fig.text(0.005, 0.5, r"Visibility$^2$", va='center', rotation='vertical')
            
            plt.gcf().set_size_inches(8, 8*(n_rows/n_rows_init))
            plt.tight_layout(pad=1.0)
            pdf.savefig()


def plot_joint_seq_paper_vis2_fits(tgt_info, results, n_rows=3, n_cols=2,
                                   rasterize=False):
    """Plot the rescaled simultaneous fits for multiple sequences.
    """

    plt.close("all")

    # ============================================================
    # Crear columnas limpias para hacer match de nombres
    # ============================================================

    tgt_info = tgt_info.copy()

    match_cols = ["Primary", "Ref_ID_1", "Ref_ID_2", "Ref_ID_3",
                  "Bayer_ID", "HD_ID"]

    for col in match_cols:
        if col in tgt_info.columns:
            tgt_info[col + "_clean"] = [x if pd.notnull(x) else ""
                for x in tgt_info[col].values
            ]

    search_cols = ["Primary_clean",
                   "Ref_ID_1_clean", "Ref_ID_2_clean", "Ref_ID_3_clean",
                   "Bayer_ID_clean", "HD_ID_clean"]

    with PdfPages("paper/joint_seq_vis2_plots.pdf") as pdf:

        num_sets = int(np.ceil(float(len(results)) / float(n_rows * n_cols)))
        n_rows_init = n_rows

        for set_i in np.arange(0, num_sets):

            # Numero de filas en esta pagina
            if set_i + 1 == num_sets:
                n_left = len(results) - set_i * n_rows_init * n_cols
                n_rows_page = int(np.ceil(float(n_left) / float(n_cols)))
            else:
                n_rows_page = n_rows_init

            fig, axes = plt.subplots(n_rows_page, n_cols)
            plt.subplots_adjust(wspace=0.3, hspace=0.4)
            axes = np.atleast_1d(axes).flatten()

            for star_i in np.arange(set_i * n_rows_init * n_cols,
                                    (set_i + 1) * n_rows_init * n_cols):

                plt_i = star_i - set_i * n_rows_init * n_cols

                if star_i >= len(results):
                    if plt_i < len(axes):
                        axes[plt_i].axis("off")
                    continue

                # ============================================================
                # Datos de la estrella
                # ============================================================

                sci = str(results.iloc[star_i]["STAR"])
                period = results.iloc[star_i]["PERIOD"]
                sequence = results.iloc[star_i]["SEQUENCE"]

                sci_clean = sci

                # Buscar sci en varias columnas
                sci_data = pd.DataFrame()
                matched_col = None

                for col in search_cols:
                    if col not in tgt_info.columns:
                        continue

                    tmp = tgt_info[tgt_info[col] == sci_clean]

                    if len(tmp) > 0:
                        sci_data = tmp
                        matched_col = col
                        break

                if len(sci_data) == 0:
                    print("WARNING: no encontre match para", sci)
                    print("sci_clean =", sci_clean)
                    print("Busque en:", search_cols)
                    axes[plt_i].axis("off")
                    continue

                hd_id = sci_data.index.values[0]
                hd_id = match_target_for_plot(tgt_info,sci,verbose=True)
                if hd_id is None:
                    axes[plt_i].axis("off")
                    continue

                stitle = "%s (%s, %s)" % (sci, sequence, period)

                print("%i, %i, [%i] %s %s %s" %
                      (set_i, plt_i, star_i, sci, period, sequence))

                print("Match:", sci, "-->", hd_id, "using", matched_col)

                # ============================================================
                # Coeficientes de limb darkening
                # ============================================================

                u_lambda_cols = ["u_lambda_%i" % ui for ui in np.arange(0, 6)]
                s_lambda_cols = ["s_lambda_%i" % ui for ui in np.arange(0, 6)]

                u_lambdas = np.asarray(tgt_info.loc[hd_id][u_lambda_cols].values,
                                       dtype=float)
                s_lambdas = np.asarray(tgt_info.loc[hd_id][s_lambda_cols].values,
                                       dtype=float)

                c_scale = np.asarray(results.iloc[star_i]["C_SCALE"],
                                     dtype=float).ravel()

                if len(c_scale) == 0:
                    print("WARNING: C_SCALE vacio para", sci)
                    axes[plt_i].axis("off")
                    continue

                # Esto asume 12 puntos por secuencia
                n_points_seq = [12] * len(c_scale)

                c_array = np.hstack([
                    c_scale[ni] * np.ones(n)
                    for ni, n in enumerate(n_points_seq)
                ])

                cmap = cm.get_cmap("magma")
                colours = [cmap(i) for i in np.arange(0.84, 0, -0.14)]

                wl_um = [1.533, 1.581, 1.629, 1.677, 1.725, 1.773]
                wl_lbl = [r"%s$\,\mu$m" % wl for wl in wl_um]

                # ============================================================
                # Panel de residuos
                # ============================================================

                divider = make_axes_locatable(axes[plt_i])
                res_ax = divider.append_axes("bottom", size="35%", pad=0.1)
                axes[plt_i].figure.add_axes(res_ax, sharex=axes[plt_i])

                residuals_all = np.array([], dtype=float)
                e_vis2_all = np.array([], dtype=float)

                # ============================================================
                # Plot por canal de longitud de onda
                # ============================================================

                for wl_i in np.arange(6):

                    bls = np.asarray(results.iloc[star_i]["BASELINE"],
                                     dtype=float)
                    wls = np.asarray(results.iloc[star_i]["WAVELENGTH"],
                                     dtype=float)

                    sfreq = (bls / wls[wl_i])[:len(c_array)]

                    vis2 = np.asarray(results.iloc[star_i]["VIS2"][:, wl_i],
                                      dtype=float)
                    e_vis2 = np.asarray(results.iloc[star_i]["e_VIS2"][:, wl_i],
                                        dtype=float)

                    ldd_fit = float(results.iloc[star_i]["LDD_FIT"])
                    e_ldd_fit = float(results.iloc[star_i]["e_LDD_FIT"])

                    valid_i = (((vis2 >= 0)
                                & (e_vis2 > 0)
                                & np.isfinite(vis2)
                                & np.isfinite(e_vis2)))[:len(c_array)]

                    vis2 = vis2[:len(c_array)] / c_array
                    e_vis2 = e_vis2[:len(c_array)]
                    ldd_fit_wl = ldd_fit * s_lambdas[wl_i]

                    u_lambda = float(u_lambdas[wl_i])
                    s_lambda = float(s_lambdas[wl_i])

                    vis2 = vis2[valid_i]
                    e_vis2 = e_vis2[valid_i]
                    sfreq = sfreq[valid_i]

                    if len(vis2) == 0:
                        print("WARNING: sin puntos validos para",
                              sci, period, sequence, "wl_i =", wl_i)
                        continue

                    x = np.arange(1 * 10**6, 25 * 10**7, 10000)
                    n_points_fit = (len(x),)

                    y_fit = rdiam.calc_vis2(x, ldd_fit_wl, 1.0, n_points_fit,
                                            u_lambda, s_lambda)

                    axes[plt_i].errorbar(sfreq, vis2, yerr=e_vis2, fmt=".",
                                          label=wl_lbl[wl_i],
                                          elinewidth=0.3,
                                          capsize=0.6,
                                          capthick=0.3,
                                          markersize=3,
                                          color=colours[wl_i],
                                          markeredgecolor="grey",
                                          markeredgewidth=0.01,
                                          rasterized=rasterize)

                    axes[plt_i].plot(x, y_fit, "--",
                                     linewidth=0.4,
                                     color=colours[wl_i])

                    # Residuals
                    n_points_res = (len(sfreq),)

                    model = rdiam.calc_vis2(sfreq, ldd_fit_wl, 1.0,
                                            n_points_res, u_lambda, s_lambda)

                    residuals = vis2 - model

                    res_ax.errorbar(sfreq, residuals, yerr=e_vis2, fmt=".",
                                    elinewidth=0.3,
                                    capsize=0.6,
                                    capthick=0.3,
                                    markersize=3,
                                    color=colours[wl_i],
                                    markeredgecolor="grey",
                                    markeredgewidth=0.01,
                                    rasterized=rasterize)

                    residuals_all = np.hstack((residuals_all, residuals))
                    e_vis2_all = np.hstack((e_vis2_all, e_vis2))

                # ============================================================
                # Labels y limites del panel principal
                # ============================================================

                xx = (axes[plt_i].get_xlim()[1] -
                      axes[plt_i].get_xlim()[0]) * 0.05

                yy = (axes[plt_i].get_ylim()[1] -
                      axes[plt_i].get_ylim()[0]) * 0.05

                axes[plt_i].text(xx, yy, rutils.format_id(sci),
                                  fontsize="small")

                axes[plt_i].set_xlim([0.0, 10E7])
                axes[plt_i].set_ylim([0.0, 1.1])

                axes[plt_i].set_xticklabels([])

                axes[plt_i].tick_params(axis="both", top=True, right=True)
                res_ax.tick_params(axis="y", right=True)

                maj_loc = plticker.MultipleLocator(base=0.2)
                min_loc = plticker.MultipleLocator(base=0.1)

                axes[plt_i].yaxis.set_major_locator(maj_loc)
                axes[plt_i].yaxis.set_minor_locator(min_loc)

                # ============================================================
                # Ticks seguros para residuos
                # ============================================================

                residuals_all = np.asarray(residuals_all, dtype=float)
                e_vis2_all = np.asarray(e_vis2_all, dtype=float)

                good_res = (np.isfinite(residuals_all)
                            & np.isfinite(e_vis2_all))

                if np.sum(good_res) == 0:

                    print("WARNING: no finite residuals for",
                          sci, period, sequence)

                    res_sep_maj = 0.1
                    res_sep_min = 0.05
                    res_ax.set_ylim([-0.3, 0.3])

                else:

                    y_low = residuals_all[good_res] - e_vis2_all[good_res]
                    y_high = residuals_all[good_res] + e_vis2_all[good_res]

                    y_min = np.nanmin(y_low)
                    y_max = np.nanmax(y_high)

                    if ((not np.isfinite(y_min))
                        or (not np.isfinite(y_max))
                        or (y_max == y_min)):

                        print("WARNING: invalid residual range for",
                              sci, period, sequence)
                        print("y_min =", y_min, "y_max =", y_max)

                        res_sep_maj = 0.1
                        res_sep_min = 0.05
                        res_ax.set_ylim([-0.3, 0.3])

                    else:

                        res_range = np.abs(y_max - y_min)

                        res_sep_maj = res_range / 4.0
                        res_sep_maj = np.round(res_sep_maj * 2, 2) / 2.0

                        if ((not np.isfinite(res_sep_maj))
                            or (res_sep_maj <= 0)):

                            res_sep_maj = 0.1

                        res_sep_min = res_sep_maj / 2.0

                        pad = 0.1 * res_range
                        res_ax.set_ylim([y_min - pad, y_max + pad])

                res_maj_loc = plticker.MultipleLocator(base=res_sep_maj)
                res_min_loc = plticker.MultipleLocator(base=res_sep_min)

                res_ax.yaxis.set_major_locator(res_maj_loc)
                res_ax.yaxis.set_minor_locator(res_min_loc)

                res_ax.set_xlim([0.0, 10E7])
                res_ax.hlines(0, 0, 25E7,
                              linestyles="dotted",
                              linewidth=0.25)

                plt.setp(axes[plt_i].get_xticklabels(), fontsize="medium")
                plt.setp(axes[plt_i].get_yticklabels(), fontsize="medium")
                plt.setp(res_ax.get_xticklabels(), fontsize="medium")
                plt.setp(res_ax.get_yticklabels(), fontsize="medium")

                res_ax.xaxis.offsetText.set_fontsize("medium")
                res_ax.yaxis.offsetText.set_fontsize("medium")

            # ================================================================
            # Finalizar pagina
            # ================================================================

            fig.text(0.5, 0.005,
                     r"Spatial Frequency (rad$^{-1})$",
                     ha="center")

            fig.text(0.005, 0.5,
                     r"Visibility$^2$",
                     va="center",
                     rotation="vertical")

            if n_rows_page == n_cols:
                plt.gcf().set_size_inches(10,
                                           6 * (float(n_rows_page)
                                                / float(n_rows_init)))
            else:
                plt.gcf().set_size_inches(8,
                                           11 * (float(n_rows_page)
                                                 / float(n_rows_init)))

            plt.tight_layout(pad=1.0)
            plt.savefig("paper/joint_seq_vis2_plots_pg%i.png" % set_i,
                        dpi=200)
            pdf.savefig(dpi=200)
            plt.close()
def plot_joint_seq_paper_vis2_fits_old(tgt_info, results, n_rows=3, n_cols=2,
                                   rasterize=False):
    """Plot the rescaled simultaneous fits for multiple sequences
    """
    plt.close("all")
    with PdfPages("paper/joint_seq_vis2_plots.pdf") as pdf:
        # Figure out how many sets of plots are needed
        num_sets = int(np.ceil(len(results) / n_rows / n_cols))
        n_rows_init = n_rows
        
        # For every set, save a page
        for set_i in np.arange(0, num_sets):
            # Ensure we don't have an incomplete set of subplots
            if set_i + 1 == num_sets:
                n_rows = int((len(results) - set_i*n_rows*n_cols) / n_cols)
            
            # Setup the axes
            fig, axes = plt.subplots(n_rows, n_cols)
            plt.subplots_adjust(wspace=0.3, hspace=0.4)
            axes = axes.flatten()
    
            for star_i in np.arange(set_i*n_rows_init*n_cols, 
                                    (set_i+1)*n_rows_init*n_cols):
                # Subplot index < n_rows
                plt_i = star_i % (n_rows * n_cols)
               
                # Might not be able to finish
                if star_i >= len(results):
                    break
            
                # Get the science target name
                sci = str(results.iloc[star_i]["STAR"])
                
                hd_id = tgt_info[tgt_info["Primary"]==sci].index.values[0]
                
                period = results.iloc[star_i]["PERIOD"]
                sequence = results.iloc[star_i]["SEQUENCE"]
            
                stitle = "%s (%s, %s)" % (sci, sequence, period)
            
                print("%i, %i, [%i] %s %s %s" % (set_i, plt_i, star_i, sci, 
                                                 period, sequence))
            
                # Get the C params, and u_lambda values
                u_lambda_cols = ["u_lambda_%i" % ui for ui in np.arange(0,6)]
                s_lambda_cols = ["s_lambda_%i" % ui for ui in np.arange(0,6)]
                
                u_lambdas = tgt_info.loc[hd_id][u_lambda_cols].values
                s_lambdas = tgt_info.loc[hd_id][s_lambda_cols].values
                
                c_scale = results.iloc[star_i]["C_SCALE"]
                
                n_points = [12] * len(c_scale)
                
                c_array = np.hstack([c_scale[ni]*np.ones(n) 
                             for ni, n in enumerate(n_points)])
                
                #colours = ["mistyrose", "coral", "orangered", "red", 
                #           "firebrick", "maroon"]
                cmap = cm.get_cmap("magma")
                colours = [cmap(i) for i in np.arange(0.84,0,-0.14)]
                           
                wl_um = [1.533, 1.581, 1.629, 1.677, 1.725, 1.773]
                wl_lbl = [r"%s$\,\mu$m" % wl for wl in wl_um]
                
                # -------------------------------------------------------------
                # Plot vis^2 fits
                # -------------------------------------------------------------
                n_bl = len(results.iloc[star_i]["BASELINE"])
                n_wl = len(results.iloc[star_i]["WAVELENGTH"])
                
                # Setup lower panel for residuals
                divider = make_axes_locatable(axes[plt_i])
                res_ax = divider.append_axes("bottom", size="35%", pad=0.1)
                axes[plt_i].figure.add_axes(res_ax, sharex=axes[plt_i])
                
                residuals_all = []
                e_vis2_all = []
                
                # For each wavelength dimension
                for wl_i in np.arange(6):
                    # Need to do 1 plot per wavelength channel
                    bls = results.iloc[star_i]["BASELINE"]
                    wls = results.iloc[star_i]["WAVELENGTH"]
                    sfreq = (bls / wls[wl_i])[:len(c_array)]
                    
                    vis2 = results.iloc[star_i]["VIS2"][:, wl_i]
                    e_vis2 = results.iloc[star_i]["e_VIS2"][:, wl_i]
                    ldd_fit = results.iloc[star_i]["LDD_FIT"]
                    e_ldd_fit = results.iloc[star_i]["e_LDD_FIT"]
                    
                    # Add a mask to not plot any bad data
                    valid_i = (((vis2 >= 0) & (e_vis2 > 0) 
                               & ~np.isnan(vis2)))[:len(c_array)]
                    
                    # Normalise vis2 and scale ldd_fit
                    # TODO: Fix the uncertainty over the length of each seq
                    vis2 = vis2[:len(c_array)] / c_array
                    e_vis2 = e_vis2[:len(c_array)]
                    ldd_fit = ldd_fit * s_lambdas[wl_i]
                    
                    u_lambda = u_lambdas[wl_i]
                    s_lambda = s_lambdas[wl_i]
                    
                    # Apply mask
                    vis2 = vis2[valid_i]
                    e_vis2 = e_vis2[valid_i]
                    sfreq = sfreq[valid_i]

                    x = np.arange(1*10**6, 25*10**7, 10000)

                    n_points = (len(x),)

                    y_fit = rdiam.calc_vis2(x, ldd_fit, 1.0, n_points,
                                            u_lambda, s_lambda) 
    
                    # Plot the data points and best fit curve
                    axes[plt_i].errorbar(sfreq, vis2, yerr=e_vis2, fmt=".", 
                                    label=wl_lbl[wl_i], elinewidth=0.3, capsize=0.6, 
                                    capthick=0.3, markersize=3, color=colours[wl_i],
                                    markeredgecolor="grey", markeredgewidth=0.01,
                                    rasterized=rasterize)
    
                    axes[plt_i].plot(x, y_fit, "--", linewidth=0.4, 
                                     color=colours[wl_i])
                
                    # Plot residuals below the vis2 plot
                    n_points = (len(sfreq),)
                    residuals = vis2 - rdiam.calc_vis2(sfreq, ldd_fit, 1.0,
                                                       n_points, u_lambda,
                                                       s_lambda)
            
                    res_ax.errorbar(sfreq, residuals, yerr=e_vis2, fmt=".", 
                                elinewidth=0.3, capsize=0.6, 
                                capthick=0.3, markersize=3, color=colours[wl_i],
                                markeredgecolor="grey", markeredgewidth=0.01,
                                rasterized=rasterize)
                                
                    #axes[plt_i].legend(loc="best", fontsize="xx-small")
                    
                    # Record all points for figuring out axis ticks later
                    residuals_all = np.hstack((residuals_all, residuals))
                    e_vis2_all = np.hstack((e_vis2_all, e_vis2))
                    
                # Annotate the sequence name
                xx = (axes[plt_i].get_xlim()[1] - axes[plt_i].get_xlim()[0]) * 0.05
                yy = (axes[plt_i].get_ylim()[1] - axes[plt_i].get_ylim()[0]) * 0.05
                axes[plt_i].text(xx, yy, rutils.format_id(sci), 
                                 fontsize="small")
                
                # Set up ticks
                axes[plt_i].set_xlim([0.0,10E7])
                axes[plt_i].set_ylim([0.0,1.1])
                
                axes[plt_i].set_xticklabels([])
                
                axes[plt_i].tick_params(axis="both", top=True, right=True)
                res_ax.tick_params(axis="y", right=True)
                
                maj_loc = plticker.MultipleLocator(base=0.2)
                min_loc = plticker.MultipleLocator(base=0.1)
                
                axes[plt_i].yaxis.set_major_locator(maj_loc)
                axes[plt_i].yaxis.set_minor_locator(min_loc)
                
                # Work out the residual axis spacing
                res_sep_maj = np.abs(np.max(residuals_all + e_vis2_all) 
                               - np.min(residuals_all - e_vis2_all)) / 4
                               
                res_sep_maj = np.round(res_sep_maj*2, 2) / 2
                res_sep_min = res_sep_maj / 2
                
                res_maj_loc = plticker.MultipleLocator(base=res_sep_maj)
                res_min_loc = plticker.MultipleLocator(base=res_sep_min)
                res_ax.yaxis.set_major_locator(res_maj_loc)
                res_ax.yaxis.set_minor_locator(res_min_loc)
                
                res_ax.set_xlim([0.0,10E7])
                res_ax.hlines(0, 0, 25E7, linestyles="dotted", linewidth=0.25)
                
                plt.setp(axes[plt_i].get_xticklabels(), fontsize="medium")
                plt.setp(axes[plt_i].get_yticklabels(), fontsize="medium")
                plt.setp(res_ax.get_xticklabels(), fontsize="medium")
                plt.setp(res_ax.get_yticklabels(), fontsize="medium")
                res_ax.xaxis.offsetText.set_fontsize("medium")
                res_ax.yaxis.offsetText.set_fontsize("medium")
                
            # -----------------------------------------------------------------
            # Finalise
            # -----------------------------------------------------------------
            fig.text(0.5, 0.005, r"Spatial Frequency (rad$^{-1})$", ha='center')
            fig.text(0.005, 0.5, r"Visibility$^2$", va='center', rotation='vertical')
            
            if n_rows == n_cols:
                plt.gcf().set_size_inches(10, 6*(n_rows/n_rows_init))
            else:
                plt.gcf().set_size_inches(8, 11*(n_rows/n_rows_init))
                
            plt.tight_layout(pad=1.0)
            plt.savefig("paper/joint_seq_vis2_plots_pg%i.png" % set_i, dpi=200)
            pdf.savefig(dpi=200)
            plt.close()    

def clean_filename(name):
    """
    Clean a string so it can be safely used as a filename.
    """

    name = str(name)

    name = name.replace(" ", "_")
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace(",", "")
    name = name.replace("'", "")
    name = name.replace('"', "")
    name = name.replace("__", "_")

    return name

def plot_sidelobe_vis2_fit(tgt_info, results, sci=None, row_i=None,
                           output_dir="paper"):
    """Plot the zoomed in fitted sidelobe
    """
    plt.close("all")
    # Setup the axes
    fig, axes = plt.subplots(1, 1)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # Get the science target name

    # -----------------------------------------------------------------
# Get the science target result
# -----------------------------------------------------------------
    if row_i is not None:

        sci_results = results.iloc[row_i]
        sci = sci_results["STAR"]

    else:

        if sci is None:
            raise ValueError("You must provide either sci or row_i")

        if len(results[results["STAR"] == sci]) == 0:
            print("WARNING: %s not found in results" % sci)
            return None

        sci_results = results[results["STAR"] == sci].iloc[0]

# -----------------------------------------------------------------
# Match science target to tgt_info
# -----------------------------------------------------------------

    hd_id = match_target_for_plot(tgt_info, sci, verbose=True)
    if hd_id is None:
        print("Skipping plot because target could not be matched: %s" % sci)
        return None
    
    # Get the C params, and u_lambda values
    u_lambda_cols = ["u_lambda_%i" % ui for ui in np.arange(0,6)]
    s_lambda_cols = ["s_lambda_%i" % ui for ui in np.arange(0,6)]
    
    u_lambdas = tgt_info.loc[hd_id][u_lambda_cols].values
    s_lambdas = tgt_info.loc[hd_id][s_lambda_cols].values
    
    c_scale = sci_results["C_SCALE"]
    
    n_points = [12] * len(c_scale)
    
    c_array = np.hstack([c_scale[ni]*np.ones(n) 
                 for ni, n in enumerate(n_points)])
    
    cmap = cm.get_cmap("magma")
    colours = [cmap(i) for i in np.arange(0.84,0,-0.14)]
               
    wl_um = [1.533, 1.581, 1.629, 1.677, 1.725, 1.773]
    wl_lbl = [r"%s$\,\mu$m" % wl for wl in wl_um]
    
    # -----------------------------------------------------------------
    # Plot vis^2 fits
    # -----------------------------------------------------------------
    n_bl = len(sci_results["BASELINE"])
    n_wl = len(sci_results["WAVELENGTH"])
    
    # Setup lower panel for residuals
    divider = make_axes_locatable(axes)
    res_ax = divider.append_axes("bottom", size="35%", pad=0.1)
    axes.figure.add_axes(res_ax, sharex=axes)
    
    # For each wavelength dimension
    for wl_i in np.arange(6):
        # Need to do 1 plot per wavelength channel
        bls = sci_results["BASELINE"]
        wls = sci_results["WAVELENGTH"]
        sfreq = (bls / wls[wl_i])[:len(c_array)]
        
        vis2 = sci_results["VIS2"][:, wl_i]
        e_vis2 = sci_results["e_VIS2"][:, wl_i]
        ldd_fit = sci_results["LDD_FIT"]
        e_ldd_fit = sci_results["e_LDD_FIT"]
        
        # Normalise vis2 and scale ldd_fit
        # TODO: Fix the uncertainty over the length of each seq
        vis2 = vis2[:len(c_array)] / c_array
        e_vis2 = e_vis2[:len(c_array)]
        ldd_fit = ldd_fit * s_lambdas[wl_i]

        x = np.arange(1*10**6, 25*10**7, 10000)

        u_lambda = u_lambdas[wl_i]
        s_lambda = s_lambdas[wl_i]
        n_points = (len(x),)

        y_fit = rdiam.calc_vis2(x, ldd_fit, 1.0, n_points, u_lambda, s_lambda) 

        # Plot the data points and best fit curve
        axes.errorbar(sfreq, vis2, yerr=e_vis2, fmt=".", 
                        label=wl_lbl[wl_i], elinewidth=0.3, capsize=0.6, 
                        capthick=0.3, markersize=6.0, color=colours[wl_i],
                        markeredgecolor="grey", markeredgewidth=0.05)

        axes.plot(x, y_fit, "--", linewidth=0.4, color=colours[wl_i])

        n_points = (len(sfreq),)

        # Plot residuals below the vis2 plot
        residuals = vis2 - rdiam.calc_vis2(sfreq, ldd_fit, 1.0, n_points,
                                           u_lambda, s_lambda)

        res_ax.errorbar(sfreq, residuals, yerr=e_vis2, fmt=".", elinewidth=0.3, 
                    capsize=0.6, capthick=0.3, markersize=6.0, 
                    color=colours[wl_i], markeredgecolor="grey",
                    markeredgewidth=0.05)
    
    # Plot the uniform disc diameter
    udd_fit = sci_results["UDD_FIT"]
    x = np.arange(1*10**6, 25*10**7, 10000)
    n_points = (len(x),)
    y_fit = rdiam.calc_vis2(x, udd_fit, 1.0, n_points, 0, s_lambda)
    axes.plot(x, y_fit, "--", linewidth=0.4, color="black", 
              label=r"$\theta_{\rm UD}$") 
    
    axes.legend(loc="best", fontsize="medium")
    
    # Set up ticks and axes
    axes.set_xlim([0E7,9.5E7])
    axes.set_ylim([0.0,1.1])
    
    axes.set_xticklabels([])
    
    axes.tick_params(axis="both", top=True, right=True)
    res_ax.tick_params(axis="y", right=True)
    
    maj_loc = plticker.MultipleLocator(base=0.2)
    min_loc = plticker.MultipleLocator(base=0.1)
    
    axes.yaxis.set_major_locator(maj_loc)
    axes.yaxis.set_minor_locator(min_loc)
    axes.set_ylabel(r"Visibility$^2$", fontsize="x-large")
    
    res_maj_loc = plticker.MultipleLocator(base=0.005)
    res_min_loc = plticker.MultipleLocator(base=0.01)
    
    res_ax.yaxis.set_major_locator(res_maj_loc)
    res_ax.yaxis.set_minor_locator(res_min_loc)
    
    res_ax.set_xlim([0E7,9.5E7])
    res_ax.set_ylim([-0.03,0.03])
    res_ax.hlines(0, 0, 25E7, linestyles="dotted", linewidth=0.25)
    res_ax.set_ylabel("Residuals", fontsize="x-large")
    
    res_ax.set_xlabel(r"Spatial Frequency (rad$^{-1})$", fontsize="x-large")
    
    plt.setp(axes.get_xticklabels(), fontsize="x-large")
    plt.setp(axes.get_yticklabels(), fontsize="x-large")
    plt.setp(res_ax.get_xticklabels(), fontsize="x-large")
    plt.setp(res_ax.get_yticklabels(), fontsize="x-large")
    res_ax.xaxis.offsetText.set_fontsize("x-large")
    res_ax.yaxis.offsetText.set_fontsize("x-large")
        
    plt.tight_layout(pad=1.0)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sequence = sci_results["SEQUENCE"]
    period = sci_results["PERIOD"]

    outname = "%s_%s_P%s_sidelobe" % (
       clean_filename(sci),
        clean_filename(sequence),
        clean_filename(period)
    )

    pdf_name = os.path.join(output_dir, "%s.pdf" % outname)
    png_name = os.path.join(output_dir, "%s.png" % outname)

    plt.savefig(pdf_name)
    plt.savefig(png_name, dpi=200)

    print("Saved:")
    print(pdf_name)
    print(png_name)

    return pdf_name

def plot_all_sidelobe_vis2_fits(tgt_info, results, output_dir="paper/sidelobes"):
    """
    Plot sidelobe visibility fits for all science targets in results.

    This loops over each row of results, so it works for bright/faint
    sequences separately.
    """

    import os
    import pandas as pd

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_files = []
    failed_rows = []

    print("\nPlotting sidelobe visibility fits for all science targets")
    print("Number of rows in results: %i" % len(results))

    for row_i in range(len(results)):

        sci = results.iloc[row_i]["STAR"]

        if pd.isnull(sci):
            print("Skipping row %i because STAR is NaN" % row_i)
            failed_rows.append((row_i, "NaN STAR"))
            continue

        sequence = results.iloc[row_i]["SEQUENCE"]
        period = results.iloc[row_i]["PERIOD"]

        print("\nPlotting row %i: %s, %s, P%s" %
              (row_i, sci, sequence, period))

        try:

            saved = plot_sidelobe_vis2_fit(
                tgt_info,
                results,
                row_i=row_i,
                output_dir=output_dir
            )

            if saved is not None:
                saved_files.append(saved)
            else:
                failed_rows.append((row_i, "Returned None"))

        except Exception as err:

            print("FAILED on row %i: %s" % (row_i, sci))
            print(str(err))

            failed_rows.append((row_i, str(err)))

    # -----------------------------------------------------------------
    # Save failed rows
    # -----------------------------------------------------------------

    failed_path = os.path.join(output_dir, "failed_sidelobe_plots.txt")

    with open(failed_path, "w") as f:

        f.write("Failed sidelobe plots\n")
        f.write("=" * 70 + "\n\n")

        for row_i, reason in failed_rows:
            f.write("row %s: %s\n" % (str(row_i), str(reason)))

    print("\nFinished sidelobe plotting")
    print("Saved %i plots" % len(saved_files))
    print("Failed %i plots" % len(failed_rows))
    print("Failed log:")
    print(failed_path)

    return saved_files, failed_rows
 

def plot_lit_diam_comp(tgt_info, xy_map=None, markers=["s","v","D","o","*"]):
    """Plot for paper comparing measured LDD vs any literature values
    """
    # Load in the literature diameters
    lit_diam_file = "data/literature_diameters.tsv"
    lit_diam_info = pd.read_csv(lit_diam_file, sep="\t", header=0)
    
    instruments = set(lit_diam_info[lit_diam_info["has_diam"]]["instrument"])
    
    markers = itertools.cycle(markers)

    plt.close("all")
    fig, ax = plt.subplots()
            
    # Setup lower panel for residuals
    divider = make_axes_locatable(ax)
    res_ax = divider.append_axes("bottom", size="20%", pad=0.1)
    ax.figure.add_axes(res_ax)
    
    # For every different instrument, plot the comparison between our results
    # and those from the literature
    for instrument in instruments:
        #print(instrument)
        mask = np.logical_and(lit_diam_info["has_diam"], 
                              lit_diam_info["instrument"]==instrument).values
        
        # Initialise arrays
        calc_diams = []
        e_calc_diams = []
        lit_diams = []
        e_lit_diams = []
        
        for index, star in lit_diam_info[mask].iterrows():
            # Get the two LDDs to compare
            lit_diams.append(star["theta_ldd"])
            e_lit_diams.append(star["e_theta_ldd"])
            calc_diams.append(tgt_info.loc[star["HD"]]["ldd_final"])
            e_calc_diams.append(tgt_info.loc[star["HD"]]["e_ldd_final"])

        marker = markers.next()

        # Plot the points
        ax.errorbar(calc_diams, lit_diams, xerr=e_calc_diams, yerr=e_lit_diams, 
                    fmt=marker, label=instrument, elinewidth=0.5,  
                    capsize=0.8, capthick=0.5, markersize=4)
            
        # Plot residuals
        ax.set_xticks([])
        residuals = np.array(lit_diams) / np.array(calc_diams)
        err_res = np.array(e_lit_diams) / np.array(calc_diams)
            
        res_ax.errorbar(calc_diams, residuals, xerr=e_calc_diams, 
                        yerr=e_lit_diams, fmt=marker, elinewidth=0.5, 
                        capsize=0.8, capthick=0.5, markersize=4)
        
        # Plot the names of the stars
        hd_ids = set(lit_diam_info[lit_diam_info["has_diam"]]["HD"])
        
        for hd_id in hd_ids:
            prim_id = tgt_info.loc[hd_id]["Primary"]
            ldd = tgt_info.loc[hd_id]["ldd_final"]
            plt.text(ldd + xy_map[prim_id][0],
                     ldd + xy_map[prim_id][1],
                     rutils.format_id(prim_id),
                     verticalalignment="center",
                     horizontalalignment="center",
                     fontsize="large")

    # Plot the two lines
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(np.arange(0, 10), np.arange(0, 10), "--", color="black")
    res_ax.hlines(1, xmin=0, xmax=10, linestyles="dashed")
                      
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    res_ax.set_xlim(xlim)
    
    # Setup residual y axis
    maj_loc = plticker.MultipleLocator(base=0.1)
    min_loc = plticker.MultipleLocator(base=0.05)
    res_ax.yaxis.set_major_locator(maj_loc)
    res_ax.yaxis.set_minor_locator(min_loc)

    # Setup the rest of the plot
    ax.set_ylabel(r"$\theta_{\rm Lit}$ (mas)", fontsize="x-large")
    res_ax.set_xlabel(r"$\theta_{\rm PIONIER}$ (mas)", fontsize="x-large")
    res_ax.set_ylabel(r"$\theta_{\rm Lit} / \theta_{\rm PIONIER}$",
                     fontsize="x-large")
    ax.legend(loc="best", fontsize="large")

    ax.tick_params(axis="both", which="major", labelsize="x-large")
    res_ax.tick_params(axis="both", which="major", labelsize="x-large")
    
    plt.tight_layout()
    plt.savefig("paper/lit_diam_comp.pdf")    
    plt.savefig("paper/lit_diam_comp.png", dpi=200) 
    


def plot_colour_rel_diam_comp(tgt_info, colour_rels=["V-W3","V-W4","B-V_feh"], 
                              cbar="feh", xy_maps=(None,None,None)):
    """Plot for paper comparing measured LDD vs Boyajian colour relation diams.
    
    Colourbar solution from here:
    stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    """
    plt.close("all")
    fig, axes = plt.subplots(1, len(colour_rels), sharex=True,sharey=True)
    
    if hasattr(axes, "__len__"):
        axes = axes.flatten()
    else:
        axes = np.array([axes])
    
    # Plot each subplot
    for ax_i, (ax, colour_rel, xy_map) in enumerate(zip(axes, colour_rels, xy_maps)):
        # Format the colour relation
        colour_rel_col = "LDD_" + colour_rel.replace("-", "")
    
        # Remove [Fe/H] if it's there
        colour_rel = colour_rel.replace("_feh", "")
    
        # Setup lower panel for residuals
        divider = make_axes_locatable(ax)
        res_ax = divider.append_axes("bottom", size="30%", pad=0.1)
        ax.figure.add_axes(res_ax)
    
        # Initialise arrays
        fit_diams = []
        e_fit_diams = []
        colour_rel_diams = []
        e_colour_rel_diams = []
        fehs = []
        teffs = []
    
        # Change the annotation rotation to prevent labels overlapping
        xy_txt = []
    
        # For every science target, plot using the given relation
        for star, star_data in tgt_info[tgt_info["Science"]].iterrows():
            # If star doesn't have a diameter using this relation, skip
            if (np.isnan(star_data[colour_rel_col]) 
                or not star_data["in_paper"]):
                continue
            elif colour_rel=="V-K" and star_data["LDD_rel"] != colour_rel_col:
                continue
        
            # Get the two LDDs to compare
            fit_diams.append(star_data["ldd_final"])
            e_fit_diams.append(star_data["e_ldd_final"])
            colour_rel_diams.append(star_data[colour_rel_col])
            e_colour_rel_diams.append(star_data["e_%s" % colour_rel_col])
            fehs.append(star_data["FeH_rel"])
            teffs.append(star_data["teff_final"])
        
            # Compare positions
            # TODO: a better solution would be sorting the stars by LDD, then
            # alternate the sign on xx and yy to plot above or below...or just 
            # hardcode it
            xy_abs = (fit_diams[-1]**2 + colour_rel_diams[-1]**2)**0.5
            xy = np.abs(np.array(xy_txt) - xy_abs)
            sep = 0.1

            # Import the provided xy_map if given
            xy_map = None
            if xy_map is not None:
                xx = xy_map["alfcmi"][0]
                yy = xy_map["alfcmi"][1]
                #xx = xy_map[star_data["Primary"]][0]
                #yy = xy_map[star_data["Primary"]][1]

            elif len(xy_txt) > 0 and (xy < sep).any():
                xx = 0.025
                yy = 0.2
            else:
                xx = 0.025
                yy = 0.15 
        
            # Plot the name of the star
            ax.annotate(rutils.format_id(star_data["Primary"]),  
                        xy=(fit_diams[-1], colour_rel_diams[-1]), 
                        xytext=(fit_diams[-1]+xx, colour_rel_diams[-1]+yy), 
                        fontsize="small", verticalalignment="center",
                        #arrowprops=dict(facecolor="black", width=0.0, 
                                    #headwidth=0.0),
                        )
                    
            xy_txt.append(xy_abs)
                        
        # Plot the points + errors
        ax.errorbar(fit_diams, colour_rel_diams, xerr=e_fit_diams, 
                    yerr=e_colour_rel_diams, fmt=".", ecolor="firebrick", 
                    elinewidth=0.5, capsize=0.8, capthick=0.5, zorder=1)
        
        # Plot residuals
        ax.set_xticklabels([])
        residuals = np.array(colour_rel_diams) / np.array(fit_diams)
        err_res = np.array(e_colour_rel_diams) / np.array(fit_diams)
        
        res_ax.errorbar(fit_diams, residuals, xerr=e_fit_diams, 
                        yerr=err_res, fmt=".", elinewidth=0.5,  
                        ecolor="firebrick", capsize=0.8, capthick=0.5, 
                        zorder=1)

        # Normalise colour scale on all scatter plots using this mask
        mask = np.logical_and(tgt_info["Science"], tgt_info["in_paper"])

        # Overplot scatter points so we can have [Fe/H] as colours
        if cbar == "feh":
            norm = plt.Normalize(tgt_info[mask]["FeH_rel"].min(), 
                                 tgt_info[mask]["FeH_rel"].max())
            scatter = ax.scatter(fit_diams, colour_rel_diams, c=fehs,  
                                 marker="o",zorder=2, norm=norm)
            res_ax.scatter(fit_diams, residuals, c=fehs, marker="o", 
                           zorder=2, norm=norm)
            cbar_label = "[Fe/H]"
    
        # Overplot scatter points so we can have Teff as colours
        elif cbar == "teff":
            norm = plt.Normalize(tgt_info[mask]["teff_final"].min(), 
                                 tgt_info[mask]["teff_final"].max())
            scatter = ax.scatter(fit_diams, colour_rel_diams, c=teffs, 
                                 marker="o", zorder=2, cmap="magma", norm=norm)
            res_ax.scatter(fit_diams, residuals, c=teffs, marker="o", zorder=2, 
                           cmap="magma", norm=norm)
            cbar_label = r"T$_{\rm eff}$"
        
        # Plot the two lines
        xlim = ax.get_xlim()
        ylim = [-1, 4.6]
        ax.plot(np.arange(0, 10), np.arange(0, 10), "--", color="black", 
                zorder=1)
        res_ax.hlines(1, xmin=0, xmax=10, linestyles="dashed", zorder=1)
                      
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        res_ax.set_xlim(xlim)
        res_ax.set_ylim([0.85, 1.2])
    
        # Set residual y ticks sensibly
        loc = plticker.MultipleLocator(base=0.1)
        res_ax.yaxis.set_major_locator(loc)
    
        # Setup the rest of the plot
        ax.set_ylabel(r"$\theta_{(%s)}$ (mas)" % colour_rel, fontsize="large")
        res_ax.set_xlabel(r"$\theta_{\rm PIONIER}$ (mas)", fontsize="large") 
        res_ax.set_ylabel(r"$\theta_{\rm %s} / \theta_{\rm PIONIER}$" 
                          % colour_rel, fontsize="large")
        
        if ax_i != 0:
            res_ax.set_yticklabels([])
        
        plt.setp(ax.get_xticklabels(), fontsize="medium")
        plt.setp(ax.get_yticklabels(), fontsize="medium")
        plt.setp(res_ax.get_xticklabels(), fontsize="medium")
        plt.setp(res_ax.get_yticklabels(), fontsize="medium")
    
    # Plot the colourbar
    cb = fig.colorbar(scatter, ax=axes.ravel().tolist())
    cb.set_label(cbar_label)
    
    plt.gcf().set_size_inches(12, 4)
    plt.savefig("paper/colour_rel_diam_comp_%s.pdf" % cbar, 
                bbox_inches="tight")     
    plt.savefig("paper/colour_rel_diam_comp_%s.png" % cbar, 
                bbox_inches="tight", dpi=200)   
        

def plot_casagrande_teff_comp(tgt_info, xy_map=None):
    """Plot for paper comparing measured LDD vs Boyajian colour relation diams.
    """
    plt.close("all")
    fig, ax = plt.subplots()
            
    # Setup lower panel for residuals
    divider = make_axes_locatable(ax)
    res_ax = divider.append_axes("bottom", size="30%", pad=0.1)
    ax.figure.add_axes(res_ax)
    
    # Initialise arrays
    final_teffs = []
    e_final_teffs = []
    casagrande_teffs = []
    e_casagrande_teffs = []
    fehs = []
    
    # Change the annotation rotation to prevent labels overlapping
    xy_txt = []
    
    # For every science target, plot using the given relation
    for star, star_data in tgt_info[tgt_info["Science"]].iterrows():
        
        if star_data["Primary"] in ["gamPav", "HD187289"]:
            continue
        
        # Get the two LDDs to compare
        final_teffs.append(star_data["teff_final"])
        e_final_teffs.append(star_data["e_teff_final"])
        casagrande_teffs.append(star_data["teff_casagrande"])
        e_casagrande_teffs.append(star_data["e_teff_casagrande"])
        fehs.append(star_data["FeH_rel"])
        
        # Compare positions
        # TODO: a better solution would be sorting the stars by LDD, then
        # alternate the sign on xx and yy to plot above or below...or just 
        # hardcode it
        xy_abs = (final_teffs[-1]**2 + casagrande_teffs[-1]**2)**0.5
        xy = np.abs(np.array(xy_txt) - xy_abs)
        sep = 50
        star_data["Primary"] = star_data["Primary"]
        
        # Import the provided xy_map if given
        if xy_map is not None:
           
            xx = xy_map["alfcmi"][0]
            yy = xy_map["alfcmi"][1]

            #xx = xy_map[star_data["Primary"]][0]
            #yy = xy_map[star_data["Primary"]][1]

        elif len(xy_txt) > 0 and (xy < sep).any():
            xx = 25
            yy = 200
        else:
            xx = 25
            yy = 100
        
        # Plot the name of the star
        ax.annotate(rutils.format_id(star_data["Primary"]),  
                    xy=(final_teffs[-1],casagrande_teffs[-1]), 
                    xytext=(final_teffs[-1]+xx, casagrande_teffs[-1]+yy), 
                    #arrowprops=dict(facecolor="black", width=0.0, 
                                    #headwidth=0.0),
                    fontsize="small", horizontalalignment="center")
                    
        xy_txt.append(xy_abs)
                        
    # Plot the points + errors
    ax.errorbar(final_teffs, casagrande_teffs, xerr=e_final_teffs, 
                yerr=e_casagrande_teffs, fmt=".", ecolor="firebrick",
                elinewidth=0.5, capsize=0.8, capthick=0.5, zorder=1)
        
    # Plot residuals
    ax.set_xticklabels([])
    residuals = np.array(casagrande_teffs) - np.array(final_teffs)
        
    res_ax.errorbar(final_teffs, residuals, xerr=e_final_teffs, 
                    yerr=e_casagrande_teffs, fmt=".", elinewidth=0.5, 
                    ecolor="firebrick", capsize=0.8, capthick=0.5, zorder=1)
    
    scatter = ax.scatter(final_teffs, casagrande_teffs, c=fehs, marker="o", 
                         zorder=2)
                         

    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("[Fe/H]", fontsize="x-large")
    cb.ax.tick_params(labelsize="x-large") 
    res_ax.scatter(final_teffs, residuals, c=fehs, marker="o", zorder=2)
    
    # Setup residual y axis
    maj_loc = plticker.MultipleLocator(base=150)
    min_loc = plticker.MultipleLocator(base=75)
    res_ax.yaxis.set_major_locator(maj_loc)
    res_ax.yaxis.set_minor_locator(min_loc)

    # Plot the two lines
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(np.arange(0, 10000), np.arange(0, 10000), "--", color="black", 
            zorder=1)
    res_ax.hlines(1, xmin=0, xmax=10000, linestyles="dashed", zorder=1)
                      
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    res_ax.set_xlim(xlim)
    
    # Set residual y ticks sensibly
    #loc = plticker.MultipleLocator(base=0.1)
    #res_ax.yaxis.set_major_locator(loc)
    #res_ax.yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    
    # Setup the rest of the plot
    ax.set_ylabel(r"$T_{\rm eff, Casagrande+2010}$ (K)", fontsize="x-large")  
    res_ax.set_xlabel(r"$T_{\rm eff, PIONIER}$ (K)", fontsize="x-large")  
    res_ax.set_ylabel(r"$T_{\rm eff, residuals}$ (K)", fontsize="x-large")  
    #ax.legend(loc="best")
    
    ax.tick_params(axis="both", which="major", labelsize="x-large")
    res_ax.tick_params(axis="both", which="major", labelsize="x-large")

    plt.tight_layout()
    plt.savefig("paper/teff_comp_casagrande.pdf")  
    plt.savefig("paper/teff_comp_casagrande.png", dpi=200)  


def plot_lit_teff_comp(tgt_info):
    """Plot for paper comparing measured LDD vs Boyajian colour relation diams.
    """
    plt.close("all")
    fig, ax = plt.subplots()
            
    # Setup lower panel for residuals
    divider = make_axes_locatable(ax)
    res_ax = divider.append_axes("bottom", size="30%", pad=0.1)
    ax.figure.add_axes(res_ax)
    
    # Initialise arrays
    final_teffs = []
    e_final_teffs = []
    lit_teffs = []
    e_lit_teffs = []
    
    # Change the annotation rotation to prevent labels overlapping
    xy_txt = []
    
    # For every science target, plot using the given relation
    for star, star_data in tgt_info[tgt_info["Science"]].iterrows():
        
        if star_data["Primary"] in ["gamPav", "HD187289"]:
            continue
        
        # Get the two LDDs to compare
        #final_teffs.append(star_data["teff_final"])
        #e_final_teffs.append(star_data["e_teff_final"])
        final_teffs.append(star_data["teff_casagrande"])
        e_final_teffs.append(star_data["e_teff_casagrande"])
        lit_teffs.append(star_data["Teff"])
        e_lit_teffs.append(star_data["e_teff"])
        
        # Compare positions
        # TODO: a better solution would be sorting the stars by LDD, then
        # alternate the sign on xx and yy to plot above or below...or just 
        # hardcode it
        xy_abs = (final_teffs[-1]**2 + lit_teffs[-1]**2)**0.5
        xy = np.abs(np.array(xy_txt) - xy_abs)
        sep = 0.1
        
        if len(xy_txt) > 0 and (xy < sep).any():
            xx = 0.025
            yy = 0.4
        else:
            xx = 0.025
            yy = 0.3 
        
        # Plot the name of the star
        ax.annotate(star_data["Primary"], xy=(final_teffs[-1], 
                    lit_teffs[-1]), 
                    xytext=(final_teffs[-1]+xx, lit_teffs[-1]-yy), 
                    arrowprops=dict(facecolor="black", width=0.1, 
                                    headwidth=0.1),
                    fontsize="xx-small")
                    
        xy_txt.append(xy_abs)
                        
    # Plot the points + errors
    ax.errorbar(final_teffs, lit_teffs, xerr=e_final_teffs, 
                yerr=e_lit_teffs, fmt=".",# label=colour_rel, 
                elinewidth=0.5, capsize=0.8, capthick=0.5, zorder=1)
        
    # Plot residuals
    ax.set_xticklabels([])
    residuals = np.array(lit_teffs) - np.array(final_teffs)
        
    res_ax.errorbar(final_teffs, residuals, xerr=e_final_teffs, 
                    yerr=e_lit_teffs, fmt=".", elinewidth=0.5, 
                    capsize=0.8, capthick=0.5, zorder=1)
    
    scatter = ax.scatter(final_teffs, lit_teffs, marker="o", 
                         zorder=2)
    
    # Plot the two lines
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(np.arange(0, 10000), np.arange(0, 10000), "--", color="black")
    res_ax.hlines(1, xmin=0, xmax=10000, linestyles="dashed")
                      
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    res_ax.set_xlim(xlim)
    
    # Set residual y ticks sensibly
    #loc = plticker.MultipleLocator(base=0.1)
    #res_ax.yaxis.set_major_locator(loc)
    #res_ax.yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    
    # Setup the rest of the plot
    ax.set_ylabel(r"T$_{\rm eff, literature}$")  
    res_ax.set_xlabel(r"T$_{\rm eff, PIONIER}$")  
    res_ax.set_ylabel(r"T$_{\rm eff, residuals}$")  
    #ax.legend(loc="best")
    
    plt.tight_layout()
    plt.savefig("plots/teff_comp_lit_casagrande.pdf")  

    
def plot_vis2(oi_fits_file, star_id):
    """
    Plot the calibrated VIS2 measurements contained in one OIFITS file.

    The OIFITS file can contain one or more observing sequences.
    Bad flagged measurements are excluded.

    Parameters
    ----------
    oi_fits_file : str
        Path to the calibrated OIFITS file.

    star_id : str
        Label displayed in the plot title.
    """

    plt.close("all")

    # ============================================================
    # Extract OIFITS information
    # ============================================================

    extracted = rdiam.extract_vis2(
        oi_fits_file
    )

    if len(extracted) != 7:
        raise ValueError(
            "extract_vis2 returned %i values instead of 7"
            % len(extracted)
        )

    mjds = extracted[0]
    pairs = extracted[1]
    vis2_all = extracted[2]
    e_vis2_all = extracted[3]
    flags_all = extracted[4]
    baselines_all = extracted[5]
    wavelengths = extracted[6]

    wavelengths = np.asarray(
        wavelengths,
        dtype=float
    ).ravel()

    if len(wavelengths) == 0:
        raise ValueError(
            "No wavelengths found in %s"
            % oi_fits_file
        )

    # ============================================================
    # Create figure
    # ============================================================

    fig, ax = plt.subplots()

    n_sequences = len(vis2_all)
    n_points_plotted = 0

    print("")
    print("Raw visibility plot:")
    print("  file        =", oi_fits_file)
    print("  sequences   =", n_sequences)
    print("  wavelengths =", len(wavelengths))

    # ============================================================
    # Plot each sequence
    # ============================================================

    for seq_i in xrange(n_sequences):

        vis2 = np.asarray(
            vis2_all[seq_i],
            dtype=float
        )

        e_vis2 = np.asarray(
            e_vis2_all[seq_i],
            dtype=float
        )

        baselines = np.asarray(
            baselines_all[seq_i],
            dtype=float
        ).ravel()

        flags = np.asarray(
            flags_all[seq_i]
        )

        
        
        print("")
        print("==========================================")
        print("FLAG DEBUG")
        print("sequence:", seq_i)
        print("VIS2 shape:", vis2.shape)
        print("VIS2 size :", vis2.size)
        print("FLAG shape:", flags.shape)
        print("FLAG size :", flags.size)
        print("FLAG True :", np.sum(flags.astype(bool)))
        print("FLAG False:", np.sum(~flags.astype(bool)))
        print("==========================================")

        n_wl = len(wavelengths)

        # --------------------------------------------------------
        # Ensure VIS2 is a 2-D array: n_baselines x n_wavelengths
        # --------------------------------------------------------

        if vis2.ndim == 1:

            if vis2.size % n_wl != 0:
                raise ValueError(
                    "Cannot reshape VIS2 for sequence %i in %s: "
                    "VIS2 size=%i, number of wavelengths=%i"
                    % (
                        seq_i,
                        oi_fits_file,
                        vis2.size,
                        n_wl
                    )
                )

            vis2 = vis2.reshape(
                (-1, n_wl)
            )

        if e_vis2.ndim == 1:

            if e_vis2.size != vis2.size:
                raise ValueError(
                    "VIS2 and e_VIS2 sizes differ for sequence %i "
                    "in %s: %i versus %i"
                    % (
                        seq_i,
                        oi_fits_file,
                        vis2.size,
                        e_vis2.size
                    )
                )

            e_vis2 = e_vis2.reshape(
                vis2.shape
            )

        # Some files could contain transposed VIS2 arrays.
        if (
            vis2.ndim == 2
            and vis2.shape[0] == n_wl
            and vis2.shape[1] == len(baselines)
        ):

            vis2 = vis2.T
            e_vis2 = e_vis2.T

            if flags.size == vis2.size:
                flags = flags.reshape(
                    e_vis2.T.shape
                ).T

        n_bl = vis2.shape[0]

        if vis2.shape[1] != n_wl:
            raise ValueError(
                "Unexpected VIS2 shape for sequence %i in %s: "
                "%s; expected second dimension=%i"
                % (
                    seq_i,
                    oi_fits_file,
                    str(vis2.shape),
                    n_wl
                )
            )

        if len(baselines) != n_bl:

            # In some files the baseline may be repeated for every
            # wavelength channel.
            if baselines.size == vis2.size:

                baselines = baselines.reshape(
                    vis2.shape
                )[:, 0]

            else:

                raise ValueError(
                    "Baseline and VIS2 dimensions differ for "
                    "sequence %i in %s: baselines=%i, VIS2 rows=%i"
                    % (
                        seq_i,
                        oi_fits_file,
                        len(baselines),
                        n_bl
                    )
                )

        # --------------------------------------------------------
        # Prepare flags
        # --------------------------------------------------------

        if flags.size == vis2.size:

            flags = flags.reshape(
                vis2.shape
            ).astype(bool)

        elif flags.size == n_bl:

            flags = np.repeat(
                flags.astype(bool)[:, np.newaxis],
                n_wl,
                axis=1
            )

        else:

            print(
                "WARNING: unexpected FLAG shape for sequence %i: %s"
                % (
                    seq_i,
                    str(flags.shape)
                )
            )

            print(
                "Ignoring flags for this sequence"
            )

            flags = np.zeros(
                vis2.shape,
                dtype=bool
            )

        # --------------------------------------------------------
        # Spatial frequency
        # --------------------------------------------------------

        bl_grid = np.repeat(
            baselines[:, np.newaxis],
            n_wl,
            axis=1
        )

        wl_grid = np.repeat(
            wavelengths[np.newaxis, :],
            n_bl,
            axis=0
        )

        sfreq = bl_grid / wl_grid

        # --------------------------------------------------------
        # Valid measurements
        # --------------------------------------------------------

        good = (
            ~flags
            & np.isfinite(sfreq)
            & np.isfinite(vis2)
            & np.isfinite(e_vis2)
            & (e_vis2 > 0)
        )

        n_good = int(
            np.sum(good)
        )

        print(
            "  sequence %i: VIS2 shape=%s, baselines=%i, valid=%i"
            % (
                seq_i + 1,
                str(vis2.shape),
                n_bl,
                n_good
            )
        )

        if n_good == 0:

            print(
                "WARNING: no valid VIS2 points in sequence %i"
                % (seq_i + 1)
            )

            continue

        sequence_label = (
            "Sequence %i"
            % (seq_i + 1)
        )

        ax.errorbar(
            sfreq[good],
            vis2[good],
            yerr=e_vis2[good],
            fmt=".",
            label=sequence_label,
            elinewidth=0.3,
            capsize=0.5,
            capthick=0.3
        )

        n_points_plotted += n_good

    # ============================================================
    # Confirm that something was plotted
    # ============================================================

    if n_points_plotted == 0:

        plt.close(fig)

        raise RuntimeError(
            "No valid VIS2 measurements could be plotted from %s"
            % oi_fits_file
        )

    # ============================================================
    # Labels and limits
    # ============================================================

    ax.set_xlabel(
        r"Spatial frequency (rad$^{-1}$)"
    )

    ax.set_ylabel(
        r"Calibrated visibility$^2$"
    )

    ax.set_title(
        r"%s (%i valid vis$^2$ points)"
        % (
            star_id,
            n_points_plotted
        )
    )

    ax.set_xlim(
        [0.0, 25E7]
    )

    # Do not force the upper limit to exactly 1 because calibrated
    # visibilities can be slightly above one.
    all_y_values = []

    for line in ax.lines:

        try:

            line_y = np.asarray(
                line.get_ydata(),
                dtype=float
            )

            line_y = line_y[
                np.isfinite(line_y)
            ]

            all_y_values.extend(
                line_y.tolist()
            )

        except Exception:
            pass

    if len(all_y_values) > 0:

        y_upper = max(
            1.1,
            np.nanpercentile(
                np.asarray(all_y_values),
                99
            ) + 0.05
        )

        y_upper = min(
            y_upper,
            2.0
        )

    else:

        y_upper = 1.1

    ax.set_ylim(
        [0.0, y_upper]
    )

    ax.grid()

    if n_sequences > 1:

        ax.legend(
            loc="best",
            fontsize="small"
        )

    plt.tight_layout()

    # Do not close the figure here.
    # plot_last_bootstrap_oifits saves it after this function.

def plot_fbol_comp(tgt_info):
    """Plot a comparison of the sampled values of fbol from each filter to 
    check whether they are consistent or not.
    """
    plt.close("all")
    fig, axis = plt.subplots()
    
    # Define bands to reference, construct new headers
    bands = ["Hp", "BT", "VT"]#, "BP", "RP"]
    band_lbl = [r"$f_{\rm bol (H_p)}$", r"$f_{\rm bol (B_T)}$", 
                r"$f_{\rm bol (V_T)}$", r"$f_{\rm bol (final)}$"]
    f_bol_bands = ["f_bol_%s" % band for band in bands] + ["f_bol_final"]
    e_f_bol_bands = ["e_f_bol_%s" % band for band in bands] + ["e_f_bol_final"]
    
    bands += ["Avg"]
    
    offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
    trans = plt.gca().transData
    
    tf = 4
    
    colours = ["green", "blue", "red", "black"]#"orange", "deepskyblue", "red", "black"]
    
    mask = np.logical_and(tgt_info["Science"], tgt_info["in_paper"])
    
    ids = [rutils.format_id(star) for star in tgt_info[mask]["Primary"].values]
    
    fbol = tgt_info[mask][f_bol_bands]
    e_fbol = tgt_info[mask][e_f_bol_bands]

    for band_i, (fband, e_fband) in enumerate(zip(f_bol_bands, e_f_bol_bands)):
        axis.errorbar(ids, fbol[fband], yerr=e_fbol[e_fband], elinewidth=0.3,
                     fmt=".", zorder=1, label="", ecolor="black",capsize=1,
                     capthick=0.3, transform=trans+offset(-tf*(band_i)+8),
                     markersize=0.1)
        axis.scatter(ids, fbol[fband], s=1**4, c=colours[band_i], label=band_lbl[band_i],
                    zorder=2, transform=trans+offset(-tf*(band_i)+8))#, 
                    #marker="$%s$" % bands[band_i])

    axis.yaxis.get_major_formatter().set_powerlimits((0,1))                   
    #axis.set_xlabel("Star", fontsize="large")
    axis.set_ylabel(r"$f_{\rm bol}$ (ergs s$^{-1}$ cm $^{-2}$)", 
                    fontsize="x-large")
    plt.setp(axis.get_yticklabels(), fontsize="x-large")
    plt.setp(axis.get_xticklabels(), fontsize="x-large", rotation="vertical")
    #plt.yscale("log")
    plt.tight_layout()
    legend = plt.legend(loc="best", fontsize="x-large")
    for handle in legend.legendHandles:
        handle.set_sizes([20])

    #plt.gcf().set_size_inches(16, 9)
    plt.savefig("paper/fbol_comp.pdf")
    

def plot_jsdc_ldd_comp(tgt_info):
    """
    """
    tgt_info[["Primary","LDD_pred","e_LDD_pred","JSDC_LDD","e_JSDC_LDD"]]
    
    plt.close("all")
    fig, ax = plt.subplots()
            
    # Setup lower panel for residuals
    divider = make_axes_locatable(ax)
    res_ax = divider.append_axes("bottom", size="30%", pad=0.1)
    ax.figure.add_axes(res_ax)
    
    # Initialise arrays
    pred_ldd = []
    e_pred_ldd = []
    lsdc_ldd = []
    e_lsdc_ldd = []
    
    # Change the annotation rotation to prevent labels overlapping
    xy_txt = []
    
    # For every science target, plot using the given relation
    for star, star_data in tgt_info[tgt_info["Quality"] != "BAD"].iterrows():
        
        if star_data["Primary"] in ["gamPav", "HD187289"]:
            continue
        
        # Get the two LDDs to compare
        pred_ldd.append(star_data["LDD_pred"])
        e_pred_ldd.append(star_data["e_LDD_pred"])
        lsdc_ldd.append(star_data["JSDC_LDD"])
        e_lsdc_ldd.append(star_data["e_JSDC_LDD"])
        
        # Compare positions
        # TODO: a better solution would be sorting the stars by LDD, then
        # alternate the sign on xx and yy to plot above or below...or just 
        # hardcode it
        xy_abs = (pred_ldd[-1]**2 + lsdc_ldd[-1]**2)**0.5
        xy = np.abs(np.array(xy_txt) - xy_abs)
        sep = 0.1
        
        if len(xy_txt) > 0 and (xy < sep).any():
            xx = 0.025
            yy = 0.4
        else:
            xx = 0.025
            yy = 0.3 
        
        # Plot the name of the star
        ax.annotate(star_data["Primary"], xy=(pred_ldd[-1], 
                    lsdc_ldd[-1]), 
                    xytext=(pred_ldd[-1]+xx, lsdc_ldd[-1]-yy), 
                    #arrowprops=dict(facecolor="black", width=0.1, 
                    #                headwidth=0.1),
                    fontsize="xx-small")
                    
        xy_txt.append(xy_abs)
                        
    # Plot the points + errors
    ax.errorbar(pred_ldd, lsdc_ldd, xerr=e_pred_ldd, 
                yerr=e_lsdc_ldd, fmt=".",# label=colour_rel, 
                elinewidth=0.5, capsize=0.8, capthick=0.5, zorder=1)
        
    # Plot residuals
    ax.set_xticklabels([])
    residuals = np.array(lsdc_ldd) / np.array(pred_ldd)
    err_res = np.array(e_lsdc_ldd) / np.array(pred_ldd)
    
    #residuals = np.array(lsdc_ldd) - np.array(pred_ldd)
        
    res_ax.errorbar(pred_ldd, residuals, xerr=e_pred_ldd, 
                    yerr=err_res, fmt=".", elinewidth=0.5, 
                    capsize=0.8, capthick=0.5, zorder=1)
    
    scatter = ax.scatter(pred_ldd, lsdc_ldd, marker="o", 
                         zorder=2)
    
    # Plot the two lines
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(np.arange(0, 10000), np.arange(0, 10000), "--", color="black")
    res_ax.hlines(1, xmin=0, xmax=10000, linestyles="dashed")
                      
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    res_ax.set_xlim(xlim)
    
    # Set residual y ticks sensibly
    #loc = plticker.MultipleLocator(base=0.1)
    #res_ax.yaxis.set_major_locator(loc)
    #res_ax.yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    
    # Setup the rest of the plot
    ax.set_ylabel(r"$\theta_{\rm JSDC}$ (mas)")  
    res_ax.set_xlabel(r"$\theta_{\rm pred}$ (mas)")  
    res_ax.set_ylabel(r"$\theta_{\rm pred} / \theta_{\rm JSDC}$ (mas)")  
    #ax.legend(loc="best")
    
    plt.tight_layout()
    plt.savefig("plots/ldd_comp_jsdc.pdf") 

def plot_claret_vs_stagger_diam_comp():
    """
    """
    diams_claret = pd.read_csv("results/paper_results/diams_claret.csv")
    diams_stagger = pd.read_csv("results/paper_results/diams_stagger.csv")

    plt.close("all")
    fig, ax = plt.subplots()
            
    # Setup lower panel for residuals
    divider = make_axes_locatable(ax)
    res_ax = divider.append_axes("bottom", size="30%", pad=0.1)
    ax.figure.add_axes(res_ax)
    
    # Change the annotation rotation to prevent labels overlapping
    xy_txt = []
    """
    # For every science target, plot using the given relation
    for star, star_data in tgt_info[tgt_info["Science"]].iterrows():
        
        # Get the two LDDs to compare
        final_teffs.append(star_data["teff_final"])
        e_final_teffs.append(star_data["e_teff_final"])
        casagrande_teffs.append(star_data["teff_casagrande"])
        e_casagrande_teffs.append(star_data["e_teff_casagrande"])
        fehs.append(star_data["FeH_rel"])
    
        
        # Import the provided xy_map if given
        if xy_map is not None:
            xx = xy_map[star_data["Primary"]][0]
            yy = xy_map[star_data["Primary"]][1]

        elif len(xy_txt) > 0 and (xy < sep).any():
            xx = 25
            yy = 200
        else:
            xx = 25
            yy = 100
        
        # Plot the name of the star
        ax.annotate(rutils.format_id(star_data["Primary"]),  
                    xy=(final_teffs[-1],casagrande_teffs[-1]), 
                    xytext=(final_teffs[-1]+xx, casagrande_teffs[-1]+yy), 
                    #arrowprops=dict(facecolor="black", width=0.0, 
                                    #headwidth=0.0),
                    fontsize="small", horizontalalignment="center")
                    
        xy_txt.append(xy_abs)
    """
    # Plot the points + errors
    ax.errorbar(diams_claret["ldd_final"], diams_stagger["ldd_final"], 
                xerr=diams_claret["e_ldd_final"], 
                yerr=diams_stagger["e_ldd_final"], fmt=".", ecolor="firebrick",
                elinewidth=0.5, capsize=0.8, capthick=0.5, zorder=1)
        
    # Plot residuals
    ax.set_xticklabels([])
    residuals = diams_stagger["ldd_final"] - diams_claret["ldd_final"]
        
    res_ax.errorbar(diams_claret["ldd_final"], residuals, 
                    xerr=diams_claret["e_ldd_final"], 
                    yerr=diams_stagger["e_ldd_final"], fmt=".", elinewidth=0.5, 
                    ecolor="firebrick", capsize=0.8, capthick=0.5, zorder=1)
    
    scatter = ax.scatter(diams_claret["ldd_final"], diams_stagger["ldd_final"], 
                         c=diams_claret["FeH_rel"], marker="o", zorder=2)
                         

    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("[Fe/H]")
    res_ax.scatter(diams_claret["ldd_final"], residuals, 
                   c=diams_claret["FeH_rel"], marker="o", zorder=2)
    
    # Setup residual y axis
    maj_loc = plticker.MultipleLocator(base=150)
    min_loc = plticker.MultipleLocator(base=75)
    res_ax.yaxis.set_major_locator(maj_loc)
    res_ax.yaxis.set_minor_locator(min_loc)

    # Plot the two lines
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(np.arange(0, 10000), np.arange(0, 10000), "--", color="black", 
            zorder=1)
    res_ax.hlines(0, xmin=0, xmax=10000, linestyles="dashed", zorder=1)
                      
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    res_ax.set_xlim(xlim)
    
    # Set residual y ticks sensibly
    #loc = plticker.MultipleLocator(base=0.1)
    #res_ax.yaxis.set_major_locator(loc)
    #res_ax.yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    
    # Setup the rest of the plot
    ax.set_ylabel(r"$\theta_{\rm Final, Stagger}$ (mas)", fontsize="large")  
    res_ax.set_xlabel(r"$\theta_{\rm Final, Claret}$ (mas)", fontsize="large")  
    res_ax.set_ylabel(r"$\theta_{\rm Final, residuals}$ (K)", fontsize="large")  
    #ax.legend(loc="best")
    
    plt.tight_layout()
    #plt.savefig("paper/teff_comp_casagrande.pdf")  
    #plt.savefig("paper/teff_comp_casagrande.png", dpi=500)  

def plot_hr_diagram(
        tgt_info,
        plot_isochrones_basti=False,
        plot_isochrones_padova=False,
        feh=0.062,
        basti_folder="data/basti",
        basti_ages_myr=None):
    """
    Plot an absolute Johnson colour-magnitude diagram:

        x = B - V
        y = absolute V magnitude

    BaSTI Johnson-Cousins files are expected to contain:

        M_ini, M_fin, logL, logTe,
        U, BX, B, V, R, I, J, H, K, Lprime, L, M

    For the BaSTI isochrones:

        B-V = B - V
        Mv  = V

    Parameters
    ----------
    tgt_info : pandas.DataFrame
        Target information.

    plot_isochrones_basti : bool
        Plot BaSTI Johnson-Cousins isochrones.

    plot_isochrones_padova : bool
        Plot the previous Padova isochrones.

    feh : float
        Requested BaSTI [M/H]. A tolerance of 0.02 dex is used.

    basti_folder : str
        Folder containing the extracted .isc_john files.

    basti_ages_myr : list or None
        Specific ages to plot. The nearest available isochrone is used.

        Example:
            [60, 100, 500, 1000, 2000, 5000, 10000, 14000]

        When None, every available isochrone is plotted.
    """

    plt.close("all")

    fig, ax = plt.subplots()

    fig.set_size_inches(
        10,
        8
    )

    # =====================================================================
    # Science-star photometry
    # =====================================================================

    mask = np.logical_and(
        np.asarray(
            tgt_info["Science"],
            dtype=bool
        ),
        np.asarray(
            tgt_info["in_paper"],
            dtype=bool
        )
    )

    selected_targets = tgt_info.loc[
        mask
    ].copy()

    apparent_vmag = np.asarray(
        selected_targets["Vmag_dr"],
        dtype=float
    )

    apparent_bmag = np.asarray(
        selected_targets["Bmag_dr"],
        dtype=float
    )

    distance_pc = np.asarray(
        selected_targets["Dist"],
        dtype=float
    )

    metallicity = np.asarray(
        selected_targets["FeH_rel"],
        dtype=float
    )

    # Absolute V magnitude.
    abs_vmag = (
        apparent_vmag
        - 5.0
        * np.log10(
            distance_pc / 10.0
        )
    )

    # Johnson B-V colour.
    b_minus_v = (
        apparent_bmag
        - apparent_vmag
    )

    valid_stars = (
        np.isfinite(
            abs_vmag
        )
        & np.isfinite(
            b_minus_v
        )
        & np.isfinite(
            metallicity
        )
    )

    star_scatter = ax.scatter(
        b_minus_v[
            valid_stars
        ],
        abs_vmag[
            valid_stars
        ],
        s=100,
        c=metallicity[
            valid_stars
        ],
        marker="o",
        zorder=5
    )

    # =====================================================================
    # Annotate the star names
    # =====================================================================

    primary_values = selected_targets[
        "Primary"
    ].values

    star_ids = rutils.format_id(
        primary_values
    )

    for star_i, star in enumerate(
            star_ids):

        if not valid_stars[
                star_i]:

            continue

        primary_name = str(
            primary_values[
                star_i
            ]
        )

        if primary_name in (
                "epsInd",
                "chiEri"):

            yy = -0.05
            xx = 0.1

        elif primary_name in (
                "37Lib",):

            yy = 0.3
            xx = 0.0

        else:

            yy = 0.2
            xx = 0.0

        ax.annotate(
            star,
            xy=(
                b_minus_v[
                    star_i
                ],
                abs_vmag[
                    star_i
                ]
            ),
            xytext=(
                b_minus_v[
                    star_i
                ] - xx,
                abs_vmag[
                    star_i
                ] - yy
            ),
            fontsize="medium",
            horizontalalignment="center"
        )

    # =====================================================================
    # Padova isochrones
    # =====================================================================

    if plot_isochrones_padova:

        isochrones_file = (
            "data/padova_isochrone_age.dat"
        )

        names = [
            "Zini",
            "Age",
            "Mini",
            "Mass",
            "logL",
            "logTe",
            "logg",
            "label",
            "McoreTP",
            "C_O",
            "period0",
            "period1",
            "pmode",
            "Mloss",
            "tau1m",
            "X",
            "Y",
            "Xc",
            "Xn",
            "Xo",
            "Cexcess",
            "Z",
            "mbolmag",
            "Gmag",
            "G_BPbrmag",
            "G_BPftmag",
            "G_RPmag",
            "B_Tmag",
            "V_Tmag",
            "Jmag",
            "Hmag",
            "Ksmag"
        ]

        isochrones = pd.read_csv(
            isochrones_file,
            delim_whitespace=True,
            names=names,
            comment="#",
            dtype="float"
        )

        ages = sorted(
            list(
                set(
                    isochrones[
                        "Age"
                    ]
                )
            )
        )

        isochrones[
            "BtVt"
        ] = (
            isochrones[
                "B_Tmag"
            ]
            - isochrones[
                "V_Tmag"
            ]
        )

        for age in ages[
                20:]:

            age_mask = (
                isochrones[
                    "Age"
                ]
                == age
            )

            ax.plot(
                isochrones.loc[
                    age_mask,
                    "BtVt"
                ].values[:-3],
                isochrones.loc[
                    age_mask,
                    "V_Tmag"
                ].values[:-3],
                linestyle="--",
                label=(
                    "%.3f Gyr"
                    % (
                        age / 1.0E9
                    )
                ),
                alpha=0.5,
                zorder=2
            )

    # =====================================================================
    # BaSTI Johnson-Cousins isochrones
    # =====================================================================

    if plot_isochrones_basti:

        # -------------------------------------------------------------
        # Actual columns in your .isc_john files
        # -------------------------------------------------------------

        basti_names = [
            "M_ini",
            "M_fin",
            "logL",
            "logTe",
            "U",
            "BX",
            "B",
            "V",
            "R",
            "I",
            "J",
            "H",
            "K",
            "Lprime",
            "L",
            "M"
        ]

        # -------------------------------------------------------------
        # Find files recursively
        # -------------------------------------------------------------

        iso_files = []

        for root_directory, directory_names, filenames in os.walk(
                basti_folder):

            for filename in filenames:

                if filename.endswith(
                        ".isc_john"):

                    iso_files.append(
                        os.path.join(
                            root_directory,
                            filename
                        )
                    )

        iso_files = sorted(
            list(
                set(
                    iso_files
                )
            )
        )

        if len(iso_files) == 0:

            raise IOError(
                "No .isc_john files found inside %s"
                % basti_folder
            )

        print("")
        print("=" * 79)
        print(
            "Found %i BaSTI Johnson-Cousins files"
            % len(iso_files)
        )
        print("=" * 79)

        basti_isochrones = []

        # -------------------------------------------------------------
        # Read each file
        # -------------------------------------------------------------

        for iso_file in iso_files:

            age_myr = np.nan
            file_mh = np.nan
            n_columns = None

            # Read age and composition from header.
            with open(
                    iso_file,
                    "r") as input_handle:

                for line in input_handle:

                    stripped_line = line.strip()

                    if stripped_line == "":
                        continue

                    if "Age (Myr)" in stripped_line:

                        try:

                            age_text = (
                                stripped_line
                                .split(
                                    "Age (Myr) ="
                                )[1]
                                .strip()
                                .split()[0]
                            )

                            age_myr = float(
                                age_text
                            )

                        except Exception:

                            age_myr = np.nan

                    if "[M/H]" in stripped_line:

                        try:

                            metallicity_text = (
                                stripped_line
                                .split(
                                    "[M/H] ="
                                )[1]
                                .split(
                                    "Z ="
                                )[0]
                                .strip()
                            )

                            file_mh = float(
                                metallicity_text
                            )

                        except Exception:

                            file_mh = np.nan

                    # Count columns in the first numerical row.
                    if not stripped_line.startswith(
                            "#"):

                        n_columns = len(
                            stripped_line.split()
                        )

                        break

            if n_columns is None:

                print(
                    "WARNING: no numerical data in %s"
                    % iso_file
                )

                continue

            if n_columns != len(
                    basti_names):

                print("")
                print(
                    "WARNING: skipping %s"
                    % iso_file
                )

                print(
                    "It contains %i columns; expected %i"
                    % (
                        n_columns,
                        len(
                            basti_names
                        )
                    )
                )

                continue

            # Filter by metallicity using the header rather than filename.
            if (
                feh is not None
                and np.isfinite(
                    file_mh
                )
                and abs(
                    file_mh - float(feh)
                ) > 0.02
            ):

                print(
                    "Skipping %s: [M/H] = %.3f"
                    % (
                        iso_file,
                        file_mh
                    )
                )

                continue

            track = pd.read_csv(
                iso_file,
                delim_whitespace=True,
                names=basti_names,
                comment="#",
                dtype="float"
            )

            # ---------------------------------------------------------
            # Calculate the quantities needed by the diagram
            # ---------------------------------------------------------

            track[
                "B-V"
            ] = (
                track[
                    "B"
                ]
                - track[
                    "V"
                ]
            )

            # BaSTI photometric magnitudes are absolute magnitudes.
            track[
                "Mv"
            ] = track[
                "V"
            ]

            valid_track = (
                np.isfinite(
                    track[
                        "B-V"
                    ]
                )
                & np.isfinite(
                    track[
                        "Mv"
                    ]
                )
            )

            track = track.loc[
                valid_track
            ].copy()

            if len(track) == 0:

                print(
                    "WARNING: no valid B-V/V points in %s"
                    % iso_file
                )

                continue

            basti_isochrones.append({
                "age_myr": age_myr,
                "mh": file_mh,
                "filename": iso_file,
                "track": track
            })

        if len(basti_isochrones) == 0:

            raise RuntimeError(
                "No valid BaSTI Johnson isochrones were read"
            )

        # -------------------------------------------------------------
        # Sort by age
        # -------------------------------------------------------------

        basti_isochrones.sort(
            key=lambda entry: (
                entry[
                    "age_myr"
                ]
                if np.isfinite(
                    entry[
                        "age_myr"
                    ]
                )
                else 1.0E99
            )
        )

        # -------------------------------------------------------------
        # Select requested ages
        # -------------------------------------------------------------

        if basti_ages_myr is not None:

                selected_isochrones = []

                # Store filenames rather than comparing dictionaries containing
                # pandas DataFrames.
                selected_filenames = set()

                finite_age_isochrones = [
                    entry
                    for entry in basti_isochrones
                    if np.isfinite(
                        entry[
                            "age_myr"
                        ]
                    )
                ]

                if len(
                        finite_age_isochrones) == 0:

                    raise RuntimeError(
                        "No BaSTI isochrones have a valid age"
                    )

                for requested_age in basti_ages_myr:

                    requested_age = float(
                        requested_age
                    )

                    closest_isochrone = min(
                        finite_age_isochrones,
                        key=lambda entry: abs(
                            float(
                                entry[
                                    "age_myr"
                                ]
                            )
                            - requested_age
                        )
                    )

                    closest_filename = str(
                        closest_isochrone[
                            "filename"
                        ]
                    )

                    # Do not compare the complete dictionary because it contains
                    # a pandas DataFrame in closest_isochrone["track"].
                    if closest_filename not in selected_filenames:

                        selected_isochrones.append(
                            closest_isochrone
                        )

                        selected_filenames.add(
                            closest_filename
                        )

                        print(
                            "Requested %.0f Myr -> using %.0f Myr"
                            % (
                                requested_age,
                                closest_isochrone[
                                    "age_myr"
                                ]
                            )
                        )

                    else:

                        print(
                            "Requested %.0f Myr -> %.0f Myr already selected"
                            % (
                                requested_age,
                                closest_isochrone[
                                    "age_myr"
                                ]
                            )
                        )

                basti_isochrones = sorted(
                    selected_isochrones,
                    key=lambda entry: float(
                        entry[
                            "age_myr"
                        ]
                    )
                )
        # -------------------------------------------------------------
        # Colour map
        # -------------------------------------------------------------

        try:

            colour_map = cm.get_cmap(
                "viridis"
            )

        except Exception:

            colour_map = cm.get_cmap(
                "jet"
            )

        n_isochrones = len(
            basti_isochrones
        )

        # -------------------------------------------------------------
        # Plot each constant-age isochrone
        # -------------------------------------------------------------

        for iso_i, entry in enumerate(
                basti_isochrones):

            track = entry[
                "track"
            ]

            age_myr = entry[
                "age_myr"
            ]

            if n_isochrones == 1:

                curve_colour = colour_map(
                    0.5
                )

            else:

                curve_colour = colour_map(
                    float(
                        iso_i
                    )
                    / float(
                        n_isochrones - 1
                    )
                )

            if np.isfinite(
                    age_myr):

                if age_myr >= 1000.0:

                    age_label = (
                        "%.1f Gyr"
                        % (
                            age_myr
                            / 1000.0
                        )
                    )

                else:

                    age_label = (
                        "%.0f Myr"
                        % age_myr
                    )

            else:

                age_label = os.path.basename(
                    entry[
                        "filename"
                    ]
                )

            ax.plot(
                track[
                    "B-V"
                ],
                track[
                    "Mv"
                ],
                linestyle="--",
                color=curve_colour,
                label=age_label,
                alpha=0.65,
                zorder=2,
                linewidth=0.8
            )

        ax.legend(
            loc="best",
            fontsize=7,
            ncol=2
        )

    # =====================================================================
    # Final formatting
    # =====================================================================

    colour_bar = fig.colorbar(
        star_scatter,
        ax=ax
    )

    colour_bar.set_label(
        r"[Fe/H]",
        fontsize="x-large"
    )

    colour_bar.ax.tick_params(
        labelsize="x-large"
    )

    ax.tick_params(
        axis="both",
        labelsize="x-large"
    )

    ax.set_xlim(
        [0.0, 1.5]
    )

    # Magnitudes increase downward.
    ax.set_ylim(
        [7.5, 0.0]
    )

    ax.set_xlabel(
        r"$(B-V)$",
        fontsize="x-large"
    )

    ax.set_ylabel(
        r"$M_V$",
        fontsize="x-large"
    )

    fig.tight_layout()

    output_file = (
        "paper/hr_diagram.pdf"
    )

    fig.savefig(
        output_file
    )

    plt.close(
        fig
    )

    print(
        "Saved HR diagram:"
    )

    print(
        output_file
    )

    return output_file
def plot_hr_diagram_old(tgt_info, plot_isochrones_basti=False, 
                    plot_isochrones_padova=False, feh=0.058):
    """Plots the Vt, (Bt-Vt) colour magnitude diagram for all science targets.
    """
    plt.close("all")
    mask = np.logical_and(tgt_info["Science"], tgt_info["in_paper"])
    

    abs_Vmag = tgt_info[mask]["Vmag_dr"] - 5*np.log10(tgt_info[mask]["Dist"]/10) 
    b_v = tgt_info[mask]["Bmag_dr"] - tgt_info[mask]["Vmag_dr"]
    
    plt.close("all")
    plt.scatter(b_v, abs_Vmag, s=100, c=tgt_info[mask]["FeH_rel"], marker="o")
    
    # Annotate the star name
    star_ids = rutils.format_id(tgt_info[mask]["Primary"].values)
    for star_i, star in enumerate(star_ids):
        if tgt_info[mask]["Primary"][star_i] in ("epsInd", "chiEri"):
            yy = -0.05
            xx = 0.1
        elif tgt_info[mask]["Primary"][star_i] in ("37Lib"):
            yy = 0.3
            xx = 0.0
        else:
            yy = 0.2
            xx = 0.0
        
        plt.annotate(star, xy=(b_v[star_i], abs_Vmag[star_i]), 
                    xytext=(b_v[star_i]-xx, abs_Vmag[star_i]-yy), 
                    fontsize="medium", horizontalalignment="center")
    
    # Plot Padova isochrones. Note that these are for constant *age*
    if plot_isochrones_padova:
        isochrones_file = "data/padova_isochrone_age.dat"
        names = ["Zini","Age","Mini","Mass","logL","logTe","logg","label",
                 "McoreTP","C_O","period0","period1","pmode","Mloss","tau1m",
                 "X","Y","Xc","Xn","Xo","Cexcess","Z","mbolmag","Gmag",
                 "G_BPbrmag", "G_BPftmag","G_RPmag","B_Tmag", "V_Tmag","Jmag",
                 "Hmag","Ksmag"]        
        
        isochrones = pd.read_csv(isochrones_file, delim_whitespace=True, 
                                 names=names, comment="#", dtype="float")
                                 
        ages = list(set(isochrones["Age"]))
        ages.sort()
        
        # Calculate colour
        isochrones["BtVt"] = isochrones["B_Tmag"] - isochrones["V_Tmag"]
        #isochrones["B-V"] = isochrones["B_Tmag"] - isochrones["V_Tmag"]
        
        for age in ages[20:]:
            plt.plot(isochrones.loc[isochrones["Age"]==age]["BtVt"][:-3], 
                     isochrones.loc[isochrones["Age"]==age]["V_Tmag"][:-3], "--", 
                     label="%0.3f Gyr" % (age/10**9), alpha=0.5, zorder=2)
    
    # Plot Basti evolutionary tracks for constant mass
    if plot_isochrones_basti:
        names = ['logage', 'M/Mo', 'logL/Lo', 'logTe', 'Mv', 'U-B', 'B-V',
                 'V-I', 'V-R', 'V-J', 'V-K', 'V-L', 'H-K']
                 
        iso_files = glob.glob("data/basti/*%s*" % str(feh))
        iso_files.sort()
        masses = [float(file.split("/")[-1].split("m")[0]) for file in iso_files]
        
        mass_data = []
        
        for mass_i, iso_file in enumerate(iso_files):
            track = pd.read_csv(iso_file, delim_whitespace=True, 
                                names=names, comment="#", dtype="float")
                                
            plt.plot(track["B-V"], track["Mv"], "--", color="black",
                     label=r"M$_\odot$=%0.2f" % masses[mass_i], alpha=0.5, 
                     zorder=2, linewidth=0.5)
            
            # Make sure track mass labels don't overlap with stars
            if track["B-V"][0] < 0.05:
                xx = 0.11
                yy = -0.05

            elif track["B-V"][0] < 0.15:
                xx = 0.075
                yy = 0.22

            elif track["B-V"][0] < 0.8 and track["Mv"][0] > 5:
                xx = -0.1
                yy = 0.0

            else:
                xx = 0.01
                yy = 0.3
            
            plt.text(track["B-V"][0]+xx, track["Mv"][0]+yy,
                     r"$%0.2f\,$M$_\odot$" % masses[mass_i], ha="center",
                     fontsize="medium", color="grey")
        
    #plt.legend(loc="best")
    
    cb = plt.colorbar()
    cb.set_label(r"[Fe/H]", fontsize="x-large")
    
    plt.xticks(fontsize="x-large")
    plt.yticks(fontsize="x-large")
    cb.ax.tick_params(labelsize="x-large") 

    plt.xlim([0, 1.5])
    plt.ylim([7.5, 0])
    plt.xlabel(r"$(B-V)$", fontsize="x-large")
    plt.ylabel(r"$V_{\rm abs}$", fontsize="x-large")
    plt.tight_layout()
    plt.savefig("paper/hr_diagram.pdf")


def plot_c_hist_old(results, n_bins=5):
    """Plot histograms of the scaling/intercept parameter C.
    """
    faint_cs = results[results["SEQUENCE"]=="faint"]["C_SCALE"].values.tolist()
    faint_cs.sort()
    faint_cs = faint_cs[:-1]
    bright_cs = results[results["SEQUENCE"]=="bright"]["C_SCALE"].values.tolist()
    
    plt.hist(faint_cs, bins=n_bins, label="Faint", alpha=0.60)
    plt.hist(bright_cs, bins=n_bins, label="Bright", alpha=0.60)
    
    plt.text(1.08, 5, r"C$_{\rm med}$ (bright) = %0.2f" % np.median(bright_cs))
    plt.text(1.08, 4.5, r"C$_{\rm med}$ (faint) = %0.2f" % np.median(faint_cs))
    
    plt.xlabel("C")
    plt.ylabel("#")
    plt.legend(loc="best")
    plt.savefig("plots/c_hist.png")
    
def plot_c_hist(results, n_bins=10):
    """
    Plot C_SCALE values, including combined fits.
    """

    bright_cs = []
    faint_cs = []
    other_cs = []

    for row_i in range(len(results)):

        row = results.iloc[row_i]

        c_values = np.asarray(
            row["C_SCALE"],
            dtype=float
        ).ravel()

        try:
            seq_order = row["SEQ_ORDER"]
        except Exception:
            seq_order = []

        for c_i, c_value in enumerate(c_values):

            if not np.isfinite(c_value):
                continue

            sequence_name = "combined"

            try:
                sequence_name = str(
                    seq_order[c_i][1]
                ).lower()
            except Exception:
                try:
                    sequence_name = str(
                        row["SEQUENCE"]
                    ).lower()
                except Exception:
                    pass

            if sequence_name == "bright":
                bright_cs.append(c_value)

            elif sequence_name == "faint":
                faint_cs.append(c_value)

            else:
                other_cs.append(c_value)

    plt.close("all")
    fig, ax = plt.subplots()

    if len(bright_cs) > 0:
        ax.hist(
            bright_cs,
            bins=n_bins,
            alpha=0.60,
            label="Bright"
        )

    if len(faint_cs) > 0:
        ax.hist(
            faint_cs,
            bins=n_bins,
            alpha=0.60,
            label="Faint"
        )

    if len(other_cs) > 0:
        ax.hist(
            other_cs,
            bins=n_bins,
            alpha=0.60,
            label="Other"
        )

    ax.axvline(
        1.0,
        linestyle="--",
        label="C = 1"
    )

    all_cs = np.asarray(
        bright_cs + faint_cs + other_cs,
        dtype=float
    )

    if len(all_cs) > 0:

        median_c = np.nanmedian(all_cs)

        ax.axvline(
            median_c,
            linestyle=":",
            label="Median C = %.3f" % median_c
        )

    ax.set_xlabel("C scale")
    ax.set_ylabel("Number of sequences")
    ax.legend(loc="best")

    plt.tight_layout()

    plt.savefig("plots/c_hist.png", dpi=200)
    plt.savefig("plots/c_hist.pdf")

    plt.close() 
    
def presentation_vis2_plot():
    """Plot spatial frequency coverage of PAVO and POINIER for use as a visial
    aid when giving talks.
    """
    # CHARA
    chara_min_bl = 34
    chara_max_bl = 330
    chara_min_lambda = 630 * 10**-9
    chara_max_lambda = 950 * 10**-9
    chara_lims = np.array([chara_min_bl/chara_max_lambda, 
                                   chara_max_bl/chara_min_lambda])
                                
    # PIONIER
    vlti_min_bl = 11
    vlti_max_bl = 132
    vlti_min_lambda = 1533 * 10**-9
    vlti_max_lambda = 1773 * 10**-9
    vlti_lims = np.array([vlti_min_bl/vlti_max_lambda, 
                                     vlti_max_bl/vlti_min_lambda])
    
    # Diameters to plot
    ldds = [4.0, 2.0, 1.0, 0.5]
    u_lld = 0.3
    c_scale = 1
    xmax = 55*10**7
    nsteps = 25
    
    freqs = np.arange(1*10**6, xmax, 10000)
    chara_freqs = np.arange(chara_lims[0], chara_lims[1], 
                           (chara_lims[1]-chara_lims[0])/nsteps)
    vlti_freqs = np.arange(vlti_lims[0], vlti_lims[1], 
                          (vlti_lims[1]-vlti_lims[0])/nsteps)
    
    plt.close("all")
    
    plt.xlim([0.0, xmax])
    plt.ylim([0.0, 1.0])
    plt.xlabel(r"Spatial Frequency (rad$^{-1})$")
    plt.ylabel(r"Visibility$^2$")
    plt.tight_layout()
    
    n_points = (len(freqs),)
    s_lambda = 1

    # First plot just the curves
    for ldd_i, ldd in enumerate(ldds):
        vis2 = rdiam.calc_vis2(freqs, ldd, c_scale, n_points, u_lld,
                               s_lambda)
        
        plt.plot(freqs, vis2, label=r"$\theta_{\rm LD}$ = %0.1f mas" % ldd)
        plt.legend(loc="best")

    
        plt.savefig("plots/presentation_vis2_vs_ldd_%i.png" % ldd_i)
    
    # Next just the PIONIER points
    plt.text(0.5*xmax, 0.95, "PIONIER: 11-132m, H band", color="darkred", 
             ha='center')
    
    n_points = (len(vlti_freqs),)

    for ldd in ldds:
        # PIONIER
        ldd_rad = ldd / 1000 / 3600 / 180 * np.pi
        vlti_vis2 = rdiam.calc_vis2(vlti_freqs, ldd, c_scale, n_points, 
                                    u_lld, s_lambda)
        plt.plot(vlti_freqs, vlti_vis2, ".", color="darkred") 
    
    plt.savefig("plots/presentation_vis2_vs_ldd_p.png")
    
    # Finally the PAVO points
    plt.text(0.5*xmax, 0.9, "PAVO: 34-330m, R band", color="blue", 
             ha='center')
    
    n_points = (len(chara_freqs),)

    for ldd in ldds:
        # CHARA
        ldd_rad = ldd / 1000 / 3600 / 180 * np.pi
        chara_vis2 = rdiam.calc_vis2(chara_freqs, ldd, c_scale, n_points,
                                     u_lld, s_lambda)
        plt.plot(chara_freqs, chara_vis2, "+", color="blue")
    
    plt.tight_layout()
    plt.savefig("plots/presentation_vis2_vs_ldd.pdf")
    plt.savefig("plots/presentation_vis2_vs_ldd_c.png")


def limb_darkened_fourier_amplitude(
        spatial_frequency,
        theta_ld_mas,
        u_lambda):
    """
    Calculate the normalized signed Fourier amplitude V(q) of a
    linearly limb-darkened circular stellar disc.

    Parameters
    ----------
    spatial_frequency : array
        Spatial frequency B/lambda in rad^-1.

    theta_ld_mas : float
        Limb-darkened angular diameter in milliarcseconds.

    u_lambda : float
        Linear limb-darkening coefficient.

    Returns
    -------
    visibility : array
        Signed normalized visibility amplitude V(q).

    Notes
    -----
    The squared visibility measured by the interferometer is V(q)^2.
    The sign of V(q) changes after each null, but this sign is lost
    in V^2.
    """

    spatial_frequency = np.asarray(
        spatial_frequency,
        dtype=float
    )

    theta_ld_mas = float(
        theta_ld_mas
    )

    u_lambda = float(
        u_lambda
    )

    # Convert milliarcseconds to radians.
    theta_rad = (
        theta_ld_mas
        / 1000.0
        / 3600.0
        / 180.0
        * np.pi
    )

    x = (
        np.pi
        * spatial_frequency
        * theta_rad
    )

    visibility = np.ones(
        x.shape,
        dtype=float
    )

    # Avoid division by zero at q=0.
    nonzero = (
        np.abs(x) > 1.0E-10
    )

    xx = x[
        nonzero
    ]

    normalization = (
        (1.0 - u_lambda) / 2.0
        + u_lambda / 3.0
    )

    visibility[nonzero] = (
        (
            (1.0 - u_lambda)
            * jv(1.0, xx)
            / xx
        )
        +
        (
            u_lambda
            * np.sqrt(np.pi / 2.0)
            * jv(1.5, xx)
            / xx**1.5
        )
    ) / normalization

    # By definition, normalized visibility is one at zero baseline.
    visibility[~nonzero] = 1.0

    return visibility

def plot_science_fourier_transforms(
        tgt_info,
        results,
        output_file="plots/science_fourier_transforms.pdf",
        q_max=2.5E8,
        n_model_points=20000,
        use_predicted_if_missing=True):
    """
    Plot the Fourier transform of every science target.

    For every science target, the function plots:

        upper panel:
            signed Fourier amplitude V(q)

        lower panel:
            squared visibility V(q)^2, together with the observed
            VIS2 points corrected by their fitted C_SCALE values

    The six PIONIER wavelength channels are shown separately.

    Parameters
    ----------
    tgt_info : pandas.DataFrame
        Target information table containing Science, LDD_pred,
        e_LDD_pred, u_lambda_i and s_lambda_i.

    results : pandas.DataFrame
        Final diameter-fitting results.

    output_file : str
        Multipage output PDF.

    q_max : float
        Maximum spatial frequency in rad^-1.

    n_model_points : int
        Number of points in the theoretical Fourier-transform curves.

    use_predicted_if_missing : bool
        Use LDD_pred when LDD_FIT is invalid.

    Returns
    -------
    output_file : str
        Path to the generated multipage PDF.
    """

    plt.close("all")

    # ============================================================
    # Output directories
    # ============================================================

    output_directory = os.path.dirname(
        output_file
    )

    if output_directory == "":
        output_directory = "."

    if not os.path.exists(
            output_directory):

        os.makedirs(
            output_directory
        )

    individual_directory = os.path.join(
        output_directory,
        "science_fourier"
    )

    if not os.path.exists(
            individual_directory):

        os.makedirs(
            individual_directory
        )

    # ============================================================
    # Model spatial-frequency grid
    # ============================================================

    q_model = np.linspace(
        0.0,
        q_max,
        int(n_model_points)
    )

    plotted_targets = set()

    n_created = 0
    n_skipped = 0

    print("")
    print("=" * 79)
    print("Creating Fourier transforms for science stars")
    print("Output:")
    print(output_file)
    print("=" * 79)

    # ============================================================
    # Multipage PDF
    # ============================================================

    with PdfPages(output_file) as pdf:

        for result_i in xrange(
                len(results)):

            row = results.iloc[
                result_i
            ]

            sci = str(
                row["STAR"]
            )

            # Avoid plotting the same science target more than once.
            sci_clean = clean_target_name_for_plot(
                sci
            )

            if sci_clean in plotted_targets:
                continue

            # ----------------------------------------------------
            # Match against tgt_info
            # ----------------------------------------------------

            hd_id = match_target_for_plot(
                tgt_info,
                sci,
                verbose=True
            )

            if hd_id is None:

                print(
                    "WARNING: could not match %s; skipping"
                    % sci
                )

                n_skipped += 1
                continue

            # ----------------------------------------------------
            # Keep science targets only
            # ----------------------------------------------------

            if "Science" in tgt_info.columns:

                try:

                    is_science = bool(
                        tgt_info.loc[
                            hd_id,
                            "Science"
                        ]
                    )

                except Exception:

                    is_science = False

                if not is_science:

                    print(
                        "Skipping calibrator/non-science target: %s"
                        % sci
                    )

                    continue

            plotted_targets.add(
                sci_clean
            )

            # ----------------------------------------------------
            # Select angular diameter
            # ----------------------------------------------------

            try:
                ldd_fit = float(
                    row["LDD_FIT"]
                )
            except Exception:
                ldd_fit = np.nan

            try:
                e_ldd_fit = float(
                    row["e_LDD_FIT"]
                )
            except Exception:
                e_ldd_fit = np.nan

            valid_fit = (
                np.isfinite(ldd_fit)
                and ldd_fit > 0
            )

            if valid_fit:

                theta_base = ldd_fit
                theta_error = e_ldd_fit
                theta_source = "fitted"

            elif use_predicted_if_missing:

                try:

                    theta_base = float(
                        tgt_info.loc[
                            hd_id,
                            "LDD_pred"
                        ]
                    )

                    theta_error = float(
                        tgt_info.loc[
                            hd_id,
                            "e_LDD_pred"
                        ]
                    )

                except Exception:

                    theta_base = np.nan
                    theta_error = np.nan

                if (
                    not np.isfinite(theta_base)
                    or theta_base <= 0
                ):

                    print(
                        "WARNING: no valid diameter for %s; skipping"
                        % sci
                    )

                    n_skipped += 1
                    continue

                theta_source = "predicted"

            else:

                print(
                    "WARNING: no valid fitted diameter for %s; skipping"
                    % sci
                )

                n_skipped += 1
                continue

            # ====================================================
            # Visibility data
            # ====================================================

            baselines = np.asarray(
                row["BASELINE"],
                dtype=float
            ).ravel()

            wavelengths = np.asarray(
                row["WAVELENGTH"],
                dtype=float
            ).ravel()

            vis2_matrix = np.asarray(
                row["VIS2"],
                dtype=float
            )

            e_vis2_matrix = np.asarray(
                row["e_VIS2"],
                dtype=float
            )

            n_bl = len(
                baselines
            )

            n_wl = len(
                wavelengths
            )

            expected_size = (
                n_bl * n_wl
            )

            if vis2_matrix.ndim == 1:

                if vis2_matrix.size != expected_size:

                    print(
                        "WARNING: invalid VIS2 dimensions for %s"
                        % sci
                    )

                    n_skipped += 1
                    continue

                vis2_matrix = vis2_matrix.reshape(
                    n_bl,
                    n_wl
                )

            if e_vis2_matrix.ndim == 1:

                if e_vis2_matrix.size != expected_size:

                    print(
                        "WARNING: invalid e_VIS2 dimensions for %s"
                        % sci
                    )

                    n_skipped += 1
                    continue

                e_vis2_matrix = e_vis2_matrix.reshape(
                    n_bl,
                    n_wl
                )

            if (
                vis2_matrix.shape
                != e_vis2_matrix.shape
            ):

                print(
                    "WARNING: VIS2/e_VIS2 shape mismatch for %s"
                    % sci
                )

                n_skipped += 1
                continue

            # ====================================================
            # Apply fitted C values to the observations
            # ====================================================

            try:

                c_values = np.asarray(
                    row["C_SCALE"],
                    dtype=float
                ).ravel()

            except Exception:

                c_values = np.array(
                    [1.0],
                    dtype=float
                )

            if len(c_values) == 0:

                c_values = np.array(
                    [1.0],
                    dtype=float
                )

            bad_c = (
                ~np.isfinite(c_values)
                | (c_values <= 0)
            )

            c_values[bad_c] = 1.0

            if n_bl % len(c_values) == 0:

                n_bl_per_c = int(
                    n_bl / len(c_values)
                )

                c_per_baseline = np.repeat(
                    c_values,
                    n_bl_per_c
                )

                vis2_corrected = (
                    vis2_matrix
                    / c_per_baseline[:, np.newaxis]
                )

                e_vis2_corrected = (
                    e_vis2_matrix
                    / c_per_baseline[:, np.newaxis]
                )

            else:

                print(
                    "WARNING: cannot map C values to %s; "
                    "observed points will not be shown"
                    % sci
                )

                vis2_corrected = None
                e_vis2_corrected = None

            # ====================================================
            # Limb-darkening and wavelength scaling
            # ====================================================

            u_lambdas = []
            s_lambdas = []

            for wl_i in xrange(
                    n_wl):

                u_column = (
                    "u_lambda_%i"
                    % wl_i
                )

                s_column = (
                    "s_lambda_%i"
                    % wl_i
                )

                if u_column in tgt_info.columns:

                    try:

                        u_value = float(
                            tgt_info.loc[
                                hd_id,
                                u_column
                            ]
                        )

                    except Exception:

                        u_value = np.nan

                else:

                    u_value = np.nan

                if (
                    not np.isfinite(u_value)
                    or u_value < 0
                    or u_value > 1
                ):

                    u_value = 0.3

                if s_column in tgt_info.columns:

                    try:

                        s_value = float(
                            tgt_info.loc[
                                hd_id,
                                s_column
                            ]
                        )

                    except Exception:

                        s_value = np.nan

                else:

                    s_value = np.nan

                if (
                    not np.isfinite(s_value)
                    or s_value <= 0
                ):

                    s_value = 1.0

                u_lambdas.append(
                    u_value
                )

                s_lambdas.append(
                    s_value
                )

            u_lambdas = np.asarray(
                u_lambdas,
                dtype=float
            )

            s_lambdas = np.asarray(
                s_lambdas,
                dtype=float
            )

            # ====================================================
            # Create figure
            # ====================================================

            fig, axes = plt.subplots(
                2,
                1,
                sharex=True
            )

            fig.set_size_inches(
                10,
                9
            )

            colour_map = cm.get_cmap(
                "viridis"
            )

            if n_wl == 1:

                colours = [
                    colour_map(0.5)
                ]

            else:

                colours = [
                    colour_map(
                        float(wl_i)
                        / float(n_wl - 1)
                    )
                    for wl_i in xrange(n_wl)
                ]

            all_observed_vis2 = []

            # ====================================================
            # Transform for each wavelength channel
            # ====================================================

            for wl_i in xrange(
                    n_wl):

                wavelength_m = wavelengths[
                    wl_i
                ]

                wavelength_um = (
                    wavelength_m * 1.0E6
                )

                u_lambda = u_lambdas[
                    wl_i
                ]

                s_lambda = s_lambdas[
                    wl_i
                ]

                # Apply the wavelength scaling exactly once.
                theta_lambda = (
                    theta_base
                    * s_lambda
                )

                fourier_amplitude = (
                    limb_darkened_fourier_amplitude(
                        q_model,
                        theta_lambda,
                        u_lambda
                    )
                )

                model_vis2 = (
                    fourier_amplitude**2
                )

                wavelength_label = (
                    r"$%.3f\,\mu{\rm m}$"
                    % wavelength_um
                )

                # Signed Fourier amplitude.
                axes[0].plot(
                    q_model,
                    fourier_amplitude,
                    label=wavelength_label,
                    color=colours[wl_i]
                )

                # Squared Fourier amplitude.
                axes[1].plot(
                    q_model,
                    model_vis2,
                    color=colours[wl_i]
                )

                # ------------------------------------------------
                # Corrected observed points
                # ------------------------------------------------

                if vis2_corrected is not None:

                    observed_sfreq = (
                        baselines
                        / wavelength_m
                    )

                    observed_vis2 = (
                        vis2_corrected[
                            :,
                            wl_i
                        ]
                    )

                    observed_e_vis2 = (
                        e_vis2_corrected[
                            :,
                            wl_i
                        ]
                    )

                    valid = (
                        np.isfinite(observed_sfreq)
                        & np.isfinite(observed_vis2)
                        & np.isfinite(observed_e_vis2)
                        & (observed_e_vis2 > 0)
                    )

                    if np.any(valid):

                        axes[1].errorbar(
                            observed_sfreq[valid],
                            observed_vis2[valid],
                            yerr=observed_e_vis2[valid],
                            fmt=".",
                            color=colours[wl_i],
                            elinewidth=0.3,
                            capsize=0.5,
                            capthick=0.3,
                            markersize=3
                        )

                        all_observed_vis2.extend(
                            observed_vis2[
                                valid
                            ].tolist()
                        )

            # ====================================================
            # Figure formatting
            # ====================================================

            axes[0].axhline(
                0.0,
                linestyle=":",
                linewidth=0.7
            )

            axes[0].set_ylabel(
                r"Fourier amplitude $V(q)$"
            )

            axes[0].set_ylim(
                [-0.40, 1.05]
            )

            axes[0].grid()

            axes[0].legend(
                loc="best",
                fontsize="small",
                ncol=2
            )

            axes[1].set_xlabel(
                r"Spatial frequency $q=B/\lambda$ (rad$^{-1}$)"
            )

            axes[1].set_ylabel(
                r"Squared visibility $V^2(q)$"
            )

            axes[1].set_xlim(
                [0.0, q_max]
            )

            if len(all_observed_vis2) > 0:

                finite_observed = np.asarray(
                    all_observed_vis2,
                    dtype=float
                )

                finite_observed = finite_observed[
                    np.isfinite(finite_observed)
                ]

                if len(finite_observed) > 0:

                    vis2_max = max(
                        1.1,
                        np.nanpercentile(
                            finite_observed,
                            99
                        ) + 0.05
                    )

                    vis2_max = min(
                        vis2_max,
                        1.5
                    )

                else:

                    vis2_max = 1.1

            else:

                vis2_max = 1.1

            axes[1].set_ylim(
                [0.0, vis2_max]
            )

            axes[1].grid()

            if np.isfinite(theta_error):

                title = (
                    "%s: %s "
                    r"$\theta_{\rm LDD}=%.4f\pm%.4f$ mas"
                    % (
                        sci,
                        theta_source,
                        theta_base,
                        theta_error
                    )
                )

            else:

                title = (
                    "%s: %s "
                    r"$\theta_{\rm LDD}=%.4f$ mas"
                    % (
                        sci,
                        theta_source,
                        theta_base
                    )
                )

            fig.suptitle(
                title
            )

            fig.tight_layout(
                rect=[0.0, 0.0, 1.0, 0.96]
            )

            # ====================================================
            # Save page and individual image
            # ====================================================

            pdf.savefig(
                fig
            )

            safe_name = (
                clean_target_name_for_plot(
                    sci
                )
            )

            individual_file = os.path.join(
                individual_directory,
                "%s_fourier.png"
                % safe_name
            )

            fig.savefig(
                individual_file,
                dpi=200
            )

            plt.close(
                fig
            )

            n_created += 1

            print(
                "Saved Fourier transform for %s"
                % sci
            )

    print("")
    print("=" * 79)
    print("Science-star Fourier transforms finished")
    print("Created: %i" % n_created)
    print("Skipped: %i" % n_skipped)
    print("Saved:")
    print(output_file)
    print("=" * 79)

    if n_created == 0:

        raise RuntimeError(
            "No science-star Fourier transforms were generated"
        )

    return output_file

def limb_darkened_intensity_map(
        
        theta_ld_mas,
        u_lambda,
        n_pixels=400,
        field_factor=1.4):
    """
    Create a two-dimensional linearly limb-darkened stellar disc.

    Parameters
    ----------
    theta_ld_mas : float
        Limb-darkened angular diameter in mas.

    u_lambda : float
        Linear limb-darkening coefficient.

    n_pixels : int
        Number of pixels along each image dimension.

    field_factor : float
        Image half-width in units of the stellar radius.

    Returns
    -------
    delta_ra : ndarray
        Delta RA grid in mas.

    delta_dec : ndarray
        Delta Dec grid in mas.

    intensity : ndarray
        Intensity normalized to the central intensity.
        Pixels outside the stellar disc are NaN.
    """

    theta_ld_mas = float(
        theta_ld_mas
    )

    u_lambda = float(
        u_lambda
    )

    radius_mas = (
        theta_ld_mas / 2.0
    )

    field_radius_mas = (
        field_factor * radius_mas
    )

    coordinate = np.linspace(
        -field_radius_mas,
        field_radius_mas,
        int(n_pixels)
    )

    delta_ra, delta_dec = np.meshgrid(
        coordinate,
        coordinate
    )

    radial_distance = np.sqrt(
        delta_ra**2
        + delta_dec**2
    )

    normalized_radius = (
        radial_distance / radius_mas
    )

    inside_disc = (
        normalized_radius <= 1.0
    )

    intensity = np.empty(
        normalized_radius.shape,
        dtype=float
    )

    intensity[:] = np.nan

    # mu = cos(theta) on the projected stellar surface.
    mu = np.sqrt(
        np.clip(
            1.0
            - normalized_radius[inside_disc]**2,
            0.0,
            1.0
        )
    )

    # Linear limb-darkening law:
    #
    # I(mu) / I(1) = 1 - u * (1 - mu)
    intensity[inside_disc] = (
        1.0
        - u_lambda
        * (1.0 - mu)
    )

    return (
        delta_ra,
        delta_dec,
        intensity
    )

def plot_science_intensity_maps(
        tgt_info,
        results,
        output_file="plots/science_intensity_maps.pdf",
        wavelength_index=2,
        n_pixels=400,
        field_factor=1.4,
        use_predicted_if_missing=True):
    """
    Create RA-Dec intensity maps for all science targets.

    Each page contains a linearly limb-darkened model image based on
    the fitted LDD. When the fitted LDD is unavailable, LDD_pred can
    optionally be used.

    Parameters
    ----------
    tgt_info : pandas.DataFrame
        Target information table.

    results : pandas.DataFrame
        Final interferometric fitting results.

    output_file : str
        Multipage PDF output path.

    wavelength_index : int
        PIONIER wavelength-channel index.
        Valid values are normally 0 through 5.

    n_pixels : int
        Image resolution along each dimension.

    field_factor : float
        Image half-width in stellar-radius units.

    use_predicted_if_missing : bool
        Use LDD_pred when LDD_FIT is invalid.

    Returns
    -------
    output_file : str
        Path to the generated PDF.
    """

    plt.close("all")

    output_directory = os.path.dirname(
        output_file
    )

    if output_directory == "":
        output_directory = "."

    if not os.path.exists(
            output_directory):

        os.makedirs(
            output_directory
        )

    individual_directory = os.path.join(
        output_directory,
        "science_intensity"
    )

    if not os.path.exists(
            individual_directory):

        os.makedirs(
            individual_directory
        )

    plotted_targets = set()

    n_created = 0
    n_skipped = 0

    print("")
    print("=" * 79)
    print("Creating science-star RA-Dec intensity maps")
    print("Output:")
    print(output_file)
    print("=" * 79)

    with PdfPages(output_file) as pdf:

        for result_i in xrange(
                len(results)):

            row = results.iloc[
                result_i
            ]

            sci = str(
                row["STAR"]
            )

            clean_name = clean_target_name_for_plot(
                sci
            )

            # Avoid plotting the same target more than once.
            if clean_name in plotted_targets:
                continue

            target_id = match_target_for_plot(
                tgt_info,
                sci,
                verbose=True
            )

            if target_id is None:

                print(
                    "WARNING: target match not found for %s"
                    % sci
                )

                n_skipped += 1
                continue

            # Keep science stars only.
            if "Science" in tgt_info.columns:

                try:

                    is_science = bool(
                        tgt_info.loc[
                            target_id,
                            "Science"
                        ]
                    )

                except Exception:

                    is_science = False

                if not is_science:
                    continue

            plotted_targets.add(
                clean_name
            )

            # ====================================================
            # Select fitted or predicted angular diameter
            # ====================================================

            try:

                theta_base = float(
                    row["LDD_FIT"]
                )

            except Exception:

                theta_base = np.nan

            valid_fitted_theta = (
                np.isfinite(theta_base)
                and theta_base > 0
            )

            if valid_fitted_theta:

                theta_source = "fitted"

                try:

                    theta_error = float(
                        row["e_LDD_FIT"]
                    )

                except Exception:

                    theta_error = np.nan

            elif use_predicted_if_missing:

                try:

                    theta_base = float(
                        tgt_info.loc[
                            target_id,
                            "LDD_pred"
                        ]
                    )

                except Exception:

                    theta_base = np.nan

                try:

                    theta_error = float(
                        tgt_info.loc[
                            target_id,
                            "e_LDD_pred"
                        ]
                    )

                except Exception:

                    theta_error = np.nan

                theta_source = "predicted"

            else:

                theta_base = np.nan
                theta_error = np.nan
                theta_source = "unavailable"

            if (
                not np.isfinite(theta_base)
                or theta_base <= 0
            ):

                print(
                    "WARNING: no valid diameter for %s"
                    % sci
                )

                n_skipped += 1
                continue

            # ====================================================
            # Select wavelength channel
            # ====================================================

            wavelengths = np.asarray(
                row["WAVELENGTH"],
                dtype=float
            ).ravel()

            if len(wavelengths) == 0:

                print(
                    "WARNING: no wavelengths for %s"
                    % sci
                )

                n_skipped += 1
                continue

            channel_index = int(
                wavelength_index
            )

            if channel_index < 0:
                channel_index = 0

            if channel_index >= len(wavelengths):
                channel_index = len(wavelengths) - 1

            wavelength_m = wavelengths[
                channel_index
            ]

            wavelength_um = (
                wavelength_m * 1.0E6
            )

            # ====================================================
            # Limb-darkening coefficient
            # ====================================================

            u_column = (
                "u_lambda_%i"
                % channel_index
            )

            try:

                u_lambda = float(
                    tgt_info.loc[
                        target_id,
                        u_column
                    ]
                )

            except Exception:

                u_lambda = np.nan

            if (
                not np.isfinite(u_lambda)
                or u_lambda < 0
                or u_lambda > 1
            ):

                print(
                    "WARNING: invalid %s for %s; using u=0.3"
                    % (
                        u_column,
                        sci
                    )
                )

                u_lambda = 0.3

            # ====================================================
            # Wavelength-dependent diameter scaling
            # ====================================================

            s_column = (
                "s_lambda_%i"
                % channel_index
            )

            try:

                s_lambda = float(
                    tgt_info.loc[
                        target_id,
                        s_column
                    ]
                )

            except Exception:

                s_lambda = np.nan

            if (
                not np.isfinite(s_lambda)
                or s_lambda <= 0
            ):

                s_lambda = 1.0

            theta_lambda = (
                theta_base
                * s_lambda
            )

            # ====================================================
            # Build intensity map
            # ====================================================

            delta_ra, delta_dec, intensity = (
                limb_darkened_intensity_map(
                    theta_lambda,
                    u_lambda,
                    n_pixels=n_pixels,
                    field_factor=field_factor
                )
            )

            image_limit = np.nanmax(
                np.abs(delta_ra)
            )

            # ====================================================
            # Plot
            # ====================================================

            fig, ax = plt.subplots()

            fig.set_size_inches(
                8,
                7
            )

            image = ax.imshow(
                intensity,
                origin="lower",
                extent=[
                    -image_limit,
                    image_limit,
                    -image_limit,
                    image_limit
                ],
                interpolation="bilinear",
                vmin=0.0,
                vmax=1.0,
                cmap="inferno"
            )

            colorbar = fig.colorbar(
                image,
                ax=ax
            )

            colorbar.set_label(
                r"Normalized intensity $I/I_{\rm center}$"
            )

            ax.set_xlabel(
                r"$\Delta$RA (mas)"
            )

            ax.set_ylabel(
                r"$\Delta$Dec (mas)"
            )

            ax.set_aspect(
                "equal"
            )

            # Astronomical convention:
            # RA increases toward the left.
            ax.invert_xaxis()

            if np.isfinite(theta_error):

                title = (
                    "%s\n"
                    r"%s $\theta_{\rm LDD}=%.4f\pm%.4f$ mas, "
                    r"$\lambda=%.3f\,\mu$m, $u=%.3f$"
                    % (
                        sci,
                        theta_source,
                        theta_lambda,
                        theta_error,
                        wavelength_um,
                        u_lambda
                    )
                )

            else:

                title = (
                    "%s\n"
                    r"%s $\theta_{\rm LDD}=%.4f$ mas, "
                    r"$\lambda=%.3f\,\mu$m, $u=%.3f$"
                    % (
                        sci,
                        theta_source,
                        theta_lambda,
                        wavelength_um,
                        u_lambda
                    )
                )

            ax.set_title(
                title
            )

            fig.tight_layout()

            # Save page in multipage PDF.
            pdf.savefig(
                fig
            )

            # Save one PNG per science target.
            individual_file = os.path.join(
                individual_directory,
                "%s_intensity.png"
                % clean_name
            )

            fig.savefig(
                individual_file,
                dpi=200
            )

            plt.close(
                fig
            )

            n_created += 1

            print(
                "Saved intensity image for %s"
                % sci
            )


    print("")
    print("=" * 79)
    print("Science intensity maps finished")
    print("Created: %i" % n_created)
    print("Skipped: %i" % n_skipped)
    print("Saved:")
    print(output_file)
    print("=" * 79)

    if n_created == 0:

        raise RuntimeError(
            "No science intensity maps were generated"
        )

    return output_file


def plot_visibility_diagnostic_summary_old(
        results,
        bs_results,
        tgt_info,
        output_file=(
            "plots/visibility_diagnostics/"
            "visibility_diagnostic_summary.pdf"
        ),
        bootstrap_index=0,
        sigma_threshold=3.0,
        raw_residual_threshold=0.05,
        e_wl_frac=0.0035,
        star_filter=None,
        highlight_night=None,
        highlight_pair=None,
        highlight_baseline_range=None,
        highlight_wavelength_index=None,
        max_annotations=10,
        use_predicted_if_missing=True):
    """
    Create point-by-point visibility diagnostics.

    The four panels are:

        1. Corrected VIS2 coloured by observing night.
        2. Corrected VIS2 coloured by telescope pair.
        3. Raw residual:
               VIS2_observed - VIS2_model
           coloured by wavelength channel.
        4. Standardized residual:
               (VIS2_observed - VIS2_model) / e_VIS2
           versus projected baseline.

    A point is marked as a removal candidate when:

        - it has an OIFITS flag; or
        - abs(standardized residual) >= sigma_threshold
          AND
          abs(raw residual) >= raw_residual_threshold.

    No data are removed by this function.
    """

    # ====================================================================
    # Compatibility
    # ====================================================================

    try:
        string_types = (basestring,)
    except NameError:
        string_types = (str,)

    # ====================================================================
    # Helper functions
    # ====================================================================

    def safe_float(value):

        try:
            return float(value)
        except Exception:
            return np.nan


    def pair_to_string(value):
        """
        Convert a telescope-pair representation to a readable string.
        """

        try:

            if isinstance(
                    value,
                    (tuple, list, np.ndarray)):

                values = list(
                    value
                )

                if len(values) >= 2:

                    return "%s-%s" % (
                        str(values[0]),
                        str(values[1])
                    )

        except Exception:
            pass

        return str(
            value
        )


    def resize_numeric_metadata(
            values,
            required_length):
        """
        Resize numerical metadata to the number of baseline rows.
        """

        output = np.empty(
            required_length,
            dtype=float
        )

        output[:] = np.nan

        try:

            values = np.asarray(
                values,
                dtype=float
            ).ravel()

        except Exception:

            values = np.array(
                [],
                dtype=float
            )

        n_copy = min(
            len(values),
            required_length
        )

        if n_copy > 0:

            output[:n_copy] = values[
                :n_copy
            ]

        return output


    def resize_pair_metadata(
            values,
            required_length):
        """
        Preserve telescope pairs without flattening tuple pairs.
        """

        output = np.empty(
            required_length,
            dtype=object
        )

        output[:] = "unknown"

        try:

            values = list(
                values
            )

        except Exception:

            values = []

        n_copy = min(
            len(values),
            required_length
        )

        for value_i in xrange(
                n_copy):

            output[value_i] = pair_to_string(
                values[value_i]
            )

        return output


    def mjd_to_night(mjd_value):
        """
        Convert MJD to an observing-night string.

        Subtracting half a day keeps observations made after midnight
        associated with the evening on which the night began.
        """

        mjd_value = safe_float(
            mjd_value
        )

        if not np.isfinite(
                mjd_value):

            return "unknown"

        try:

            mjd_epoch = datetime(
                1858,
                11,
                17
            )

            date_value = (
                mjd_epoch
                + timedelta(
                    days=mjd_value - 0.5
                )
            )

            return date_value.strftime(
                "%Y-%m-%d"
            )

        except Exception:

            return "unknown"


    def resolve_bs_key(
            science_name,
            sequence_name,
            period_value):
        """
        Match one results row to the corresponding bs_results key.
        """

        science_clean = clean_target_name_for_plot(
            science_name
        )

        sequence_clean = str(
            sequence_name
        ).strip().lower()

        period_clean = str(
            period_value
        ).strip()

        # Direct lookup for combined fits.
        if science_name in bs_results:

            return science_name

        # Direct tuple possibilities.
        direct_keys = [
            (
                science_name,
                sequence_name,
                period_value
            ),
            (
                science_name,
                str(sequence_name),
                period_value
            ),
        ]

        for direct_key in direct_keys:

            if direct_key in bs_results:

                return direct_key

        # Robust matching.
        for candidate_key in bs_results.keys():

            if isinstance(
                    candidate_key,
                    tuple):

                candidate_star = candidate_key[0]

                candidate_sequence = ""

                candidate_period = ""

                if len(candidate_key) > 1:

                    candidate_sequence = str(
                        candidate_key[1]
                    ).strip().lower()

                if len(candidate_key) > 2:

                    candidate_period = str(
                        candidate_key[2]
                    ).strip()

                same_star = (
                    clean_target_name_for_plot(
                        candidate_star
                    )
                    == science_clean
                )

                same_sequence = (
                    candidate_sequence
                    == sequence_clean
                )

                same_period = (
                    candidate_period
                    == period_clean
                )

                if (
                    same_star
                    and same_sequence
                    and same_period
                ):

                    return candidate_key

            else:

                same_star = (
                    clean_target_name_for_plot(
                        candidate_key
                    )
                    == science_clean
                )

                if same_star:

                    return candidate_key

        return None


    def get_wavelength_coefficients(
            result_row,
            target_id,
            coefficient_name,
            n_wavelengths,
            default_value):
        """
        Obtain U_LAMBDA or S_LAMBDA from results, falling back to
        wavelength-specific columns in tgt_info.
        """

        if coefficient_name == "u":

            result_column = "U_LAMBDA"
            target_prefix = "u_lambda_"

        else:

            result_column = "S_LAMBDA"
            target_prefix = "s_lambda_"

        coefficient_values = np.empty(
            n_wavelengths,
            dtype=float
        )

        coefficient_values[:] = np.nan

        # First use the final values stored in results.
        if result_column in result_row.index:

            try:

                result_values = np.asarray(
                    result_row[
                        result_column
                    ],
                    dtype=float
                ).ravel()

                n_copy = min(
                    len(result_values),
                    n_wavelengths
                )

                coefficient_values[
                    :n_copy
                ] = result_values[
                    :n_copy
                ]

            except Exception:
                pass

        # Fill missing values from tgt_info.
        for wavelength_i in xrange(
                n_wavelengths):

            if np.isfinite(
                    coefficient_values[
                        wavelength_i
                    ]):

                continue

            column_name = (
                target_prefix
                + str(wavelength_i)
            )

            if column_name in tgt_info.columns:

                try:

                    coefficient_values[
                        wavelength_i
                    ] = float(
                        tgt_info.loc[
                            target_id,
                            column_name
                        ]
                    )

                except Exception:
                    pass

        invalid_values = (
            ~np.isfinite(
                coefficient_values
            )
        )

        coefficient_values[
            invalid_values
        ] = float(
            default_value
        )

        return coefficient_values


    def plot_grouped_visibilities(
            axis,
            group_values,
            group_label,
            spatial_frequency,
            vis2_values,
            e_vis2_values,
            valid_mask,
            colour_map):
        """
        Plot visibility measurements grouped by night or telescope pair.
        """

        unique_groups = sorted(
            set(
                group_values.tolist()
            )
        )

        group_colours = {}

        for group_i, group_value in enumerate(
                unique_groups):

            denominator = max(
                1.0,
                float(
                    len(unique_groups) - 1
                )
            )

            group_colour = colour_map(
                float(group_i)
                / denominator
            )

            group_colours[
                group_value
            ] = group_colour

            group_mask = (
                valid_mask
                & (
                    group_values
                    == group_value
                )
            )

            if not np.any(
                    group_mask):

                continue

            finite_error = (
                group_mask
                & np.isfinite(
                    e_vis2_values
                )
                & (
                    e_vis2_values > 0
                )
            )

            no_error = (
                group_mask
                & ~finite_error
            )

            if np.any(
                    finite_error):

                axis.errorbar(
                    spatial_frequency[
                        finite_error
                    ],
                    vis2_values[
                        finite_error
                    ],
                    xerr=(
                        spatial_frequency[
                            finite_error
                        ]
                        * e_wl_frac
                    ),
                    yerr=e_vis2_values[
                        finite_error
                    ],
                    fmt=".",
                    color=group_colour,
                    label=str(
                        group_value
                    ),
                    elinewidth=0.3,
                    capsize=0.5,
                    capthick=0.3,
                    markersize=4,
                    zorder=3
                )

            if np.any(
                    no_error):

                axis.scatter(
                    spatial_frequency[
                        no_error
                    ],
                    vis2_values[
                        no_error
                    ],
                    s=12,
                    color=group_colour,
                    label=(
                        str(group_value)
                        if not np.any(finite_error)
                        else None
                    ),
                    zorder=3
                )

        axis.set_title(
            group_label
        )

        axis.set_xlabel(
            r"Spatial frequency (rad$^{-1}$)"
        )

        axis.set_ylabel(
            r"Corrected visibility$^2$"
        )

        axis.set_xlim(
            [0.0, 2.5E8]
        )

        axis.set_ylim(
            [0.0, 1.5]
        )

        axis.grid()

        handles, labels = (
            axis.get_legend_handles_labels()
        )

        if len(handles) > 0:

            axis.legend(
                loc="best",
                fontsize=7,
                ncol=2
            )

        return group_colours


    # ====================================================================
    # Validate inputs
    # ====================================================================

    if results is None or len(
            results) == 0:

        raise ValueError(
            "results is empty"
        )

    if bs_results is None or len(
            bs_results) == 0:

        raise ValueError(
            "bs_results is empty"
        )

    if tgt_info is None:

        raise ValueError(
            "tgt_info must be provided"
        )

    # ====================================================================
    # Output directories
    # ====================================================================

    output_directory = os.path.dirname(
        output_file
    )

    if output_directory == "":

        output_directory = "."

    if not os.path.exists(
            output_directory):

        os.makedirs(
            output_directory
        )

    individual_directory = os.path.join(
        output_directory,
        "individual"
    )

    if not os.path.exists(
            individual_directory):

        os.makedirs(
            individual_directory
        )

    all_points_csv = os.path.join(
        output_directory,
        "visibility_point_diagnostics.csv"
    )

    candidate_csv = os.path.join(
        output_directory,
        "visibility_removal_candidates.csv"
    )

    # ====================================================================
    # Star filtering
    # ====================================================================

    if star_filter is None:

        selected_star_names = None

    else:

        if isinstance(
                star_filter,
                string_types):

            star_filter = [
                star_filter
            ]

        selected_star_names = set([
            clean_target_name_for_plot(
                target_name
            )
            for target_name in star_filter
        ])

    diagnostic_rows = []

    n_created = 0
    n_failed = 0

    plt.close(
        "all"
    )

    print("")
    print("=" * 79)
    print("Creating visibility diagnostics")
    print("Output:")
    print(output_file)
    print("=" * 79)

    # ====================================================================
    # Multipage PDF
    # ====================================================================

    with PdfPages(
            output_file) as pdf:

        for result_i in xrange(
                len(results)):

            result_row = results.iloc[
                result_i
            ]

            science_name = str(
                result_row[
                    "STAR"
                ]
            )

            sequence_name = str(
                result_row.get(
                    "SEQUENCE",
                    "combined"
                )
            )

            period_value = result_row.get(
                "PERIOD",
                ""
            )

            science_clean = (
                clean_target_name_for_plot(
                    science_name
                )
            )

            if (
                selected_star_names is not None
                and science_clean
                not in selected_star_names
            ):

                continue

            print("")
            print(
                "Diagnostic for %s, %s, %s"
                % (
                    science_name,
                    sequence_name,
                    str(period_value)
                )
            )

            try:

                # ============================================================
                # Match target information
                # ============================================================

                target_id = result_row.get(
                    "HD",
                    None
                )

                if target_id not in tgt_info.index:

                    target_id = match_target_for_plot(
                        tgt_info,
                        science_name,
                        verbose=True
                    )

                if target_id is None:

                    raise ValueError(
                        "Could not match target in tgt_info"
                    )

                # ============================================================
                # Match bootstrap metadata
                # ============================================================

                bootstrap_key = resolve_bs_key(
                    science_name,
                    sequence_name,
                    period_value
                )

                if bootstrap_key is None:

                    raise ValueError(
                        "Could not find matching bs_results key"
                    )

                bootstrap_table = bs_results[
                    bootstrap_key
                ]

                if len(bootstrap_table) == 0:

                    raise ValueError(
                        "Bootstrap table is empty"
                    )

                metadata_index = int(
                    bootstrap_index
                )

                if metadata_index < 0:

                    metadata_index = (
                        len(bootstrap_table)
                        + metadata_index
                    )

                metadata_index = max(
                    0,
                    min(
                        metadata_index,
                        len(bootstrap_table) - 1
                    )
                )

                metadata_row = bootstrap_table.iloc[
                    metadata_index
                ]

                # ============================================================
                # Final VIS2 arrays
                # ============================================================

                baselines = np.asarray(
                    result_row[
                        "BASELINE"
                    ],
                    dtype=float
                ).ravel()

                wavelengths = np.asarray(
                    result_row[
                        "WAVELENGTH"
                    ],
                    dtype=float
                ).ravel()

                vis2_matrix = np.asarray(
                    result_row[
                        "VIS2"
                    ],
                    dtype=float
                )

                e_vis2_matrix = np.asarray(
                    result_row[
                        "e_VIS2"
                    ],
                    dtype=float
                )

                n_baselines = len(
                    baselines
                )

                n_wavelengths = len(
                    wavelengths
                )

                if n_baselines == 0:

                    raise ValueError(
                        "No baseline values"
                    )

                if n_wavelengths == 0:

                    raise ValueError(
                        "No wavelength values"
                    )

                expected_size = (
                    n_baselines
                    * n_wavelengths
                )

                if vis2_matrix.size != expected_size:

                    raise ValueError(
                        "VIS2 size=%i, expected=%i"
                        % (
                            vis2_matrix.size,
                            expected_size
                        )
                    )

                if e_vis2_matrix.size != expected_size:

                    raise ValueError(
                        "e_VIS2 size=%i, expected=%i"
                        % (
                            e_vis2_matrix.size,
                            expected_size
                        )
                    )

                vis2_matrix = vis2_matrix.reshape(
                    n_baselines,
                    n_wavelengths
                )

                e_vis2_matrix = e_vis2_matrix.reshape(
                    n_baselines,
                    n_wavelengths
                )

                # ============================================================
                # Recover MJD, observing night and telescope pair
                # ============================================================

                mjds = resize_numeric_metadata(
                    metadata_row[
                        "MJD"
                    ],
                    n_baselines
                )

                telescope_pairs = resize_pair_metadata(
                    metadata_row[
                        "TEL_PAIR"
                    ],
                    n_baselines
                )

                night_labels = np.asarray([
                    mjd_to_night(
                        mjd_value
                    )
                    for mjd_value in mjds
                ])

                # ============================================================
                # Recover flags
                # ============================================================

                try:

                    flag_matrix = np.asarray(
                        metadata_row[
                            "FLAG"
                        ]
                    )

                    if flag_matrix.size == expected_size:

                        flag_matrix = flag_matrix.reshape(
                            n_baselines,
                            n_wavelengths
                        )

                        bad_flag_matrix = (
                            flag_matrix.astype(
                                bool
                            )
                        )

                    else:

                        print(
                            "WARNING: FLAG shape does not match VIS2"
                        )

                        bad_flag_matrix = np.zeros(
                            vis2_matrix.shape,
                            dtype=bool
                        )

                except Exception:

                    bad_flag_matrix = np.zeros(
                        vis2_matrix.shape,
                        dtype=bool
                    )

                # ============================================================
                # Map C_SCALE to baseline rows
                # ============================================================

                try:

                    c_values = np.asarray(
                        result_row[
                            "C_SCALE"
                        ],
                        dtype=float
                    ).ravel()

                except Exception:

                    c_values = np.array(
                        [1.0],
                        dtype=float
                    )

                if len(c_values) == 0:

                    c_values = np.array(
                        [1.0],
                        dtype=float
                    )

                invalid_c = (
                    ~np.isfinite(
                        c_values
                    )
                    | (
                        c_values <= 0
                    )
                )

                c_values[
                    invalid_c
                ] = 1.0

                if len(c_values) == 1:

                    c_per_baseline = np.repeat(
                        c_values[0],
                        n_baselines
                    )

                elif (
                    n_baselines
                    % len(c_values)
                    == 0
                ):

                    baselines_per_c = int(
                        n_baselines
                        / len(c_values)
                    )

                    c_per_baseline = np.repeat(
                        c_values,
                        baselines_per_c
                    )

                else:

                    print(
                        "WARNING: cannot map C_SCALE to baselines; "
                        "using C=1 for %s"
                        % science_name
                    )

                    c_per_baseline = np.ones(
                        n_baselines,
                        dtype=float
                    )

                c_matrix = np.repeat(
                    c_per_baseline[
                        :, np.newaxis
                    ],
                    n_wavelengths,
                    axis=1
                )

                vis2_corrected_matrix = (
                    vis2_matrix
                    / c_matrix
                )

                e_vis2_corrected_matrix = (
                    e_vis2_matrix
                    / c_matrix
                )

                # ============================================================
                # Spatial-frequency arrays
                # ============================================================

                baseline_matrix = np.repeat(
                    baselines[
                        :, np.newaxis
                    ],
                    n_wavelengths,
                    axis=1
                )

                wavelength_matrix = np.repeat(
                    wavelengths[
                        np.newaxis, :
                    ],
                    n_baselines,
                    axis=0
                )

                spatial_frequency_matrix = (
                    baseline_matrix
                    / wavelength_matrix
                )

                # ============================================================
                # Angular diameter
                # ============================================================

                fitted_ldd = safe_float(
                    result_row[
                        "LDD_FIT"
                    ]
                )

                fitted_ldd_error = safe_float(
                    result_row[
                        "e_LDD_FIT"
                    ]
                )

                if (
                    np.isfinite(fitted_ldd)
                    and fitted_ldd > 0
                ):

                    model_ldd = fitted_ldd
                    diameter_source = "fitted"

                elif use_predicted_if_missing:

                    model_ldd = safe_float(
                        tgt_info.loc[
                            target_id,
                            "LDD_pred"
                        ]
                    )

                    diameter_source = "predicted"

                else:

                    model_ldd = np.nan
                    diameter_source = "unavailable"

                if (
                    not np.isfinite(model_ldd)
                    or model_ldd <= 0
                ):

                    raise ValueError(
                        "No valid angular diameter"
                    )

                # ============================================================
                # Wavelength-dependent model coefficients
                # ============================================================

                u_lambdas = get_wavelength_coefficients(
                    result_row,
                    target_id,
                    "u",
                    n_wavelengths,
                    0.3
                )

                s_lambdas = get_wavelength_coefficients(
                    result_row,
                    target_id,
                    "s",
                    n_wavelengths,
                    1.0
                )

                u_matrix = np.repeat(
                    u_lambdas[
                        np.newaxis, :
                    ],
                    n_baselines,
                    axis=0
                )

                s_matrix = np.repeat(
                    s_lambdas[
                        np.newaxis, :
                    ],
                    n_baselines,
                    axis=0
                )

                # ============================================================
                # Flatten arrays
                # ============================================================

                spatial_frequency = (
                    spatial_frequency_matrix.flatten()
                )

                baseline_per_point = (
                    baseline_matrix.flatten()
                )

                wavelength_per_point = (
                    wavelength_matrix.flatten()
                )

                vis2_corrected = (
                    vis2_corrected_matrix.flatten()
                )

                e_vis2_corrected = (
                    e_vis2_corrected_matrix.flatten()
                )

                u_per_point = (
                    u_matrix.flatten()
                )

                s_per_point = (
                    s_matrix.flatten()
                )

                bad_flag = (
                    bad_flag_matrix.flatten()
                )

                wavelength_indices = np.tile(
                    np.arange(
                        n_wavelengths
                    ),
                    n_baselines
                )

                night_per_point = np.repeat(
                    night_labels,
                    n_wavelengths
                )

                pair_per_point = np.repeat(
                    telescope_pairs,
                    n_wavelengths
                )

                mjd_per_point = np.repeat(
                    mjds,
                    n_wavelengths
                )

                c_per_point = np.repeat(
                    c_per_baseline,
                    n_wavelengths
                )

                point_indices = np.arange(
                    expected_size
                )

                # ============================================================
                # Model VIS2
                # ============================================================

                model_vis2 = rdiam.calc_vis2(
                    spatial_frequency,
                    model_ldd,
                    1.0,
                    (
                        len(spatial_frequency),
                    ),
                    u_per_point,
                    s_per_point
                )

                # ============================================================
                # Raw and standardized residuals
                # ============================================================

                raw_residual = (
                    vis2_corrected
                    - model_vis2
                )

                standardized_residual = np.empty(
                    len(raw_residual),
                    dtype=float
                )

                standardized_residual[:] = np.nan

                valid_uncertainty = (
                    np.isfinite(
                        e_vis2_corrected
                    )
                    & (
                        e_vis2_corrected > 0
                    )
                )

                standardized_residual[
                    valid_uncertainty
                ] = (
                    raw_residual[
                        valid_uncertainty
                    ]
                    / e_vis2_corrected[
                        valid_uncertainty
                    ]
                )

                valid_data = (
                    np.isfinite(
                        spatial_frequency
                    )
                    & np.isfinite(
                        vis2_corrected
                    )
                    & np.isfinite(
                        model_vis2
                    )
                )

                large_sigma_residual = (
                    np.isfinite(
                        standardized_residual
                    )
                    & (
                        np.abs(
                            standardized_residual
                        )
                        >= sigma_threshold
                    )
                )

                large_raw_residual = (
                    np.isfinite(
                        raw_residual
                    )
                    & (
                        np.abs(
                            raw_residual
                        )
                        >= raw_residual_threshold
                    )
                )

                combined_outlier = (
                    large_sigma_residual
                    & large_raw_residual
                )

                removal_candidate = (
                    bad_flag
                    | combined_outlier
                )

                # ============================================================
                # Optional highlight mask
                # ============================================================

                highlight_mask = np.ones(
                    expected_size,
                    dtype=bool
                )

                has_highlight_filter = False

                if highlight_night is not None:

                    has_highlight_filter = True

                    highlight_mask &= (
                        night_per_point
                        == str(highlight_night)
                    )

                if highlight_pair is not None:

                    has_highlight_filter = True

                    highlight_mask &= (
                        pair_per_point
                        == str(highlight_pair)
                    )

                if highlight_baseline_range is not None:

                    has_highlight_filter = True

                    baseline_min = float(
                        highlight_baseline_range[0]
                    )

                    baseline_max = float(
                        highlight_baseline_range[1]
                    )

                    highlight_mask &= (
                        baseline_per_point
                        >= baseline_min
                    )

                    highlight_mask &= (
                        baseline_per_point
                        <= baseline_max
                    )

                if highlight_wavelength_index is not None:

                    has_highlight_filter = True

                    highlight_mask &= (
                        wavelength_indices
                        == int(
                            highlight_wavelength_index
                        )
                    )

                if not has_highlight_filter:

                    highlight_mask[:] = False

                # ============================================================
                # Add rows to output CSV
                # ============================================================

                for point_i in xrange(
                        expected_size):

                    diagnostic_rows.append({
                        "star": science_name,
                        "sequence": sequence_name,
                        "period": period_value,
                        "point_id": int(point_i),
                        "night": str(
                            night_per_point[
                                point_i
                            ]
                        ),
                        "mjd": mjd_per_point[
                            point_i
                        ],
                        "telescope_pair": str(
                            pair_per_point[
                                point_i
                            ]
                        ),
                        "baseline_m": baseline_per_point[
                            point_i
                        ],
                        "wavelength_index": int(
                            wavelength_indices[
                                point_i
                            ]
                        ),
                        "wavelength_um": (
                            wavelength_per_point[
                                point_i
                            ]
                            * 1.0E6
                        ),
                        "spatial_frequency_rad_inv":
                            spatial_frequency[
                                point_i
                            ],
                        "c_scale": c_per_point[
                            point_i
                        ],
                        "vis2_corrected": vis2_corrected[
                            point_i
                        ],
                        "e_vis2_corrected":
                            e_vis2_corrected[
                                point_i
                            ],
                        "model_vis2": model_vis2[
                            point_i
                        ],
                        "raw_residual": raw_residual[
                            point_i
                        ],
                        "standardized_residual":
                            standardized_residual[
                                point_i
                            ],
                        "oifits_flag": bool(
                            bad_flag[
                                point_i
                            ]
                        ),
                        "large_raw_residual": bool(
                            large_raw_residual[
                                point_i
                            ]
                        ),
                        "large_sigma_residual": bool(
                            large_sigma_residual[
                                point_i
                            ]
                        ),
                        "candidate_remove": bool(
                            removal_candidate[
                                point_i
                            ]
                        ),
                        "highlighted": bool(
                            highlight_mask[
                                point_i
                            ]
                        )
                    })

                # ============================================================
                # Figure
                # ============================================================

                fig, axes = plt.subplots(
                    2,
                    2
                )

                fig.set_size_inches(
                    16,
                    11
                )

                axes = axes.flatten()

                # ------------------------------------------------------------
                # Smooth visibility model curves
                # ------------------------------------------------------------

                model_frequency = np.linspace(
                    1.0E6,
                    2.5E8,
                    10000
                )

                for wavelength_i in xrange(
                        n_wavelengths):

                    model_curve = rdiam.calc_vis2(
                        model_frequency,
                        model_ldd,
                        1.0,
                        (
                            len(model_frequency),
                        ),
                        u_lambdas[
                            wavelength_i
                        ],
                        s_lambdas[
                            wavelength_i
                        ]
                    )

                    axes[0].plot(
                        model_frequency,
                        model_curve,
                        color="0.75",
                        linewidth=0.7,
                        zorder=1
                    )

                    axes[1].plot(
                        model_frequency,
                        model_curve,
                        color="0.75",
                        linewidth=0.7,
                        zorder=1
                    )

                # ------------------------------------------------------------
                # Top-left: colour by night
                # ------------------------------------------------------------

                night_colours = plot_grouped_visibilities(
                    axes[0],
                    night_per_point,
                    "Visibility points coloured by night",
                    spatial_frequency,
                    vis2_corrected,
                    e_vis2_corrected,
                    valid_data,
                    cm.get_cmap(
                        "jet"
                    )
                )

                # ------------------------------------------------------------
                # Top-right: colour by telescope pair
                # ------------------------------------------------------------

                pair_colours = plot_grouped_visibilities(
                    axes[1],
                    pair_per_point,
                    "Visibility points coloured by telescope pair",
                    spatial_frequency,
                    vis2_corrected,
                    e_vis2_corrected,
                    valid_data,
                    cm.get_cmap(
                        "jet"
                    )
                )

                # ------------------------------------------------------------
                # Bottom-left: raw residual by wavelength channel
                # ------------------------------------------------------------

                wavelength_colour_map = cm.get_cmap(
                    "viridis"
                )

                for wavelength_i in xrange(
                        n_wavelengths):

                    wavelength_mask = (
                        np.isfinite(
                            raw_residual
                        )
                        & (
                            wavelength_indices
                            == wavelength_i
                        )
                    )

                    if not np.any(
                            wavelength_mask):

                        continue

                    denominator = max(
                        1.0,
                        float(
                            n_wavelengths - 1
                        )
                    )

                    wavelength_colour = (
                        wavelength_colour_map(
                            float(wavelength_i)
                            / denominator
                        )
                    )

                    axes[2].scatter(
                        spatial_frequency[
                            wavelength_mask
                        ],
                        raw_residual[
                            wavelength_mask
                        ],
                        s=18,
                        color=wavelength_colour,
                        label=(
                            "ch %i: %.3f um"
                            % (
                                wavelength_i,
                                wavelengths[
                                    wavelength_i
                                ] * 1.0E6
                            )
                        )
                    )

                axes[2].axhline(
                    0.0,
                    linestyle="-",
                    linewidth=0.7,
                    color="0.3"
                )

                axes[2].axhline(
                    raw_residual_threshold,
                    linestyle="--",
                    linewidth=0.8,
                    color="red"
                )

                axes[2].axhline(
                    -raw_residual_threshold,
                    linestyle="--",
                    linewidth=0.8,
                    color="red"
                )

                axes[2].set_title(
                    "Raw visibility residual by wavelength channel"
                )

                axes[2].set_xlabel(
                    r"Spatial frequency (rad$^{-1}$)"
                )

                axes[2].set_ylabel(
                    r"$V^2_{\rm obs}-V^2_{\rm model}$"
                )

                axes[2].grid()

                axes[2].legend(
                    loc="best",
                    fontsize=7,
                    ncol=2
                )

                # ------------------------------------------------------------
                # Bottom-right: standardized residual versus baseline
                # ------------------------------------------------------------

                unique_pairs = sorted(
                    set(
                        pair_per_point.tolist()
                    )
                )

                for pair_value in unique_pairs:

                    pair_mask = (
                        np.isfinite(
                            standardized_residual
                        )
                        & (
                            pair_per_point
                            == pair_value
                        )
                    )

                    if not np.any(
                            pair_mask):

                        continue

                    pair_colour = pair_colours.get(
                        pair_value,
                        "0.4"
                    )

                    axes[3].scatter(
                        baseline_per_point[
                            pair_mask
                        ],
                        standardized_residual[
                            pair_mask
                        ],
                        s=20,
                        color=pair_colour,
                        label=str(
                            pair_value
                        )
                    )

                axes[3].axhline(
                    0.0,
                    linestyle="-",
                    linewidth=0.7,
                    color="0.3"
                )

                axes[3].axhline(
                    sigma_threshold,
                    linestyle="--",
                    linewidth=0.8,
                    color="red"
                )

                axes[3].axhline(
                    -sigma_threshold,
                    linestyle="--",
                    linewidth=0.8,
                    color="red"
                )

                axes[3].set_title(
                    "Standardized residual versus projected baseline"
                )

                axes[3].set_xlabel(
                    "Projected baseline (m)"
                )

                axes[3].set_ylabel(
                    r"$(V^2_{\rm obs}-V^2_{\rm model})/\sigma$"
                )

                axes[3].grid()

                axes[3].legend(
                    loc="best",
                    fontsize=7,
                    ncol=2
                )

                # ============================================================
                # Mark flagged measurements
                # ============================================================

                flagged_valid = (
                    bad_flag
                    & valid_data
                )

                if np.any(
                        flagged_valid):

                    axes[0].scatter(
                        spatial_frequency[
                            flagged_valid
                        ],
                        vis2_corrected[
                            flagged_valid
                        ],
                        marker="x",
                        s=50,
                        color="black",
                        zorder=7
                    )

                    axes[1].scatter(
                        spatial_frequency[
                            flagged_valid
                        ],
                        vis2_corrected[
                            flagged_valid
                        ],
                        marker="x",
                        s=50,
                        color="black",
                        zorder=7
                    )

                # ============================================================
                # Mark removal candidates
                # ============================================================

                candidate_valid = (
                    removal_candidate
                    & valid_data
                )

                if np.any(
                        candidate_valid):

                    axes[0].scatter(
                        spatial_frequency[
                            candidate_valid
                        ],
                        vis2_corrected[
                            candidate_valid
                        ],
                        marker="o",
                        s=80,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.2,
                        zorder=8
                    )

                    axes[1].scatter(
                        spatial_frequency[
                            candidate_valid
                        ],
                        vis2_corrected[
                            candidate_valid
                        ],
                        marker="o",
                        s=80,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.2,
                        zorder=8
                    )

                    axes[2].scatter(
                        spatial_frequency[
                            candidate_valid
                        ],
                        raw_residual[
                            candidate_valid
                        ],
                        marker="o",
                        s=80,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.2,
                        zorder=8
                    )

                    candidate_with_sigma = (
                        candidate_valid
                        & np.isfinite(
                            standardized_residual
                        )
                    )

                    axes[3].scatter(
                        baseline_per_point[
                            candidate_with_sigma
                        ],
                        standardized_residual[
                            candidate_with_sigma
                        ],
                        marker="o",
                        s=80,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.2,
                        zorder=8
                    )

                # ============================================================
                # Highlight requested subset
                # ============================================================

                highlighted_valid = (
                    highlight_mask
                    & valid_data
                )

                if np.any(
                        highlighted_valid):

                    axes[0].scatter(
                        spatial_frequency[
                            highlighted_valid
                        ],
                        vis2_corrected[
                            highlighted_valid
                        ],
                        marker="s",
                        s=100,
                        facecolors="none",
                        edgecolors="red",
                        linewidths=1.4,
                        zorder=9
                    )

                    axes[1].scatter(
                        spatial_frequency[
                            highlighted_valid
                        ],
                        vis2_corrected[
                            highlighted_valid
                        ],
                        marker="s",
                        s=100,
                        facecolors="none",
                        edgecolors="red",
                        linewidths=1.4,
                        zorder=9
                    )

                    axes[2].scatter(
                        spatial_frequency[
                            highlighted_valid
                        ],
                        raw_residual[
                            highlighted_valid
                        ],
                        marker="s",
                        s=100,
                        facecolors="none",
                        edgecolors="red",
                        linewidths=1.4,
                        zorder=9
                    )

                    highlighted_sigma = (
                        highlighted_valid
                        & np.isfinite(
                            standardized_residual
                        )
                    )

                    axes[3].scatter(
                        baseline_per_point[
                            highlighted_sigma
                        ],
                        standardized_residual[
                            highlighted_sigma
                        ],
                        marker="s",
                        s=100,
                        facecolors="none",
                        edgecolors="red",
                        linewidths=1.4,
                        zorder=9
                    )

                # ============================================================
                # Annotate strongest candidates
                # ============================================================

                candidate_indices = np.where(
                    removal_candidate
                )[0]

                if len(candidate_indices) > 0:

                    candidate_strength = np.abs(
                        raw_residual[
                            candidate_indices
                        ]
                    )

                    finite_sigma_candidate = np.isfinite(
                        standardized_residual[
                            candidate_indices
                        ]
                    )

                    candidate_strength[
                        finite_sigma_candidate
                    ] *= np.maximum(
                        1.0,
                        np.abs(
                            standardized_residual[
                                candidate_indices[
                                    finite_sigma_candidate
                                ]
                            ]
                        )
                    )

                    sorted_candidate_indices = (
                        candidate_indices[
                            np.argsort(
                                candidate_strength
                            )[::-1]
                        ]
                    )

                    sorted_candidate_indices = (
                        sorted_candidate_indices[
                            :int(max_annotations)
                        ]
                    )

                    for point_i in sorted_candidate_indices:

                        point_label = (
                            "P%i"
                            % point_i
                        )

                        axes[2].annotate(
                            point_label,
                            xy=(
                                spatial_frequency[
                                    point_i
                                ],
                                raw_residual[
                                    point_i
                                ]
                            ),
                            xytext=(3, 3),
                            textcoords="offset points",
                            fontsize=6
                        )

                        if np.isfinite(
                                standardized_residual[
                                    point_i
                                ]):

                            axes[3].annotate(
                                point_label,
                                xy=(
                                    baseline_per_point[
                                        point_i
                                    ],
                                    standardized_residual[
                                        point_i
                                    ]
                                ),
                                xytext=(3, 3),
                                textcoords="offset points",
                                fontsize=6
                            )

                # ============================================================
                # Candidate text
                # ============================================================

                candidate_text = []

                candidate_text.append(
                    "Candidate rule: FLAG or "
                    "(|raw residual| >= %.3f and |residual/sigma| >= %.1f)"
                    % (
                        raw_residual_threshold,
                        sigma_threshold
                    )
                )

                if len(candidate_indices) == 0:

                    candidate_text.append(
                        "No removal candidates."
                    )

                else:

                    candidate_text.append(
                        "ID | night | pair | B(m) | channel | raw | sigma"
                    )

                    for point_i in sorted_candidate_indices[
                            :8]:

                        candidate_text.append(
                            "P%i | %s | %s | %.1f | %i | %.3f | %s"
                            % (
                                point_i,
                                night_per_point[
                                    point_i
                                ],
                                pair_per_point[
                                    point_i
                                ],
                                baseline_per_point[
                                    point_i
                                ],
                                wavelength_indices[
                                    point_i
                                ],
                                raw_residual[
                                    point_i
                                ],
                                (
                                    "%.2f"
                                    % standardized_residual[
                                        point_i
                                    ]
                                    if np.isfinite(
                                        standardized_residual[
                                            point_i
                                        ]
                                    )
                                    else "nan"
                                )
                            )
                        )

                # ============================================================
                # Final figure formatting
                # ============================================================

                if np.isfinite(
                        fitted_ldd_error):

                    diameter_text = (
                        "%.4f +/- %.4f mas"
                        % (
                            model_ldd,
                            fitted_ldd_error
                        )
                    )

                else:

                    diameter_text = (
                        "%.4f mas"
                        % model_ldd
                    )

                fig.suptitle(
                    (
                        "%s, %s, %s\n"
                        "%s LDD = %s; "
                        "%i measurements; %i removal candidates"
                    )
                    % (
                        science_name,
                        sequence_name,
                        str(period_value),
                        diameter_source,
                        diameter_text,
                        expected_size,
                        int(
                            np.sum(
                                removal_candidate
                            )
                        )
                    ),
                    fontsize=14
                )

                fig.text(
                    0.5,
                    0.01,
                    "\n".join(
                        candidate_text
                    ),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=7,
                    family="monospace"
                )

                fig.tight_layout(
                    rect=[
                        0.0,
                        0.11,
                        1.0,
                        0.93
                    ]
                )

                # ============================================================
                # Save page and PNG
                # ============================================================

                pdf.savefig(
                    fig
                )

                safe_sequence = (
                    clean_target_name_for_plot(
                        sequence_name
                    )
                )

                individual_filename = (
                    "%s_%s_%s_visibility_diagnostic.png"
                    % (
                        science_clean,
                        safe_sequence,
                        str(period_value)
                    )
                )

                individual_output = os.path.join(
                    individual_directory,
                    individual_filename
                )

                fig.savefig(
                    individual_output,
                    dpi=200
                )

                plt.close(
                    fig
                )

                n_created += 1

                print(
                    "Saved diagnostic for %s"
                    % science_name
                )

            except Exception as error:

                n_failed += 1

                print("")
                print(
                    "FAILED visibility diagnostic for %s"
                    % science_name
                )

                print(
                    "Error: %s"
                    % str(error)
                )

                traceback.print_exc()

                plt.close(
                    "all"
                )

    # ====================================================================
    # Save diagnostic CSV files
    # ====================================================================

    diagnostic_table = pd.DataFrame(
        diagnostic_rows
    )

    diagnostic_table.to_csv(
        all_points_csv,
        index=False
    )

    if (
        len(diagnostic_table) > 0
        and "candidate_remove"
        in diagnostic_table.columns
    ):

        candidate_table = diagnostic_table[
            diagnostic_table[
                "candidate_remove"
            ]
        ].copy()

    else:

        candidate_table = pd.DataFrame()

    candidate_table.to_csv(
        candidate_csv,
        index=False
    )

    print("")
    print("=" * 79)
    print("Visibility diagnostics finished")
    print(
        "Created pages: %i"
        % n_created
    )
    print(
        "Failed pages: %i"
        % n_failed
    )
    print("PDF:")
    print(output_file)
    print("Complete point table:")
    print(all_points_csv)
    print("Removal-candidate table:")
    print(candidate_csv)
    print("=" * 79)

    if n_created == 0:

        raise RuntimeError(
            "No visibility diagnostic pages were generated"
        )

    return output_file


def plot_visibility_diagnostic_summary(
        results,
        bs_results,
        tgt_info,
        output_file=(
            "plots/visibility_diagnostics/"
            "visibility_diagnostic_summary.pdf"
        ),
        bootstrap_index=0,
        sigma_threshold=3.0,
        raw_residual_threshold=0.05,
        e_wl_frac=0.0035,
        star_filter=None,
        highlight_night=None,
        highlight_pair=None,
        highlight_baseline_range=None,
        highlight_wavelength_index=None,
        max_annotations=10,
        use_predicted_if_missing=True):
    """
    Create point-by-point visibility diagnostics.

    The four panels are:

        1. Corrected VIS2 coloured by observing night.
        2. Corrected VIS2 coloured by telescope pair.
        3. Raw residual:
               VIS2_observed - VIS2_model
           coloured by wavelength channel.
        4. Standardized residual:
               (VIS2_observed - VIS2_model) / e_VIS2
           versus projected baseline.

    A point is marked as a removal candidate when:

        - it has an OIFITS flag; or
        - abs(standardized residual) >= sigma_threshold
          AND
          abs(raw residual) >= raw_residual_threshold.

    No data are removed by this function.
    """

    # ====================================================================
    # Compatibility
    # ====================================================================

    try:
        string_types = (basestring,)
    except NameError:
        string_types = (str,)

    # ====================================================================
    # Helper functions
    # ====================================================================

    def safe_float(value):

        try:
            return float(value)
        except Exception:
            return np.nan


    def pair_to_string(value):
        """
        Convert a telescope-pair representation to a readable string.
        """

        try:

            if isinstance(
                    value,
                    (tuple, list, np.ndarray)):

                values = list(
                    value
                )

                if len(values) >= 2:

                    return "%s-%s" % (
                        str(values[0]),
                        str(values[1])
                    )

        except Exception:
            pass

        return str(
            value
        )


    def resize_numeric_metadata(
            values,
            required_length):
        """
        Resize numerical metadata to the number of baseline rows.
        """

        output = np.empty(
            required_length,
            dtype=float
        )

        output[:] = np.nan

        try:

            values = np.asarray(
                values,
                dtype=float
            ).ravel()

        except Exception:

            values = np.array(
                [],
                dtype=float
            )

        n_copy = min(
            len(values),
            required_length
        )

        if n_copy > 0:

            output[:n_copy] = values[
                :n_copy
            ]

        return output


    def resize_pair_metadata(
            values,
            required_length):
        """
        Preserve telescope pairs without flattening tuple pairs.
        """

        output = np.empty(
            required_length,
            dtype=object
        )

        output[:] = "unknown"

        try:

            values = list(
                values
            )

        except Exception:

            values = []

        n_copy = min(
            len(values),
            required_length
        )

        for value_i in xrange(
                n_copy):

            output[value_i] = pair_to_string(
                values[value_i]
            )

        return output


    def mjd_to_night(mjd_value):
        """
        Convert MJD to an observing-night string.

        Subtracting half a day keeps observations made after midnight
        associated with the evening on which the night began.
        """

        mjd_value = safe_float(
            mjd_value
        )

        if not np.isfinite(
                mjd_value):

            return "unknown"

        try:

            mjd_epoch = datetime(
                1858,
                11,
                17
            )

            date_value = (
                mjd_epoch
                + timedelta(
                    days=mjd_value - 0.5
                )
            )

            return date_value.strftime(
                "%Y-%m-%d"
            )

        except Exception:

            return "unknown"


    def resolve_bs_key(
            science_name,
            sequence_name,
            period_value):
        """
        Match one results row to the corresponding bs_results key.
        """

        science_clean = clean_target_name_for_plot(
            science_name
        )

        sequence_clean = str(
            sequence_name
        ).strip().lower()

        period_clean = str(
            period_value
        ).strip()

        # Direct lookup for combined fits.
        if science_name in bs_results:

            return science_name

        # Direct tuple possibilities.
        direct_keys = [
            (
                science_name,
                sequence_name,
                period_value
            ),
            (
                science_name,
                str(sequence_name),
                period_value
            ),
        ]

        for direct_key in direct_keys:

            if direct_key in bs_results:

                return direct_key

        # Robust matching.
        for candidate_key in bs_results.keys():

            if isinstance(
                    candidate_key,
                    tuple):

                candidate_star = candidate_key[0]

                candidate_sequence = ""

                candidate_period = ""

                if len(candidate_key) > 1:

                    candidate_sequence = str(
                        candidate_key[1]
                    ).strip().lower()

                if len(candidate_key) > 2:

                    candidate_period = str(
                        candidate_key[2]
                    ).strip()

                same_star = (
                    clean_target_name_for_plot(
                        candidate_star
                    )
                    == science_clean
                )

                same_sequence = (
                    candidate_sequence
                    == sequence_clean
                )

                same_period = (
                    candidate_period
                    == period_clean
                )

                if (
                    same_star
                    and same_sequence
                    and same_period
                ):

                    return candidate_key

            else:

                same_star = (
                    clean_target_name_for_plot(
                        candidate_key
                    )
                    == science_clean
                )

                if same_star:

                    return candidate_key

        return None


    def get_wavelength_coefficients(
            result_row,
            target_id,
            coefficient_name,
            n_wavelengths,
            default_value):
        """
        Obtain U_LAMBDA or S_LAMBDA from results, falling back to
        wavelength-specific columns in tgt_info.
        """

        if coefficient_name == "u":

            result_column = "U_LAMBDA"
            target_prefix = "u_lambda_"

        else:

            result_column = "S_LAMBDA"
            target_prefix = "s_lambda_"

        coefficient_values = np.empty(
            n_wavelengths,
            dtype=float
        )

        coefficient_values[:] = np.nan

        # First use the final values stored in results.
        if result_column in result_row.index:

            try:

                result_values = np.asarray(
                    result_row[
                        result_column
                    ],
                    dtype=float
                ).ravel()

                n_copy = min(
                    len(result_values),
                    n_wavelengths
                )

                coefficient_values[
                    :n_copy
                ] = result_values[
                    :n_copy
                ]

            except Exception:
                pass

        # Fill missing values from tgt_info.
        for wavelength_i in xrange(
                n_wavelengths):

            if np.isfinite(
                    coefficient_values[
                        wavelength_i
                    ]):

                continue

            column_name = (
                target_prefix
                + str(wavelength_i)
            )

            if column_name in tgt_info.columns:

                try:

                    coefficient_values[
                        wavelength_i
                    ] = float(
                        tgt_info.loc[
                            target_id,
                            column_name
                        ]
                    )

                except Exception:
                    pass

        invalid_values = (
            ~np.isfinite(
                coefficient_values
            )
        )

        coefficient_values[
            invalid_values
        ] = float(
            default_value
        )

        return coefficient_values


    def plot_grouped_visibilities(
            axis,
            group_values,
            group_label,
            spatial_frequency,
            vis2_values,
            e_vis2_values,
            valid_mask,
            colour_map):
        """
        Plot visibility measurements grouped by night or telescope pair.
        """

        unique_groups = sorted(
            set(
                group_values.tolist()
            )
        )

        group_colours = {}

        for group_i, group_value in enumerate(
                unique_groups):

            denominator = max(
                1.0,
                float(
                    len(unique_groups) - 1
                )
            )

            group_colour = colour_map(
                float(group_i)
                / denominator
            )

            group_colours[
                group_value
            ] = group_colour

            group_mask = (
                valid_mask
                & (
                    group_values
                    == group_value
                )
            )

            if not np.any(
                    group_mask):

                continue

            finite_error = (
                group_mask
                & np.isfinite(
                    e_vis2_values
                )
                & (
                    e_vis2_values > 0
                )
            )

            no_error = (
                group_mask
                & ~finite_error
            )

            if np.any(
                    finite_error):

                axis.errorbar(
                    spatial_frequency[
                        finite_error
                    ],
                    vis2_values[
                        finite_error
                    ],
                    xerr=(
                        spatial_frequency[
                            finite_error
                        ]
                        * e_wl_frac
                    ),
                    yerr=e_vis2_values[
                        finite_error
                    ],
                    fmt=".",
                    color=group_colour,
                    label=str(
                        group_value
                    ),
                    elinewidth=0.3,
                    capsize=0.5,
                    capthick=0.3,
                    markersize=4,
                    zorder=3
                )

            if np.any(
                    no_error):

                axis.scatter(
                    spatial_frequency[
                        no_error
                    ],
                    vis2_values[
                        no_error
                    ],
                    s=12,
                    color=group_colour,
                    label=(
                        str(group_value)
                        if not np.any(finite_error)
                        else None
                    ),
                    zorder=3
                )

        axis.set_title(
            group_label
        )

        axis.set_xlabel(
            r"Spatial frequency (rad$^{-1}$)"
        )

        axis.set_ylabel(
            r"Corrected visibility$^2$"
        )

        axis.set_xlim(
            [0.0, 2.5E8]
        )

        axis.set_ylim(
            [0.0, 1.5]
        )

        axis.grid()

        handles, labels = (
            axis.get_legend_handles_labels()
        )

        if len(handles) > 0:

            axis.legend(
                loc="best",
                fontsize=7,
                ncol=2
            )

        return group_colours


    # ====================================================================
    # Validate inputs
    # ====================================================================

    if results is None or len(
            results) == 0:

        raise ValueError(
            "results is empty"
        )

    if bs_results is None or len(
            bs_results) == 0:

        raise ValueError(
            "bs_results is empty"
        )

    if tgt_info is None:

        raise ValueError(
            "tgt_info must be provided"
        )

    # ====================================================================
    # Output directories
    # ====================================================================

    output_directory = os.path.dirname(
        output_file
    )

    if output_directory == "":

        output_directory = "."

    if not os.path.exists(
            output_directory):

        os.makedirs(
            output_directory
        )

    individual_directory = os.path.join(
        output_directory,
        "individual"
    )

    if not os.path.exists(
            individual_directory):

        os.makedirs(
            individual_directory
        )

    all_points_csv = os.path.join(
        output_directory,
        "visibility_point_diagnostics.csv"
    )

    candidate_csv = os.path.join(
        output_directory,
        "visibility_removal_candidates.csv"
    )

    v2_summary_pdf = os.path.join(
        output_directory,
        "visibility_v2_above_one_summary.pdf"
    )

    v2_summary_csv = os.path.join(
        output_directory,
        "visibility_v2_above_one_summary.csv"
    )

    v2_overview_png = os.path.join(
        output_directory,
        "visibility_v2_above_one_overview.png"
    )

    v2_individual_directory = os.path.join(
        output_directory,
        "v2_above_one_individual"
    )

    if not os.path.exists(
            v2_individual_directory):

        os.makedirs(
            v2_individual_directory
        )

    # ====================================================================
    # Star filtering
    # ====================================================================

    if star_filter is None:

        selected_star_names = None

    else:

        if isinstance(
                star_filter,
                string_types):

            star_filter = [
                star_filter
            ]

        selected_star_names = set([
            clean_target_name_for_plot(
                target_name
            )
            for target_name in star_filter
        ])

    diagnostic_rows = []
    v2_summary_rows = []

    n_created = 0
    n_failed = 0

    plt.close(
        "all"
    )

    print("")
    print("=" * 79)
    print("Creating visibility diagnostics")
    print("Output:")
    print(output_file)
    print("=" * 79)

    # ====================================================================
    # Multipage PDFs
    # ====================================================================

    v2_pdf = PdfPages(
        v2_summary_pdf
    )

    with PdfPages(
            output_file) as pdf:

        for result_i in xrange(
                len(results)):

            result_row = results.iloc[
                result_i
            ]

            science_name = str(
                result_row[
                    "STAR"
                ]
            )

            sequence_name = str(
                result_row.get(
                    "SEQUENCE",
                    "combined"
                )
            )

            period_value = result_row.get(
                "PERIOD",
                ""
            )

            science_clean = (
                clean_target_name_for_plot(
                    science_name
                )
            )

            if (
                selected_star_names is not None
                and science_clean
                not in selected_star_names
            ):

                continue

            print("")
            print(
                "Diagnostic for %s, %s, %s"
                % (
                    science_name,
                    sequence_name,
                    str(period_value)
                )
            )

            try:

                # ============================================================
                # Match target information
                # ============================================================

                target_id = result_row.get(
                    "HD",
                    None
                )

                if target_id not in tgt_info.index:

                    target_id = match_target_for_plot(
                        tgt_info,
                        science_name,
                        verbose=True
                    )

                if target_id is None:

                    raise ValueError(
                        "Could not match target in tgt_info"
                    )

                # ============================================================
                # Match bootstrap metadata
                # ============================================================

                bootstrap_key = resolve_bs_key(
                    science_name,
                    sequence_name,
                    period_value
                )

                if bootstrap_key is None:

                    raise ValueError(
                        "Could not find matching bs_results key"
                    )

                bootstrap_table = bs_results[
                    bootstrap_key
                ]

                if len(bootstrap_table) == 0:

                    raise ValueError(
                        "Bootstrap table is empty"
                    )

                metadata_index = int(
                    bootstrap_index
                )

                if metadata_index < 0:

                    metadata_index = (
                        len(bootstrap_table)
                        + metadata_index
                    )

                metadata_index = max(
                    0,
                    min(
                        metadata_index,
                        len(bootstrap_table) - 1
                    )
                )

                metadata_row = bootstrap_table.iloc[
                    metadata_index
                ]

                # ============================================================
                # Final VIS2 arrays
                # ============================================================

                baselines = np.asarray(
                    result_row[
                        "BASELINE"
                    ],
                    dtype=float
                ).ravel()

                wavelengths = np.asarray(
                    result_row[
                        "WAVELENGTH"
                    ],
                    dtype=float
                ).ravel()

                vis2_matrix = np.asarray(
                    result_row[
                        "VIS2"
                    ],
                    dtype=float
                )

                e_vis2_matrix = np.asarray(
                    result_row[
                        "e_VIS2"
                    ],
                    dtype=float
                )

                n_baselines = len(
                    baselines
                )

                n_wavelengths = len(
                    wavelengths
                )

                if n_baselines == 0:

                    raise ValueError(
                        "No baseline values"
                    )

                if n_wavelengths == 0:

                    raise ValueError(
                        "No wavelength values"
                    )

                expected_size = (
                    n_baselines
                    * n_wavelengths
                )

                if vis2_matrix.size != expected_size:

                    raise ValueError(
                        "VIS2 size=%i, expected=%i"
                        % (
                            vis2_matrix.size,
                            expected_size
                        )
                    )

                if e_vis2_matrix.size != expected_size:

                    raise ValueError(
                        "e_VIS2 size=%i, expected=%i"
                        % (
                            e_vis2_matrix.size,
                            expected_size
                        )
                    )

                vis2_matrix = vis2_matrix.reshape(
                    n_baselines,
                    n_wavelengths
                )

                e_vis2_matrix = e_vis2_matrix.reshape(
                    n_baselines,
                    n_wavelengths
                )

                # ============================================================
                # Recover MJD, observing night and telescope pair
                # ============================================================

                mjds = resize_numeric_metadata(
                    metadata_row[
                        "MJD"
                    ],
                    n_baselines
                )

                telescope_pairs = resize_pair_metadata(
                    metadata_row[
                        "TEL_PAIR"
                    ],
                    n_baselines
                )

                night_labels = np.asarray([
                    mjd_to_night(
                        mjd_value
                    )
                    for mjd_value in mjds
                ])

                # ============================================================
                # Recover flags
                # ============================================================

                try:

                    flag_matrix = np.asarray(
                        metadata_row[
                            "FLAG"
                        ]
                    )

                    if flag_matrix.size == expected_size:

                        flag_matrix = flag_matrix.reshape(
                            n_baselines,
                            n_wavelengths
                        )

                        bad_flag_matrix = (
                            flag_matrix.astype(
                                bool
                            )
                        )

                    else:

                        print(
                            "WARNING: FLAG shape does not match VIS2"
                        )

                        bad_flag_matrix = np.zeros(
                            vis2_matrix.shape,
                            dtype=bool
                        )

                except Exception:

                    bad_flag_matrix = np.zeros(
                        vis2_matrix.shape,
                        dtype=bool
                    )

                # ============================================================
                # Map C_SCALE to baseline rows
                # ============================================================

                try:

                    c_values = np.asarray(
                        result_row[
                            "C_SCALE"
                        ],
                        dtype=float
                    ).ravel()

                except Exception:

                    c_values = np.array(
                        [1.0],
                        dtype=float
                    )

                if len(c_values) == 0:

                    c_values = np.array(
                        [1.0],
                        dtype=float
                    )

                invalid_c = (
                    ~np.isfinite(
                        c_values
                    )
                    | (
                        c_values <= 0
                    )
                )

                c_values[
                    invalid_c
                ] = 1.0

                if len(c_values) == 1:

                    c_per_baseline = np.repeat(
                        c_values[0],
                        n_baselines
                    )

                elif (
                    n_baselines
                    % len(c_values)
                    == 0
                ):

                    baselines_per_c = int(
                        n_baselines
                        / len(c_values)
                    )

                    c_per_baseline = np.repeat(
                        c_values,
                        baselines_per_c
                    )

                else:

                    print(
                        "WARNING: cannot map C_SCALE to baselines; "
                        "using C=1 for %s"
                        % science_name
                    )

                    c_per_baseline = np.ones(
                        n_baselines,
                        dtype=float
                    )

                c_matrix = np.repeat(
                    c_per_baseline[
                        :, np.newaxis
                    ],
                    n_wavelengths,
                    axis=1
                )

                vis2_corrected_matrix = (
                    vis2_matrix
                    / c_matrix
                )

                e_vis2_corrected_matrix = (
                    e_vis2_matrix
                    / c_matrix
                )

                # ============================================================
                # Spatial-frequency arrays
                # ============================================================

                baseline_matrix = np.repeat(
                    baselines[
                        :, np.newaxis
                    ],
                    n_wavelengths,
                    axis=1
                )

                wavelength_matrix = np.repeat(
                    wavelengths[
                        np.newaxis, :
                    ],
                    n_baselines,
                    axis=0
                )

                spatial_frequency_matrix = (
                    baseline_matrix
                    / wavelength_matrix
                )

                # ============================================================
                # Angular diameter
                # ============================================================

                fitted_ldd = safe_float(
                    result_row[
                        "LDD_FIT"
                    ]
                )

                fitted_ldd_error = safe_float(
                    result_row[
                        "e_LDD_FIT"
                    ]
                )

                if (
                    np.isfinite(fitted_ldd)
                    and fitted_ldd > 0
                ):

                    model_ldd = fitted_ldd
                    diameter_source = "fitted"

                elif use_predicted_if_missing:

                    model_ldd = safe_float(
                        tgt_info.loc[
                            target_id,
                            "LDD_pred"
                        ]
                    )

                    diameter_source = "predicted"

                else:

                    model_ldd = np.nan
                    diameter_source = "unavailable"

                if (
                    not np.isfinite(model_ldd)
                    or model_ldd <= 0
                ):

                    raise ValueError(
                        "No valid angular diameter"
                    )

                # ============================================================
                # Wavelength-dependent model coefficients
                # ============================================================

                u_lambdas = get_wavelength_coefficients(
                    result_row,
                    target_id,
                    "u",
                    n_wavelengths,
                    0.3
                )

                s_lambdas = get_wavelength_coefficients(
                    result_row,
                    target_id,
                    "s",
                    n_wavelengths,
                    1.0
                )

                u_matrix = np.repeat(
                    u_lambdas[
                        np.newaxis, :
                    ],
                    n_baselines,
                    axis=0
                )

                s_matrix = np.repeat(
                    s_lambdas[
                        np.newaxis, :
                    ],
                    n_baselines,
                    axis=0
                )

                # ============================================================
                # Flatten arrays
                # ============================================================

                spatial_frequency = (
                    spatial_frequency_matrix.flatten()
                )

                baseline_per_point = (
                    baseline_matrix.flatten()
                )

                wavelength_per_point = (
                    wavelength_matrix.flatten()
                )

                vis2_corrected = (
                    vis2_corrected_matrix.flatten()
                )

                e_vis2_corrected = (
                    e_vis2_corrected_matrix.flatten()
                )

                u_per_point = (
                    u_matrix.flatten()
                )

                s_per_point = (
                    s_matrix.flatten()
                )

                bad_flag = (
                    bad_flag_matrix.flatten()
                )

                wavelength_indices = np.tile(
                    np.arange(
                        n_wavelengths
                    ),
                    n_baselines
                )

                night_per_point = np.repeat(
                    night_labels,
                    n_wavelengths
                )

                pair_per_point = np.repeat(
                    telescope_pairs,
                    n_wavelengths
                )

                mjd_per_point = np.repeat(
                    mjds,
                    n_wavelengths
                )

                c_per_point = np.repeat(
                    c_per_baseline,
                    n_wavelengths
                )

                point_indices = np.arange(
                    expected_size
                )

                # ============================================================
                # Model VIS2
                # ============================================================

                model_vis2 = rdiam.calc_vis2(
                    spatial_frequency,
                    model_ldd,
                    1.0,
                    (
                        len(spatial_frequency),
                    ),
                    u_per_point,
                    s_per_point
                )

                # ============================================================
                # Raw and standardized residuals
                # ============================================================

                raw_residual = (
                    vis2_corrected
                    - model_vis2
                )

                standardized_residual = np.empty(
                    len(raw_residual),
                    dtype=float
                )

                standardized_residual[:] = np.nan

                valid_uncertainty = (
                    np.isfinite(
                        e_vis2_corrected
                    )
                    & (
                        e_vis2_corrected > 0
                    )
                )

                standardized_residual[
                    valid_uncertainty
                ] = (
                    raw_residual[
                        valid_uncertainty
                    ]
                    / e_vis2_corrected[
                        valid_uncertainty
                    ]
                )

                valid_data = (
                    np.isfinite(
                        spatial_frequency
                    )
                    & np.isfinite(
                        vis2_corrected
                    )
                    & np.isfinite(
                        model_vis2
                    )
                )

                # ============================================================
                # V2 > 1 summary statistics
                # ============================================================

                # Count every finite corrected V2 point, including OIFITS-
                # flagged points. This makes the summary describe the complete
                # corrected VIS2 array shown in the diagnostic plots.
                finite_vis2 = np.isfinite(
                    vis2_corrected
                )

                v2_above_one = (
                    finite_vis2
                    & (
                        vis2_corrected > 1.0
                    )
                )

                n_v2_points = int(
                    np.sum(
                        finite_vis2
                    )
                )

                n_v2_above_one = int(
                    np.sum(
                        v2_above_one
                    )
                )

                if n_v2_points > 0:

                    fraction_v2_above_one = (
                        float(
                            n_v2_above_one
                        )
                        / float(
                            n_v2_points
                        )
                    )

                    finite_point_indices = np.where(
                        finite_vis2
                    )[0]

                    max_point_index = int(
                        finite_point_indices[
                            np.argmax(
                                vis2_corrected[
                                    finite_point_indices
                                ]
                            )
                        ]
                    )

                    max_vis2 = float(
                        vis2_corrected[
                            max_point_index
                        ]
                    )

                    max_night = str(
                        night_per_point[
                            max_point_index
                        ]
                    )

                    max_pair = str(
                        pair_per_point[
                            max_point_index
                        ]
                    )

                    max_baseline = float(
                        baseline_per_point[
                            max_point_index
                        ]
                    )

                    max_wavelength_index = int(
                        wavelength_indices[
                            max_point_index
                        ]
                    )

                    max_wavelength_um = float(
                        wavelength_per_point[
                            max_point_index
                        ]
                        * 1.0E6
                    )

                else:

                    fraction_v2_above_one = np.nan
                    max_point_index = -1
                    max_vis2 = np.nan
                    max_night = "unknown"
                    max_pair = "unknown"
                    max_baseline = np.nan
                    max_wavelength_index = -1
                    max_wavelength_um = np.nan

                v2_above_one_indices = np.where(
                    v2_above_one
                )[0]

                v2_summary_rows.append({
                    "star": science_name,
                    "sequence": sequence_name,
                    "period": period_value,
                    "max_vis2_corrected": max_vis2,
                    "max_point_id": (
                        "P%i"
                        % max_point_index
                        if max_point_index >= 0
                        else ""
                    ),
                    "max_night": max_night,
                    "max_telescope_pair": max_pair,
                    "max_baseline_m": max_baseline,
                    "max_wavelength_index": max_wavelength_index,
                    "max_wavelength_um": max_wavelength_um,
                    "has_vis2_above_one": bool(
                        n_v2_above_one > 0
                    ),
                    "n_finite_vis2_points": n_v2_points,
                    "n_vis2_above_one": n_v2_above_one,
                    "fraction_vis2_above_one":
                        fraction_v2_above_one,
                    "percent_vis2_above_one": (
                        100.0
                        * fraction_v2_above_one
                        if np.isfinite(
                            fraction_v2_above_one
                        )
                        else np.nan
                    ),
                    "point_ids_vis2_above_one": ";".join([
                        "P%i"
                        % int(point_i)
                        for point_i
                        in v2_above_one_indices
                    ])
                })

                large_sigma_residual = (
                    np.isfinite(
                        standardized_residual
                    )
                    & (
                        np.abs(
                            standardized_residual
                        )
                        >= sigma_threshold
                    )
                )

                large_raw_residual = (
                    np.isfinite(
                        raw_residual
                    )
                    & (
                        np.abs(
                            raw_residual
                        )
                        >= raw_residual_threshold
                    )
                )

                combined_outlier = (
                    large_sigma_residual
                    & large_raw_residual
                )

                removal_candidate = (
                    bad_flag
                    | combined_outlier
                )

                # ============================================================
                # Optional highlight mask
                # ============================================================

                highlight_mask = np.ones(
                    expected_size,
                    dtype=bool
                )

                has_highlight_filter = False

                if highlight_night is not None:

                    has_highlight_filter = True

                    highlight_mask &= (
                        night_per_point
                        == str(highlight_night)
                    )

                if highlight_pair is not None:

                    has_highlight_filter = True

                    highlight_mask &= (
                        pair_per_point
                        == str(highlight_pair)
                    )

                if highlight_baseline_range is not None:

                    has_highlight_filter = True

                    baseline_min = float(
                        highlight_baseline_range[0]
                    )

                    baseline_max = float(
                        highlight_baseline_range[1]
                    )

                    highlight_mask &= (
                        baseline_per_point
                        >= baseline_min
                    )

                    highlight_mask &= (
                        baseline_per_point
                        <= baseline_max
                    )

                if highlight_wavelength_index is not None:

                    has_highlight_filter = True

                    highlight_mask &= (
                        wavelength_indices
                        == int(
                            highlight_wavelength_index
                        )
                    )

                if not has_highlight_filter:

                    highlight_mask[:] = False

                # ============================================================
                # Add rows to output CSV
                # ============================================================

                for point_i in xrange(
                        expected_size):

                    diagnostic_rows.append({
                        "star": science_name,
                        "sequence": sequence_name,
                        "period": period_value,
                        "point_id": int(point_i),
                        "night": str(
                            night_per_point[
                                point_i
                            ]
                        ),
                        "mjd": mjd_per_point[
                            point_i
                        ],
                        "telescope_pair": str(
                            pair_per_point[
                                point_i
                            ]
                        ),
                        "baseline_m": baseline_per_point[
                            point_i
                        ],
                        "wavelength_index": int(
                            wavelength_indices[
                                point_i
                            ]
                        ),
                        "wavelength_um": (
                            wavelength_per_point[
                                point_i
                            ]
                            * 1.0E6
                        ),
                        "spatial_frequency_rad_inv":
                            spatial_frequency[
                                point_i
                            ],
                        "c_scale": c_per_point[
                            point_i
                        ],
                        "vis2_corrected": vis2_corrected[
                            point_i
                        ],
                        "e_vis2_corrected":
                            e_vis2_corrected[
                                point_i
                            ],
                        "model_vis2": model_vis2[
                            point_i
                        ],
                        "raw_residual": raw_residual[
                            point_i
                        ],
                        "standardized_residual":
                            standardized_residual[
                                point_i
                            ],
                        "oifits_flag": bool(
                            bad_flag[
                                point_i
                            ]
                        ),
                        "large_raw_residual": bool(
                            large_raw_residual[
                                point_i
                            ]
                        ),
                        "large_sigma_residual": bool(
                            large_sigma_residual[
                                point_i
                            ]
                        ),
                        "candidate_remove": bool(
                            removal_candidate[
                                point_i
                            ]
                        ),
                        "highlighted": bool(
                            highlight_mask[
                                point_i
                            ]
                        )
                    })

                # ============================================================
                # Figure
                # ============================================================

                fig, axes = plt.subplots(
                    2,
                    2
                )

                fig.set_size_inches(
                    16,
                    11
                )

                axes = axes.flatten()

                # ------------------------------------------------------------
                # Smooth visibility model curves
                # ------------------------------------------------------------

                model_frequency = np.linspace(
                    1.0E6,
                    2.5E8,
                    10000
                )

                for wavelength_i in xrange(
                        n_wavelengths):

                    model_curve = rdiam.calc_vis2(
                        model_frequency,
                        model_ldd,
                        1.0,
                        (
                            len(model_frequency),
                        ),
                        u_lambdas[
                            wavelength_i
                        ],
                        s_lambdas[
                            wavelength_i
                        ]
                    )

                    axes[0].plot(
                        model_frequency,
                        model_curve,
                        color="0.75",
                        linewidth=0.7,
                        zorder=1
                    )

                    axes[1].plot(
                        model_frequency,
                        model_curve,
                        color="0.75",
                        linewidth=0.7,
                        zorder=1
                    )

                # ------------------------------------------------------------
                # Top-left: colour by night
                # ------------------------------------------------------------

                night_colours = plot_grouped_visibilities(
                    axes[0],
                    night_per_point,
                    "Visibility points coloured by night",
                    spatial_frequency,
                    vis2_corrected,
                    e_vis2_corrected,
                    valid_data,
                    cm.get_cmap(
                        "jet"
                    )
                )

                # ------------------------------------------------------------
                # Top-right: colour by telescope pair
                # ------------------------------------------------------------

                pair_colours = plot_grouped_visibilities(
                    axes[1],
                    pair_per_point,
                    "Visibility points coloured by telescope pair",
                    spatial_frequency,
                    vis2_corrected,
                    e_vis2_corrected,
                    valid_data,
                    cm.get_cmap(
                        "jet"
                    )
                )

                # ------------------------------------------------------------
                # Bottom-left: raw residual by wavelength channel
                # ------------------------------------------------------------

                wavelength_colour_map = cm.get_cmap(
                    "viridis"
                )

                for wavelength_i in xrange(
                        n_wavelengths):

                    wavelength_mask = (
                        np.isfinite(
                            raw_residual
                        )
                        & (
                            wavelength_indices
                            == wavelength_i
                        )
                    )

                    if not np.any(
                            wavelength_mask):

                        continue

                    denominator = max(
                        1.0,
                        float(
                            n_wavelengths - 1
                        )
                    )

                    wavelength_colour = (
                        wavelength_colour_map(
                            float(wavelength_i)
                            / denominator
                        )
                    )

                    axes[2].scatter(
                        spatial_frequency[
                            wavelength_mask
                        ],
                        raw_residual[
                            wavelength_mask
                        ],
                        s=18,
                        color=wavelength_colour,
                        label=(
                            "ch %i: %.3f um"
                            % (
                                wavelength_i,
                                wavelengths[
                                    wavelength_i
                                ] * 1.0E6
                            )
                        )
                    )

                axes[2].axhline(
                    0.0,
                    linestyle="-",
                    linewidth=0.7,
                    color="0.3"
                )

                axes[2].axhline(
                    raw_residual_threshold,
                    linestyle="--",
                    linewidth=0.8,
                    color="red"
                )

                axes[2].axhline(
                    -raw_residual_threshold,
                    linestyle="--",
                    linewidth=0.8,
                    color="red"
                )

                axes[2].set_title(
                    "Raw visibility residual by wavelength channel"
                )

                axes[2].set_xlabel(
                    r"Spatial frequency (rad$^{-1}$)"
                )

                axes[2].set_ylabel(
                    r"$V^2_{\rm obs}-V^2_{\rm model}$"
                )

                axes[2].grid()

                axes[2].legend(
                    loc="best",
                    fontsize=7,
                    ncol=2
                )

                # ------------------------------------------------------------
                # Bottom-right: standardized residual versus baseline
                # ------------------------------------------------------------

                unique_pairs = sorted(
                    set(
                        pair_per_point.tolist()
                    )
                )

                for pair_value in unique_pairs:

                    pair_mask = (
                        np.isfinite(
                            standardized_residual
                        )
                        & (
                            pair_per_point
                            == pair_value
                        )
                    )

                    if not np.any(
                            pair_mask):

                        continue

                    pair_colour = pair_colours.get(
                        pair_value,
                        "0.4"
                    )

                    axes[3].scatter(
                        baseline_per_point[
                            pair_mask
                        ],
                        standardized_residual[
                            pair_mask
                        ],
                        s=20,
                        color=pair_colour,
                        label=str(
                            pair_value
                        )
                    )

                axes[3].axhline(
                    0.0,
                    linestyle="-",
                    linewidth=0.7,
                    color="0.3"
                )

                axes[3].axhline(
                    sigma_threshold,
                    linestyle="--",
                    linewidth=0.8,
                    color="red"
                )

                axes[3].axhline(
                    -sigma_threshold,
                    linestyle="--",
                    linewidth=0.8,
                    color="red"
                )

                axes[3].set_title(
                    "Standardized residual versus projected baseline"
                )

                axes[3].set_xlabel(
                    "Projected baseline (m)"
                )

                axes[3].set_ylabel(
                    r"$(V^2_{\rm obs}-V^2_{\rm model})/\sigma$"
                )

                axes[3].grid()

                axes[3].legend(
                    loc="best",
                    fontsize=7,
                    ncol=2
                )

                # ============================================================
                # Mark flagged measurements
                # ============================================================

                flagged_valid = (
                    bad_flag
                    & valid_data
                )

                if np.any(
                        flagged_valid):

                    axes[0].scatter(
                        spatial_frequency[
                            flagged_valid
                        ],
                        vis2_corrected[
                            flagged_valid
                        ],
                        marker="x",
                        s=50,
                        color="black",
                        zorder=7
                    )

                    axes[1].scatter(
                        spatial_frequency[
                            flagged_valid
                        ],
                        vis2_corrected[
                            flagged_valid
                        ],
                        marker="x",
                        s=50,
                        color="black",
                        zorder=7
                    )

                # ============================================================
                # Mark removal candidates
                # ============================================================

                candidate_valid = (
                    removal_candidate
                    & valid_data
                )

                if np.any(
                        candidate_valid):

                    axes[0].scatter(
                        spatial_frequency[
                            candidate_valid
                        ],
                        vis2_corrected[
                            candidate_valid
                        ],
                        marker="o",
                        s=80,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.2,
                        zorder=8
                    )

                    axes[1].scatter(
                        spatial_frequency[
                            candidate_valid
                        ],
                        vis2_corrected[
                            candidate_valid
                        ],
                        marker="o",
                        s=80,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.2,
                        zorder=8
                    )

                    axes[2].scatter(
                        spatial_frequency[
                            candidate_valid
                        ],
                        raw_residual[
                            candidate_valid
                        ],
                        marker="o",
                        s=80,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.2,
                        zorder=8
                    )

                    candidate_with_sigma = (
                        candidate_valid
                        & np.isfinite(
                            standardized_residual
                        )
                    )

                    axes[3].scatter(
                        baseline_per_point[
                            candidate_with_sigma
                        ],
                        standardized_residual[
                            candidate_with_sigma
                        ],
                        marker="o",
                        s=80,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.2,
                        zorder=8
                    )

                # ============================================================
                # Highlight requested subset
                # ============================================================

                highlighted_valid = (
                    highlight_mask
                    & valid_data
                )

                if np.any(
                        highlighted_valid):

                    axes[0].scatter(
                        spatial_frequency[
                            highlighted_valid
                        ],
                        vis2_corrected[
                            highlighted_valid
                        ],
                        marker="s",
                        s=100,
                        facecolors="none",
                        edgecolors="red",
                        linewidths=1.4,
                        zorder=9
                    )

                    axes[1].scatter(
                        spatial_frequency[
                            highlighted_valid
                        ],
                        vis2_corrected[
                            highlighted_valid
                        ],
                        marker="s",
                        s=100,
                        facecolors="none",
                        edgecolors="red",
                        linewidths=1.4,
                        zorder=9
                    )

                    axes[2].scatter(
                        spatial_frequency[
                            highlighted_valid
                        ],
                        raw_residual[
                            highlighted_valid
                        ],
                        marker="s",
                        s=100,
                        facecolors="none",
                        edgecolors="red",
                        linewidths=1.4,
                        zorder=9
                    )

                    highlighted_sigma = (
                        highlighted_valid
                        & np.isfinite(
                            standardized_residual
                        )
                    )

                    axes[3].scatter(
                        baseline_per_point[
                            highlighted_sigma
                        ],
                        standardized_residual[
                            highlighted_sigma
                        ],
                        marker="s",
                        s=100,
                        facecolors="none",
                        edgecolors="red",
                        linewidths=1.4,
                        zorder=9
                    )

                # ============================================================
                # Annotate strongest candidates
                # ============================================================

                candidate_indices = np.where(
                    removal_candidate
                )[0]

                if len(candidate_indices) > 0:

                    candidate_strength = np.abs(
                        raw_residual[
                            candidate_indices
                        ]
                    )

                    finite_sigma_candidate = np.isfinite(
                        standardized_residual[
                            candidate_indices
                        ]
                    )

                    candidate_strength[
                        finite_sigma_candidate
                    ] *= np.maximum(
                        1.0,
                        np.abs(
                            standardized_residual[
                                candidate_indices[
                                    finite_sigma_candidate
                                ]
                            ]
                        )
                    )

                    sorted_candidate_indices = (
                        candidate_indices[
                            np.argsort(
                                candidate_strength
                            )[::-1]
                        ]
                    )

                    sorted_candidate_indices = (
                        sorted_candidate_indices[
                            :int(max_annotations)
                        ]
                    )

                    for point_i in sorted_candidate_indices:

                        point_label = (
                            "P%i"
                            % point_i
                        )

                        axes[2].annotate(
                            point_label,
                            xy=(
                                spatial_frequency[
                                    point_i
                                ],
                                raw_residual[
                                    point_i
                                ]
                            ),
                            xytext=(3, 3),
                            textcoords="offset points",
                            fontsize=6
                        )

                        if np.isfinite(
                                standardized_residual[
                                    point_i
                                ]):

                            axes[3].annotate(
                                point_label,
                                xy=(
                                    baseline_per_point[
                                        point_i
                                    ],
                                    standardized_residual[
                                        point_i
                                    ]
                                ),
                                xytext=(3, 3),
                                textcoords="offset points",
                                fontsize=6
                            )

                # ============================================================
                # Candidate text
                # ============================================================

                candidate_text = []

                candidate_text.append(
                    "Candidate rule: FLAG or "
                    "(|raw residual| >= %.3f and |residual/sigma| >= %.1f)"
                    % (
                        raw_residual_threshold,
                        sigma_threshold
                    )
                )

                if len(candidate_indices) == 0:

                    candidate_text.append(
                        "No removal candidates."
                    )

                else:

                    candidate_text.append(
                        "ID | night | pair | B(m) | channel | raw | sigma"
                    )

                    for point_i in sorted_candidate_indices[
                            :8]:

                        candidate_text.append(
                            "P%i | %s | %s | %.1f | %i | %.3f | %s"
                            % (
                                point_i,
                                night_per_point[
                                    point_i
                                ],
                                pair_per_point[
                                    point_i
                                ],
                                baseline_per_point[
                                    point_i
                                ],
                                wavelength_indices[
                                    point_i
                                ],
                                raw_residual[
                                    point_i
                                ],
                                (
                                    "%.2f"
                                    % standardized_residual[
                                        point_i
                                    ]
                                    if np.isfinite(
                                        standardized_residual[
                                            point_i
                                        ]
                                    )
                                    else "nan"
                                )
                            )
                        )

                # ============================================================
                # Final figure formatting
                # ============================================================

                if np.isfinite(
                        fitted_ldd_error):

                    diameter_text = (
                        "%.4f +/- %.4f mas"
                        % (
                            model_ldd,
                            fitted_ldd_error
                        )
                    )

                else:

                    diameter_text = (
                        "%.4f mas"
                        % model_ldd
                    )

                fig.suptitle(
                    (
                        "%s, %s, %s\n"
                        "%s LDD = %s; "
                        "%i measurements; %i removal candidates"
                    )
                    % (
                        science_name,
                        sequence_name,
                        str(period_value),
                        diameter_source,
                        diameter_text,
                        expected_size,
                        int(
                            np.sum(
                                removal_candidate
                            )
                        )
                    ),
                    fontsize=14
                )

                fig.text(
                    0.5,
                    0.01,
                    "\n".join(
                        candidate_text
                    ),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=7,
                    family="monospace"
                )

                fig.tight_layout(
                    rect=[
                        0.0,
                        0.11,
                        1.0,
                        0.93
                    ]
                )

                # ============================================================
                # Save page and PNG
                # ============================================================

                pdf.savefig(
                    fig
                )

                safe_sequence = (
                    clean_target_name_for_plot(
                        sequence_name
                    )
                )

                individual_filename = (
                    "%s_%s_%s_visibility_diagnostic.png"
                    % (
                        science_clean,
                        safe_sequence,
                        str(period_value)
                    )
                )

                individual_output = os.path.join(
                    individual_directory,
                    individual_filename
                )

                fig.savefig(
                    individual_output,
                    dpi=200
                )

                plt.close(
                    fig
                )

                # ============================================================
                # Separate V2 > 1 plot
                # ============================================================

                v2_figure, v2_axis = plt.subplots()

                v2_figure.set_size_inches(
                    13,
                    7
                )

                point_number = np.arange(
                    expected_size
                )

                v2_at_or_below_one = (
                    finite_vis2
                    & ~v2_above_one
                )

                if np.any(
                        v2_at_or_below_one):

                    v2_axis.scatter(
                        point_number[
                            v2_at_or_below_one
                        ],
                        vis2_corrected[
                            v2_at_or_below_one
                        ],
                        s=22,
                        color="0.35",
                        label=r"$V^2 \leq 1$",
                        zorder=3
                    )

                if np.any(
                        v2_above_one):

                    v2_axis.scatter(
                        point_number[
                            v2_above_one
                        ],
                        vis2_corrected[
                            v2_above_one
                        ],
                        s=55,
                        marker="o",
                        facecolors="none",
                        edgecolors="red",
                        linewidths=1.4,
                        label=r"$V^2 > 1$",
                        zorder=5
                    )

                v2_axis.axhline(
                    1.0,
                    linestyle="--",
                    linewidth=1.0,
                    color="red",
                    label=r"$V^2 = 1$",
                    zorder=2
                )

                if max_point_index >= 0:

                    v2_axis.scatter(
                        [max_point_index],
                        [max_vis2],
                        marker="*",
                        s=180,
                        color="black",
                        label="Maximum",
                        zorder=7
                    )

                    v2_axis.annotate(
                        "P%i: %.4f"
                        % (
                            max_point_index,
                            max_vis2
                        ),
                        xy=(
                            max_point_index,
                            max_vis2
                        ),
                        xytext=(6, 7),
                        textcoords="offset points",
                        fontsize=9,
                        fontweight="bold"
                    )

                if n_v2_points > 0:

                    finite_vis2_values = vis2_corrected[
                        finite_vis2
                    ]

                    v2_y_min = min(
                        1.0,
                        float(
                            np.min(
                                finite_vis2_values
                            )
                        )
                    )

                    v2_y_max = max(
                        1.0,
                        float(
                            np.max(
                                finite_vis2_values
                            )
                        )
                    )

                    v2_y_margin = max(
                        0.05,
                        0.08
                        * (
                            v2_y_max
                            - v2_y_min
                        )
                    )

                    v2_axis.set_ylim(
                        [
                            v2_y_min - v2_y_margin,
                            v2_y_max + v2_y_margin
                        ]
                    )

                v2_axis.set_xlim(
                    [
                        -1,
                        max(
                            1,
                            expected_size
                        )
                    ]
                )

                v2_axis.set_xlabel(
                    "Point ID"
                )

                v2_axis.set_ylabel(
                    r"Corrected visibility$^2$"
                )

                v2_axis.set_title(
                    (
                        "%s, %s, %s: corrected V2 values"
                    )
                    % (
                        science_name,
                        sequence_name,
                        str(period_value)
                    )
                )

                v2_axis.grid()

                v2_axis.legend(
                    loc="best",
                    fontsize=9
                )

                if max_point_index >= 0:

                    maximum_detail_text = (
                        "Maximum point details:\n"
                        "  night: %s\n"
                        "  pair: %s\n"
                        "  baseline: %.2f m\n"
                        "  channel: %i\n"
                        "  wavelength: %.4f um"
                        % (
                            max_night,
                            max_pair,
                            max_baseline,
                            max_wavelength_index,
                            max_wavelength_um
                        )
                    )

                else:

                    maximum_detail_text = (
                        "Maximum point details:\n"
                        "  no finite V2 values"
                    )

                if np.isfinite(
                        fraction_v2_above_one):

                    fraction_text = (
                        "%.4f (%.2f%%)"
                        % (
                            fraction_v2_above_one,
                            100.0
                            * fraction_v2_above_one
                        )
                    )

                else:

                    fraction_text = "nan"

                v2_summary_text = (
                    "V2 summary\n"
                    "-----------------------------\n"
                    "Maximum V2: %s\n"
                    "Maximum point: %s\n"
                    "Any V2 > 1: %s\n"
                    "Number of finite points: %i\n"
                    "Number of points V2 > 1: %i\n"
                    "Fraction of points V2 > 1: %s\n\n"
                    "%s"
                    % (
                        (
                            "%.6f"
                            % max_vis2
                            if np.isfinite(
                                max_vis2
                            )
                            else "nan"
                        ),
                        (
                            "P%i"
                            % max_point_index
                            if max_point_index >= 0
                            else "none"
                        ),
                        (
                            "YES"
                            if n_v2_above_one > 0
                            else "NO"
                        ),
                        n_v2_points,
                        n_v2_above_one,
                        fraction_text,
                        maximum_detail_text
                    )
                )

                v2_figure.subplots_adjust(
                    left=0.08,
                    right=0.67,
                    bottom=0.12,
                    top=0.88
                )

                v2_figure.text(
                    0.70,
                    0.84,
                    v2_summary_text,
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize=10,
                    family="monospace",
                    bbox=dict(
                        boxstyle="round",
                        facecolor="0.95",
                        edgecolor="0.5"
                    )
                )

                v2_pdf.savefig(
                    v2_figure
                )

                safe_period = (
                    clean_target_name_for_plot(
                        str(period_value)
                    )
                )

                v2_individual_filename = (
                    "%s_%s_%s_v2_above_one.png"
                    % (
                        science_clean,
                        safe_sequence,
                        safe_period
                    )
                )

                v2_individual_output = os.path.join(
                    v2_individual_directory,
                    v2_individual_filename
                )

                v2_figure.savefig(
                    v2_individual_output,
                    dpi=200
                )

                plt.close(
                    v2_figure
                )

                n_created += 1

                print(
                    "Saved diagnostic for %s"
                    % science_name
                )

            except Exception as error:

                n_failed += 1

                print("")
                print(
                    "FAILED visibility diagnostic for %s"
                    % science_name
                )

                print(
                    "Error: %s"
                    % str(error)
                )

                traceback.print_exc()

                plt.close(
                    "all"
                )

    # ====================================================================
    # Global V2 > 1 overview
    # ====================================================================

    if len(
            v2_summary_rows) > 0:

        overview_rows = sorted(
            v2_summary_rows,
            key=lambda row: (
                row[
                    "fraction_vis2_above_one"
                ]
                if np.isfinite(
                    row[
                        "fraction_vis2_above_one"
                    ]
                )
                else -1.0
            ),
            reverse=True
        )

        overview_labels = []
        overview_percentages = []
        overview_annotations = []

        for overview_row in overview_rows:

            overview_labels.append(
                "%s | %s | %s"
                % (
                    overview_row[
                        "star"
                    ],
                    overview_row[
                        "sequence"
                    ],
                    str(
                        overview_row[
                            "period"
                        ]
                    )
                )
            )

            overview_fraction = overview_row[
                "fraction_vis2_above_one"
            ]

            if np.isfinite(
                    overview_fraction):

                overview_percentage = (
                    100.0
                    * overview_fraction
                )

            else:

                overview_percentage = 0.0

            overview_percentages.append(
                overview_percentage
            )

            overview_annotations.append(
                (
                    "%i/%i; max=%.4f (%s)"
                    % (
                        overview_row[
                            "n_vis2_above_one"
                        ],
                        overview_row[
                            "n_finite_vis2_points"
                        ],
                        overview_row[
                            "max_vis2_corrected"
                        ],
                        overview_row[
                            "max_point_id"
                        ]
                    )
                )
            )

        overview_y = np.arange(
            len(
                overview_rows
            )
        )

        overview_height = max(
            6.0,
            0.45
            * len(
                overview_rows
            )
            + 2.0
        )

        overview_figure, overview_axis = plt.subplots()

        overview_figure.set_size_inches(
            14,
            overview_height
        )

        overview_bars = overview_axis.barh(
            overview_y,
            overview_percentages,
            color="0.55"
        )

        overview_axis.set_yticks(
            overview_y
        )

        overview_axis.set_yticklabels(
            overview_labels,
            fontsize=8
        )

        overview_axis.invert_yaxis()

        overview_axis.set_xlabel(
            r"Fraction of finite corrected $V^2$ points above 1 (percent)"
        )

        overview_axis.set_title(
            (
                "Corrected V2 > 1 overview\n"
                "Labels show N(V2 > 1)/N(total), maximum V2 and maximum point"
            )
        )

        maximum_overview_percentage = max(
            overview_percentages
        )

        overview_x_max = max(
            5.0,
            maximum_overview_percentage
            * 1.35
            + 2.0
        )

        overview_axis.set_xlim(
            [
                0.0,
                overview_x_max
            ]
        )

        overview_axis.grid(
            axis="x"
        )

        for overview_i, overview_bar in enumerate(
                overview_bars):

            annotation_x = max(
                0.3,
                overview_percentages[
                    overview_i
                ]
                + 0.3
            )

            overview_axis.text(
                annotation_x,
                overview_bar.get_y()
                + overview_bar.get_height()
                / 2.0,
                overview_annotations[
                    overview_i
                ],
                verticalalignment="center",
                horizontalalignment="left",
                fontsize=8
            )

        overview_figure.tight_layout()

        v2_pdf.savefig(
            overview_figure
        )

        overview_figure.savefig(
            v2_overview_png,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(
            overview_figure
        )

    v2_pdf.close()

    # ====================================================================
    # Save diagnostic CSV files
    # ====================================================================

    diagnostic_table = pd.DataFrame(
        diagnostic_rows
    )

    diagnostic_table.to_csv(
        all_points_csv,
        index=False
    )

    if (
        len(diagnostic_table) > 0
        and "candidate_remove"
        in diagnostic_table.columns
    ):

        candidate_table = diagnostic_table[
            diagnostic_table[
                "candidate_remove"
            ]
        ].copy()

    else:

        candidate_table = pd.DataFrame()

    candidate_table.to_csv(
        candidate_csv,
        index=False
    )

    v2_summary_table = pd.DataFrame(
        v2_summary_rows
    )

    v2_summary_table.to_csv(
        v2_summary_csv,
        index=False
    )

    print("")
    print("=" * 79)
    print("Visibility diagnostics finished")
    print(
        "Created pages: %i"
        % n_created
    )
    print(
        "Failed pages: %i"
        % n_failed
    )
    print("PDF:")
    print(output_file)
    print("Complete point table:")
    print(all_points_csv)
    print("Removal-candidate table:")
    print(candidate_csv)
    print("V2 > 1 summary PDF:")
    print(v2_summary_pdf)
    print("V2 > 1 summary CSV:")
    print(v2_summary_csv)
    print("V2 > 1 overview PNG:")
    print(v2_overview_png)
    print("Individual V2 > 1 plots:")
    print(v2_individual_directory)
    print("=" * 79)

    if n_created == 0:

        raise RuntimeError(
            "No visibility diagnostic pages were generated"
        )

    return output_file

def extract_constant_mass_points(
        basti_folder,
        masses_to_follow=None):

    """
    Extract the point nearest to each requested initial mass
    from every BaSTI constant-age isochrone.

    This approximates constant-initial-mass evolutionary sequences
    using the available isochrones.
    """

    if masses_to_follow is None:

        masses_to_follow = [
            0.6,
            0.8,
            1.0,
            1.2,
            1.5,
            2.0,
            3.0,
            5.0
        ]

    column_names = [
        "M_ini",
        "M_fin",
        "logL",
        "logTe",
        "U",
        "BX",
        "B",
        "V",
        "R",
        "I",
        "J",
        "H",
        "K",
        "Lprime",
        "L",
        "M"
    ]

    iso_files = []

    for root, directories, filenames in os.walk(
            basti_folder):

        for filename in filenames:

            if filename.endswith(
                    ".isc_john"):

                iso_files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    iso_files.sort()

    output_rows = []

    for iso_file in iso_files:

        age_myr = np.nan

        with open(
                iso_file,
                "r") as handle:

            for line in handle:

                if "Age (Myr)" in line:

                    try:

                        age_myr = float(
                            line.split(
                                "Age (Myr) ="
                            )[1]
                            .strip()
                            .split()[0]
                        )

                    except Exception:

                        age_myr = np.nan

                    break

        track = pd.read_csv(
            iso_file,
            delim_whitespace=True,
            names=column_names,
            comment="#",
            dtype=float
        )

        track["B-V"] = (
            track["B"]
            - track["V"]
        )

        for requested_mass in masses_to_follow:

            valid = (
                np.isfinite(
                    track["M_ini"]
                )
                & np.isfinite(
                    track["B-V"]
                )
                & np.isfinite(
                    track["V"]
                )
            )

            valid_track = track.loc[
                valid
            ]

            if len(valid_track) == 0:

                continue

            differences = np.abs(
                valid_track["M_ini"]
                - float(requested_mass)
            )

            nearest_index = differences.idxmin()

            nearest_row = valid_track.loc[
                nearest_index
            ]

            output_rows.append({
                "age_myr": age_myr,
                "requested_mass": float(
                    requested_mass
                ),
                "actual_M_ini": float(
                    nearest_row["M_ini"]
                ),
                "M_fin": float(
                    nearest_row["M_fin"]
                ),
                "logL": float(
                    nearest_row["logL"]
                ),
                "logTe": float(
                    nearest_row["logTe"]
                ),
                "B-V": float(
                    nearest_row["B-V"]
                ),
                "Mv": float(
                    nearest_row["V"]
                ),
                "filename": iso_file
            })

    output_table = pd.DataFrame(
        output_rows
    )

    return output_table

def plot_basti_constant_mass_evolution(
        basti_folder="data/basti",
        masses_to_plot=None,
        feh=0.062,
        feh_tolerance=0.02,
        absolute_mass_tolerance=0.03,
        relative_mass_tolerance=0.02,
        minimum_points=3,
        output_csv="paper/basti_constant_mass_tracks.csv",
        axis=None):
    """
    Reconstruct approximately constant-initial-mass evolutionary
    sequences from a collection of BaSTI constant-age isochrones.

    Each .isc_john file corresponds to one age and contains many
    initial masses. For every requested mass, this function finds
    the nearest M_ini value in every available isochrone.

    Parameters
    ----------
    basti_folder : str
        Directory containing the BaSTI .isc_john files.

    masses_to_plot : list
        Initial stellar masses in solar masses.

    feh : float or None
        Requested [M/H]. Set to None to accept all metallicities.

    feh_tolerance : float
        Allowed difference between requested and file [M/H].

    absolute_mass_tolerance : float
        Minimum allowed difference in initial mass, in solar masses.

    relative_mass_tolerance : float
        Relative mass tolerance. The final tolerance is:

            max(
                absolute_mass_tolerance,
                relative_mass_tolerance * requested_mass
            )

    minimum_points : int
        Minimum number of ages required to draw a mass sequence.

    output_csv : str or None
        CSV containing every selected mass-age point.

    axis : matplotlib axis or None
        Axis where the tracks are drawn. Uses plt.gca() when None.

    Returns
    -------
    mass_track_table : pandas.DataFrame
        Table containing all accepted points.
    """

    if masses_to_plot is None:

        masses_to_plot = [
            0.6,
            0.8,
            1.0,
            1.2,
            1.5,
            2.0,
            3.0,
            5.0
        ]

    if axis is None:

        axis = plt.gca()

    # =====================================================================
    # Actual BaSTI Johnson-Cousins columns
    # =====================================================================

    column_names = [
        "M_ini",
        "M_fin",
        "logL",
        "logTe",
        "U",
        "BX",
        "B",
        "V",
        "R",
        "I",
        "J",
        "H",
        "K",
        "Lprime",
        "L",
        "M"
    ]

    # =====================================================================
    # Find all isochrone files recursively
    # =====================================================================

    iso_files = []

    for root_directory, directory_names, filenames in os.walk(
            basti_folder):

        for filename in filenames:

            if filename.endswith(
                    ".isc_john"):

                iso_files.append(
                    os.path.join(
                        root_directory,
                        filename
                    )
                )

    iso_files = sorted(
        list(
            set(
                iso_files
            )
        )
    )

    if len(iso_files) == 0:

        raise IOError(
            "No .isc_john files found inside %s"
            % basti_folder
        )

    print("")
    print("=" * 79)
    print("Reading BaSTI isochrones for constant-mass evolution")
    print(
        "Found %i .isc_john files"
        % len(iso_files)
    )
    print("=" * 79)

    # =====================================================================
    # Read all constant-age isochrones
    # =====================================================================

    isochrone_entries = []

    for iso_file in iso_files:

        age_myr = np.nan
        file_mh = np.nan
        number_of_columns = None

        # -----------------------------------------------------------------
        # Read header information
        # -----------------------------------------------------------------

        with open(
                iso_file,
                "r") as input_handle:

            for line in input_handle:

                stripped_line = line.strip()

                if stripped_line == "":

                    continue

                if "Age (Myr)" in stripped_line:

                    try:

                        age_text = (
                            stripped_line
                            .split(
                                "Age (Myr) ="
                            )[1]
                            .strip()
                            .split()[0]
                        )

                        age_myr = float(
                            age_text
                        )

                    except Exception:

                        age_myr = np.nan

                if "[M/H]" in stripped_line:

                    try:

                        metallicity_text = (
                            stripped_line
                            .split(
                                "[M/H] ="
                            )[1]
                            .split(
                                "Z ="
                            )[0]
                            .strip()
                        )

                        file_mh = float(
                            metallicity_text
                        )

                    except Exception:

                        file_mh = np.nan

                # First numerical row.
                if not stripped_line.startswith(
                        "#"):

                    number_of_columns = len(
                        stripped_line.split()
                    )

                    break

        if not np.isfinite(
                age_myr):

            print(
                "WARNING: age not found in %s"
                % iso_file
            )

            continue

        if number_of_columns is None:

            print(
                "WARNING: no numerical data in %s"
                % iso_file
            )

            continue

        if number_of_columns != len(
                column_names):

            print(
                "WARNING: skipping %s"
                % iso_file
            )

            print(
                "Found %i columns; expected %i"
                % (
                    number_of_columns,
                    len(column_names)
                )
            )

            continue

        # -----------------------------------------------------------------
        # Filter the chemical composition
        # -----------------------------------------------------------------

        if (
            feh is not None
            and np.isfinite(
                file_mh
            )
            and abs(
                file_mh - float(feh)
            ) > float(feh_tolerance)
        ):

            continue

        # -----------------------------------------------------------------
        # Read numerical data
        # -----------------------------------------------------------------

        isochrone = pd.read_csv(
            iso_file,
            delim_whitespace=True,
            names=column_names,
            comment="#",
            dtype=float
        )

        # Required colour-magnitude quantities.
        isochrone[
            "B-V"
        ] = (
            isochrone[
                "B"
            ]
            - isochrone[
                "V"
            ]
        )

        isochrone[
            "Mv"
        ] = isochrone[
            "V"
        ]

        valid_rows = (
            np.isfinite(
                isochrone[
                    "M_ini"
                ]
            )
            & np.isfinite(
                isochrone[
                    "M_fin"
                ]
            )
            & np.isfinite(
                isochrone[
                    "B-V"
                ]
            )
            & np.isfinite(
                isochrone[
                    "Mv"
                ]
            )
        )

        isochrone = isochrone.loc[
            valid_rows
        ].copy()

        if len(isochrone) == 0:

            continue

        # Sort by initial mass.
        isochrone = isochrone.sort_values(
            "M_ini"
        )

        isochrone_entries.append({
            "age_myr": float(
                age_myr
            ),
            "mh": float(
                file_mh
            ),
            "filename": iso_file,
            "isochrone": isochrone
        })

    if len(isochrone_entries) == 0:

        raise RuntimeError(
            "No valid BaSTI isochrones were read"
        )

    # Sort files chronologically.
    isochrone_entries.sort(
        key=lambda entry: entry[
            "age_myr"
        ]
    )

    print(
        "Accepted %i BaSTI ages"
        % len(isochrone_entries)
    )

    print(
        "Age range: %.1f - %.1f Myr"
        % (
            isochrone_entries[0][
                "age_myr"
            ],
            isochrone_entries[-1][
                "age_myr"
            ]
        )
    )

    # =====================================================================
    # Extract equal-initial-mass points
    # =====================================================================

    output_rows = []

    for requested_mass in masses_to_plot:

        requested_mass = float(
            requested_mass
        )

        allowed_mass_difference = max(
            float(
                absolute_mass_tolerance
            ),
            float(
                relative_mass_tolerance
            )
            * requested_mass
        )

        for entry in isochrone_entries:

            isochrone = entry[
                "isochrone"
            ]

            masses_available = np.asarray(
                isochrone[
                    "M_ini"
                ],
                dtype=float
            )

            if len(masses_available) == 0:

                continue

            mass_differences = np.abs(
                masses_available
                - requested_mass
            )

            nearest_position = int(
                np.argmin(
                    mass_differences
                )
            )

            nearest_difference = float(
                mass_differences[
                    nearest_position
                ]
            )

            # Do not associate a completely different mass when
            # the requested star no longer exists in an old isochrone.
            if (
                nearest_difference
                > allowed_mass_difference
            ):

                continue

            nearest_row = isochrone.iloc[
                nearest_position
            ]

            output_rows.append({
                "requested_M_ini": requested_mass,
                "actual_M_ini": float(
                    nearest_row[
                        "M_ini"
                    ]
                ),
                "mass_difference": nearest_difference,
                "M_fin": float(
                    nearest_row[
                        "M_fin"
                    ]
                ),
                "age_myr": float(
                    entry[
                        "age_myr"
                    ]
                ),
                "age_gyr": float(
                    entry[
                        "age_myr"
                    ]
                ) / 1000.0,
                "B-V": float(
                    nearest_row[
                        "B-V"
                    ]
                ),
                "Mv": float(
                    nearest_row[
                        "Mv"
                    ]
                ),
                "logL": float(
                    nearest_row[
                        "logL"
                    ]
                ),
                "logTe": float(
                    nearest_row[
                        "logTe"
                    ]
                ),
                "mh": entry[
                    "mh"
                ],
                "filename": entry[
                    "filename"
                ]
            })

    mass_track_table = pd.DataFrame(
        output_rows
    )

    if len(mass_track_table) == 0:

        raise RuntimeError(
            "No constant-mass points passed the mass tolerance"
        )

    # =====================================================================
    # Save selected points
    # =====================================================================

    if output_csv is not None:

        output_directory = os.path.dirname(
            output_csv
        )

        if (
            output_directory != ""
            and not os.path.exists(
                output_directory
            )
        ):

            os.makedirs(
                output_directory
            )

        mass_track_table.to_csv(
            output_csv,
            index=False
        )

        print(
            "Saved constant-mass table:"
        )

        print(
            output_csv
        )

    # =====================================================================
    # Plot one evolutionary sequence per initial mass
    # =====================================================================

    unique_masses = sorted(
        mass_track_table[
            "requested_M_ini"
        ].unique()
    )

    try:

        colour_map = cm.get_cmap(
            "viridis"
        )

    except Exception:

        colour_map = cm.get_cmap(
            "jet"
        )

    plotted_masses = 0

    for mass_i, requested_mass in enumerate(
            unique_masses):

        mass_data = mass_track_table[
            mass_track_table[
                "requested_M_ini"
            ]
            == requested_mass
        ].copy()

        mass_data = mass_data.sort_values(
            "age_myr"
        )

        if len(mass_data) < int(
                minimum_points):

            print(
                "Skipping %.2f M_sun: only %i valid ages"
                % (
                    requested_mass,
                    len(mass_data)
                )
            )

            continue

        if len(unique_masses) == 1:

            curve_colour = colour_map(
                0.5
            )

        else:

            curve_colour = colour_map(
                float(
                    mass_i
                )
                / float(
                    len(unique_masses) - 1
                )
            )

        axis.plot(
            mass_data[
                "B-V"
            ],
            mass_data[
                "Mv"
            ],
            linestyle="-",
            linewidth=1.2,
            color=curve_colour,
            label=(
                r"$M_{\rm ini}=%.2f\,M_{\odot}$"
                % requested_mass
            ),
            zorder=2
        )

        # Show individual age points.
        marker_step = max(
            1,
            int(
                len(mass_data) / 20
            )
        )

        axis.plot(
            mass_data[
                "B-V"
            ].values[
                ::marker_step
            ],
            mass_data[
                "Mv"
            ].values[
                ::marker_step
            ],
            linestyle="None",
            marker=".",
            markersize=3,
            color=curve_colour,
            zorder=3
        )

        plotted_masses += 1

        print(
            "%.2f M_sun: %i ages, %.1f-%.1f Myr"
            % (
                requested_mass,
                len(mass_data),
                mass_data[
                    "age_myr"
                ].min(),
                mass_data[
                    "age_myr"
                ].max()
            )
        )

    if plotted_masses == 0:

        raise RuntimeError(
            "No constant-mass tracks had enough points"
        )

    return mass_track_table



def plot_complete_sequence_vis2(
        night_directory,
        science_target,
        output_file,
        y_min=0.0,
        y_max=1.3,
        low_v2_threshold=0.70):
    """
    Plot all calibrated VIS2 measurements present in a REACH
    complete_sequences night.

    SCI and CAL are shown together, but the science target is
    also diagnosed separately.

    This is a diagnostic only. It does NOT modify the data used
    by the diameter fit.
    """

    import os
    import glob
    import numpy as np
    import matplotlib.pyplot as plt

    from astropy.io import fits


    print("")
    print("=" * 100)
    print("COMPLETE-SEQUENCE VIS2 DIAGNOSTIC")
    print("=" * 100)

    print("Night directory:")
    print(night_directory)

    print("Science target:")
    print(science_target)


    # ============================================================
    # Normalise target names
    # ============================================================

    def normalise_name(name):

        if isinstance(name, bytes):
            name = name.decode("utf-8")

        return (
            str(name)
            .strip()
            .replace("_", "")
            .replace(" ", "")
            .lower()
        )


    sci_clean = normalise_name(
        science_target
    )


    # ============================================================
    # Find calibrated products
    #
    # IMPORTANT:
    # only non-bootstrap current calibrated files:
    #
    # *_oidataCalibrated.fits
    #
    # This deliberately does NOT read _00/_01.
    # ============================================================

    fits_files = sorted(
        glob.glob(
            os.path.join(
                night_directory,
                "*_oidataCalibrated.fits"
            )
        )
    )


    print("")
    print(
        "Calibrated files found:",
        len(fits_files)
    )


    if len(fits_files) == 0:

        print(
            "No *_oidataCalibrated.fits files found."
        )

        return


    # ============================================================
    # Storage
    # ============================================================

    all_points = []

    science_points = []


    # ============================================================
    # Read calibrated OIFITS
    # ============================================================

    for filename in fits_files:

        basename = os.path.basename(
            filename
        )


        # --------------------------------------------------------
        # SCI / CAL from filename
        # --------------------------------------------------------

        if "_SCI_" in basename:

            role = "SCI"

        elif "_CAL_" in basename:

            role = "CAL"

        else:

            continue


        try:

            hdul = fits.open(
                filename
            )

        except Exception as error:

            print(
                "WARNING: cannot open %s: %s"
                % (
                    filename,
                    str(error)
                )
            )

            continue


        try:

            # ====================================================
            # Required extensions
            # ====================================================

            if (
                "OI_VIS2" not in hdul
                or
                "OI_WAVELENGTH" not in hdul
                or
                "OI_ARRAY" not in hdul
            ):

                continue


            vis2 = hdul[
                "OI_VIS2"
            ].data


            wave = np.asarray(
                hdul[
                    "OI_WAVELENGTH"
                ].data[
                    "EFF_WAVE"
                ],
                dtype=float
            )


            array = hdul[
                "OI_ARRAY"
            ].data


            # ====================================================
            # Station mapping
            # ====================================================

            station_map = {}


            for row in array:

                index = int(
                    row[
                        "STA_INDEX"
                    ]
                )


                # Prefer station name:
                # A0, G1, J2, J3...
                try:

                    name = row[
                        "STA_NAME"
                    ]

                except Exception:

                    name = row[
                        "TEL_NAME"
                    ]


                if isinstance(
                        name,
                        bytes):

                    name = name.decode(
                        "utf-8"
                    )


                station_map[
                    index
                ] = str(
                    name
                ).strip()


            # ====================================================
            # TARGET mapping
            # ====================================================

            target_map = {}


            if "OI_TARGET" in hdul:

                target_table = hdul[
                    "OI_TARGET"
                ].data


                for row in target_table:

                    target_id = int(
                        row[
                            "TARGET_ID"
                        ]
                    )

                    target_name = row[
                        "TARGET"
                    ]


                    if isinstance(
                            target_name,
                            bytes):

                        target_name = (
                            target_name.decode(
                                "utf-8"
                            )
                        )


                    target_map[
                        target_id
                    ] = str(
                        target_name
                    ).strip()


            # ====================================================
            # VIS2 rows
            # ====================================================

            for row_i in range(
                    len(vis2)):


                target_id = int(
                    vis2[
                        "TARGET_ID"
                    ][row_i]
                )


                target_name = (
                    target_map.get(
                        target_id,
                        "UNKNOWN"
                    )
                )


                target_clean = (
                    normalise_name(
                        target_name
                    )
                )


                pair = vis2[
                    "STA_INDEX"
                ][row_i]


                sta1 = station_map.get(
                    int(pair[0]),
                    str(pair[0])
                )

                sta2 = station_map.get(
                    int(pair[1]),
                    str(pair[1])
                )


                baseline_name = (
                    "%s-%s"
                    % (
                        sta1,
                        sta2
                    )
                )


                mjd = float(
                    vis2[
                        "MJD"
                    ][row_i]
                )


                ucoord = float(
                    vis2[
                        "UCOORD"
                    ][row_i]
                )

                vcoord = float(
                    vis2[
                        "VCOORD"
                    ][row_i]
                )


                baseline_m = np.sqrt(
                    ucoord ** 2
                    +
                    vcoord ** 2
                )


                values = np.asarray(
                    vis2[
                        "VIS2DATA"
                    ][row_i],
                    dtype=float
                )


                errors = np.asarray(
                    vis2[
                        "VIS2ERR"
                    ][row_i],
                    dtype=float
                )


                flags = np.asarray(
                    vis2[
                        "FLAG"
                    ][row_i],
                    dtype=bool
                )


                # =================================================
                # Wavelength channels
                # =================================================

                for channel_i in range(
                        len(wave)):


                    value = values[
                        channel_i
                    ]


                    error = errors[
                        channel_i
                    ]


                    flag = flags[
                        channel_i
                    ]


                    if flag:
                        continue


                    if not np.isfinite(
                            value):
                        continue


                    wavelength = float(
                        wave[
                            channel_i
                        ]
                    )


                    spatial_frequency = (
                        baseline_m
                        /
                        wavelength
                    )


                    point = {

                        "role":
                            role,

                        "target":
                            target_name,

                        "target_clean":
                            target_clean,

                        "file":
                            basename,

                        "row":
                            row_i,

                        "channel":
                            channel_i,

                        "mjd":
                            mjd,

                        "baseline":
                            baseline_name,

                        "baseline_m":
                            baseline_m,

                        "wavelength":
                            wavelength,

                        "spatial_frequency":
                            spatial_frequency,

                        "vis2":
                            value,

                        "e_vis2":
                            error,

                    }


                    all_points.append(
                        point
                    )


                    # =================================================
                    # Science target only
                    # =================================================

                    if (
                        role == "SCI"
                        and
                        target_clean
                        ==
                        sci_clean
                    ):

                        science_points.append(
                            point
                        )


        finally:

            hdul.close()


    # ============================================================
    # Diagnostics
    # ============================================================

    print("")
    print(
        "TOTAL valid SCI+CAL points:",
        len(all_points)
    )

    print(
        "SCI points for %s:"
        % science_target,
        len(science_points)
    )


    if len(science_points) > 0:

        sci_vis2 = np.asarray(
            [
                p["vis2"]
                for p
                in science_points
            ],
            dtype=float
        )


        print(
            "SCI min V2 = %.8f"
            % np.nanmin(
                sci_vis2
            )
        )

        print(
            "SCI max V2 = %.8f"
            % np.nanmax(
                sci_vis2
            )
        )


        print("")
        print(
            "SCI points with V2 < %.2f"
            % low_v2_threshold
        )

        print("-" * 100)


        low_found = False


        for point in science_points:

            if (
                point[
                    "vis2"
                ]
                <
                low_v2_threshold
            ):

                low_found = True

                print(
                    "file=%s  "
                    "row=%i  "
                    "ch=%i  "
                    "baseline=%s  "
                    "MJD=%.8f  "
                    "V2=%.8f"
                    % (
                        point[
                            "file"
                        ],
                        point[
                            "row"
                        ],
                        point[
                            "channel"
                        ],
                        point[
                            "baseline"
                        ],
                        point[
                            "mjd"
                        ],
                        point[
                            "vis2"
                        ]
                    )
                )


        if not low_found:

            print(
                "NONE"
            )


    # ============================================================
    # Plot
    # ============================================================

    if len(
            all_points) == 0:

        print(
            "No valid data to plot."
        )

        return


    fig, ax = plt.subplots(
        figsize=(10, 7)
    )


    # ============================================================
    # Find targets
    # ============================================================

    target_names = []


    for point in all_points:

        name = point[
            "target"
        ]


        if name not in target_names:

            target_names.append(
                name
            )


    # Let matplotlib choose colours automatically
    target_color = {}


    for target_i, target in enumerate(
            target_names):

        target_color[
            target
        ] = "C%i" % (
            target_i % 10
        )


    # ============================================================
    # Plot points
    # ============================================================

    used_labels = set()


    for point in all_points:

        role = point[
            "role"
        ]

        target = point[
            "target"
        ]


        label = "%s %s" % (
            role,
            target
        )


        if label in used_labels:

            plot_label = None

        else:

            plot_label = label

            used_labels.add(
                label
            )


        if role == "SCI":

            ax.scatter(

                point[
                    "spatial_frequency"
                ],

                point[
                    "vis2"
                ],

                marker="o",

                s=35,

                color=target_color[
                    target
                ],

                label=plot_label

            )


        else:

            ax.scatter(

                point[
                    "spatial_frequency"
                ],

                point[
                    "vis2"
                ],

                marker="x",

                s=40,

                color=target_color[
                    target
                ],

                label=plot_label

            )


    ax.set_xlabel(
        r"Spatial frequency [rad$^{-1}$]"
    )

    ax.set_ylabel(
        r"$V^2$"
    )

    ax.set_ylim(
        y_min,
        y_max
    )

    ax.grid()

    ax.legend(
        fontsize=8,
        loc="best"
    )


    ax.set_title(
        "%s - complete calibrated sequence"
        % science_target
    )


    fig.tight_layout()


    # ============================================================
    # Save
    # ============================================================

    output_directory = os.path.dirname(
        output_file
    )


    if (
        output_directory
        and
        not os.path.exists(
            output_directory
        )
    ):

        os.makedirs(
            output_directory
        )


    fig.savefig(
        output_file
    )


    plt.close(
        fig
    )


    print("")
    print(
        "Saved complete-sequence plot:"
    )

    print(
        output_file
    )

    print("=" * 100)