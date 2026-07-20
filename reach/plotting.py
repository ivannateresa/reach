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


def plot_hr_diagram(tgt_info, plot_isochrones_basti=False, 
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
    