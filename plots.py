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
combined_fit = False                 # Fit for LDD for multiple seq at once
load_saved_results = False          # Load or do fitting fresh
assign_default_uncertainties = True # Give default errors to stars without
force_claret_params = False         # Force use of Claret+11 limb d. params
n_bootstraps = 2
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
results_folder = "26-06-15_i2"       # Final run for 1st draft
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




rplt.plot_casagrande_teff_comp(tgt_info, xy_map.teff)
rplt.plot_lit_diam_comp(tgt_info, xy_map.lit_diam)
 
def plot_sidelobe_vis2_fit(tgt_info, results, sci):
    """Plot the zoomed in fitted sidelobe
    """
    plt.close("all")
    # Setup the axes
    fig, axes = plt.subplots(1, 1)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # Get the science target name

    hd_id = tgt_info[tgt_info["Primary"]==sci].index.values[0]
    
    # And the science target results
    sci_results = results[results["STAR"]==sci].iloc[0]
    
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
    axes.set_xlim([5.5E7,9.5E7])
    axes.set_ylim([0.0,0.018])
    
    axes.set_xticklabels([])
    
    axes.tick_params(axis="both", top=True, right=True)
    res_ax.tick_params(axis="y", right=True)
    
    maj_loc = plticker.MultipleLocator(base=0.004)
    min_loc = plticker.MultipleLocator(base=0.00025)
    
    axes.yaxis.set_major_locator(maj_loc)
    axes.yaxis.set_minor_locator(min_loc)
    axes.set_ylabel(r"Visibility$^2$", fontsize="x-large")
    
    res_maj_loc = plticker.MultipleLocator(base=0.005)
    res_min_loc = plticker.MultipleLocator(base=0.001)
    
    res_ax.yaxis.set_major_locator(res_maj_loc)
    res_ax.yaxis.set_minor_locator(res_min_loc)
    
    res_ax.set_xlim([5.5E7,9.5E7])
    res_ax.set_ylim([-0.008,0.008])
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
    plt.savefig("paper/lam_sgr_sidelobe.pdf") 
    plt.savefig("paper/lam_sgr_sidelobe.png", dpi=200)   
 

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
            if xy_map is not None:
                xx = xy_map[star_data["Primary"]][0]
                yy = xy_map[star_data["Primary"]][1]

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
        ylim = [0.9, 4.6]
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
        star_data["Primary"] = clean_name_for_match(star_data["Primary"])
        
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
    """
    
    #fig, ax = plt.figure()
    plt.close("all")
    
    vis2, e_vis2, baselines, wavelengths = rdiam.extract_vis2(oi_fits_file)
    
    n_bl = len(baselines)
    n_wl = len(wavelengths)
    bl_grid = np.tile(baselines, n_wl).reshape([n_wl, n_bl]).T
    wl_grid = np.tile(wavelengths, n_bl).reshape([n_bl, n_wl])
            
    sfreq = (bl_grid / wl_grid).flatten()
    
    plt.errorbar(sfreq, vis2.flatten(), yerr=e_vis2.flatten(), fmt=".")
    
    plt.xlabel(r"Spatial Frequency (rad$^{-1})$")
    plt.ylabel(r"Visibility$^2$")
    plt.title(r"%s (%i vis$^2$ points)" % (star_id, len(vis2.flatten())))
    #plt.legend(loc="best")
    plt.xlim([0.0,25E7])
    plt.ylim([0.0,1.0])
    plt.grid()
    #plt.gcf().set_size_inches(16, 9)
    #plt.savefig("plots/vis2_fit.pdf")


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


def plot_c_hist(results, n_bins=5):
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
    
    n_points = (len(freqs))
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
    