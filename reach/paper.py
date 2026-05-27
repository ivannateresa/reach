"""Helper functions to assist with writing the PIONIER paper (e.g. making 
LaTeX tables).

Note that it might be helpful to plot dates the tables were generated to ease
copying and pasting?
"""
from __future__ import division, print_function
import pandas as pd
import numpy as np
import reach.utils as rutils
from collections import OrderedDict
import os
        
# -----------------------------------------------------------------------------
# Making tables
# ----------------------------------------------------------------------------- 
def make_table_final_results(tgt_info):
    """Make the final results table to display the angular diameters and 
    derived fundamental parameters.
    """
    exp_scale = -8
    
    columns = OrderedDict([("Star", ""),
                           #("HD", ""),
                           #(r"u$_\lambda$", ""),
                           #(r"s$_\lambda$", ""),
                           (r"$\theta_{\rm UD}$", "(mas)"),
                           (r"$\theta_{\rm LD}$", "(mas)"),
                           (r"$R$", "($R_\odot$)"), 
                           (r"$f_{\rm bol}$", 
                            r"(10$^{%i}\,$ergs s$^{-1}$ cm $^{-2}$)" % exp_scale),
                           (r"$T_{\rm eff}$", "(K)"),
                           (r"$L$", ("($L_\odot$)"))])
                           
    header = []
    table_rows = []
    footer = []
    
    # Construct the header of the table
    header.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    header.append("\hline")
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.values()))
    header.append("\hline")
    
    # Populate the table for every science target
    for star_i, row in tgt_info[tgt_info["Science"]].iterrows():
        
        # Only continue if we have data on this particular star
        if not row["in_paper"]:
            continue
        
        table_row = ""
        
        # Step through column by column
        table_row += "%s & " % rutils.format_id(row["Primary"])
        table_row += r"%0.3f $\pm$ %0.3f & " % (row["udd_final"], row["e_udd_final"])
        table_row += r"%0.3f $\pm$ %0.3f & " % (row["ldd_final"], row["e_ldd_final"])
        table_row += r"%0.3f $\pm$ %0.3f &" % (row["r_star_final"], row["e_r_star_final"])
        
        # For fbol representation, split mantissa and exponent
        table_row += r"%5.1f $\pm$ %0.1f &" % (row["f_bol_final"] / 10**exp_scale, 
                                               row["e_f_bol_final"] / 10**exp_scale)
        table_row += r"%0.0f $\pm$ %0.0f & " % (row["teff_final"], row["e_teff_final"])
        table_row += r"%0.2f $\pm$ %0.2f " % (row["L_star_final"], row["e_L_star_final"])
        
        table_rows.append(table_row + r"\\")
    
    
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    
    # Write the tables
    table_1 = header + table_rows + footer
    
    np.savetxt("paper/table_final_results.tex", table_1, fmt="%s")


def make_table_limb_darkening(tgt_info):
    """Make the table to display both kinds of limb darkening coefficients
    (Claret and STAGGER), as well as the scaling parameters associated with
    the latter.
    """
    columns = OrderedDict([("", ""),
                           (r"u$_{\lambda}$", ""),
                           (r"u$_{\lambda_1}$", ""),
                           (r"u$_{\lambda_2}$", ""),
                           (r"u$_{\lambda_3}$", ""),
                           (r"u$_{\lambda_4}$", ""),
                           (r"u$_{\lambda_5}$", ""),
                           (r"u$_{\lambda_6}$", ""),
                           (r"s$_{\lambda_1}$", ""),
                           (r"s$_{\lambda_2}$", ""),
                           (r"s$_{\lambda_3}$", ""),
                           (r"s$_{\lambda_4}$", ""),
                           (r"s$_{\lambda_5}$", ""),
                           (r"s$_{\lambda_6}$", ""),])
    
    # Get the limb darkening and scaling parameters
    u_lambda_cols = ["u_lambda_%i" % ui for ui in np.arange(0,6)]
    e_u_lambda_cols = ["e_u_lambda_%i" % ui for ui in np.arange(0,6)]
    s_lambda_cols = ["s_lambda_%i" % ui for ui in np.arange(0,6)]
    e_s_lambda_cols = ["e_s_lambda_%i" % ui for ui in np.arange(0,6)]
                           
    header = []
    table_rows = []
    footer = []
    
    # Construct the header of the table
    header.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    header.append("\hline")
    header.append((r"Star & CB11 &"
                   r"\multicolumn{6}{c}{Equivalent Linear Limb Darkening Coefficient} &" 
                   r"\multicolumn{6}{c}{$\theta_{\rm LD}$ Scaling Term} \\")) 
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    #header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.values()))
    header.append("\hline")
    
    # Populate the table for every science target
    for star_i, row in tgt_info[tgt_info["Science"]].iterrows():
        
        # Only continue if we have data on this particular star
        if not row["in_paper"]:
            continue
        
        table_row = ""
        
        # Step through column by column
        table_row += "%s & " % rutils.format_id(row["Primary"])
        
        # Not using STAGGER grid
        if len(set(row[u_lambda_cols])) == 1:
            table_row += r"%0.3f $\pm$ %0.3f & " % (row[u_lambda_cols[0]], 
                                                    row[e_u_lambda_cols[0]])
            # Empty values for 6 lambda and scale params
            for u_i in np.arange(12):
                table_row += "-&" 
        
        # Using STAGGER grid
        else:
            # Claret param
            table_row += "-&" 
            
            for u_i in np.arange(6):
                table_row += (r"%0.3f $\pm$ %0.3f & " 
                              % (row[u_lambda_cols[u_i]], 
                                 row[e_u_lambda_cols[u_i]]))
            
            for s_i in np.arange(6):
                table_row += r"%0.3f &" % (row[s_lambda_cols[s_i]])
        
        table_rows.append(table_row[:-1] + r" \\")
    
    
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    
    # Write the tables
    table_1 = header + table_rows + footer
    
    np.savetxt("paper/table_limb_darkening.tex", table_1, fmt="%s")


def make_table_seq_results(results):
    """Make the table to display the C intercept parameter associated with
    each sequence.
    """
    columns = OrderedDict([("Star", ""),
                           #("HD", ""),
                           ("Period", ""),
                           ("Sequence", ""),
                           (r"$C_{\rm LD}$", ""),
                           (r"$C_{\rm UD}$", "")])
                           
    header = []
    table_rows = []
    footer = []
    
    # Construct the header of the table
    header.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    header.append("\hline")
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    #header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.values()))
    header.append("\hline")
    
    # Populate the table for every science target
    for star, row in results.iterrows():
        # Give each row its own sequence
        for seq_i in np.arange(len(row["SEQ_ORDER"])):
            id = row["STAR"]
            period = row["SEQ_ORDER"][seq_i][2]
            sequence = row["SEQ_ORDER"][seq_i][1]
            
            table_row = ""
            
            # Step through column by column
            table_row += "%s & " % rutils.format_id(str(id))
            table_row += "%s & " % period
            table_row += "%s & " % sequence
            table_row += "%0.3f $\pm$ %0.3f & " % (row["C_SCALE"][seq_i],
                                                   row["e_C_SCALE"][seq_i])
            table_row += "%0.3f $\pm$ %0.3f" % (row["C_SCALE_UDD"][seq_i],
                                                row["e_C_SCALE_UDD"][seq_i])
        
            table_rows.append(table_row + r"\\")
    
    
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    
    # Write the tables
    table_1 = header + table_rows + footer
    
    np.savetxt("paper/table_sequence_results.tex", table_1, fmt="%s")
    
    
def make_table_fbol(tgt_info):
    """Make the table to display the derived bolometric fluxes from each band,
    as well as their errors.
    """
    exp_scale = -8
    
    columns = OrderedDict([("Star", ""),
                           ("HD", ""),
                           (r"$f_{\rm bol}$ (MARCS)", 
                           r"(10$^{%i}\,$ergs s$^{-1}$ cm $^{-2}$)" % exp_scale),
                           (r"$\sigma_{f_{\rm bol}} (\zeta)$", r"(\%)")])
                           #(r"f$_{\rm bol} (avg)$", r"(ergs s$^{-1}$ cm $^{-2}$)")])
                     
    header = []
    table_rows = []
    footer = []
    
    # Construct the header of the table
    header.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    header.append("\hline")
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.values()))
    header.append("\hline")    
    
    
    bands = [r"H$_p$", r"B$_T$", r"V$_T$"]#, r"B$_P$", r"R$_P$"]
    
    f_bols = ["f_bol_Hp", "f_bol_BT", "f_bol_VT"]#, "f_bol_BP", "f_bol_RP"]
    e_f_bols = ["e_f_bol_Hp", "e_f_bol_BT", "e_f_bol_VT"]#, "e_f_bol_BP", 
                #"e_f_bol_RP"]
    
     # Populate the table for every science target
    for star_i, star in tgt_info[tgt_info["Science"]].iterrows():
        table_row = ""
        
        # Only continue if we have data on this particular star
        if not star["in_paper"]:
            continue
        
        # Step through column by column
        table_row += "%s & " % rutils.format_id(star["Primary"])
        table_row += "%s & " % star.name.replace("HD", "")
        
        # Add final average value
        table_row += r"$<>$: %.3f & " % (star["f_bol_final"] / 10**exp_scale)
    
        e_pc_f_bol_final = star["e_f_bol_final"] / star["f_bol_final"]
    
        table_row += r"%.2f \\" % (e_pc_f_bol_final * 100)
        
        table_rows.append(table_row)
        
        # Now have a separate row for each of the remaining filters
        for band_i in np.arange(len(bands)):
            table_row = r" & & %s: %.3f &" % (bands[band_i], 
                                        star[f_bols[band_i]] / 10**exp_scale)
            
            e_pc_f_bol = star[e_f_bols[band_i]] / star[f_bols[band_i]]
        
            table_row += r"%.2f \\" % (e_pc_f_bol*100)
            
            table_rows.append(table_row)
    
    
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    
    # Write the tables
    table_1 = header + table_rows + footer
    #table_2 = header + table_rows[30:] + footer
    
    # Write the table
    np.savetxt("paper/table_fbol_1.tex", table_1, fmt="%s")      
    #np.savetxt("paper/table_fbol_2.tex", table_2, fmt="%s")     
    

def make_table_claret_stagger_comp():
    """
    """
    diams_claret = pd.read_csv("results/paper_results/diams_claret.csv")
    diams_stagger = pd.read_csv("results/paper_results/diams_stagger.csv")
    
    assert (tuple(diams_claret["Primary"].values) 
        == tuple(diams_stagger["Primary"].values))

    stars = diams_claret["Primary"].values
    ldd_claret = diams_claret["ldd_final"].values
    e_ldd_claret = diams_claret["e_ldd_final"].values
    ldd_stagger = diams_stagger["ldd_final"].values
    e_ldd_stagger = diams_stagger["e_ldd_final"].values

    columns = OrderedDict([("Star", ""),
                           (r"$\theta_{\rm LD, CB11}$", "(mas)"),
                           (r"$\theta_{\rm LD, \textsc{stagger}}$", "(mas)"),
                           (r"$\sigma_{\theta_{\rm LD}}$", r"(\%)"),
                           ])
                     
    header = []
    table_rows = []
    footer = []
    
    # Construct the header of the table
    header.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    header.append("\hline")
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.values()))
    header.append("\hline")    
    
     # Populate the table for every science target
    for star_i, (star, ldd_c, e_ldd_c, ldd_s, e_ldd_s) in enumerate(
            zip(stars, ldd_claret, e_ldd_claret, ldd_stagger, e_ldd_stagger)):
        # ....
        table_row = ""
        
        # Step through column by column
        table_row += "%s & " % rutils.format_id(star)
        
        # Claret & Bloemen 2011
        table_row += r"%.3f $\pm$ %0.3f & " % (ldd_c, e_ldd_c)

        # Stagger
        table_row += r"%.3f $\pm$ %0.3f & " % (ldd_s, e_ldd_s)

        # Percent
        table_row += r"%.2f \\" % ((ldd_c - ldd_s)/ldd_c * 100)
        
        table_rows.append(table_row)
    
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    
    # Write the tables
    table_1 = header + table_rows + footer
    #table_2 = header + table_rows[30:] + footer
    
    # Write the table
    np.savetxt("paper/table_claret_stagger_comp.tex", table_1, fmt="%s")


def make_table_observation_log(tgt_info, complete_sequences, sequences):
    """Make the table to summarise the observations, including what star was
    in what sequence.
    """
    columns = OrderedDict([("Star", ""),
                           ("UT Date", ""),
                           ("ESO", "Period"),
                           ("Sequence", "Type"),
                           ("Baseline", ""), 
                           #("Spectral", "channels"),
                           ("Calibrator", "HD"),
                           ("Calibrators", "Used")])
                           
    header = []
    table_rows = {}
    footer = []
    
    # Construct the header of the table
    header.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    header.append("\hline")
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.values()))
    header.append("\hline")
    
    
    # Populate the table for every science target
    for seq in complete_sequences:
        table_row = ""
        
        star_id = rutils.format_id(seq[1])
        ut_date = complete_sequences[seq][0]
        period = seq[0]
        seq_type = seq[2]
        baselines = complete_sequences[seq][2][0][9]
        cals = [target.replace("_", "").replace(".","").replace(" ", "") 
                for target in sequences[seq][::2]]
        cals_hd = tuple(rutils.get_unique_key(tgt_info, cals))
        cals = ("%s, %s, %s" % cals_hd).replace("HD", "")
        
        # Figure out how many calibrators we used (i.e. weren't 'BAD')
        cal_quality = [tgt_info.loc[cal]["Quality"] for cal in cals_hd]
        cals_used = 3 - np.sum(np.array(cal_quality)=="BAD")
        
        table_row = ("%s & "*len(columns)) % (star_id, ut_date, period,  
                                              seq_type, baselines, cals, 
                                              str(cals_used))
                
        table_rows[(ut_date, star_id, seq_type)] = table_row[:-2] + r"\\"
        
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    
    # Sort by UT
    ut_sorted = table_rows.keys()
    ut_sorted.sort()
    
    sorted_rows = [table_rows[row] for row in ut_sorted]
    
    # Write the table
    table = header + sorted_rows + footer
    np.savetxt("paper/table_observations.tex", table, fmt="%s")
    

def make_table_targets(tgt_info):
    """Make the table to summarise the target information and literature
    stellar parameters.
    """
    # Column names and associated units
    columns = OrderedDict([("Star", ""), 
                           ("HD", ""),
                           ("RA$^a$", "(hh mm ss.ss)"),
                           ("DEC$^a$", "(dd mm ss.ss)"),
                           ("SpT$^b$", ""),
                           ("$V_{\\rm T}^c$", "(mag)"), 
                           ("$H^d$", "(mag)"),
                           ("$T_{\\rm eff}$", "(K)"),
                           (r"$\log g$", "(dex)"), 
                           ("[Fe/H]", "(dex)"),
                           (r"$v \sin i$", r"(km$\,$s$^{-1}$)"),
                           ("Plx$^a$", "(mas)"),
                           ("Refs", "")])#,
                           #("Mission", "")])         
    
    table_rows = []
    
    # Construct the header of the table
    table_rows.append("\\begin{landscape}")
    table_rows.append("\\begin{table}")
    table_rows.append("\\centering")
    table_rows.append("\\caption{Science targets}")
    table_rows.append("\\label{tab:science_targets}")
    
    table_rows.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    table_rows.append("\hline")
    table_rows.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    table_rows.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.values()))
    table_rows.append("\hline")
    
    ref_i = 0
    references = []
    
    # Populate the table for every science target
    for star_i, star in tgt_info[tgt_info["Science"]].iterrows():
        table_row = ""
        
        # Only continue if we have data on this particular star
        if not star["in_paper"]:
            continue
        
        # Format RA and DEC

        ra_hr = np.floor(star["RA"] / 15)
        ra_min = np.floor((star["RA"] / 15 - ra_hr) * 60)
        ra_sec = ((star["RA"] / 15 - ra_hr) * 60 - ra_min) * 60
        ra = "%02i %02i %05.2f" % (ra_hr, ra_min, ra_sec)
        
        dec_deg = np.floor(star["DEC"])
        dec_min = np.floor((star["DEC"] - dec_deg) * 60)
        dec_sec = ((star["DEC"] - dec_deg) * 60 - dec_min) * 60
        dec = "%02i %02i %05.2f" % (dec_deg, dec_min, dec_sec)
        
        # Step through column by column
        table_row += "%s & " % rutils.format_id(star["Primary"])
        table_row += "%s & " % star.name.replace("HD", "")
        table_row += "%s & " % ra
        table_row += "%s & " % dec
        table_row += "%s & " % star["SpT"]
        table_row += "%0.2f & " % star["VTmag"]
        table_row += "%0.1f & " % star["Hmag"]
        table_row += r"%0.0f $\pm$ %0.0f & " % (star["Teff"], star["e_teff"])
        table_row += r"%0.2f $\pm$ %0.2f &" % (star["logg"], star["e_logg"])
        table_row += r"%0.2f $\pm$ %0.2f &" % (star["FeH_rel"], star["e_FeH_rel"])
        table_row += r"%0.2f & " % star["vsini"]
        
        # Parallax is not from Gaia DR2
        if np.isnan(star["Plx"]):
            table_row += r"%0.2f $\pm$ %0.2f & " % (star["Plx_alt"], star["e_Plx_alt"])
            #table_row += "\\textit{Hipparcos}"
        
        # From Gaia DR2
        else:
            table_row += r"%0.2f $\pm$ %0.2f & " % (star["Plx"], star["e_Plx"])
            #table_row += "\\textit{Gaia}"
            
        # Now do references
        refs = [star["teff_bib_ref"], star["logg_bib_ref"], 
                star["feh_bib_ref"], star["vsini_bib_ref"]]
         
        for ref in refs:   
            if ref == "":
                table_row += "-,"   
                  
            elif ref not in references:
                references.append(ref)
                ref_i = np.argwhere(np.array(references)==ref)[0][0] + 1
                table_row += "%s," % ref_i
            
            elif ref in references:
                ref_i = np.argwhere(np.array(references)==ref)[0][0] + 1
                table_row += "%s," % ref_i
        
        # Remove the final comma and append (Replace any nans with '-')
        table_rows.append(table_row[:-1].replace("nan", "-")  + r"\\")
        
    # Finish the table
    table_rows.append("\\hline")
    table_rows.append("\\end{tabular}")
    
    # Add notes section with references
    table_rows.append("\\begin{minipage}{\linewidth}")
    table_rows.append("\\vspace{0.1cm}")
    
    table_rows.append("\\textbf{Notes:} $^a$Gaia \citet{brown_gaia_2018} - "
                      " note that Gaia parallaxes listed here have not been "
                      "corrected for the zeropoint offset, "
                      "$^b$SIMBAD, $^c$Tycho \citet{hog_tycho-2_2000}, "
                      "$^d$2MASS \citet{skrutskie_two_2006} \\\\")
    table_rows.append(" \\textbf{References for spectroscopic $T_{\\rm eff}$, "
                      "$\\log g$, [Fe/H], and $v \\sin i$:}") 
    
    for ref_i, ref in enumerate(references):
        table_rows.append("%i. \\citet{%s}, " % (ref_i+1, ref))
    
    # Remove last comma
    table_rows[-1] = table_rows[-1][:-1]
    
    table_rows.append("\\end{minipage}")
    table_rows.append("\\end{table}")
    table_rows.append("\\end{landscape}")
    
    # Write the table
    if not os.path.exists("paper"):
        os.makedirs("paper")

    np.savetxt("paper/table_targets.tex", table_rows, fmt="%s")


def make_table_calibrators(tgt_info, sequences):
    """Summarise the calibrators used (or not).
    """
    # Column names and associated units
    columns = OrderedDict([("HD", ""),
                           ("SpT$^a$", "(Actual)$^a$"),
                           ("SpT$^b$", "(Adopted)$^b$"),
                           (r"$V_{\rm T}^c$", "(mag)"), 
                           (r"$H^d$", "(mag)"),
                           (r"$E(B-V)$", "(mag)"),
                           ("$\\theta_{\\rm pred}$", "(mas)"),
                           ("$\\theta_{\\rm LD}$ Rel", ""),
                           ("Used", ""),
                           ("Plx", "(mas)"),
                           ("Target/s", "")])
    
    labels = ["index", "SpT", "VTmag", "Hmag", "Quality", "Target/s"]
    
    exclusion_codes = OrderedDict([("Binarity", "f"), 
                                   ("IR excess","g"), 
                                   ("Inconsistent photometry","h")])           
    header = []
    table_rows = []
    footer = []
    notes = []
    
    # Construct the header of the table
    #header.append("\\begin{landscape}")
    header.append("\\begin{tabular}{%s}" % ("c"*len(columns)))
    header.append("\hline")
    header.append((r"HD & \multicolumn{2}{c}{SpT} & " +
                  ("%s & "*(len(columns)-3))[:-2] + r"\\") 
                  % tuple(columns.keys()[3:]))
    
    #header.append((("%s & "*len(columns))[:-2] + r"\\") % tuple(columns.keys()))
    header.append((("%s & "*len(columns))[:-2] + r"\\") 
                     % tuple(columns.values()))
    header.append("\hline")
    
    # Populate the table for every science target
    for star_i, star in tgt_info[~tgt_info["Science"]].iterrows():
        table_row = ""
        
        # Only continue if we have data on this particular star
        if not star["in_paper"]:
            continue
        
        # Find which science target/s the calibrator is associated with
        scis = []
        for seq in sequences:
            cals = [target.replace("_", "").replace(".","").replace(" ", "") 
                    for target in sequences[seq]]
                    
            if star.name in rutils.get_unique_key(tgt_info, cals):
                scis.append(rutils.format_id(seq[1]))
                
        scis = list(set(scis))
        scis.sort()
        
        # Step through column by column
        table_row += "%s & " % star.name.replace("HD", "")
        
        # Make SpT have a smaller font if it's long
        #if len(str(star["SpT"])) > 5:
        #    table_row += "{\\tiny %s } & " % star["SpT"]
        #else:
        table_row += "%s & " % star["SpT"]
        table_row += "%s & " % star["SpT_simple"]
            
        table_row += "%0.2f & " % star["VTmag"]
        table_row += "%0.2f & " % star["Hmag"]
        table_row += "%0.3f & " % star["eb_v"]
        
        # Remove placeholder LDD
        if star["LDD_pred"] == 1.0 or np.isnan(star["LDD_pred"]):
            table_row += "- & "
        else:
            table_row += r"%0.3f $\pm$ %0.2f & " % (star["LDD_pred"], star["e_LDD_pred"])
            
        table_row += ("%s & " % star["LDD_rel"]).split("LDD_")[-1].replace("_", "-")
        
        # Determine whether the star was used as a calibrator
        if star["Quality"] == "BAD":
            reason_code = exclusion_codes[star["exclusion_reason"]]
            table_row += r"N$^%s$ & " % reason_code
        else:
            table_row += "Y & "
        
        # Both parallaxes are nan, placeholder value
        if np.isnan(star["Plx"]) and np.isnan(star["Plx_alt"]):
            table_row += "- & "
        
        # If Gaia plx is nan, use Tycho
        elif np.isnan(star["Plx"]):
            table_row += r"%0.2f $\pm$ %0.2f$^c$ &" % (star["Plx_alt"], star["e_Plx_alt"])
        
        # From Gaia DR2
        else:
            table_row += r"%0.2f $\pm$ %0.2f$^e$ &" % (star["Plx"], star["e_Plx"])    
        
        table_row += ("%s, "*len(scis) % tuple(scis))[:-2]
        
        # Replace any nans with '-'
        table_rows.append(table_row.replace("nan", "-") + r"\\")
        
    # Finish the table
    footer.append("\hline")
    footer.append("\end{tabular}")
    #footer.append("\\end{landscape}")
    
    # Add notes section
    notes.append("\\begin{minipage}{\linewidth}")
    notes.append("\\vspace{0.1cm}")
    
    notes.append("\\textbf{Notes:} $^a$SIMBAD, "
                 "$^b$Adopted for intrinsic colour grid interpolation,"
                 "$^c$Tycho \citet{hog_tycho-2_2000}, "
                 "$^d$2MASS \citet{skrutskie_two_2006}, "
                 "$^e$Gaia \citet{brown_gaia_2018}")
                  
    for er in exclusion_codes.keys():
        notes[-1] += r", $^%s$%s" % (exclusion_codes[er], er)
    
    notes[-1] += "\\\\"
    
    notes.append("\\end{minipage}")
    #notes.append("\\end{table*}")
    #notes.append("\\end{landscape}")
    
    # Write the tables
    table_1 = header + table_rows[:45] + footer
    table_2 = header + table_rows[45:] + footer + notes
    
    np.savetxt("paper/table_calibrators_1.tex", table_1, fmt="%s")
    np.savetxt("paper/table_calibrators_2.tex", table_2, fmt="%s")