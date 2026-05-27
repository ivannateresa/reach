"""Miscellaneous function module for reach
"""
from __future__ import division, print_function

import os
import csv
import glob
import pickle
import numpy as np
import pandas as pd
from decimal import Decimal
import reach.photometry as rphot
import reach.diameters as rdiam
from collections import OrderedDict

# -----------------------------------------------------------------------------
# Utilities Functions
# -----------------------------------------------------------------------------
def summarise_sequences():
    """Creates a dictionary summary of each sequence, specified with the unique
    key (period, science, bright/faint).
    
    Returns
    -------
    sequences: OrderedDict
        Dict mapping key (period, science, bright/faint) with list of target 
        IDs in the sequence.
    """
    bright_list_files = sorted(glob.glob("data/p106_bright.txt"))

    faint_list_files  = sorted(glob.glob("data/p106_faint.txt"))
    period = [106]

    target_list = []

    for p_i, bright_list_file in enumerate(bright_list_files):
        with open(bright_list_file) as csv_file:
            for line in csv.reader(csv_file):
                target_list.append((period[p_i], line[0].replace(" ", ""),
                                    "bright"))

    for p_i, faint_list_file in enumerate(faint_list_files):
        with open(faint_list_file) as csv_file:
            for line in csv.reader(csv_file):
                target_list.append((period[p_i], line[0].replace(" ", ""),
                                    "faint"))
        
    # Order each sequence
    sequences = OrderedDict()
    
    for tgt_i in xrange(0, len(target_list), 4):
        # All targets must share a sequence and period
        assert (target_list[tgt_i][::2] == target_list[tgt_i+1][::2] 
                and target_list[tgt_i][::2] == target_list[tgt_i+2][::2] 
                and target_list[tgt_i][::2] == target_list[tgt_i+3][::2])
        
        sequences[target_list[tgt_i]] = [target_list[tgt_i+1][1], 
                                         target_list[tgt_i][1],
                                         target_list[tgt_i+2][1], 
                                         target_list[tgt_i][1],
                                         target_list[tgt_i+3][1]]
    
    pkl_sequences = open("data/sequences.pkl", "wb")
    pickle.dump(sequences, pkl_sequences)
    pkl_sequences.close()
    
    return sequences
    
    
def load_target_information(filepath="data/target_info.tsv", 
                            assign_default_uncertainties=True, def_e_logg=0.2,
                            def_e_teff=100, def_e_feh=0.1, 
                            remove_unused_targets=False):
    """Loads in the target information tsv (tab separated) as a pandas 
    dataframne with appropriate column labels and rows labelled as each star.
    
    https://pandas.pydata.org/pandas-docs/version/0.21/generated/
    pandas.read_csv.html#pandas.read_csv
    """
    # Import (TODO: specify dtypes)
    tgt_info = pd.read_csv(filepath, sep="\t", header=1, index_col=8,
                              skiprows=0)
    
    print(list(tgt_info.columns))

    # Organise dataframe by removing duplicates
    # Note that the tilde is a bitwise not operation on the mask
    tgt_info = tgt_info[~tgt_info.index.duplicated(keep="first")]
    
    # Force primary and Bayer IDs to standard no-space format
    tgt_info["Primary"] = [id.replace(" ", "").replace(".", "").replace("_","")
                           for id in tgt_info["Primary"]]
                           
    tgt_info["Bayer_ID"] = [id.replace(" ", "").replace("_","") 
                            if type(id)==str else None
                            for id in tgt_info["Bayer_ID"]]

                            
    # Use None as empty value for IDs
    # Note: this is possibly unnecessary since dataframes have a notnull method
    tgt_info["Ref_ID_1"].where(tgt_info["Ref_ID_1"].notnull(), None, 
                               inplace=True)
    tgt_info["Ref_ID_2"].where(tgt_info["Ref_ID_2"].notnull(), None, 
                               inplace=True)
    tgt_info["Ref_ID_3"].where(tgt_info["Ref_ID_3"].notnull(), None, 
                               inplace=True)

    # Use empty string for empty references
    tgt_info["vsini_bib_ref"].where(tgt_info["vsini_bib_ref"].notnull(), "", 
                                    inplace=True)                           
    
    # Assign default uncertainties
    if assign_default_uncertainties:
        # logg
        logg_mask = np.logical_and(np.isnan(tgt_info["e_logg"]), 
                                   tgt_info["Science"])
        tgt_info["e_logg"].where(~logg_mask, def_e_logg, inplace=True)
        
        # Teff
        teff_mask = np.logical_and(np.isnan(tgt_info["e_teff"]), 
                                   tgt_info["Science"])
        tgt_info["e_teff"].where(~teff_mask, def_e_teff, inplace=True)     
        
        # [Fe/H]
        feh_mask = np.logical_and(np.isnan(tgt_info["e_FeH_rel"]), 
                                  tgt_info["Science"])
        tgt_info["e_FeH_rel"].where(~feh_mask, def_e_feh, inplace=True)    
    
    # Remove any targets not used
        
    if remove_unused_targets:
        tgt_info = tgt_info[tgt_info["in_paper"]]
                               
    # Return result
    tgt_info.index.name = "HD_ID"
    tgt_info["HD_ID"] = tgt_info.index
    return tgt_info
    
    
def combine_independent_boostrap_runs(pkl_list):
    """Function to combine results from independent bootstrapping runs for the
    purpose of plotting histograms/estimating uncertainties.
    
    Parameters
    ----------
    pkl_list: list
        List of pickle files to combine.
        
    Returns
    -------
    bs_results: dict of pandas dataframes
        Dictionary with science targets as keys, containing pandas dataframes
        recording the results of each bootstrapping iteration as rows.
    """
    # Initialise structure to store results in
    bs_result_list = []   
    
    # Open each pickle and join together into n_ldd_fit_all
    for pkl_fn in pkl_list:
        pkl = open(pkl_fn)
        bs_result_list.append(pickle.load(pkl))
        pkl.close()
        
    # Get a reference to the dataframe we want to join to    
    all_bs_results = bs_result_list[0]     
        
    # For every science target, combine all bootstrapping iterations
    for bs_result_n in bs_result_list[1:]:
        for sci in bs_result_n.keys():
            # Increment the index of the data to be joined
            orig_n_bs = len(bs_result_n[sci])
            base_n = len(all_bs_results[sci])
            bs_result_n[sci].set_index(np.arange(base_n, base_n + orig_n_bs),
                                       inplace=True)
            
            # Join
            all_bs_results[sci] = pd.concat([all_bs_results[sci], 
                                             bs_result_n[sci]])
                
    return all_bs_results
    

def complete_obs_diagnostics(complete_sequences):
    """Prints complete_complete sequences in a human readable format for 
    troubleshooting.
    """
    for seq in complete_sequences.keys():
        print("-"*30)
        print(seq,len(complete_sequences[seq][2]))
        for i, yy in enumerate(complete_sequences[seq][2]):
            print("%02i" % i, yy[2], yy[3], yy[-1])
          
            
def night_log_diagnostics(night_log):
    """Prints the night log in a human readable format for troubleshooting.
    """
    for night in night_log.keys():
        print("-"*30)
        print(night, len(night_log[night]))
        for i, yy in enumerate(night_log[night]):
            print("%02i" % i, yy[2], yy[3], yy[-1])
    
            
def get_unique_key(tgt_info, id_list):
    """Some stars were observed multiple times under different names (e.g. a
    Bayer designation once, and a HD number another time). This complicates
    uniquely IDing each star, so this method serves to take an ID that we may
    have referenced a star with, and take the value that allows us to easily
    reference tgt_info.
    
    Parameters
    ----------
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
    
    id_list: list
        List of string IDs.
    
    Returns
    -------
    unique_ids: list
        List of IDs.
    """
    unique_ids = []
    
    if isinstance(id_list, str):
        id_list = [id_list]
    
    # Grab the primary IDs
    # Note that several stars are observed multiple times under different
    # primary IDs, so we need to check HD and Bayer IDs as well
    for star in id_list:
        # Remove non-alpha-numeric characters 
        star = star.replace("_", "").replace(" ", "").replace(".", "")
        
        prim_id = tgt_info[tgt_info["Primary"]==star].index
        
        if len(prim_id)==0:
            prim_id = tgt_info[tgt_info["Bayer_ID"]==star].index
            
        if len(prim_id)==0:
            prim_id = tgt_info[tgt_info.index==star].index
        
        try:
            assert len(prim_id) > 0
        except:
            print("...failed on %s, %s" % (star))
            failed = True
            break
        unique_ids.append(prim_id[0])
        
    return unique_ids
    
    
def compute_dist(tgt_info, use_plx_systematic=True):
    """Calculate distances and distance errors for both stars with Gaia and HIP
    parallaxes. 
    
    Incorporates the systematic offset in Gaia DR2 by subtracting the offset
    from the parallax, then adding its uncertainty in quadrature. This makes 
    the parallax *bigger*.

    https://ui.adsabs.harvard.edu/abs/2018ApJ...862...61S/abstract
    """
    # Stassun & Torres systematic offsets
    if use_plx_systematic:
        plx_off = -0.082    # mas
        e_plx_off = 0.033   # mas

        # Incorporate the systematic
        plx = tgt_info["Plx"] - plx_off
        e_plx = np.sqrt(tgt_info["e_Plx"]**2 + e_plx_off**2)
    
    # Not using offsets
    else:
        plx = tgt_info["Plx"]
        e_plx = tgt_info["e_Plx"]

    # Compute distance
    tgt_info["Dist"] = 1000 / plx
    tgt_info["Dist"].where(~np.isnan(tgt_info["Dist"]), 
                       1000/tgt_info["Plx_alt"][np.isnan(tgt_info["Dist"])],
                       inplace=True)
    
    # Compute distance error
    # e_dist = |D*-1*e_plx / plx|
    tgt_info["e_Dist"] = np.abs(tgt_info["Dist"] * e_plx / plx)
    tgt_info["e_Dist"].where(~np.isnan(tgt_info["e_Dist"]),
                        np.abs(tgt_info["Dist"] * tgt_info["e_Plx"] 
                               / tgt_info["Plx"]))
    
    
def initialise_tgt_info(assign_default_uncertainties=True, lb_pc=70,
                        use_plx_systematic=True):
    """
    """
    # Import the base target info sans calculations
    tgt_info = load_target_information(
                assign_default_uncertainties=assign_default_uncertainties)        

    # Calculate distances and distance errors
    compute_dist(tgt_info, use_plx_systematic)

    # -------------------------------------------------------------------------
    # Convert Tycho magnitudes to Johnson-Cousins magnitudes
    # -------------------------------------------------------------------------
    # Convert Tycho V to Johnson system using Bessell 2000

    # For simplification during testing, remove any stars that fall outside the 
    # VT --> V conversion from Bessell 2000, or those science targets not
    # in use
    #tgt_info = tgt_info.drop(["GJ551","HD133869", "HD203608"])

    # Convert VT and BT to V and B
    # TODO: proper treatment of magnitude errors

    Bmag, Vmag = rphot.convert_vtbt_to_vb(tgt_info["BTmag"], tgt_info["VTmag"])

    tgt_info["Bmag"] = Bmag   
    tgt_info["e_Bmag"] = tgt_info["e_BTmag"]

    tgt_info["Vmag"] = Vmag   
    tgt_info["e_Vmag"] = tgt_info["e_VTmag"]

    # -------------------------------------------------------------------------
    # Correct photometry for interstellar extinction
    # -------------------------------------------------------------------------
    # These are the filter effective wavelengths *not* considering the effect  
    # of spectral type (Angstroms):
    # B = 4450, V = 5510, Hp = 5280, Bt = 4200, Vt = 5320, Bp = 5320, Rp = 7970
    # (https://ui.adsabs.harvard.edu/abs/2010A%26A...523A..48J/abstract)
    filter_eff_lambda = np.array([4450., 5510., 5280., 4200., 5320., 5320., 
                                  7970.])

    # Import/create the SpT vs B-V grid
    grid = rphot.create_spt_uv_grid()

    # Create a mask which has values of 1 for stars outside the local bubble,
    # values of 0 for stars within it. This is multiplied by the calculated 
    # extinction in each band, treating it as zero for stars within the bubble,
    # as calculated for those stars outside it.
    lb_mask = (tgt_info["Dist"] > lb_pc).astype(int)

    # Calculate selective extinction (i.e. (B-V) colour excess) only for stars
    # outside the local bubble
    print(tgt_info.columns)
    eb_v = rphot.calculate_selective_extinction(tgt_info["Bmag"], 
                                                tgt_info["Vmag"], 
                                                tgt_info["SpT_simple"], grid)
    eb_v *= lb_mask
    eb_v[eb_v==0] = 0.      # Remove -0 values
    tgt_info["eb_v"] = eb_v

    # Calculate V band extinction
    tgt_info["A_V"] = rphot.calculate_v_band_extinction(tgt_info["eb_v"])

    # Determine extinction
    a_mags = rphot.deredden_photometry(tgt_info[["Bmag", "Vmag", "Hpmag", 
                                                 "BTmag", "VTmag", "BPmag", 
                                                 "RPmag"]], 
                                      tgt_info[["e_Bmag", "e_Vmag", "e_Hpmag", 
                                                 "e_BTmag", "e_VTmag", 
                                                 "e_BPmag", "e_RPmag"]], 
                                      filter_eff_lambda, tgt_info["A_V"])

    # Correct for extinction only for those stars outside the Local Bubble
    tgt_info["Bmag_dr"] = tgt_info["Bmag"] - a_mags[:,0]# * lb_mask
    tgt_info["Vmag_dr"] = tgt_info["Vmag"] - a_mags[:,1]# * lb_mask
    tgt_info["Hpmag_dr"] = tgt_info["Hpmag"] - a_mags[:,2]# * lb_mask
    tgt_info["BTmag_dr"] = tgt_info["BTmag"] - a_mags[:,3]# * lb_mask
    tgt_info["VTmag_dr"] = tgt_info["VTmag"] - a_mags[:,4]# * lb_mask
    tgt_info["BPmag_dr"] = tgt_info["BPmag"] - a_mags[:,5]# * lb_mask
    tgt_info["RPmag_dr"] = tgt_info["RPmag"] - a_mags[:,6]# * lb_mask

    # Calculate predicted V-K colour
    tgt_info["V-K_calc"] = rphot.calc_vk_colour(tgt_info["VTmag_dr"], 
                                                tgt_info["RPmag_dr"])

    # -------------------------------------------------------------------------
    # Estimate angular diameters
    # -------------------------------------------------------------------------
    # Estimate angular diameters using colour relations. 
    # TODO: Is not correcting reddening for W1-3 appropriate given the laws 
    # don't extend that far?
    rdiam.predict_all_ldd(tgt_info)
    
    # Compute Teffs from Casagrande et al. 2010 relations
    import reach.parameters as rparam
    teffs, e_teffs = rparam.compute_casagrande_2010_teff(tgt_info["BTmag_dr"],  
                                                         tgt_info["VTmag_dr"], 
                                                         tgt_info["FeH_rel"])
    tgt_info["teff_casagrande"] = teffs
    tgt_info["e_teff_casagrande"] = e_teffs
    
    return tgt_info

# -----------------------------------------------------------------------------
# Saving and loading
# -----------------------------------------------------------------------------    
def load_sequence_logs():
    """Load the sequence and complete_sequence data structures.
    """
    pkl_obslog = open("data/pionier_observing_log.pkl", "r")
    complete_sequences = pickle.load(pkl_obslog)
    pkl_obslog.close()

    pkl_sequences = open("data/sequences.pkl", "r")
    sequences = pickle.load(pkl_sequences)
    pkl_sequences.close()
    
    return complete_sequences, sequences
    
    
def save_results(bs_results, results, folder):
    """Save the bootstrap and final results datastructures.
    """
    pkl_bs_results = open("results/%s/bs_results.pkl" % folder, "wb")
    pickle.dump(bs_results, pkl_bs_results)
    pkl_bs_results.close()

    pkl_results = open("results/%s/results.pkl" % folder, "wb")
    pickle.dump(results, pkl_results)
    pkl_results.close()


def load_results(folder):
    """Load the bootstrapped and final results pickle.
    """
    pkl_bs_results = open("results/%s/bs_results.pkl" % folder, "r")
    bs_results = pickle.load(pkl_bs_results)
    pkl_bs_results.close()

    pkl_results = open("results/%s/results.pkl" % folder, "r")
    results = pickle.load(pkl_results)
    pkl_results.close()
    
    return bs_results, results    
    
    
def save_sampled_params(sampled_params, folder, force_claret_params=False, 
                        final_teff_sample=False):
    """Save the sampled stellar parameters.
    """
    # Using literature teffs (but not forcing Claret params)
    if not force_claret_params and not final_teff_sample:
        pkl_params = open("/home2/ihernand/Desktop/reach/results/%s/sampled_params.pkl" % folder, "wb")
    
    # Using literature teffs (but forcing Claret params)
    elif force_claret_params and not final_teff_sample:
        pkl_params = open("/home2/ihernand/Desktop/reach/results/%s/sampled_params_claret.pkl" % folder, "wb")

    # Using interferometric teffs
    else:
        pkl_params = open("/home2/ihernand/Desktop/reach/results/%s/sampled_params_final.pkl" % folder, "wb")
        
    pickle.dump(sampled_params, pkl_params)
    pkl_params.close()
    
    
def load_sampled_params(folder, force_claret_params=False, 
                        final_teff_sample=False):
    """Load the sampled stellar parameters.
    """
    # Force use of claret params
    if not force_claret_params and not final_teff_sample:
        pkl_params = open("/home2/ihernand/Desktop/reach/results/%s/sampled_params.pkl" % folder, "r")
    
    # Standard load in using literature values
    elif force_claret_params and not final_teff_sample:
        pkl_params = open("/home2/ihernand/Desktop/reach/results/%s/sampled_params_claret.pkl" % folder, "r")
    
    # Loading in sample using interferometric teffs
    elif final_teff_sample:
        pkl_params = open("/home2/ihernand/Desktop/reach/results/%s/sampled_params_final.pkl" % folder, "r")
    
    sampled_params = pickle.load(pkl_params)
    pkl_params.close()    
    
    return sampled_params
    
    
def save_sampled_ldd(n_pred_ldd, e_pred_ldd, folder):
    """Save the sampled values of LDD and the corresponding uncertainties.
    """
    pkl_pred_ldd = open("results/%s/n_pred_ldd.pkl" % folder, "wb")
    pickle.dump(n_pred_ldd, pkl_pred_ldd)
    pkl_pred_ldd.close()
    
    pkl_e_pred_ldd = open("results/%s/e_pred_ldd.pkl" % folder, "wb")
    pickle.dump(e_pred_ldd, pkl_e_pred_ldd)
    pkl_e_pred_ldd.close()
    
    
def load_sampled_ldd(folder):
    """Load in the sampled values of LDD and the corresponding uncertainties.
    """
    pkl_pred_ldd = open("results/%s/n_pred_ldd.pkl" % folder, "r")
    n_pred_ldd = pickle.load(pkl_pred_ldd)
    pkl_pred_ldd.close()

    pkl_e_pred_ldd = open("results/%s/e_pred_ldd.pkl" % folder, "r")
    e_pred_ldd = pickle.load(pkl_e_pred_ldd)
    pkl_e_pred_ldd.close()  
    
    return n_pred_ldd, e_pred_ldd


def sampling_already_done(folder, force_claret_params=False):
    """Return True if both the diameters, their uncertainties, and science
    target parameters have already been sampled, false if otherwise.
    """
    if (os.path.exists("results/%s/n_pred_ldd.pkl" % folder) 
        and os.path.exists("results/%s/e_pred_ldd.pkl" % folder)
        and ((os.path.exists("results/%s/sampled_params.pkl" % folder) 
              and not force_claret_params)) 
            or ((os.path.exists("results/%s/sampled_params_claret.pkl" % folder) 
              and force_claret_params))):
        return True
    else:
        return False
          
# -----------------------------------------------------------------------------
# Scientific Notation formatting
# -----------------------------------------------------------------------------
def fexp(number):
    """Calculate the base 10 exponent of the input for use in scientific 
    notation representation.
    
    Per: https://stackoverflow.com/questions/45332056/
    decompose-a-float-into-mantissa-and-exponent-in-base-10-without-strings
    """
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def fman(number):
    """Calculate the base 10 mantissa of the input for use in scientific 
    notation representation.
    
    Per: https://stackoverflow.com/questions/45332056/
    decompose-a-float-into-mantissa-and-exponent-in-base-10-without-strings
    """
    return Decimal(number).scaleb(-fexp(number)).normalize()
    
# -----------------------------------------------------------------------------
# Collating results pdf
# -----------------------------------------------------------------------------
def collate_cal_pdfs():
    """Merge diagnostic pdfs together for easy checking.
    
    Code from:
    https://stackoverflow.com/questions/3444645/merge-pdf-files
    """
    from PyPDF2 import PdfFileMerger

    pdfs = glob.glob("/priv/mulga1/arains/pionier/complete_sequences/"
                     "*_abcd/*-*-*/*CAL*vis2*")
    
    # Sort by star name (otherwise will just be sorted by date)
    pdfs = np.array(pdfs)
    targets = np.array([pdf.split("CAL_")[-1] for pdf in pdfs])
    pdfs = pdfs[np.argsort(targets)]
    
    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(open(pdf, 'rb'))

    with open('cal_vis2_summary.pdf', 'wb') as fout:
        merger.write(fout)
       
       
def get_mean_delta_h(tgt_info, complete_sequences, sequences):
    """
    """
    used_cals = {}
    
    for seq in complete_sequences.keys():
        # Get the science target id
        sci = get_unique_key(tgt_info, [seq[1]])[0]
        sci_h = tgt_info.loc[sci]["Hmag"]
        
        # Bright/faint?
        seq_kind = seq[2]
        
        # Get the calibrator IDs
        cals = get_unique_key(tgt_info, sequences[seq][0::2])
        
        # Take only the used calibrators
        cals = tgt_info.loc[cals][tgt_info.loc[cals]["Quality"]!="BAD"].index
        
        if (sci, seq_kind) in used_cals.keys():
            used_cals[(sci, seq_kind)] += list(cals)
        else:
            used_cals[(sci, seq_kind)] = list(cals)    
        
    bright_delta_h = []
    faint_delta_h = []
    
    bright_ldd_frac = []
    faint_ldd_frac = []
    
    for seq in used_cals.keys():
        # Get science info
        sci = seq[0]
        sci_h = tgt_info.loc[sci]["Hmag"]
        sci_ldd = tgt_info.loc[sci]["LDD_pred"]
        
        cals = list(set(used_cals[seq]))
        
        cal_delta_h = list(tgt_info.loc[cals]["Hmag"].values - sci_h)
        cal_frac_ldd = list(tgt_info.loc[cals]["LDD_pred"].values / sci_ldd)
        
        if "bright" in seq:
            bright_delta_h += cal_delta_h
            bright_ldd_frac += cal_frac_ldd
        
        elif "faint" in seq:
            faint_delta_h += cal_delta_h
            faint_ldd_frac += cal_frac_ldd
    
    print("Bright delta H = %0.2f" % np.nanmean(bright_delta_h))
    print("Faint delta H = %0.2f\n" % np.nanmean(faint_delta_h))
    
    print("Bright LDD frac = %0.2f" % np.nanmean(bright_ldd_frac))
    print("Faint LDD frac = %0.2f" % np.nanmean(faint_ldd_frac))


def summarise_cs(results):
    """
    """
    cs = np.hstack(results["C_SCALE"].values)
    seq_order = np.vstack(results["SEQ_ORDER"].values)    
    
    bmask = [i for i, seq in enumerate(seq_order) if "bright" in seq]   
    fmask = [i for i, seq in enumerate(seq_order) if "faint" in seq]   
    
    print("AVG: %0.2f \pm %0.2f" % (cs.mean(), cs.std()))
    print("Bright: %0.2f \pm %0.2f" % (cs[bmask].mean(), cs[bmask].std()))  
    print("Faint: %0.2f \pm %0.2f" % (cs[fmask].mean(), cs[fmask].std())) 


def summarise_percentages(tgt_info):
    """
    """
    mask = tgt_info["Science"] & tgt_info["in_paper"]

    udd_pc = tgt_info[mask]["e_udd_final"]/tgt_info[mask]["udd_final"] * 100
    ldd_pc = tgt_info[mask]["e_ldd_final"]/tgt_info[mask]["ldd_final"] * 100
    rad_pc = tgt_info[mask]["e_r_star_final"]/tgt_info[mask]["r_star_final"] * 100
    fb_pc = tgt_info[mask]["e_f_bol_final"]/tgt_info[mask]["f_bol_final"] * 100
    l_pc = tgt_info[mask]["e_L_star_final"]/tgt_info[mask]["L_star_final"] * 100
    teff_pc = tgt_info[mask]["e_teff_final"]/tgt_info[mask]["teff_final"] * 100

    print("Mean UDD precision: %0.2f %%" % udd_pc.mean())
    print("Mean LDD precision: %0.2f %%" % ldd_pc.mean())
    print("Mean R precision: %0.2f %%" % rad_pc.mean())
    print("Mean fbol precision: %0.2f %%" % fb_pc.mean())
    print("Mean L precision: %0.2f %%" % l_pc.mean())
    print("Mean Teff precision: %0.2f %%" % teff_pc.mean())


# -----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------- 
def format_id(text_ids):
    """Swap in an ID better for LateX formatting. Recursive.
    """
    star_greek_map = {"TauCet":r"$\tau$ Cet",
                      "alfHyi":r"$\alpha$ Hyi",
                      "chiEri":r"$\chi$ Eri",
                      "95Cet":r"95 Cet A",
                      "epsEri":r"$\epsilon$ Eri",
                      "delEri":r"$\delta$ Eri",
                      "omi2Eri":r"40 Eri A",
                      "37Lib":r"37 Lib",
                      "betTrA":r"$\beta$ TrA",
                      "lamSgr":r"$\lambda$ Sgr",
                      "delPav":r"$\delta$ Pav",
                      "epsInd":r"$\epsilon$ Ind",
                      "HD131977":r"HD131977",
                      "etaSco":r"$\eta$ Sco",
                      "betAql":r"$\beta$ Aql",
                      "HR7221":r"HR7221",}
    
    # Single ID
    if type(text_ids)==str:
        if text_ids in star_greek_map:
            return star_greek_map[text_ids]
        else:
            return text_ids
     
    # List of IDs
    elif hasattr(text_ids, "__len__"):
        return [format_id(tid) for tid in text_ids] 
        
    else:
        return text_ids       