"""
"""
from __future__ import division, print_function
import os
import sys
import numpy as np
import pandas as pd
import astropy.constants as apc
import reach.limb_darkening as rld
import reach.utils as rutils
# -----------------------------------------------------------------------------
# Sampling Parameters
# -----------------------------------------------------------------------------   
def sample_stellar_params_pd(tgt_info, n_bootstraps, 
                             assign_default_uncertainties=True):
    """Sample stellar parameters for use when calculating the limb-darkening
    coefficient.
    """
    # Make new dataframes for each stellar parameter
    ids = tgt_info[tgt_info["Science"]].index.values
    
    n_logg = pd.DataFrame(np.zeros([n_bootstraps, len(ids)]), columns=ids)
    n_teff = pd.DataFrame(np.zeros([n_bootstraps, len(ids)]), columns=ids)
    n_feh = pd.DataFrame(np.zeros([n_bootstraps, len(ids)]), columns=ids)
    
    if assign_default_uncertainties:
        # logg
        logg_mask = np.logical_and(np.isnan(tgt_info["e_logg"]), 
                                   tgt_info["Science"])
        tgt_info["e_logg"].where(~logg_mask, 0.2, inplace=True)
        
        # [Fe/H]
        feh_mask = np.logical_and(np.isnan(tgt_info["e_FeH_rel"]), 
                                  tgt_info["Science"])
        tgt_info["e_FeH_rel"].where(~feh_mask, 0.1, inplace=True)
        
        # Teff
        teff_mask = np.logical_and(np.isnan(tgt_info["e_teff"]), 
                                   tgt_info["Science"])
        tgt_info["e_teff"].where(~teff_mask, 100, inplace=True)      
    
    
    for id in ids:
        n_logg[id] = np.random.normal(tgt_info.loc[id, "logg"],
                                      tgt_info.loc[id, "e_logg"],
                                      n_bootstraps)   
        n_teff[id] = np.random.normal(tgt_info.loc[id, "Teff"],
                                      tgt_info.loc[id, "e_teff"],
                                      n_bootstraps) 
        n_feh[id] = np.random.normal(tgt_info.loc[id, "FeH_rel"],
                                      tgt_info.loc[id, "e_FeH_rel"],
                                      n_bootstraps)                                             
    return n_logg, n_teff, n_feh 


def sample_parameters(tgt_info, n_bootstraps, force_claret_params=False, 
                      use_literature_teffs=True):
    """Sample stellar parameters (teff, logg, feh) and derived parameters 
    (u_lambda, s_lambda) N times and save in a single datastructure. The result
    is a 3D pandas dataframe, with N frames of:
        [star, logg, teff, feh, u_elc 1-6, s_lambda, u_ftc 1-6]
        
    This 3D pandas dataframe will have the structure:
        [star, [bs_i, logg, teff, feh, u_lld_1, u_lld_2, u_lld_3, u_lld_4, 
                u_lld_5, u_lld_6, u_scale]]
    """
    # Get the science target IDs - these will be the rows/index column
    ids = tgt_info[np.logical_and(tgt_info["Science"], 
                                  tgt_info["in_paper"])].index.values
    
    u_lambda_cols = ["u_lambda_%i" % ui for ui in np.arange(0,6)]
    s_lambda_cols = ["s_lambda_%i" % ui for ui in np.arange(0,6)]
    cols = ["logg", "teff", "feh"] + u_lambda_cols + s_lambda_cols
    
    # Initialise list that will hold all frames until we combine at the end
    frames = []
    
    #data = np.zeros( (len(ids), n_bootstraps , len(cols)) )
    
    print("Sampling Teff, logg, [Fe/H], u_lambda, and s_lambda for %i stars..." 
          % len(ids)) 
    
    if use_literature_teffs:
        print("Using **literature** values for Teff")
    else:
        print("Using **interferometric** values for Teff")
    
    for id_i, id in enumerate(ids):
        # Initialise
        data = np.zeros( (n_bootstraps , len(cols)) )
        
        # Sample stellar parameters
        n_logg = np.random.normal(tgt_info.loc[id, "logg"],
                                  tgt_info.loc[id, "e_logg"], n_bootstraps)
        
        # First time we sample, want to use the literature values of Teff
        if use_literature_teffs:
            n_teff = np.random.normal(tgt_info.loc[id, "Teff"],
                                      tgt_info.loc[id, "e_teff"], n_bootstraps)
        
        # After we have fit for LDD and derived an interferometric Teff using
        # fbol we can iterate the fitting procedure by using these new 
        # (hopefully) more realistic values
        else:
            n_teff = np.random.normal(tgt_info.loc[id, "teff_final"],
                                      tgt_info.loc[id, "e_teff_final"], 
                                      n_bootstraps)
        
        n_feh = np.random.normal(tgt_info.loc[id, "FeH_rel"],
                                 tgt_info.loc[id, "e_FeH_rel"], n_bootstraps)
                                  
        # Sample equivalent linear coefficients + scaling parameter
        print("Sampling parameters for %s..." % tgt_info.loc[id]["Primary"], 
              end="")
        sys.stdout.flush()
        n_u_lambda = rld.sample_lld_coeff(n_logg, n_teff, n_feh, 
                                            force_claret_params)
        
        # Assemble
        data[:, 0] = n_logg
        data[:, 1] = n_teff
        data[:, 2] = n_feh
        data[:, 3:] = n_u_lambda
        
        frames.append(pd.DataFrame(data, columns=cols))
    
    # Construct hierarchical dataframe
    sampled_sci_params = pd.concat(frames, keys=ids)
        
    return sampled_sci_params
        

def sample_bc_magnitudes(sampled_sci_params, tgt_info):
    """Sample the magnitudes used for bolometric corrections.
    """
    # Get the star IDs and do this one star at a time
    star_ids = set(np.vstack(sampled_sci_params.index)[:,0])
    
    # Initialise the new columns
    mag_labels = ["Hpmag_dr", "BTmag_dr", "VTmag_dr", "BPmag_dr", "RPmag_dr"]
    e_mag_labels = ["e_Hpmag", "e_BTmag", "e_VTmag", "e_BPmag", "e_RPmag"]
    for mag in mag_labels:
        sampled_sci_params[mag] = 0
    
    print("Sampling magnitudes for %i stars..." % len(star_ids)) 
        
    # Go through star by star and populate
    for star in star_ids:
        for mag, e_mag in zip(mag_labels, e_mag_labels):
            mags = np.random.normal(tgt_info.loc[star][mag], 
                                    tgt_info.loc[star][e_mag], 
                                    len(sampled_sci_params.loc[star]))

            sampled_sci_params.loc[star, mag] = mags


def compute_sampled_fbol(sampled_sci_params, band_mask=[1, 0, 0, 0, 0]):
    """Derive the sampled value of fbol from the sampled magnitude and BC from
    Casagrande BC code.
    
    Currently just averages those bands specified in the band mask.
    """
    # Get the star IDs and do this one star at a time
    star_ids = set(np.vstack(sampled_sci_params.index)[:,0])
    
    # Initialise the new columns
    bc_labels = ["BC_Hp", "BC_BT", "BC_VT", "BC_BP", "BC_RP"]
    mag_labels = ["Hpmag_dr", "BTmag_dr", "VTmag_dr", "BPmag_dr", "RPmag_dr"]
    fbol_labels = ["f_bol_Hp", "f_bol_BT", "f_bol_VT", "f_bol_BP", "f_bol_RP"]
    e_fbol_labels = ["e_f_bol_Hp", "e_f_bol_BT", "e_f_bol_VT", "e_f_bol_BP", 
                     "e_f_bol_RP"]
    
    for fbol in fbol_labels:
        sampled_sci_params[fbol] = 0
    
    # Plus the representative fbol and its error
    sampled_sci_params["f_bol_final"] = 0
    #sampled_sci_params["e_f_bol_final"] = 0
        
    masked_fbol = np.array(fbol_labels)[band_mask]
    e_masked_fbol = np.array(e_fbol_labels)[band_mask]    
        
    print("Computing sampled fbol for %i stars..." % len(star_ids))    
        
    # Go through star by star and populate
    for star in star_ids:
        
        
        for mag, bc, fbol in zip(mag_labels, bc_labels, fbol_labels):
            print("DEBUG compute_sampled_fbol")
            print("star =", star)
            print("bc column =", bc)
            print("mag column =", mag)
            bc_raw = sampled_sci_params.loc[star, bc]
            mag_raw = sampled_sci_params.loc[star, mag]

            bcs = np.atleast_1d(bc_raw)
            mags = np.atleast_1d(mag_raw)
            print("bcs first values =", bcs[:5])
            print("mags first values =", mags[:5])
            bcs = pd.to_numeric(pd.Series(bcs), errors="coerce").values
            mags = pd.to_numeric(pd.Series(mags), errors="coerce").values


            good = np.isfinite(bcs) & np.isfinite(mags)

            if np.sum(good) == 0:
                print("WARNING: no valid BC/mag values for")
                print("  star =", star)
                print("  bc column =", bc)
                print("  mag column =", mag)
                sampled_sci_params.loc[star, fbol] = np.nan
            else:
                sampled_sci_params.loc[star, fbol] = calc_f_bol(
                bcs[good],
                mags[good]
            )
        
        # Compute the "final" value of fbol for each iteration
        #weights = sampled_sci_params.loc[star][e_masked_fbosl].values**(-2)
        f_bol_avg = np.average(sampled_sci_params.loc[star][masked_fbol].values, 
                               axis=1)
                               #weights=weights, axis=1)
        #e_f_bol_avg = (np.sum(weights, axis=1)**-1)**0.5
        
        sampled_sci_params.loc[star, "f_bol_final"] = f_bol_avg
        #sampled_sci_params.loc[star, "e_f_bol_final"] = e_f_bol_avg
        

def sample_distance(sampled_sci_params, tgt_info):
    """Sample the distance to the star.
    """
    # Get the star IDs and do this one star at a time
    star_ids = set(np.vstack(sampled_sci_params.index)[:,0])
    
    # Initialise the new columns
    sampled_sci_params["Dist"] = 0
    
    print("Sampling distances for %i stars..." % len(star_ids))    
        
    # Go through star by star and populate
    for star in star_ids:
        dist = np.random.normal(tgt_info.loc[star]["Dist"], 
                                tgt_info.loc[star]["e_Dist"], 
                                len(sampled_sci_params.loc[star]))

        sampled_sci_params.loc[star, "Dist"] = dist


def sample_all(tgt_info, n_bootstraps, bc_path, force_claret_params=False, 
               band_mask=[1, 0, 0, 0, 0], use_literature_teffs=True):
    """Sample each of teff, logg, [Fe/H], u_lambda, s_lambda, Hp, VT, BT, BP,
    RP, BC_Hp, BC_VT, BC_BT, BC_BP, BC_RP, fbol_Hp, fbol_VT, fbol_BT, fbol_BP, 
    fbol_RP, fbol_final, L_star, r_star, and distance using literature values
    and their errors, or the Casagrande BC code.
    """
    # Sample stellar parameters and limb darkening coefficients (and initialise
    # the 3D pandas dataframe
    sampled_sci_params = sample_parameters(tgt_info, n_bootstraps, 
                                    force_claret_params, use_literature_teffs)
                                    
    # Then sample distance, magnitudes and bolometric corrections, and fbol
    sample_distance(sampled_sci_params, tgt_info)
    
    sample_bc_magnitudes(sampled_sci_params, tgt_info)

    
    sample_casagrande_bc(sampled_sci_params, bc_path)
    
    compute_sampled_fbol(sampled_sci_params, band_mask)
    
    calc_all_L_bol(sampled_sci_params)
    
    return sampled_sci_params

# -----------------------------------------------------------------------------
# Combining distributions
# -----------------------------------------------------------------------------   
def update_target_info_with_ldd_fits(tgt_info, results):
    """Combine independent measures of LDD from multiple different sequences to
    a single measurement of LDD +/- e_LDD
    """
    stars = set(results["HD"])
    
    tgt_info["ldd_final"] = np.zeros(len(tgt_info))
    tgt_info["e_ldd_final"] = np.zeros(len(tgt_info))
    
    tgt_info["udd_final"] = np.zeros(len(tgt_info))
    tgt_info["e_udd_final"] = np.zeros(len(tgt_info))
    
    # For every star, update tgt_info with its angular diameter info.
    for star_i, star in enumerate(stars):
        star_data = results[results["HD"]==star].iloc[0]
        
        # Limb darkened disc diameter
        tgt_info.loc[star, "ldd_final"] = star_data["LDD_FIT"]
        tgt_info.loc[star, "e_ldd_final"] = star_data["e_LDD_FIT"]
        
        # Uniform disc diameter
        tgt_info.loc[star, "udd_final"] = star_data["UDD_FIT"]
        tgt_info.loc[star, "e_udd_final"] = star_data["e_UDD_FIT"]
        

def combine_u_s_lambda(tgt_info, sampled_sci_params):
    """Add the mean and standard deviation of each of the 6 u_lambda and 
    s_lambda parameters to tgt_info.
    """
    # Determine u_lld from its distribution
    scis = tgt_info[np.logical_and(tgt_info["Science"], 
                                   tgt_info["in_paper"])].index.values

    # Initialise columns
    u_lambda_cols = ["u_lambda_%i" % ui for ui in np.arange(0,6)]
    e_u_lambda_cols = ["e_u_lambda_%i" % ui for ui in np.arange(0,6)]
    s_lambda_cols = ["s_lambda_%i" % ui for ui in np.arange(0,6)]
    e_s_lambda_cols = ["e_s_lambda_%i" % ui for ui in np.arange(0,6)]

    all_cols = (u_lambda_cols + e_u_lambda_cols + s_lambda_cols 
                + e_s_lambda_cols)

    for col in all_cols:
        tgt_info[col] = np.zeros(len(tgt_info))*np.nan

    # Populate tgt_info
    for sci in scis:
        tgt_info.loc[sci, u_lambda_cols] = \
            np.nanmean(sampled_sci_params.loc[sci][u_lambda_cols].values, axis=0)
        tgt_info.loc[sci, e_u_lambda_cols] = \
            np.nanstd(sampled_sci_params.loc[sci][u_lambda_cols].values, axis=0)
        tgt_info.loc[sci, s_lambda_cols] = \
            np.nanmean(sampled_sci_params.loc[sci][s_lambda_cols].values, axis=0)
        tgt_info.loc[sci, e_s_lambda_cols] = \
            np.nanstd(sampled_sci_params.loc[sci][s_lambda_cols].values, axis=0)


# -----------------------------------------------------------------------------
# Calculating physical parameters
# -----------------------------------------------------------------------------
def calc_all_f_bol(tgt_info, sampled_sci_params, band_mask=[1, 1, 1, 0, 0]):
    """f_bol in ergs s^-1 cm^-2
    """
    # Define bands to reference, construct new headers
    bands = ["Hp", "BT", "VT", "BP", "RP"]
    e_bands = ["e_%s_dr" % band for band in bands] 
    f_bol_bands = ["f_bol_%s" % band for band in bands] 
    e_f_bol_bands = ["e_f_bol_%s" % band for band in bands] 
    
    for band in bands:
        tgt_info["f_bol_%s" % band] = np.zeros(len(tgt_info))
        tgt_info["e_f_bol_%s" % band] = np.zeros(len(tgt_info))
        
    # And the averaged fbol value
    tgt_info["f_bol_final"] = np.zeros(len(tgt_info))
    tgt_info["e_f_bol_final"] = np.zeros(len(tgt_info))
    
    # Get the star IDs and do this one star at a time
    star_ids = set(np.vstack(sampled_sci_params.index)[:,0])
    
    masked_fbol = np.array(f_bol_bands)[band_mask]
    e_masked_fbol = np.array(e_f_bol_bands)[band_mask]
    
    # Calculate bolometric fluxes for each band for every star
    for star in star_ids:
        for fbol_lbl, e_fbol_lbl in zip(f_bol_bands, e_f_bol_bands):
            fbol = np.mean(sampled_sci_params.loc[star, fbol_lbl].values)
            e_fbol = np.std(sampled_sci_params.loc[star, fbol_lbl].values)
            
            tgt_info.loc[star, fbol_lbl] = fbol
            tgt_info.loc[star, e_fbol_lbl] = e_fbol
            
    # Now use a weighted average to work out fbol, using the reciprocal of
    # the variance as weights. Only use those bands specified in the mask
    masked_fbol = np.array(f_bol_bands)[band_mask]
    e_masked_fbol = np.array(f_bol_bands)[band_mask]
    
    for star_i, (star, row) in enumerate(tgt_info[tgt_info["Science"]].iterrows()):
        weights = row[e_masked_fbol][row[e_masked_fbol] > 0].values**(-2)
        f_bol_avg = np.average(row[masked_fbol][row[masked_fbol] > 0].values, 
                               weights=weights)
        e_f_bol_avg = (np.sum(weights)**-1)**0.5
        
        tgt_info.loc[star, "f_bol_final"] = f_bol_avg
        tgt_info.loc[star, "e_f_bol_final"] = e_f_bol_avg
        

def calc_all_r_star(sampled_sci_params):
    """Calculate the radius of each science target in units of Solar radii.
    """
    # Constants
    pc = apc.pc.value         # m / pc
    r_sun = apc.R_sun.value   # m
    
    # Get the star IDs and do this one star at a time
    star_ids = set(np.vstack(sampled_sci_params.index)[:,0])
    
    sampled_sci_params["r_star_final"] = 0
    
    # Compute the physical radii
    for star in star_ids:
        # Convert to km and radians
        dist_m = sampled_sci_params.loc[star]["Dist"].values * pc
        #e_dist_km = row["e_Dist"] * pc
        ldds = sampled_sci_params.loc[star]["LDD_FIT"].values 
        ldds_rad = ldds * np.pi/180/3600/1000
        #e_ldd_rad = row["e_ldd_final"] * np.pi/180/3600/1000
        
        # Calculate the stellar radii
        r_stars = 0.5 * ldds_rad * dist_m / r_sun
        #e_r_star = r_star * ((e_ldd_rad/ldd_rad)**2
        #                     + (e_dist_km/dist_km)**2)**0.5
    
        sampled_sci_params.loc[star, "r_star_final"] = r_stars
        #tgt_info.loc[star, "e_r_star_final"] = e_r_star        
    
    
def calc_all_teff(sampled_sci_params):
    """Calculate the effective temperature for all stars
    """ 
    # Stefan-Boltzmann constant
    sigma =  apc.sigma_sb.cgs.value #erg cm^-2 s^-1 K^-4
    
    # Get the star IDs and do this one star at a time
    star_ids = set(np.vstack(sampled_sci_params.index)[:,0])
    
    # And the averaged fbol value
    sampled_sci_params["teff_final"] = 0
    
    # Calculate the Teff for every star using an MC sampling approach         
    for star in star_ids:
        # Put LDD in radians
        ldds = sampled_sci_params.loc[star]["LDD_FIT"] * np.pi/180/3600/1000
        
        # Sample fbol
        f_bols = sampled_sci_params.loc[star]["f_bol_final"]
        
        # Calculate Teff
        teffs = (4*f_bols / (sigma * ldds**2))**0.25 
        
        # Store final value
        sampled_sci_params.loc[star, "teff_final"] = teffs.values


def calc_all_L_bol(sampled_sci_params):
    """Calculate the stellar luminosity with respect to Solar.
    """
    # Constants
    L_sun = apc.L_sun.cgs.value   # erg s^-1
    pc = apc.pc.cgs.value         # cm / pc
    
    # Get the star IDs and do this one star at a time
    star_ids = set(np.vstack(sampled_sci_params.index)[:,0])
    
    # Initialise L_star column
    sampled_sci_params["L_star_final"] = 0
    #sampled_sci_params["e_L_star_final"] = 0
    
    # Calculate the Teff for every star using an MC sampling approach         
    for star in star_ids:
        # Sample fbol
        f_bols = sampled_sci_params.loc[star]["f_bol_final"].values
        
        # Sample distances
        dists = sampled_sci_params.loc[star]["Dist"].values * pc
        
        # Calculate luminosities
        L_stars = 4 * np.pi * f_bols * dists**2
        
        # Calculate final L_star (in solar units) and error
        #L_star = np.mean(L_stars) / L_sun
        #e_L_star = np.std(L_stars) / L_sun
        
        # Store final value
        sampled_sci_params.loc[star, "L_star_final"] = L_stars / L_sun
        #tgt_info.loc[star, "e_L_star_final"] = e_L_star

  
def calc_f_bol(bc, mag):
    """Calculate the bolometric flux from a bolometric correction and mag.
    """
    L_sun = apc.L_sun.cgs.value # erg s^-1
    au = apc.au.cgs.value       # cm
    M_bol_sun = 4.75
    
    exp = -0.4 * (bc - M_bol_sun + mag - 10)
    
    f_bol = (np.pi * L_sun / (1.296 * 10**9 * au)**2) * 10**exp
    
    return f_bol      
    

def calc_L_star(tgt_info):
    """Calculate the absolute stellar luminosity using the bolometric magnitude 
    """
    L_sun = apc.L_sun.cgs.value # erg s^-1
    au = apc.au.cgs.value       # cm
    M_bol_sun = 4.75
    
    tgt_info["L_star"] = 10**(-0.4 * (tgt_info["M_bol"] - M_bol_sun))


def calc_final_params(tgt_info, sampled_sci_params):
    """Average sampled parameters and get uncertainties from std.
    """
    # Get the star IDs and do this one star at a time
    star_ids = set(np.vstack(sampled_sci_params.index)[:,0])
    
    value_cols = ["f_bol_Hp", "f_bol_BT", "f_bol_VT", "f_bol_BP", "f_bol_RP",
              "f_bol_final", "teff_final", "L_star_final", "r_star_final"]
              
    error_cols = ["e_%s" % value for value in value_cols] 
    
    # Initialise
    for value, error in zip(value_cols, error_cols):
        tgt_info[value] = 0
        tgt_info[error] = 0
    
    for star in star_ids:
        # Populate
        values = np.nanmean(sampled_sci_params.loc[star][value_cols].values, axis=0)
        errors = np.nanstd(sampled_sci_params.loc[star][value_cols].values, axis=0)
        
        tgt_info.loc[star, value_cols] = values
        tgt_info.loc[star, error_cols] = errors


def calc_sample_and_final_params(tgt_info, sampled_sci_params, bs_results,
                                 results):
    """
    """
    # Combine sampled u_lambda and s_lambda
    combine_u_s_lambda(tgt_info, sampled_sci_params)

    # Add final diameter fits to tgt_info
    update_target_info_with_ldd_fits(tgt_info, results)

    # Merge (i.e. get LDD fits per iteration info in sampled_sci_params)
    merge_sampled_params_and_results(sampled_sci_params, bs_results, 
                                            tgt_info)
    # Calculate the physical radii for each iteration
    calc_all_r_star(sampled_sci_params)

    # Compute temperatures for each iteration
    calc_all_teff(sampled_sci_params)

    # Now get final parameters from the distributions in sampled_sci_params
    calc_final_params(tgt_info, sampled_sci_params)

# -----------------------------------------------------------------------------
# Empirical relations
# -----------------------------------------------------------------------------   
def compute_casagrande_2010_teff(BTmag, VTmag, fehs):
    """Compute Teff based on the Casagrande et al. 2010 empirical colour 
    relations using (B-V).
    """
    # [Fe/H] range: -2.7 - 0.4
    # (Bt-Vt) range: 0.19 - 1.49
    # N_stars = 251
    feh_bounds = [-2.7, 0.4]
    bt_vt_bounds = [0.19, 1.49]
    
    e_teff = 79
    coeff = [0.5839, 0.4000, -0.0067, -0.0282, -0.0346, -0.0087]
    
    bt_vt = BTmag-VTmag
    
    valid_bt_vt_i = np.logical_and(bt_vt > bt_vt_bounds[0], 
                                   bt_vt < bt_vt_bounds[1])
    valid_feh_i = np.logical_and(fehs > feh_bounds[0], 
                                 fehs < feh_bounds[1])
                                 
    valid_i = np.logical_and(valid_bt_vt_i, valid_feh_i)
    
    teff = 5040 / (coeff[0] + coeff[1]*bt_vt + coeff[2]*bt_vt**2 
           + coeff[3]*bt_vt*fehs + coeff[4]*fehs + coeff[5]*bt_vt*fehs**2)
    
    e_teff = np.ones(len(teff)) * e_teff
    #temp = np.polynomial.polynomial.polyval(BTmag-VTmag, coeff)
    
    # If outside of bounds, force to nan
    teff[~valid_i] = np.nan
    e_teff[~valid_i] = np.nan
    
    return teff, e_teff


# -----------------------------------------------------------------------------
# Working with Casagrande BC code
# ----------------------------------------------------------------------------- 
def sample_casagrande_bc(sampled_sci_params, bc_path):
    """Sample stellar parameters for use with the bolometric correction code
    from Casagrande & VandenBerg (2014, 2018a, 2018b):
    
    https://github.com/casaluca/bolometric-corrections
    
    Check that selectbc.data looks like this to Hp, Bt, Vt, Bp, Rp:
      1  = ialf (= [alpha/Fe] variation: select from choices listed below)
      5  = nfil (= number of filter bandpasses to be considered; maximum = 5)
     21 76  =  photometric system and filter (select from menu below)
     21 77  =  photometric system and filter (select from menu below)
     21 78  =  photometric system and filter (select from menu below)
     27 86  =  photometric system and filter (select from menu below)
     27 88  =  photometric system and filter (select from menu below)
    """
    # Get the star IDs and do this one star at a time
    star_ids = set(np.vstack(sampled_sci_params.index)[:,0])
    
    # Initialise the new columns
    bc_labels = ["BC_Hp", "BC_BT", "BC_VT", "BC_BP", "BC_RP"]
    for bc in bc_labels:
        sampled_sci_params[bc] = 0
    
    print("Sampling Casagrande bolometric corrections for %i stars..." 
          % len(star_ids))
        
    # Go through star by star and populate
    for star in star_ids:
        print("Getting BC for %s" % star)
        n_bs = len(sampled_sci_params.loc[star])
        id_fmt = star + "_%0" + str(int(np.log10(n_bs)) + 1) + "i"
        ids = [id_fmt % s for s in np.arange(0, n_bs)]
        #ids = np.arange(n_bs)
        ebvs = np.zeros(n_bs)
    
        cols = ["logg", "feh", "teff"]
        
        data = np.vstack((ids, sampled_sci_params.loc[star]["logg"], 
                          sampled_sci_params.loc[star]["feh"],
                          sampled_sci_params.loc[star]["teff"], ebvs)).T
    
        np.savetxt("%s/input.sample.all" % bc_path, data, delimiter=" ", fmt="%s")#, 
                   #fmt=["%s", "%0.3f", "%0.3f", "%0.2f", "%0.2f"])
    
        os.system("cd %s; ./bcall" % bc_path)
    
        # Load in the result
        results = pd.read_csv("%s/output.file.all" % bc_path, 
                              delim_whitespace=True)
        print("Reading BC file:", results)                    
        # Save the bolometric corrections
        bc_num_cols = ["BC_1", "BC_2", "BC_3", "BC_4", "BC_5"]
        sampled_sci_params.loc[star, bc_labels] = results[bc_num_cols].values


def sample_stellar_params(tgt_info, n_samples):
    """Sample stellar parameters for use with the bolometric correction code
    from Casagrande & VandenBerg (2014, 2018a, 2018b):
    
    https://github.com/casaluca/bolometric-corrections
    """
    loggs = []
    fehs = []
    teffs = []
    
    # Assign default errors to params
    tgt_info["e_logg"][np.logical_and(np.isnan(tgt_info["e_logg"]), 
                                      tgt_info["Science"])] = 0.1
    tgt_info["e_FeH_rel"][np.logical_and(np.isnan(tgt_info["e_FeH_rel"]), 
                                      tgt_info["Science"])] = 0.1
    tgt_info["e_teff"][np.logical_and(np.isnan(tgt_info["e_teff"]), 
                                      tgt_info["Science"])] = 100
    
    for star, row in tgt_info[tgt_info["Science"]].iterrows():
        loggs.append(np.random.normal(row["logg"], row["e_logg"], n_samples))
        fehs.append(np.random.normal(row["FeH_rel"], row["e_FeH_rel"], 
                                     n_samples))
        teffs.append(np.random.normal(row["Teff"], row["e_teff"], n_samples))
        
    params = np.vstack((np.array(loggs).flatten(), np.array(fehs).flatten(), 
                        np.array(teffs).flatten())).T
    
    np.savetxt("data/input.sample", params, delimiter=" ", 
               fmt=["%0.2f","%0.2f","%i"])
    
    return params    

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------   
def save_params(tgt_info):
    """Save parameters to a text file with uncertainties
    """
    # logg
    logg_mask = np.logical_and(np.isnan(tgt_info["e_logg"]), 
                               tgt_info["Science"])
    tgt_info["e_logg"].where(~logg_mask, 0.2, inplace=True)
    
    # [Fe/H]
    feh_mask = np.logical_and(np.isnan(tgt_info["e_FeH_rel"]), 
                              tgt_info["Science"])
    tgt_info["e_FeH_rel"].where(~feh_mask, 0.1, inplace=True)
    
    # Teff
    teff_mask = np.logical_and(np.isnan(tgt_info["e_teff"]), 
                               tgt_info["Science"])
    tgt_info["e_teff"].where(~teff_mask, 100, inplace=True)    
    
    # Save
    path = "white_ld/pionier_targets_new.txt"
    cols = ["Primary", "Teff", "e_teff", "logg", "e_logg", "FeH_rel", 
            "e_FeH_rel"]
    tgt_info[cols][tgt_info["Science"]].to_csv(path, sep="\t",index=False)    
    
    
def print_mean_flux_errors(tgt_info):
    """
    """
    bands = ["Hp", "BT", "VT", "BP", "RP"]
    f_bols = ["f_bol_Hpmag", "f_bol_BTmag", "f_bol_VTmag", "f_bol_BPmag", 
             "f_bol_RPmag"]
    e_f_bols = ["e_f_bol_Hpmag", "e_f_bol_BTmag", "e_f_bol_VTmag", 
               "e_f_bol_BPmag", "e_f_bol_RPmag"]       
     
    print("Band | % err")           
    for f_i in np.arange(0, len(f_bols)):
        med_e_f_bol = (tgt_info[e_f_bols[f_i]][tgt_info["Science"]]
                       / tgt_info[f_bols[f_i]][tgt_info["Science"]]).median()
        print("%s --- %0.2f" % (bands[f_i], med_e_f_bol*100))         
    
    # For the averaged Fbol
    med_e_f_bol = (tgt_info["e_f_bol_avg"][tgt_info["Science"]]
                       / tgt_info["f_bol_avg"][tgt_info["Science"]]).median()
    print("\nAVG --- %0.2f" % (med_e_f_bol*100))
  
  
def merge_sampled_params_and_results(sampled_sci_params, bs_results, tgt_info):
    """For simplicity, merge sampled_sci_params and bs_results
    """
    # Get the IDs from bs_results
    prim_ids = bs_results.keys()
    
    # And the matching HD IDs
    hd_ids = rutils.get_unique_key(tgt_info, bs_results.keys())
    
    # Get the list of new columns. For now don't worry about the vectors
    #bs_cols = ['MJD', 'TEL_PAIR', 'VIS2', 'FLAG', 'BASELINE', 'WAVELENGTH',
    #           'LDD_FIT', 'LDD_PRED', 'C_SCALE']
    bs_cols = ["LDD_FIT"]
    
    for col in bs_cols:
        sampled_sci_params[col] = np.nan
    
    # Now go through and combine
    for prim_id, hd_id in zip(prim_ids, hd_ids):
        # Add to sampled_sci_params
        # Extra contingencies in case we're looking at results as we go
        n_total = len(sampled_sci_params.loc[hd_id])
        n_samples = len(bs_results[prim_id])
        
        results = np.zeros(( n_total, len(bs_cols))) * np.nan
        results[:n_samples,:] = bs_results[prim_id][bs_cols].values
        
        sampled_sci_params.loc[hd_id, bs_cols] = results