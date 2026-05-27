"""Module for angular diameter prediction, calculation, and vis^2 fitting
"""
from __future__ import division, print_function
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from collections import Counter
from astropy.io import fits
from scipy.special import jv
from scipy.optimize import curve_fit, fmin
from scipy.interpolate import LinearNDInterpolator
from scipy.odr import ODR, Model, Data, RealData

class UnknownOIFitsFileFormat(Exception):
    pass

class UnknownFittingRoutine(Exception):
    pass


def clean_name_for_match(name):
        name = str(name).strip()
        name = name.replace(" ", "")
        name = name.replace("_", "")
        name = name.lower()
        return name


# -----------------------------------------------------------------------------
# Predicting LDD
# -----------------------------------------------------------------------------
def predict_ldd_boyajian(F1_mag, F1_mag_err, F2_mag, F2_mag_err, colour=None,
                         colour_rel="V-W3"):
    """Calculate the limb darkened angular diameter as predicted by 
    colour-diameter relations from Boyajian et al. 2014:
     - http://adsabs.harvard.edu/abs/2014AJ....147...47B
        
    Parameters
    ----------
    F1_mag: float
        Magnitude of the star in the first filter.
    F1_mag_err: float
        Error in magnitude of the star in the first filter.
    F2_mag: float
        Magnitude of the star in the second filter.
    F2_mag_err: float
        Error in magnitude of the star in the second filter.
    colour_rel: string
        The colour relation to use for predicting the LDD (i.e. V-K)
        
    Returns
    -------
    ldd: float
        The predicted limb darkened angular diameter
    """
    # Convert to numpy arrays
    F1_mag = np.array(F1_mag)
    F1_mag_err = np.array(F1_mag_err)
    F2_mag = np.array(F2_mag)
    F2_mag_err = np.array(F2_mag_err)
    
    # Determine whether we have been provided with a pre-existing colour, and
    # if not compute it. If given a colour, we will use that instead.
    if colour is None:
        colour = F1_mag - F2_mag
    
    num_exponents = 4
    
    # Import the Boyajian relations, columns are:
    # [Colour Index, Num Points, Range (mag), a0, a0_err, a1, a1_err,
    #  a2, a2_err, a3, a3_err, Reduced chi^2, err (%)]
    dd = os.path.dirname(__file__)[:-5]
    boyajian_2014_rel_file = os.path.join(dd, 
                            "data/boyajian_2014_colour_diameter_relations.csv")
    
    diam_rel = np.loadtxt(boyajian_2014_rel_file, delimiter=",", 
                          skiprows=1, dtype="str")
    # Create dictionary for coefficients and for error on relation
    diam_rel_coeff = {}
    diam_rel_err = {}
    
    for rel in diam_rel:
        diam_rel_coeff[rel[0]] = rel[3:11].astype(float)
        diam_rel_err[rel[0]] = rel[-1].astype(float)
    
    # Calculate diameters
    # Relationship is log_diam = Sigma(i=0) a_i * (F1_mag-F2_mag) ** i
    exponents = np.arange(0, num_exponents)
    
    # Repeat along new direction so operation is vectorised
    colour = np.repeat(np.array(colour)[:,None], num_exponents, 1)
    
    log_diam = np.sum(diam_rel_coeff[colour_rel][::2]*colour**exponents, 1)
    
    ldd = 10**(-0.2*F1_mag) * 10**log_diam
    
    # Calculate the error. Currently this is done solely using the percentage
    # error given in the paper, and does not treat errors in either magnitude
    e_ldd = ldd * diam_rel_err[colour_rel]/100
    
    return ldd, e_ldd


def predict_ldd_bv_feh_boyajian(Bmag, Vmag, feh):
    """Predict theta_ld using a [Fe/H] dependent (B-V) colour relation from
    Boyajian et al. 2014:
     - http://adsabs.harvard.edu/abs/2014AJ....147...47B
    """
    b_v = np.repeat((Bmag - Vmag)[:, None], 4, 1)
    
    exponents = np.arange(4)
    coeffs = np.array([0.52005, 0.90209, -0.67448, 0.39767, -0.08476])
    e_coeffs = np.array([0.00121, 0.01348, 0.03676, 0.02611, 0.00161])
    log_diam = np.sum(coeffs[:-1]*b_v**exponents,1) + coeffs[-1]*feh
    
    ldd = 10**(-0.2*Vmag) * 10**log_diam
    e_ldd = ldd * 0.045
    
    return ldd, e_ldd
    
    
def predict_ldd_kervella(V_mag, V_mag_err, K_mag, K_mag_err):
    """Calculate the limb darkened angular diameter as predicted by 
    colour-diameter relations from Kervella et al. 2004:
     - http://adsabs.harvard.edu/abs/2004A%26A...426..297K
    """
    # Calculate the LDD
    log_ldd = 0.0755 * (V_mag - K_mag) + 0.5170 - 0.2 * K_mag

    ldd = 10**log_ldd

    # Calculate the error on this value (assuming no covariance)
    log_ldd_err = np.sqrt((0.0755*V_mag_err)**2 + (0.2755*K_mag_err)**2)

    ldd_err = 10**(np.log(10)*log_ldd_err)
                                       
    return log_ldd, log_ldd_err, ldd, ldd_err   


def predict_all_ldd(tgt_info):
    """
    
    Given 2MASS is saturated, there are three colour relations available:
     1 - V-W3 (where V is converted from Vt through Bessell 2000 relations)
     2 - V-W4 (where V is converted from Vt through Bessell 2000 relations)
     3 - V-K (where V-K is computed using Vt-Rp/V-K relation)
     
    This is also the order of preference for the relations, with most stars
    having V-W3. V-W4 is only used for stars with saturated W3, and V-K for
    those stars without ALLWISE data.
    """
    # V-K from Vt-Rp relation
    ldd_vk, e_ldd_vk = predict_ldd_boyajian(tgt_info["Vmag_dr"],
                                            tgt_info["e_Vmag"], None, None,
                                            tgt_info["V-K_calc"], "V-K")
    # V-W3                                              
    ldd_vw3, e_ldd_vw3 = predict_ldd_boyajian(tgt_info["Vmag_dr"], 
                                              tgt_info["e_Vmag"], 
                                              tgt_info["W3mag"], 
                                              tgt_info["e_W3mag"], None, 
                                              "V-W3") 
    # V-W4                                                
    ldd_vw4, e_ldd_vw4 = predict_ldd_boyajian(tgt_info["Vmag_dr"], 
                                              tgt_info["e_Vmag"], 
                                              tgt_info["W4mag"], 
                                              tgt_info["e_W4mag"], None, 
                                              "V-W4") 
    
    # B-V
    ldd_bv, e_ldd_bv = predict_ldd_boyajian(tgt_info["Bmag_dr"], 
                                              tgt_info["e_Bmag"], 
                                              tgt_info["Vmag_dr"], 
                                              tgt_info["e_Vmag"], None, 
                                              "B-V") 
    
    # B-V, [Fe/H] dependent
    ldd_bv_feh, e_ldd_bv_feh = predict_ldd_bv_feh_boyajian(tgt_info["Bmag_dr"], 
                                                  tgt_info["Vmag_dr"], 
                                                  tgt_info["FeH_rel"]) 
    
    # Save these values
    tgt_info["LDD_VK"] = ldd_vk
    tgt_info["e_LDD_VK"] = e_ldd_vk

    tgt_info["LDD_VW3"] = ldd_vw3
    tgt_info["e_LDD_VW3"] = e_ldd_vw3
    
    tgt_info["LDD_VW4"] = ldd_vw4
    tgt_info["e_LDD_VW4"] = e_ldd_vw4
    
    tgt_info["LDD_BV"] = ldd_bv
    tgt_info["e_LDD_BV"] = e_ldd_bv
    
    tgt_info["LDD_BV_feh"] = ldd_bv_feh
    tgt_info["e_LDD_BV_feh"] = e_ldd_bv_feh
    
    ldd_pred = []
    e_ldd_pred = []
    
    # Save the 
    for star, star_data in tgt_info.iterrows():
        ldd_rel = star_data["LDD_rel"]
        
        if type(ldd_rel) is str:
            ldd_pred.append(star_data[ldd_rel])
            e_ldd_pred.append(star_data["e_%s" % ldd_rel])
        
        # For those stars without a relation (entirely bad calibrators that 
        # will be excluded), assign placeholder diameters so the PIONIER
        # pipeline is happy    
        else:
            ldd_pred.append(1.0)
            e_ldd_pred.append(0.1)
            
    tgt_info["LDD_pred"] = ldd_pred
    tgt_info["e_LDD_pred"] = e_ldd_pred
        
     
# -----------------------------------------------------------------------------
# Fitting LDD
# -----------------------------------------------------------------------------
def vis2_fit_loss(params, sfreq, vis2, e_vis2, n_points, u_lambda, s_lambda):
    """
    """
    # Unpack params
    #n_seq = len(params) // 2
    ldd = params[0]
    c_scale = params[1:]

    # Calculate the predicted V^2 values using current values of C and LDD
    vis2_pred = calc_vis2(sfreq, ldd, c_scale, n_points, u_lambda, s_lambda)

    # Calculate and function to minimise
    return np.sum((vis2_pred - vis2)**2 / e_vis2)


def calc_vis2(sfreq, ldd, c_scale, n_points, u_lambda, s_lambda):
    """Calculates squared fringe visibility assuming a linearly limb-darkened 
    disk. As outlined in Hanbury Brown et al. 1974: 
     - http://adsabs.harvard.edu/abs/1974MNRAS.167..475H
        
    This function is called from fit_for_ldd using scipy.optimize.curve_fit.
        
    Parameters
    ----------
    sfreq: float or float array
        Spatial frequency, equivalent to projected baseline B (m), divided by 
        the wavelength lambda (m).
    
    ldd: float or float array
        The limb-darkened angular diameter (mas)
        
    c_scale: float
        Scaling parameter to not force the fit to be anchored at 1.
    
    n_points:
        ...

    u_lambda: float or float array
        Array of wavelength dependent limb-darkening coefficients

    s_lambda: 
        ...
        
    Returns
    -------
    vis2: float or float array
        Calibrated squared fringe visibility
    """
    # Ensure C is a list
    if not hasattr(c_scale, '__len__'):
        c_scale = [c_scale]

    # Reformat c_scale
    if len(n_points) > 1:
        # There is one intercept parameter per set of data, so we create an
        # array equal to the total number of points, but this array has n_seq
        # different c_scale values within it mapped to those points.
        # e.g. for a standard bright/faint sequence, we'll have an array of
        # length 144, but with the first 72 entries being c_scale[0] and the
        # final 72 entries being c_scale[1].
        c_array = np.hstack([c_scale[ni]*np.ones(n) 
                             for ni, n in enumerate(n_points)])
        
    # Only one value of c_scale to worry about    
    else:
        c_array = c_scale[0] * np.ones(n_points[0])
        
    # Calculate x and convert ldd to radians (this serves two purposes: making
    # the input/output more human readable, and the fitting function performs
    # better
    x = np.pi * sfreq * (ldd / 1000 / 3600 / 180 * np.pi) * s_lambda
    
    vis = (((1 - u_lambda)/2 + u_lambda/3)**-1 * 
          ((1 - u_lambda)*jv(1,x)/x 
            + u_lambda*(np.pi/2)**0.5 * jv(3/2,x)/x**(3/2)))
    
    # Square visibility and return       
    return c_array * vis**2


def calc_vis2_odr(beta, sfreq):
    """Calculates squared fringe visibility assuming a linearly limb-darkened 
    disk. As outlined in Hanbury Brown et al. 1974: 
     - http://adsabs.harvard.edu/abs/1974MNRAS.167..475H
        
    This function calls calc_vis2 after unpacking the inputs from the format
    required of scipy.odr.ODR. Called from fit_for_ldd. 
        
    Parameters
    ----------
    beta: tuple
        Tuple containing ldd, c_scale, n_seq, n_points, u_lambda, s_lambda.
        
    sfreq: float or float array
        The projected baseline B (m), divided by the wavelength lambda (m)
        
    Returns
    -------
    vis2: float or float array
        Calibrated squared fringe visibility
    """
    # Unpack beta, whose length depends on how many sequences of data have 
    # been passed in [ldd, u_lld, s_lambda, c_scale, n_points] where there is 
    # a single value of ldd, 6 values for each of u_lld and s_lambda, and n_seq
    # values of c_scale and n_points.
    n_seq = (len(beta) - 2*len(sfreq) - 1)//2
    
    ldd = beta[0]                                  # Limb-darkened diameter
    c_scale = beta[1:n_seq+1]                      # Intercept
    n_points = np.array(beta[n_seq+1:2*n_seq+1], dtype=int) # Points/vis2 seq
    u_lambda = beta[-2*len(sfreq):-len(sfreq)]        # Limb darkening coeffs
    s_lambda = beta[-len(sfreq):]                  # LDD scaling parameter
    
    vis2 = calc_vis2(sfreq, ldd, c_scale, n_points, u_lambda, s_lambda)
         
    return vis2
         

def format_vis2_data(vis2, e_vis2, baselines, wavelengths, e_wl_frac,
                     u_lambda_6, s_lambda_6):
    """Given a set of vis2, baseline, and wavelength, and e_wl data, construct
    flattened arrays of only valid (i.e. non-nan) data to pass to the fitting
    functions.
    
    To allow for using different limb-darkening coefficients for each 
    wavelength channel, we can't lose track of the wavelength channels. That 
    is, initially, all arrays will be divisble by 6 (the number of wavelength
    channels), and if we exclude bad data without keeping track of this, we
    won't be able to do the fit properly. The (dodgy) solution is to renove the
    bad data from the vis2, e_vis2, and e_sfreq vectors, but put nans in the 
    corresponding sfreq vector which will be used during the fit to remove the
    correct corresponding wavelength channels. Not ideal, but simplest solution
    given the current implementation.
    """
    # Create and reshape the spatial frequency data
    n_bl = len(baselines)
    n_wl = len(wavelengths)
    bl_grid = np.tile(baselines, n_wl).reshape([n_wl, n_bl]).T
    wl_grid = np.tile(wavelengths, n_bl).reshape([n_bl, n_wl])
    sfreq = (bl_grid / wl_grid).flatten()         
    
    # Flatten
    e_sfreq = e_wl_frac * sfreq
    vis2 = vis2.flatten()
    e_vis2 = e_vis2.flatten()
    
    # The total number of points should always be divisible by 6 (the number
    # of wavelength channels, so for u_lld and s_lambda simply tile them by 
    # the number of baselines observed
    n_bl = len(sfreq) // 6
    u_lambda = np.tile(u_lambda_6, n_bl)
    s_lambda = np.tile(s_lambda_6, n_bl)
    
    # Don't consider bad data during fitting process
    valid_i = ((vis2 >= 0) & (e_vis2 > 0) & ~np.isnan(vis2))
                   #& (~np.isnan(vis2.flatten())))
    
    # Only take only valid data
    sfreq = sfreq[valid_i]
    e_sfreq = e_sfreq[valid_i]
    vis2 = vis2[valid_i]
    e_vis2 = e_vis2[valid_i]
    u_lambda = u_lambda[valid_i]
    s_lambda = s_lambda[valid_i]
    
    assert len(sfreq) == len(u_lambda)
    
    return sfreq, e_sfreq, vis2, e_vis2, u_lambda, s_lambda
          
          
def fit_for_ldd(vis2, e_vis2, baselines, wavelengths, sampled_params, ldd_pred,
                method="ls", e_wl_frac=0.02, do_uniform_disc_fit=False,
                ls_ldd_lims=(0.1,10), ls_c_lims=(0.1,2)):
    """Fit to calibrated squared visibilities to obtain the measured limb-
    darkened stellar diameter in mas.
    
    Lambda function per:
     - https://stackoverflow.com/questions/12208634/fitting-only-one-
        parameter-of-a-function-with-many-parameters-in-python
        
    ODR per:
     - https://stackoverflow.com/questions/26058792/correct-fitting-with-
       scipy-curve-fit-including-errors-in-x
        
    Parameters
    ----------
    vis2: float array
        Calibrated squared visibiity measurements
        
    e_vis2: float array
        Error on the calibrated squared visibility measurements
        
    baselines: float array
        Projected interferometric baselines (m)
        
    wavelengths: float array
        Wavelengths the observations were taken at (m)

    sampled_params: pandas dataframe
        Sampled coefficents of each star where the limb darkening coefficients
        are stored.

    u_lld: float
         Wavelength dependent linear limb-darkening coefficient 
         
    ldd_pred: float
        Predicted limb-darkened stellar angular diameter (mas)
    
    method: string
        "odr" or "ls" for orthogonal distance regression or least squares
        fitting respectively.

    e_wl_frac: float
        Fractional error on wavelength scale.

    do_uniform_disc_fit: boolean
        Whether to do a uniform disc fit, or make use of the limb darkening
        coefficents.
    
    ls_ldd_lims: tuple
        Tuple of form (ldd_min, ldd_max) to give reasonable limits to least
        squares fitting.

    Returns
    -------
    popt: float
        Optimal values for limb-darkened diameter (mas) and scaling param C
        
    pcov: float
        Errors (one standard deviation) oon LDD and C
    """
    # Get limb darkening coefficients and scaling parameters
    u_lambda_cols = ["u_lambda_%i" % ui for ui in np.arange(0,6)]
    s_lambda_cols = ["s_lambda_%i" % ui for ui in np.arange(0,6)]
    
    # If doing a uniform disc fit, use zeroes for the limb darkening coeffs, 
    # and ones for the scaling parameter
    if do_uniform_disc_fit:
        u_lambda_6 = np.zeros(6)
        s_lambda_6 = np.ones(6)
    else:
        u_lambda_6 = sampled_params[u_lambda_cols].values
        s_lambda_6 = sampled_params[s_lambda_cols].values
    
    # Two cases for actual data:
    # 1 - We're fitting multiple sequences, in which case vis2, e_vis2, and 
    #     baseline vectors will have 3 dimensions [n_seq, n_exp, n_bl]. In this
    #     case, we need to have a separate C intercept parameter per seq.
    # 2 - We're fitting to sequences independently, so the same vectors won't
    #     have the n_seq dimension, and C, whilst still a vector, will only 
    #     contain a single value.
    
    # Case 1: multiple sequences
    if type(vis2) == list:
        n_seq = len(vis2)
        
        # This will be passed in as a parameter to tell the fitting function
        # how to reshape the c_scale vector to simultaneously fit to all points
        n_points = np.zeros(n_seq, dtype=int)
        
        # Construct new lists to hold the formatted/flattened results
        sfreq = []
        e_sfreq = []
        fvis2 = []
        e_fvis2 = []
        u_lambda = []
        s_lambda = []
        
        for seq_i in np.arange(0, n_seq):
            sf, e_sf, v2, e_v2, u_l, s_l = format_vis2_data(vis2[seq_i], 
                                            e_vis2[seq_i], baselines[seq_i], 
                                            wavelengths, e_wl_frac, u_lambda_6, 
                                            s_lambda_6)
                                            
            # Stack to "flatten" the data from multiple sequences
            sfreq = np.hstack((sfreq, sf))
            e_sfreq = np.hstack((e_sfreq, e_sf))
            fvis2 = np.hstack((fvis2, v2))
            e_fvis2 = np.hstack((e_fvis2, e_v2))
            u_lambda = np.hstack((u_lambda, u_l))
            s_lambda = np.hstack((s_lambda, s_l))
            
            
            n_points[seq_i] = len(sf)
        
        n_points = tuple(n_points)
        
    # Case 2: single sequence
    else:
        n_seq = 1
    
        sfreq, e_sfreq, fvis2, e_fvis2, u_lambda, s_lambda = \
                format_vis2_data(vis2, e_vis2, baselines, wavelengths, 
                                 e_wl_frac, u_lambda_6, s_lambda_6)
        n_points = (len(sfreq),)
    
    # Initial C param
    c_scale = np.ones(n_seq)

    # Fit for LDD using least-squares, chi2 metric to take into account errors
    # This is implemented using a loss function, and scipy.optimize.fmin 
    if method == "ls":
        params = np.hstack(([ldd_pred], c_scale))

        xopt, chi2, n_it, n_fc, wf = fmin(vis2_fit_loss, params,
                                          args=(sfreq, fvis2, e_fvis2, 
                                                n_points, u_lambda, s_lambda), 
                                          full_output=True, disp=False)
        
        print("chi2=%0.2f, nit=%i, " % (chi2, n_it), end="")

        return xopt, np.zeros_like(xopt)

    # Run instead using Orthogonal Distance Regression (ODR) so we can have 
    # uncertainties on the wavelength calibration. Value of 0 in ifixb fixes
    # the parameter (in this case u_lld and n_points).
    elif method == "odr":
        data = RealData(sfreq, fvis2, e_sfreq, e_fvis2)
        model = Model(calc_vis2_odr)
        
        # Construct coefficient vector, which requires "unpacking" all params
        # into a single 1D vector. Have order: [ldd, u_lld, c_scale, n_points]
        if n_seq == 1:
            params = np.hstack(([ldd_pred], c_scale, n_points, u_lambda, 
                                 s_lambda))
            ifixb = [1] + [1] + [0] + [0]*2*len(sfreq)
        
        else:
            params = np.hstack(([ldd_pred], c_scale, n_points, u_lambda, 
                                 s_lambda))
            ifixb = [1] + [1]*n_seq + [0]*n_seq + [0]*2*len(sfreq)
        
        odr = ODR(data, model, params, ifixb=ifixb)
        odr.set_job(fit_type=2)
        output = odr.run()   
        
        return output.beta, output.sd_beta 

    else:
        raise UnknownFittingRoutine()


def fit_all_ldd(vis2, e_vis2, baselines, wavelengths, tgt_info, pred_ldd_col,
                sampled_params, bs_i, method="ls", e_wl_frac=0.02,
                do_uniform_disc_fit=False):
    """Fits limb-darkened diameters to all science targets using all available
    vis^2, e_vis^2, and projected baseline data.
    
    Parameters
    ----------
    vis2: dict
        Dictionary mapping science target ID to all vis^2 values
    
    e_vis2: dict
        Dictionary mapping science target ID to all e_vis^2 values
        
    baselines: dict
        Dictionary mapping science target ID to all projected baselines (m)
    
    wavelengths: list
        List recording the wavelengths observed at (m)
    
    Returns
    -------
    successful_fits: list
        List containing ldd_opt, e_ldd_opt, c_scale, e_c_scale.
    """
    successful_fits = {}
    
    for sci in vis2.keys():
        # Only take the ID part of sci - could have " (Sequence)" after it
        tgt_info["Primary"] = [id.replace(" ", "").replace(".", "").replace("_","")
                       for id in tgt_info["Primary"]]

        tgt_info["Primary"] = [clean_name_for_match(x) for x in tgt_info["Primary"]]
        if type(sci) == tuple:

            sci_data = tgt_info[tgt_info["Primary"]==sci[0]]
        else:
            sci_data = tgt_info[tgt_info["Primary"]==sci]
        print(tgt_info["Primary"], sci)

        id = sci_data.index.values[0]
        
        # Print depending on what diameter we're fitting, and if not science
        if not sci_data["Science"].values:
            print("%s is not science target, aborting fit" % str(sci))
            continue
        else:
            if do_uniform_disc_fit:
                print("\tFitting uniform-disc to %s --> " % str(sci), end="")
            else:
                print("\tFitting limb-darkened disc to %s --> " % str(sci), 
                      end="")
        print(sampled_params.index.values[:10])
        popt, pstd = fit_for_ldd(vis2[sci], e_vis2[sci], 
                                 baselines[sci], wavelengths[sci], 
                                 sampled_params.loc[id].iloc[bs_i], 
                                 sci_data[pred_ldd_col].values[0], 
                                 method=method, e_wl_frac=e_wl_frac,
                                 do_uniform_disc_fit=do_uniform_disc_fit)
        
        # Extract parameters from fit. Parameters are ordered as follows, where
        # n is the number of sequences [LDD, u_lld, C_n, N_n]
        # Expectation is that this will break for the least-squares method, but
        # can fix here if that is needed again.
        if type(vis2[sci]) == list:
            n_seq = len(vis2[sci])
        else:
            n_seq = 1
        
        ldd_opt = popt[0]
        e_ldd_opt = pstd[0]
        
        c_scale = popt[1:n_seq+1] 
        e_c_scale = pstd[1:n_seq+1] 
        
        # Everything else is constant, so don't need!
        
        print("%i seq, LDD=%0.2f, C=%s" 
              % (n_seq, ldd_opt, c_scale))
        
        successful_fits[sci] = [ldd_opt, e_ldd_opt, c_scale, e_c_scale]
            
    return successful_fits


# -----------------------------------------------------------------------------
# Working with oiFits files
# -----------------------------------------------------------------------------
def extract_vis2(oi_fits_file):
    """Read the calibrated squared visibility + errors, baseline, and 
    wavelength information from a given oifits file.
    
    Parameters
    ----------
    oi_fits_file: string
        Filepath to the oifits file
        
    Returns
    -------
    mjds: float array
        MJDs of the observations.
    
    pairs: string array
        Telescope pairs for each each baseline.
    
    vis2: float array
        Calibrated squared visibiity measurements
        
    e_vis2: float array
        Error on the calibrated squared visibility measurements
    
    flags: float array
        Quality flags for each observation.
        
    baseline: float array
        Projected interferometric baselines (m)
        
    wavelengths: float array
        Wavelengths the observations were taken at (m)
    """
    # The format of the oiFits file varies depending on how many CAL-SCI
    # CAL-SCI-CAL-SCI-CAL sequences were observed in the same night. For a
    # single sequence, the fits extensions are as follows:
    # [imageHDU, target info, wavelengths, telescopes, vis^2, t3phi]
    # When multiple sequences are observed, there are extra extensions for
    # each wavelength, vis^2, and t3phi set (e.g. two observed sequences 
    # would have two of each of these in a row)
    with fits.open(oi_fits_file, memmap=False) as oifits:
        #oifits = fits.open(oi_fits_file)
        n_extra_seq = (len(oifits) - 6) // 3
        
        mjds = []
        pairs = []
        vis2 = []
        e_vis2 = []
        flags = []
        baselines = []
        wavelengths = []
    
        # Retrieve visibility and baseline information for an arbitrary (>=1) 
        # number of sequences within a given night
        for seq_i in xrange(0, n_extra_seq+1):
            oidata = oifits[4 + n_extra_seq + seq_i].data
            
            # Figure out how large the chunk is. For the majority of cases
            # there will be two different MJDs (barring any weird sequences or
            # dropped baselines). Want to figure out how many of the maximum
            # 6 baselines per observation are available. If we don't have six
            # baselines, we need to insert an empty placeholder set to keep the
            # ordering and ensure we can compute means/standard deviations 
            # later.
            
            mjd_counts = Counter(oidata["MJD"])
            unique_mjds = list(set(oidata["MJD"]))
            unique_mjds.sort()
            n_1st_mjd = mjd_counts[unique_mjds[0]]
            
            # Sometimes the MJDs are split trivially in time (< 1 minute),
            # which splits the baselines into chunks smaller than 6. Science
            # observations actually split in time are actually 15 mins+ apart.
            # Thus we want to count any close in time as occurring at the same
            # time.
            for mjd in unique_mjds[1:]:
                if ((mjd - unique_mjds[0]) * 24 * 60) < 5: # < 5 mins in time
                    n_1st_mjd += mjd_counts[mjd]
            
            expected_pairs = set(["1-2", "1-3", "1-4", "2-3", "2-4", "3-4"])
            
            observed_pairs = np.array(["%i-%i" % (tel[0], tel[1]) 
                                  for tel in oidata["STA_INDEX"][:n_1st_mjd]])
            
            # Grab the relevant info prior to modification
            mjds_obs = oidata["MJD"]
            pairs_obs = np.array(["%i-%i" % (tel[0], tel[1]) 
                                  for tel in oidata["STA_INDEX"]])
            vis2_obs = oidata["VIS2DATA"]
            e_vis2_obs = oidata["VIS2ERR"]
            flags_obs = oidata["FLAG"]
            baselines_obs = np.sqrt(oidata["UCOORD"]**2 + oidata["VCOORD"]**2)
            
            # For every missing baseline, insert dummy NaN data to keep array
            # dimensions the same
            for missing_bl in list(expected_pairs - set(observed_pairs)):
                print("\tAdding missing info for first science block on %s" 
                      % oi_fits_file)
                mjds_obs = np.insert(mjds_obs, n_1st_mjd, np.nan)
                pairs_obs = np.insert(pairs_obs, n_1st_mjd, missing_bl)
                vis2_obs = np.insert(vis2_obs, n_1st_mjd, [np.nan]*6, axis=0)
                e_vis2_obs = np.insert(e_vis2_obs, n_1st_mjd, [np.nan]*6, 
                                       axis=0)
                flags_obs = np.insert(flags_obs, n_1st_mjd, [np.nan]*6, 
                                      axis=0)
                baselines_obs = np.insert(baselines_obs, n_1st_mjd, np.nan, 
                                          axis=0)
                
            # Now do this again for the other expected observation
            observed_pairs = np.array(["%i-%i" % (tel[0], tel[1]) 
                                  for tel in oidata["STA_INDEX"][6:]])
            
            for missing_bl in list(expected_pairs - set(observed_pairs)):
                print("\tAdding missing info for first science block on %s" 
                      % oi_fits_file)
                mjds_obs = np.insert(mjds_obs, 6, np.nan)
                pairs_obs = np.insert(pairs_obs, 6, missing_bl)
                vis2_obs = np.insert(vis2_obs, 6, [np.nan]*6, axis=0)
                e_vis2_obs = np.insert(e_vis2_obs, 6, [np.nan]*6, axis=0)
                flags_obs = np.insert(flags_obs, 6, [np.nan]*6, axis=0)
                baselines_obs = np.insert(baselines_obs, 6, np.nan, axis=0)
            
            # Sort baselines within each observation (chunk of 6) to ensure 
            # ordering is the same for bootstrapping. Given there are two 
            # observations of each science target within the 
            # CAL1-SCI1-CAL2-SCI2-CAL3 sequence, there will be 2 sets of six
            # per sequence. To simplify the sorting procedure, convert the 
            # tuple pairs of telescope IDs to a string.
            #tel_pairs = np.array(["%i-%i" % (tel[0], tel[1]) 
                                  #for tel in oidata["STA_INDEX"]])
                                  
            order = np.concatenate((pairs_obs[:6].argsort(), 
                                    pairs_obs[6:].argsort() + 6))
            
            # New solution to keep sequences separate is to simply append them
            # to the list as they come in, which per the oifits standard are
            # already sorted in time
            mjds.append(mjds_obs[order])
            pairs.append(pairs_obs[order])
            vis2.append(vis2_obs[order])
            e_vis2.append(e_vis2_obs[order])
            flags.append(flags_obs[order])
            baselines.append(baselines_obs[order])
            
        # Assume that we'll always be using same wavelength mode within a night
        wavelengths = oifits[2].data["EFF_WAVE"]
    
    return mjds, pairs, vis2, e_vis2, flags, baselines, wavelengths


def collate_vis2_from_file(results_path, bs_i=None, separate_sequences=False):
    """Collates calibrated squared visibilities, errors, baselines, and 
    wavelengths for each science target in the specified results folder.
    
    Parameters
    ----------
    results_path: string
        Directory where the calibrated oifits results files are stored.
        
    Returns
    -------
    all_mjds: dict
        Dictionary mapping science target ID to all observation MJDs (times).
    
    all_tel_pairs: dict
         Dictionary mapping science target ID to all telescope pairs.
    
    all_vis2: dict
        Dictionary mapping science target ID to all vis^2 values
    
    all_e_vis2: dict
        Dictionary mapping science target ID to all e_vis^2 values
        
    all_flags: dict    
         Dictionary mapping science target ID to all quality flags.
        
    all_baselines: dict
        Dictionary mapping science target ID to all projected baselines (m)
    
    wavelengths: list
        List recording the wavelengths observed at (m)
    """
    # Initialise data structures to store calibrated results, where dict keys
    # are the science target IDs. Note that the wavelengths are common to all.
    all_mjds = {}
    all_tel_pairs = {}
    all_vis2 = {}
    all_e_vis2 = {}
    all_flags = {}
    all_baselines = {}
    all_wavelengths = {}
    sequence_order = {}
    
    ith_bs_oifits = glob.glob(results_path 
                              + "*SCI*oidataCalibrated_%02i.fits" % bs_i)
    ith_bs_oifits.sort()
    
    # We want to keep the bright and faint sequences separate for 
    # diagnostic purposes, but still need to collate in the instance
    # that a star has duplicate sequences on the same night
    dates_obs = pd.read_csv("data/dates_observed.tsv", sep="\t")

    dates_obs["star_clean"] = dates_obs["star"].apply(clean_name_for_match)
    dates_obs["b_night"] = dates_obs["b_night"].astype(str).str.strip()
    dates_obs["f_night"] = dates_obs["f_night"].astype(str).str.strip()

    print(dates_obs)
    
    print("\nFound %i oifits file/s for bootstrap %i" % (len(ith_bs_oifits), 
                                                       bs_i+1))
    
    for oifits in ith_bs_oifits:
        # Get the target name from the file name - this is clunky, but more
        # robust than the former method of slicing using static indices which
        # inherently assumes a constant file length (which changes when we
        # begin bootstrapping)
        sci_raw = oifits.split("SCI")[1].split("oidata")[0].replace("_", "")
        sci = clean_name_for_match(sci_raw)
        
        # Extract data from oifits file. If multiple sequences were observed
        # on the same night, each of the retuned lists will contain more than
        # one list of results
        mjds, pairs, vis2, e_vis2, flags, baselines, wavelengths = \
            extract_vis2(oifits)
        
        for seq_i in np.arange(0, len(mjds)):
            # Figure out what sequence we're dealing with
            night = oifits.split("/")[-1].split("_SCI")[0]
        
            faint_entry = dates_obs[np.logical_and(dates_obs["star"]==sci, 
                                            dates_obs["f_night"]==night)]
        
            bright_entry = dates_obs[np.logical_and(dates_obs["star"]==sci, 
                                            dates_obs["b_night"]==night)]
            
            # If returning both a faint and bright entry, need to define 
            # which is which - create a tuple of form (id, seq, period)
            print(bright_entry, faint_entry)
            if len(bright_entry) > 0 and len(faint_entry) > 0:
                # Bright
                if seq_i == bright_entry["b_order"].values[0]:
                    seq_tup = (sci, "bright", 
                               bright_entry["period"].values[0])
                
                elif seq_i == faint_entry["f_order"].values[0]:
                    seq_tup = (sci, "faint", 
                              faint_entry["period"].values[0])
                                
            elif len(bright_entry) > 0 and len(faint_entry) == 0:
                seq_tup = (sci, "bright", bright_entry["period"].values[0])
            
            elif len(bright_entry) == 0 and len(faint_entry) > 0:
                seq_tup = (sci, "faint", faint_entry["period"].values[0])
            
            # If keeping the sequences separate, the ID we use will be the 
            # sequence tuple (star, bright/faint, period) as the ID, otherwise
            # just the star ID
            
            if separate_sequences:
                seq_id = seq_tup
                
            # Just use the science target    
            else:
                seq_id = sci
                
        # Extract data from oifits file and stack as appropriate
        #mjds, pairs, vis2, e_vis2, flags, baselines, wavelengths = \
            #extract_vis2(oifits)

            if seq_id not in all_vis2.keys():
                all_mjds[seq_id] = mjds[seq_i]
                all_tel_pairs[seq_id] = pairs[seq_i]
                all_vis2[seq_id] = vis2[seq_i]
                all_e_vis2[seq_id] = e_vis2[seq_i]
                all_flags[seq_id] = flags[seq_i]
                all_baselines[seq_id] = baselines[seq_i]
                all_wavelengths[seq_id] = wavelengths
                sequence_order[seq_id] = [seq_tup]
            
            else:
                all_mjds[seq_id] = np.hstack((all_mjds[seq_id], mjds[seq_i]))
                all_tel_pairs[seq_id] = np.hstack((all_tel_pairs[seq_id], 
                                                   pairs[seq_i]))
                all_vis2[seq_id] = np.vstack((all_vis2[seq_id], 
                                              vis2[seq_i]))
                all_e_vis2[seq_id] = np.vstack((all_e_vis2[seq_id], 
                                                e_vis2[seq_i]))
                all_flags[seq_id] = np.vstack((all_flags[seq_id], 
                                               flags[seq_i]))
                all_baselines[seq_id] = np.hstack((all_baselines[seq_id], 
                                                   baselines[seq_i]))
                all_wavelengths[seq_id] = wavelengths 
                sequence_order[seq_id].append(seq_tup)
                                                   
    return all_mjds, all_tel_pairs, all_vis2, all_e_vis2, all_flags, \
           all_baselines, all_wavelengths, sequence_order
    
    
# -----------------------------------------------------------------------------
# Sampling diameters
# -----------------------------------------------------------------------------    
def sample_n_pred_ldd(tgt_info, n_bootstraps, pred_ldd_col="LDD_pred", 
                      e_pred_ldd_col="e_LDD_pred",
                      do_gaussian_diam_sampling=True):
    """Prepares a pandas dataframe of predicted target diameters for 
    bootstrapping over. Each row will either be sampled from a Gaussian 
    distribution if doing calibrator bootstrapping, otherwise will simply be N
    repeats of the actual predicted diameters.
    
    Parameters
    ----------
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
        
    n_bootstraps: int
        The number of bootstrapping iterations to run.
        
    pred_ldd_col: string
        The column to use from tgt_info for the predicted diameters.
        
    e_pred_ldd_col: string
        The column to use from tgt_info for the predicted diameter 
        uncertainties.
        
    do_gaussian_diam_sampling: bool
        Boolean indicating whether to sample the n_bootstraps LDD from a 
        Gaussian distribution constructed from pred_ldd_col and e_pred_ldd_col,
        or simply make n_bootstraps repeats of the predicted diameters without
        sampling.
        
    Returns
    -------
    n_pred_ldd: pandas dataframe
        Pandas dataframe with columns being stars, and each row being a set of
        LDD for a given bootstrapping iteration. If not doing calibrator 
        bootstrapping (do_gaussian_diam_sampling=False), each row will be the 
        same, but otherwise the calibrator angular diameters are drawn from a 
        Gaussian distribution as part of the bootstrapping.
    
    e_pred_ldd: pandas dataframe
        Pandas dataframe with columns being stars, and the values being the 
        uncertainties corresponding to n_pred_ldd. Only one row.    
    """
    # Get the IDs
    ids = tgt_info.index.values
    
    e_pred_ldd = pd.DataFrame([tgt_info[e_pred_ldd_col].values], columns=ids)
    
    # If not running second stage of bootstrapping on calibrator predicted 
    # diameters create a dataframe with duplicate rows simply as the predicted
    # calibrator diameters, rather than drawing from a Gaussian distribution
    if not do_gaussian_diam_sampling:
        print("No calibrator bootstrapping --> using actual predicted LDD")
        n_pred_ldd = pd.DataFrame([tgt_info[pred_ldd_col].values], 
                                      columns=ids)
        n_pred_ldd = pd.concat([n_pred_ldd]*n_bootstraps, ignore_index=True)
        return n_pred_ldd, e_pred_ldd
        
    # Otherwise we are running cal bootstrapping, draw LDD from a Gaussian dist
    # Make a new pandas dataframe with columns representing an individual star,
    # and each row being the predicted LDD (pulled from a Gaussian 
    # distribution) for the ith bootstrapping iteration
    print("Calibrator bootstrapping --> drawing LDD from Gaussian dist")
    ldds = np.zeros([n_bootstraps, len(ids)])
    
    n_pred_ldd = pd.DataFrame(ldds, columns=ids)
    
    for id in ids:
        n_pred_ldd[id] = np.random.normal(tgt_info.loc[id, pred_ldd_col],
                                              tgt_info.loc[id, e_pred_ldd_col],
                                              n_bootstraps)
    return n_pred_ldd, e_pred_ldd
    
    
    
# -----------------------------------------------------------------------------
# Aggregating results from bootstrapping runs
# -----------------------------------------------------------------------------    
def fit_ldd_for_all_bootstraps(tgt_info, n_bootstraps, results_path, 
                               sampled_params, pred_ldd_col="LDD_pred", 
                               method="ls", e_wl_frac=0.02, 
                               prune_errant_baselines=True, 
                               separate_sequences=True, combined_fit=True):
    """Collates all bootstrapped oifits files within results_path into
    sumarising pandas dataframes. 
    
    Parameters
    ----------
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
        
    n_bootstraps: int
        The number of bootstrapping iterations to run.
    
    results_path: string
        Path to store the bootstrapped oifits files.
            
    pred_ldd_col: string
        The column to use from tgt_info for the predicted diameters.
        
    prune_errant_baselines: boolean
        Whether to replace non-matching vis2 and baseline data with NaNs to
        facillitate computation of vis2 errors. This does not affect the fitted
        LDD or its error computed from the distribution of fitted LDDs, and is
        purely to allow plotting of vis2 curves with errors.
            
    Returns
    -------
    bs_results: dict of pandas dataframes
        Dictionary with science targets as keys, containing pandas dataframes
        recording the results of each bootstrapping iteration as rows.
    """
    # Determine the stars that we have results on
    #oifits_files = glob.glob(results_path + "*SCI*.fits")
    #oifits_files.sort()
    mjds, pairs, vis2, e_vis2, flags, baselines, wavelengths, seq_order = \
            collate_vis2_from_file(results_path, 0, separate_sequences)
    
    #stars = set([file.split("SCI")[-1].split("oidata")[0].replace("_","")
                 #for file in oifits_files])
                 
    sequence_ids = mjds.keys()
    sequence_ids.sort()
    
    # If combining, take only the star IDs
    if combined_fit:
        sequence_ids = set(np.array(sequence_ids)[:,0])
                
    # Initialise a pandas dataframe for each star. At present it's hard to
    # entirely preallocate memory, but we'll try to at least preallocate the
    # rows
    cols1 = ["MJD", "TEL_PAIR", "VIS2", "FLAG", "BASELINE", 
            "WAVELENGTH", "LDD_FIT",  "LDD_PRED", "e_LDD_PRED", "u_LLD",
            "C_SCALE"]
            
    # Store the results for each star in a pandas dataframe, accessed by key 
    # from a dictionary
    bs_results = {}
        
    for star in sequence_ids:
        bs_results[star] = pd.DataFrame(index=np.arange(0, n_bootstraps), 
                                     columns=cols1)
        # TEL_PAIR --> array of tuples, MJD --> array of floats, VIS2 -->
        # array of 6 length arrays, BASELINE --> array of floats, WAVELENGTH 
        # --> array of 6 length arrays, FLAG --> array of 6 length arrays,
        # LDD --> array of floats
        bs_results[star]["MJD"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["TEL_PAIR"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["VIS2"] = np.zeros((n_bootstraps, 0)).tolist()
        #bs_results[star]["e_VIS2"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["FLAG"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["BASELINE"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["WAVELENGTH"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["LDD_FIT"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["C_SCALE"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["SEQ_ORDER"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["UDD_FIT"] = np.zeros((n_bootstraps, 0)).tolist()
        bs_results[star]["C_SCALE_UDD"] = np.zeros((n_bootstraps, 0)).tolist()
    
    # Fit a LDD for every bootstrap iteration, and save the vis2, time, 
    # baseline, and wavelength information from each iteration
    for bs_i in np.arange(0, n_bootstraps):
        # Collate the information
        mjds, pairs, vis2, e_vis2, flags, baselines, wavelengths, seq_order = \
            collate_vis2_from_file(results_path, bs_i, separate_sequences)
        
        # If doing combined fit, combine (stack in new dimension) data from the 
        # same star
        if combined_fit:
            # Combine like stars, add to dict, pop old values 
            sequence_ids = mjds.keys()
            sequence_ids.sort()
            
            for seq_id in sequence_ids:
                star = seq_id[0]
                
                # Create new dict entry, make 
                if star not in mjds:
                    mjds[star] = [mjds[seq_id]]
                    pairs[star] = [pairs[seq_id]]
                    vis2[star] = [vis2[seq_id]]
                    e_vis2[star] = [e_vis2[seq_id]]
                    flags[star] = [flags[seq_id]]
                    baselines[star] = [baselines[seq_id]]
                    wavelengths[star] = wavelengths[seq_id]
                    seq_order[star] = seq_order[seq_id]
                
                # Append to old dict entry, don't stack wl dimension
                else:
                    mjds[star].append(mjds[seq_id])
                    pairs[star].append(pairs[seq_id])
                    vis2[star].append(vis2[seq_id])
                    e_vis2[star].append(e_vis2[seq_id])
                    flags[star].append(flags[seq_id])
                    baselines[star].append(baselines[seq_id])
                    seq_order[star].append(seq_order[seq_id])
                    #wavelengths[star] = np.hstack((pairs[star], pairs[seq_id]))
                    
                # Regardless of what happened, pop old keys
                mjds.pop(seq_id, None)
                pairs.pop(seq_id, None)
                vis2.pop(seq_id, None)
                e_vis2.pop(seq_id, None)
                flags.pop(seq_id, None)
                baselines.pop(seq_id, None)
                wavelengths.pop(seq_id, None)
                seq_order.pop(seq_id, None)
            
        # Fit LDD, ldd_fits = [ldd_opt, e_ldd_opt, c_scale, e_c_scale]
        print("\nFitting limb-darkened diameters for bootstrap %i" % (bs_i+1))
        ldd_fits = fit_all_ldd(vis2, e_vis2, baselines, wavelengths, tgt_info, 
                               pred_ldd_col, sampled_params, bs_i, 
                               method=method, e_wl_frac=e_wl_frac)  
       
        # Fit LDD, ldd_fits = [ldd_opt, e_ldd_opt, c_scale, e_c_scale]
        print("\nFitting uniform-disc diameters for bootstrap %i" % (bs_i+1))
        udd_fits = fit_all_ldd(vis2, e_vis2, baselines, wavelengths, tgt_info, 
                               pred_ldd_col, sampled_params, bs_i,
                               method=method, e_wl_frac=e_wl_frac, 
                               do_uniform_disc_fit=True)  
        
        # Fitting done, no need to have the vis2 results separate anymore                 
        # Populate
        for star in mjds.keys():
            if combined_fit:
                bs_results[star]["MJD"][bs_i] = np.hstack(mjds[star])
                bs_results[star]["TEL_PAIR"][bs_i] = np.hstack(pairs[star])
                bs_results[star]["VIS2"][bs_i] = np.vstack(vis2[star])
                bs_results[star]["FLAG"][bs_i] = np.vstack(flags[star])
                bs_results[star]["BASELINE"][bs_i] = np.hstack(baselines[star])
                bs_results[star]["WAVELENGTH"][bs_i] = wavelengths[star]
                bs_results[star]["LDD_FIT"][bs_i] = ldd_fits[star][0]
                bs_results[star]["C_SCALE"][bs_i] = np.vstack(ldd_fits[star][2])
                bs_results[star]["SEQ_ORDER"][bs_i] = np.vstack(seq_order[star])
                bs_results[star]["UDD_FIT"][bs_i] = udd_fits[star][0]
                bs_results[star]["C_SCALE_UDD"][bs_i] = np.vstack(udd_fits[star][2])
            
            else:
                bs_results[star]["MJD"][bs_i] = mjds[star]
                bs_results[star]["TEL_PAIR"][bs_i] = pairs[star]
                bs_results[star]["VIS2"][bs_i] = vis2[star]
                bs_results[star]["FLAG"][bs_i] = flags[star]
                bs_results[star]["BASELINE"][bs_i] = baselines[star]
                bs_results[star]["WAVELENGTH"][bs_i] = wavelengths[star]
                bs_results[star]["LDD_FIT"][bs_i] = ldd_fits[star][0]
                bs_results[star]["C_SCALE"][bs_i] = ldd_fits[star][2]
                bs_results[star]["SEQ_ORDER"][bs_i] = np.vstack(seq_order[star])
                bs_results[star]["UDD_FIT"][bs_i] = udd_fits[star][0]
                bs_results[star]["C_SCALE_UDD"][bs_i] = udd_fits[star][2]

    
    # A minority of bootstraps result in a different number of observed 
    # baseline/vis2 measurements, which cannot be stacked to produce vis2 
    # errors. This step prunes them to enable plotting.
    if prune_errant_baselines:
        # Get the most common baseline count, and remove any not adhering
        shape_dict = {}
        for star in bs_results.keys():                   
            shape_dict[star] = []              
            for vis2 in bs_results[star]["TEL_PAIR"]:
                shape_dict[star].append(vis2.shape)
            shape_dict[star] = Counter(shape_dict[star])    
                
            # Get a list of the indices to drop
            num_most_common = shape_dict[star].most_common(1)[0][0][0]
            i_to_drop = [i_ob for i_ob, ob 
                         in enumerate(bs_results[star]["VIS2"].values)
                         if len(ob) != num_most_common]
                         
            # Now replace the errant vis2 and baseline data with nans
            for ob_i in i_to_drop:
                bs_results[star].iloc[ob_i]["VIS2"] = \
                    np.ones([num_most_common, 6])*np.nan
                bs_results[star].iloc[ob_i]["BASELINE"] = \
                    np.ones(num_most_common)*np.nan

    return bs_results


def summarise_results(bs_results, tgt_info, e_wl_frac, add_e_wl_to_ldd_in_quad,
                      pred_ldd_col="LDD_pred", e_pred_ldd_col="e_LDD_pred"):
    """Summarise N boostrapping results by computing mean and standard 
    deviations for each distribution.
    
    Parameters
    ----------
    bs_results: dict of pandas dataframes
        Dictionary with science targets as keys, containing pandas dataframes
        recording the results of each bootstrapping iteration as rows.
    
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
        
    pred_ldd_col: string
        The column to use from tgt_info for the predicted diameters.
        
    e_pred_ldd_col: string
        The column to use from tgt_info for the predicted diameter 
        uncertainties.
            
    Returns
    -------
    results: pandas dataframe
        Summarised results of the bootstrapping with mean and std values
        computed from respective parameter distributions.
    """    
    # Initialise
    cols = ["STAR", "HD", "PERIOD", "SEQUENCE", "VIS2", "e_VIS2", "BASELINE", 
            "WAVELENGTH", "LDD_FIT", "e_LDD_FIT", "C_SCALE", "e_C_SCALE", 
            "SEQ_ORDER", "U_LAMBDA", "e_U_LAMBDA", "S_LAMBDA", "e_S_LAMBDA", 
            "UDD_FIT", "e_UDD_FIT", "C_SCALE_UDD", "e_C_SCALE_UDD"]
            
    results = pd.DataFrame(index=np.arange(0, len(bs_results.keys())), 
                           columns=cols)  
    
    star_ids = bs_results.keys()
    star_ids.sort()
    
    # All done collating, combine bootstrapped values into mean and std
    for star_i, star in enumerate(star_ids):
        # Set the common ID, and get the primary ID
        if type(star) == tuple:
            results.iloc[star_i]["STAR"] = star[0]
            pid = tgt_info[tgt_info["Primary"]==star[0]].index.values[0]
            
            sequence = star[1]
            period = int(star[2])
        else:
            results.iloc[star_i]["STAR"] = star
            pid = tgt_info[tgt_info["Primary"]==star].index.values[0]
            
            sequence = "combined"
            period = ""
        
        results.iloc[star_i]["HD"] = pid
        results.iloc[star_i]["PERIOD"] = period
        results.iloc[star_i]["SEQUENCE"] = sequence
        
        # Stack and compute mean and standard deviations 
        ldds = np.nanmean(np.hstack(bs_results[star]["LDD_FIT"]), axis=0)
        udds = np.nanmean(np.hstack(bs_results[star]["UDD_FIT"]), axis=0)

        e_ldd_fit = np.nanstd(np.hstack(bs_results[star]["LDD_FIT"]), axis=0)
        e_udd_fit = np.nanstd(np.hstack(bs_results[star]["UDD_FIT"]), axis=0)

        results.iloc[star_i]["LDD_FIT"] = ldds
        results.iloc[star_i]["UDD_FIT"] = udds

        # If doing least squares fitting, add the wavelength uncertainty to the
        # final diameter here in quadrature
        if add_e_wl_to_ldd_in_quad:
            print("Doing LS fitting, add lambda uncertainty in quadrature...")
            results.iloc[star_i]["e_LDD_FIT"] = np.sqrt(e_ldd_fit**2
                                                        + (ldds*e_wl_frac)**2)
            results.iloc[star_i]["e_UDD_FIT"] = np.sqrt(e_udd_fit**2
                                                        + (udds*e_wl_frac)**2)
        # Doing ODR fitting, errors are simply the standard deviations
        else:
            print("Doing ODR fitting, add lambda uncertainty in quadrature...")
            results.iloc[star_i]["e_LDD_FIT"] = e_ldd_fit
            results.iloc[star_i]["e_UDD_FIT"] = e_udd_fit

        results.iloc[star_i]["VIS2"] = \
            np.nanmean(np.dstack(bs_results[star]["VIS2"]), axis=2)
            
        results.iloc[star_i]["e_VIS2"] = \
            np.nanstd(np.dstack(bs_results[star]["VIS2"]), axis=2)

        results.iloc[star_i]["BASELINE"] = \
            np.nanmean(np.vstack(bs_results[star]["BASELINE"]), axis=0)

        # Note that this shouldn't change within a given night
        results.iloc[star_i]["WAVELENGTH"] = \
            np.nanmedian(np.vstack(bs_results[star]["WAVELENGTH"]), axis=0)

        # Combined seq case
        if len(bs_results[star]["C_SCALE"][0]) > 1:
            results.iloc[star_i]["C_SCALE"] = \
                np.nanmean(np.hstack(bs_results[star]["C_SCALE"]), axis=1)
     
            results.iloc[star_i]["e_C_SCALE"] = \
                np.nanstd(np.hstack(bs_results[star]["C_SCALE"]), axis=1) 
            
            results.iloc[star_i]["C_SCALE_UDD"] = \
                np.nanmean(np.hstack(bs_results[star]["C_SCALE_UDD"]), axis=1)
     
            results.iloc[star_i]["e_C_SCALE_UDD"] = \
                np.nanstd(np.hstack(bs_results[star]["C_SCALE_UDD"]), axis=1) 
        # Split seq case
        else:
            results.iloc[star_i]["C_SCALE"] = \
                np.nanmean(np.vstack(bs_results[star]["C_SCALE"]), axis=0)
     
            results.iloc[star_i]["e_C_SCALE"] = \
                np.nanstd(np.vstack(bs_results[star]["C_SCALE"]), axis=0)
            
            results.iloc[star_i]["C_SCALE_UDD"] = \
                np.nanmean(np.vstack(bs_results[star]["C_SCALE_UDD"]), axis=0)
     
            results.iloc[star_i]["e_C_SCALE_UDD"] = \
                np.nanstd(np.vstack(bs_results[star]["C_SCALE_UDD"]), axis=0) 
        
        # If we're bootstrapping, the order of the sequences shouldn't change
        # from iteration to iteration, so just take the first value
        results.iloc[star_i]["SEQ_ORDER"] = \
            bs_results[star].iloc[0]["SEQ_ORDER"]
        
        # Print some simple diagnostics                
        sci_percent_fit = (results.iloc[star_i]["e_LDD_FIT"]
                           / results.iloc[star_i]["LDD_FIT"]) * 100
           
        print("%-12s\tLDD = %f +/- %f (%0.2f%%), C=%s" 
              % (star, results.iloc[star_i]["LDD_FIT"], 
                 results.iloc[star_i]["e_LDD_FIT"], sci_percent_fit, 
                 results.iloc[star_i]["C_SCALE"]))
    
    return results