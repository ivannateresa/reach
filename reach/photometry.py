"""Module for handling photometry, such as converting between filters and doing
extinction correction.
"""
from __future__ import division, print_function
import extinction
import numpy as np
import pandas as pd
#import reach.plotting as rplt
from scipy.interpolate import interp1d


class ColourOutOfBoundsException(Exception):
    pass

# -----------------------------------------------------------------------------
# Colour Conversion
# -----------------------------------------------------------------------------
def calc_vk_colour(VTmag, RPmag):
    """Calculate a synthetic V-K colour given Vt and Rp
    """
    # Import Casagrande 1M simulated stars and their BC
    bc_stars = pd.read_csv("data/casagrande_bc_ext.csv", header=0)
    
    # Compute Vt-Rp and V-K from these relations, keeping in mind that since
    # the bolometric correction takes the form: BC_F1 = M_b - M_1, this 
    # translates to a colour relation of BC_1 - BC_2 = M2 - M1
    vtrp_colour = bc_stars["BC_RP"] - bc_stars["BC_Vt"]
    vk_colour = bc_stars["BC_K"] - bc_stars["BC_V"]
    
    # Fit (Vt-Rp) vs (V-K) with a polynomial (third order)
    coeff = np.polynomial.polynomial.polyfit(vtrp_colour, vk_colour, 3)
    
    # Determine equivalent V-K colour using relation
    vk_real = np.polynomial.polynomial.polyval(VTmag-RPmag, coeff)
    
    return vk_real
    
    
def calc_2mass_h():
    """Use colour relations to compute H for given photometry
    """
    pass
    

def convert_vtbt_to_vb(BTmag, VTmag):
    """Convert Tycho BT and VT band magnitudes to Cousins-Johnson B and V band  
    using relations from Bessell 2000:
     - http://adsabs.harvard.edu/abs/2000PASP..112..961B)
    
    Import the relation points from file, fit a cubic spline to them, and use
    that to predict the Cousins-Johnson B and V band magnitudes.
        
    Parameters
    ----------
    BTmag: float or float array
        Tycho B_T band magnitude.
    VTmag: float or float array
        Tycho V_T band magnitude.
    
    Returns
    -------
    Bmag: float or float array
        The predicted B band magnitude in the Cousins-Johnson system.
    Vmag: float or float array
        The predicted V band magnitude in the Cousins-Johnson system.
    """
    # Load the stored colour relations table in from Bessell 2000 (Table 2)
    # Columns are: [BT-VT    V-VT    (B-V)-(BT-VT)    V-H_P]
    bessell_2000_rel_file = "data/bessell_2000_bt_colour_relations.csv"
    colour_rel = np.loadtxt(bessell_2000_rel_file, delimiter=",", skiprows=1)
    
    # Create cubic splines interpolators to predict (V-VT) and delta(V-T)
    # from (BT-VT) - second, third, and first column of the csv respectively
    predict_V_minus_VT = interp1d(colour_rel[:,0], colour_rel[:,1], 
                                  kind="cubic") 
    predict_delta_B_minus_V = interp1d(colour_rel[:,0], colour_rel[:,2], 
                                       kind="cubic") 
    
    # Calculate (BT-VT)
    BT_minus_VT = BTmag - VTmag

    
    # Interpolation only works for colour range in Bessell 2000 - reject values
    # that fall outside of this
    if not np.min(BT_minus_VT) > np.min(colour_rel[:,0]): 
        raise ColourOutOfBoundsException("Minimum (B_T-V_T) colour must be >"
                                        " %f20" % np.min(colour_rel[:,0]))
    elif not np.max(BT_minus_VT) < np.max(colour_rel[:,0]):
        raise ColourOutOfBoundsException("Maximum (B_T-V_T) colour must be <"
                                        " %f0" % np.max(colour_rel[:,0]))
    
    # Predict (V-VT) from (BT-VT)
    V_minus_VT= predict_V_minus_VT(BT_minus_VT)
    
    # Determine V from the (V-VT) prediction
    Vmag = V_minus_VT + VTmag
    
    # Predict delta(B-V)
    delta_B_minus_V = predict_delta_B_minus_V(BT_minus_VT)
    
    # Determine B from delta(B-V), BT, VT, and V
    Bmag = delta_B_minus_V + BT_minus_VT + Vmag
    
    return Bmag, Vmag


# -----------------------------------------------------------------------------
# Extinction
# -----------------------------------------------------------------------------
def create_spt_uv_grid(do_interpolate=True):
    """Create a grid of stellar instrinic (B-V) colours across spectral types.
    This is currently done for dwarfs using the following table, originally 
    from Pecaut & Mamajek 2013: 
     - http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt 
    
    Colours for stars not on the main sequence come from Schmidt-Kaler 1982:
     - http://adsabs.harvard.edu/abs/1982lbg6.conf.....A
    Which does not list Teff (currently we assume dwarf SpT for non-MS stars,
    which is not physically realistic), nor the subgiant branch (which is 
    interpolated as simply being halway between dwarfs and giants). As such, 
    this function is still a work in progress.
    
    Parameters
    ----------
    do_interpolate: boolean
        Escape parameter to construct the grid with solely the information
        provided in the input tables without interpolation.
        
    Returns
    -------
    grid: pandas dataframe
        Pandas dataframe of instrinsic (B-V) colours of form:
        [SpT, Teff, V, IV, III, Ib, Iab, Ia]
    """
    # Import the relations to be used
    m_colour_relations = "data/EEM_dwarf_UBVIJHK_colors_Teff.txt"
    sk_colour_relations = "data/schmidt-kaler_bv_colours.csv"
    
    mcr = pd.read_csv(m_colour_relations, comment="#", nrows=123, 
                      delim_whitespace=True, engine="python", index_col=0, 
                      na_values="...")
    
    skcr = pd.read_csv(sk_colour_relations, sep=",", index_col=0)
    
    # Initialise new pandas dataframe to store the entire grid. This should
    # be of the form:
    # | SpT | Teff |              (B-V)_0              |
    # |     |      | V | IV | III | II | Ib | Iab | Ia |
    # The values for dwarfs should come from the Mamajek table, as should the
    # labels and temperatures for the spectral types themselves. The (numeric)
    # temperatures will then be interpolated over to cover the Mamajek rang of
    # spectral types for the older (and less complete) Schmidt-Kaler dataset.
    
    # Remove the V from the Mamajek spectral types
    mcr.index = [spt[:-1] for spt in mcr.index]
    
    # Initialise the grid, with nans for empty spaces (take care to consider 
    # the difference between pandas views vs copy
    grid = mcr[["Teff", "B-V"]].copy()
    grid.rename(index=str, columns={"B-V":"V"}, inplace=True)
    grid["IV"] = np.nan
    grid["III"] = np.nan
    grid["II"] = np.nan
    grid["Ib"] = np.nan
    grid["Iab"] = np.nan
    grid["Ia"] = np.nan
    
    # Step through the Schmidt-Kaler relations and fill in the appropriate SpT
    for row_i, row in skcr.iterrows():
        if row.name in grid.index:
            # Add values for each spectral type
            grid.loc[row.name, "skV"] = row["V"]
            grid.loc[row.name, "III"] = row["III"]
            grid.loc[row.name, "II"] = row["II"]
            grid.loc[row.name, "Ib"] = row["Ia"]
            grid.loc[row.name, "Iab"] = row["Iab"]
            grid.loc[row.name, "Ia"] = row["Ia"]
    
    # Option to abort in case we only want the raw data sans interpolation
    if not do_interpolate:
        return grid
            
    # Only interpolate for spectral types without values *within* the 
    # interpolation range
    for col in ["III", "II", "Ib", "Iab", "Ia"]:
        # Using the temperatures, interpolate each along each spectral type and 
        # fill in the missing values
        teff = grid["Teff"][~np.isnan(grid[col])]
        b_minus_v = grid[col][~np.isnan(grid[col])]
        calc_b_minus_v = interp1d(teff, b_minus_v, kind="linear") 
        
        unknown_i = (np.isnan(grid[col]) & (grid["Teff"] > np.min(teff)) 
                          & (grid["Teff"] < np.max(teff)))

        grid.loc[unknown_i, col] = calc_b_minus_v(grid["Teff"][unknown_i])
    
    # Interpolate across the spectral types to fill in the values for subgiants
    # TODO - this is *bad*
    grid["IV"] = (grid["V"] + grid["III"]) / 2
    
    # Save and return the grid
    return grid
    
    
def calculate_selective_extinction(B_mag, V_mag, sptypes, grid):
    """Calculate the selective extinction (i.e. colour excess). This takes the
    form:
    
    E(B-V) = A(B) - A(V)
           = (B-V) - (B-V)_0
           
    Where E(B-V) is the selective extinction (i.e. the additional colour excess
    caused by extinction), A(B) and A(V) are the extinctions in the B and V 
    bands respectively, (B-V) is the observed B minus V colour, and (B-V)_0 is
    the unextincted B minus V colour.
    
    See http://w.astro.berkeley.edu/~ay216/08/NOTES/Lecture05-08.pdf
    
    Parameters
    ----------
    B_mag: float array
        Apparent B band magnitude.
    
    V_mag: float array
        Apparent V band magnitude
        
    sptypes: string array
        Spectral type/s
    
    grid: pandas dataframe
        Pandas dataframe of instrinsic (B-V) colours of form:
        [SpT, Teff, V, IV, III, Ib, Iab, Ia]
        
    Returns
    -------
    e_bv: float array
        Selective extinction, E(B-V) = (B-V) - (B-V)_0
    """
    # In the order listed, look for the luminosity class of the star in its 
    # full SpT. When a match is found, break from the look to avoid partial 
    # matches (e.g. both IV and V are in G4IV). This is required as SpT is 
    # typically written as a single string, whereas it is split over two
    # dimensions in the (B-V) grid.
    classes = ["IV", "V", "III", "II", "Ib", "Iab", "Ia"]
    
    bv_0_all = np.zeros(len(B_mag))
    
    lum_class_matched = False
    
    for spt_i, spt_full in enumerate(sptypes):
        # Determine class
        for lum_class in classes:
            if lum_class in spt_full:
                lum_class_matched = True
                break
               
        assert lum_class_matched
        
        spt = spt_full.replace(lum_class, "")
        
        # SpT has been identified and split, get (B-V) colour
        bv_0_all[spt_i] = grid.loc[spt, lum_class]
        
    ebv = (B_mag - V_mag) - bv_0_all
    
    return ebv
        
    
def calculate_v_band_extinction(e_bv, r_v=3.1):
    """Calculate A(V) from: 
    
    R_V = A(V) / [A(B) - A(V)]
        = A(V) / E(B-V)
        
    Where 1/R_V is the normalised extinction, and measures the steepness of the 
    extinction curve. R_V = 3.1 +/- 0.2 is for the diffuse ISM, R ~= 5 is for 
    dark interstellar clouds.
    
    A(V) is the V band extinction and serves as a scaling parameter. Thus:
    
    A(V) = R_V * E(B-V) 
    
    Parameters
    ----------
    e_bv: float
        Selective extinction, E(B-V).
    r_v: float
        Ratio of total to selective extinction, A_V / E(B-V).
    
    Returns
    -------
    a_v: float
        V band extinction.
    """
    a_v = r_v * e_bv
    
    return a_v
    
    
def calculate_effective_wavelength(spt, filter):
    """Given a stellar spectral type, and a particular photometric filter, 
    return the effective wavelength.
    
    This is per discussion in Bessell et al. 1998. As an example: 
     - "In broad-band photometry the nominal wavelength associated with a 
        passband (the effective wavelength) shifts with the color of the star.
        For Vega the effective wavelength of the V band is 5448 A and for the 
        sun it is 5502 A"
        
    Ideally this function should just reference a grid of effective wavelengths
    from a table comparing filters and spectral types.
    
    Parameters
    ----------
    spt: string
        Spectral type of the star/s.
        
    filter: string
        Name of the photometric band
    
    Returns
    -------
    filter_eff_lambda: float
    
    """
    pass
    
    
def deredden_photometry(ext_mag, ext_mag_err, filter_eff_lambda, a_v, r_v=3.1):
    """Use an extinction law to deredden photometry from a given band.
    
    Relies on:
        https://github.com/kbarbary/extinction
    With documentation at:
        https://extinction.readthedocs.io/en/latest/
    
    Parameters
    ----------
    ext_mag: np.array of type float
        The extincted magnitude.
    
    ext_mag_err: np.array of type float
        Error in the extincted magnitude
    
    filter_eff_lamda: np.array of type float
        Effective wavelength of the broad-band photometric filter specific to
        stellar spectral type.
    
    a_v : np.array of type float
        Scaling parameter, A_V: extinction in magnitudes at characteristic
        V band wavelength.
    r_v : np.array of type float
        Ratio of total to selective extinction, A_V / E(B-V).
        
    Returns
    -------
    a_mags: float array
        Array of photometric extinction of form [W, S], where W is the number
        of wavelengths, and S is the number of stars.
    
    de_ext_mag_err: np.array of type float
        Error in the de-extincted magnitude.
    """
    # Create grid of extinction
    a_mags = np.zeros(ext_mag.shape)
    
    # Use the Cardelli, Clayton, & Mathis 1989 extinction model. The extinction
    # module is not vectorised, so we have to work with one star at a time
    for star_i, star in enumerate(ext_mag.itertuples(index=False)):
        a_mags[star_i,:] = extinction.ccm89(filter_eff_lambda, a_v[star_i], 
                                            r_v)
    
    return a_mags
    
# -----------------------------------------------------------------------------
# Other
# -----------------------------------------------------------------------------
def inspect_dr_photometry(tgt_info):
    """Diagnostic function to inspect for issues with reddening/diameters. WIP. 
    """
    print("%7s \t %7s \t %7s \t %7s \t %7s \t %7s \t %7s \t %7s \t %7s \t %7s \t %7s" %
          ("ID", "B_a_mag", "V_a_mag", "J_a_mag", "H_a_mag", "K_a_mag", "Flag",
           "Dist", "LDD (V-K)", "LDD (V-W3)", "ID"))
    
    num_flagged = 0
    
    for star, row in tgt_info.iterrows():
        b_a_mag = row["Bmag"] - row["Bmag_dr"]
        v_a_mag = row["Vmag"] - row["Vmag_dr"]
        j_a_mag = row["Jmag"] - row["Jmag_dr"]
        h_a_mag = row["Hmag"] - row["Hmag_dr"]
        k_a_mag = row["Kmag"] - row["Kmag_dr"]
        primary = row["Primary"]
        vk_ldd = row["LDD_VK_dr"]
        vw3_ldd = row["LDD_VW3_dr"]
        flag = ""
        dist = row["Dist"]
        
        if np.max(np.abs([v_a_mag, j_a_mag, h_a_mag, k_a_mag])) > 0.1:
            flag = "***"
            num_flagged += 1
        
        print(("%8s \t %0.4f \t %0.4f \t %0.4f \t %0.4f \t %0.4f \t %7s \t"
               "%4.2f \t %5.3f \t %5.3f \t %s") 
                % (star, b_a_mag, v_a_mag, j_a_mag, h_a_mag, k_a_mag, flag, 
                   dist, vk_ldd, vw3_ldd, primary))
                       
    print("\nFlagged Stars: %i/%i" % (num_flagged, len(tgt_info)))