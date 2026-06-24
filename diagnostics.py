import os
import numpy as np
import pandas as pd




def clean_target_id(x):
    """
    Clean target names for robust matching.

    Examples:
    HD_79810  -> hd79810
    HD  63734 -> hd63734
    psi Vel A -> psivela
    psi_Vel   -> psivel
    HR_3729   -> hr3729
    """

    import pandas as pd

    if pd.isnull(x):
        return ""

    x = str(x)

    x = x.replace("_", "")
    x = x.replace(" ", "")
    x = x.replace(".", "")
    x = x.replace("-", "")
    x = x.replace("\t", "")
    x = x.lower()

    return x


def match_target_name(tgt_info, name, verbose=False):
    """
    Match one target name to the tgt_info index using cleaned identifiers.
    """

    import pandas as pd

    name_clean = clean_target_id(name)

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

    # Search inside columns
    for col in search_cols:

        col_clean = tgt_info[col].apply(clean_target_id)
        this_match = tgt_info.index[col_clean == name_clean]

        if len(this_match) > 0:
            matches.extend(list(this_match))

    # Search also in dataframe index
    index_clean = []

    for idx in tgt_info.index:
        index_clean.append(clean_target_id(idx))

    for i, idx_clean in enumerate(index_clean):
        if idx_clean == name_clean:
            matches.append(tgt_info.index[i])

    # Remove duplicates
    matches_unique = []

    for m in matches:
        if m not in matches_unique:
            matches_unique.append(m)

    if len(matches_unique) == 0:

        if verbose:
            print("NO MATCH: %s  cleaned as  %s" % (name, name_clean))

        return None

    if len(matches_unique) > 1:
        print("WARNING: multiple matches for %s:" % name)
        print(matches_unique)
        print("Using first match: %s" % matches_unique[0])

    return matches_unique[0]


def match_target_list(tgt_info, names, label="target", verbose=True):
    """
    Match a list of target names to tgt_info indices.
    """

    ids = []
    failed = []

    for name in names:

        this_id = match_target_name(tgt_info, name, verbose=False)

        if this_id is None:
            failed.append(name)

            if verbose:
                print("WARNING: could not match %s name: %s" % (label, name))

        else:
            ids.append(this_id)

    return ids, failed



def extract_vis2_flexible(oifits_file):
    """
    Extract V2 data from an OIFITS file without assuming a fixed array shape.

    Returns
    -------
    x : numpy array
        Spatial frequency B/lambda if possible, otherwise point index.

    vis2 : numpy array
        Flattened V2 values.

    e_vis2 : numpy array
        Flattened V2 uncertainties.

    flags : numpy array
        Flattened boolean flags.
    """

    import numpy as np
    from astropy.io import fits

    hdul = fits.open(oifits_file)

    # -------------------------------------------------------------------------
    # Get wavelength information
    # -------------------------------------------------------------------------
    wavelengths = None

    for hdu in hdul:

        extname = hdu.header.get("EXTNAME", "")

        if extname == "OI_WAVELENGTH":

            if hdu.data is not None and "EFF_WAVE" in hdu.data.names:
                wavelengths = np.asarray(hdu.data["EFF_WAVE"]).astype(float)
                break

    all_x = []
    all_vis2 = []
    all_evis2 = []
    all_flags = []

    # -------------------------------------------------------------------------
    # Read all OI_VIS2 extensions
    # -------------------------------------------------------------------------
    for hdu in hdul:

        extname = hdu.header.get("EXTNAME", "")

        if extname != "OI_VIS2":
            continue

        data = hdu.data

        if data is None:
            continue

        names = data.names

        if "VIS2DATA" not in names:
            continue

        vis2 = np.asarray(data["VIS2DATA"]).astype(float)

        if "VIS2ERR" in names:
            e_vis2 = np.asarray(data["VIS2ERR"]).astype(float)
        else:
            e_vis2 = np.zeros_like(vis2) + np.nan

        if "FLAG" in names:
            flags = np.asarray(data["FLAG"]).astype(bool)
        else:
            flags = np.zeros_like(vis2).astype(bool)

        # ---------------------------------------------------------------------
        # Get baseline per row
        # ---------------------------------------------------------------------
        if "UCOORD" in names and "VCOORD" in names:
            ucoord = np.asarray(data["UCOORD"]).astype(float)
            vcoord = np.asarray(data["VCOORD"]).astype(float)
            baselines = np.sqrt(ucoord**2 + vcoord**2)
        else:
            baselines = None

        # ---------------------------------------------------------------------
        # Flatten V2 arrays
        # ---------------------------------------------------------------------
        vis2_flat = vis2.flatten()
        e_vis2_flat = e_vis2.flatten()
        flags_flat = flags.flatten()

        n_total = len(vis2_flat)

        # ---------------------------------------------------------------------
        # Construct spatial frequency axis robustly
        # ---------------------------------------------------------------------
        x = None

        # Case 1: VIS2DATA is 2D: n_rows x n_wavelengths
        if baselines is not None and wavelengths is not None:

            n_rows = len(baselines)
            n_wl = len(wavelengths)

            if n_rows * n_wl == n_total:

                # This matches normal OIFITS structure:
                # one baseline value per row, several wavelength channels
                bl_grid = np.repeat(baselines, n_wl)
                wl_grid = np.tile(wavelengths, n_rows)

                x = bl_grid / wl_grid

        # Case 2: baseline and wavelength are already flattened
        if x is None:

            if baselines is not None:

                baselines_flat = np.asarray(baselines).flatten()

                if wavelengths is not None:
                    wavelengths_flat = np.asarray(wavelengths).flatten()
                else:
                    wavelengths_flat = None

                if len(baselines_flat) == n_total:

                    if wavelengths_flat is not None and len(wavelengths_flat) == n_total:
                        x = baselines_flat / wavelengths_flat
                    else:
                        x = baselines_flat

        # Case 3: fallback to point number
        if x is None:
            x = np.arange(n_total).astype(float)

        all_x.extend(list(x))
        all_vis2.extend(list(vis2_flat))
        all_evis2.extend(list(e_vis2_flat))
        all_flags.extend(list(flags_flat))

    hdul.close()

    if len(all_vis2) == 0:
        raise ValueError("No OI_VIS2 data found in file")

    return (np.asarray(all_x),
            np.asarray(all_vis2),
            np.asarray(all_evis2),
            np.asarray(all_flags))

def get_calibrator_name_from_filename(filename):
    """
    Extract the target name from a calibrated diagnostic filename.

    Example
    -------
    2019-11-25_SCI_HD61574_oidataCalibrated_00.fits
    returns HD61574
    """

    import os

    base = os.path.basename(filename)

    if "_SCI_" in base:
        name = base.split("_SCI_")[-1].split("_oidata")[0]
    elif "SCI" in base:
        name = base.split("SCI")[-1].split("oidata")[0]
    else:
        name = base.split("_oidata")[0]

    name = name.replace("_bad", "")
    name = name.replace(".fits", "")
    name = name.strip("_")

    return name

def save_initialise_diagnostics(
        tgt_info,
        outdir="data/diagnostics",
        prefix="initialise_tgt_info_nan_diagnostic"):
    """
    Guarda un diagnostico de las estrellas que tienen NaN
    despues de usar initialise_tgt_info().
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Columnas finales donde normalmente aparecen NaN
    output_cols = [
        "Bmag", "Vmag",
        "eb_v", "A_V",
        "Bmag_dr", "Vmag_dr",
        "Hpmag_dr", "BTmag_dr", "VTmag_dr",
        "BPmag_dr", "RPmag_dr",
        "Dist", "e_Dist",
        "V-K_calc",
        "LDD_VK", "e_LDD_VK",
        "LDD_VW3", "e_LDD_VW3",
        "LDD_VW4", "e_LDD_VW4",
        "LDD_BV", "e_LDD_BV",
        "LDD_BV_feh", "e_LDD_BV_feh",
        "LDD_pred", "e_LDD_pred",
        "teff_casagrande", "e_teff_casagrande"
    ]

    # Columnas de entrada importantes para entender por que sale NaN
    input_cols = [
        "Primary", "Science", "Sequence", "SpT", "SpT_simple",
        "BTmag", "e_BTmag",
        "VTmag", "e_VTmag",
        "Jmag", "e_Jmag",
        "Kmag", "e_Kmag",
        "W1mag", "e_W1mag",
        "W3mag", "e_W3mag",
        "W4mag", "e_W4mag",
        "Plx", "e_Plx",
        "Plx_alt", "e_Plx_alt",
        "FeH_rel", "e_FeH_rel",
        "Quality", "LDD_rel"
    ]

    # Usar solo columnas que existen en tu tabla
    output_cols = [c for c in output_cols if c in tgt_info.columns]
    input_cols = [c for c in input_cols if c in tgt_info.columns]

    # Filas que tienen al menos un NaN en columnas calculadas
    nan_mask = tgt_info[output_cols].isnull().any(axis=1)
    diagnostic = tgt_info.loc[nan_mask, input_cols + output_cols].copy()

    # Guardar que columnas calculadas salieron NaN
    diagnostic["nan_outputs"] = diagnostic.apply(
        lambda row: "; ".join([c for c in output_cols if pd.isnull(row[c])]),
        axis=1
    )

    # Diagnostico simple de la causa probable
    reasons = []

    for star, row in diagnostic.iterrows():
        reason = []

        if "BTmag" in diagnostic.columns and "VTmag" in diagnostic.columns:
            if pd.isnull(row["BTmag"]) or pd.isnull(row["VTmag"]):
                reason.append("Missing BTmag or VTmag: cannot compute Bmag/Vmag")

        if "Plx" in diagnostic.columns and "Plx_alt" in diagnostic.columns:
            if pd.isnull(row["Plx"]) and pd.isnull(row["Plx_alt"]):
                reason.append("Missing Plx and Plx_alt: cannot compute Dist")

        if "e_Plx" in diagnostic.columns and "e_Plx_alt" in diagnostic.columns:
            if pd.isnull(row["e_Plx"]) and pd.isnull(row["e_Plx_alt"]):
                reason.append("Missing e_Plx and e_Plx_alt: cannot compute e_Dist")

        if "FeH_rel" in diagnostic.columns:
            if pd.isnull(row["FeH_rel"]):
                reason.append("Missing FeH_rel: Casagrande Teff or FeH-dependent LDD may be NaN")

        if "Jmag" in diagnostic.columns and "Kmag" in diagnostic.columns:
            if pd.isnull(row["Jmag"]) or pd.isnull(row["Kmag"]):
                reason.append("Missing Jmag or Kmag: cannot compute V-K relation")

        if "W3mag" in diagnostic.columns:
            if pd.isnull(row["W3mag"]):
                reason.append("Missing W3mag: cannot compute V-W3 relation")

        if "W4mag" in diagnostic.columns:
            if pd.isnull(row["W4mag"]):
                reason.append("Missing W4mag: cannot compute V-W4 relation")

        if len(reason) == 0:
            reason.append("Inputs look present; check data type, empty strings, or validity range of calibration")

        reasons.append("; ".join(reason))

    diagnostic["probable_reason"] = reasons

    # Guardar CSV
    csv_path = os.path.join(outdir, prefix + ".csv")
    diagnostic.to_csv(csv_path, index=True)

    # Guardar TXT legible
    txt_path = os.path.join(outdir, prefix + ".txt")

    with open(txt_path, "w") as f:
        f.write("Diagnostic file for initialise_tgt_info()\n")
        f.write("=" * 70 + "\n\n")
        f.write("Number of targets with at least one NaN: {}\n\n".format(len(diagnostic)))

        for star, row in diagnostic.iterrows():
            f.write("-" * 70 + "\n")
            f.write("HD_ID: {}\n".format(star))

            if "Primary" in diagnostic.columns:
                f.write("Primary: {}\n".format(row["Primary"]))

            if "Science" in diagnostic.columns:
                f.write("Science: {}\n".format(row["Science"]))

            if "SpT_simple" in diagnostic.columns:
                f.write("SpT_simple: {}\n".format(row["SpT_simple"]))

            f.write("NaN outputs: {}\n".format(row["nan_outputs"]))
            f.write("Probable reason: {}\n".format(row["probable_reason"]))

            f.write("\nRelevant inputs:\n")
            for col in input_cols:
                f.write("  {}: {}\n".format(col, row[col]))

            f.write("\n")

    print("Saved diagnostic CSV:")
    print(csv_path)
    print("Saved diagnostic TXT:")
    print(txt_path)

    return diagnostic







def save_sequence_logs_diagnostics(complete_sequences, sequences,
                                   outdir="data/diagnostics",
                                   prefix="sequence_logs"):
    """
    Save readable diagnostic files for PIONIER sequence logs.

    Parameters
    ----------
    complete_sequences : dict or OrderedDict
        Output from rutils.load_sequence_logs()[0]

    sequences : dict or OrderedDict
        Output from rutils.load_sequence_logs()[1]

    outdir : str
        Output directory where diagnostic files will be saved.

    prefix : str
        Prefix for the output files.
    """

    import os
    import pandas as pd

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # -------------------------------------------------------------------------
    # Save sequences in CSV format
    # -------------------------------------------------------------------------
    sequence_rows = []

    for key, target_list in sequences.items():

        period = key[0]
        science_target = key[1]
        sequence_type = key[2]

        row = {
            "Period": period,
            "Science_target": science_target,
            "Sequence_type": sequence_type,
            "Full_sequence": " - ".join(target_list),
            "N_targets": len(target_list)
        }

        for i, target in enumerate(target_list):
            row["target_%02d" % (i + 1)] = target

        # Typical PIONIER sequence:
        # calibrator - science - calibrator - science - calibrator
        if len(target_list) >= 5:
            row["calibrator_1"] = target_list[0]
            row["science_1"] = target_list[1]
            row["calibrator_2"] = target_list[2]
            row["science_2"] = target_list[3]
            row["calibrator_3"] = target_list[4]

        sequence_rows.append(row)

    sequences_df = pd.DataFrame(sequence_rows)

    sequences_csv = os.path.join(outdir, prefix + "_sequences.csv")
    sequences_df.to_csv(sequences_csv, index=False)

    # -------------------------------------------------------------------------
    # Save sequences in TXT format
    # -------------------------------------------------------------------------
    sequences_txt = os.path.join(outdir, prefix + "_sequences.txt")

    with open(sequences_txt, "w") as f:

        f.write("Readable summary of PIONIER sequence definitions\n")
        f.write("=" * 70 + "\n\n")

        for key, target_list in sequences.items():

            period = key[0]
            science_target = key[1]
            sequence_type = key[2]

            f.write("-" * 70 + "\n")
            f.write("Period: %s\n" % period)
            f.write("Science target: %s\n" % science_target)
            f.write("Sequence type: %s\n" % sequence_type)
            f.write("Number of targets in sequence: %i\n" % len(target_list))
            f.write("Sequence:\n")

            for i, target in enumerate(target_list):
                f.write("  %02d  %s\n" % (i + 1, target))

            f.write("\n")

    # -------------------------------------------------------------------------
    # Save complete_sequences summary
    # -------------------------------------------------------------------------
    complete_txt = os.path.join(outdir, prefix + "_complete_sequences.txt")

    with open(complete_txt, "w") as f:

        f.write("Readable summary of complete PIONIER sequences\n")
        f.write("=" * 70 + "\n\n")

        for seq in complete_sequences.keys():

            f.write("-" * 70 + "\n")
            f.write("Sequence key: %s\n" % str(seq))

            try:
                f.write("Number of entries: %i\n" % len(complete_sequences[seq]))
            except:
                f.write("Number of entries: unknown\n")

            # This follows the structure used in complete_obs_diagnostics()
            try:
                f.write("Number of observations: %i\n" % len(complete_sequences[seq][2]))

                for i, obs in enumerate(complete_sequences[seq][2]):
                    f.write("  %02i  %s  %s  %s\n" %
                            (i, str(obs[2]), str(obs[3]), str(obs[-1])))

            except:
                f.write("Could not parse detailed observations. Raw entry:\n")
                f.write(str(complete_sequences[seq]))
                f.write("\n")

            f.write("\n")

    print("Saved sequence diagnostics:")
    print(sequences_csv)
    print(sequences_txt)
    print(complete_txt)

    return sequences_df


def diagnose_calibrator_quality(tgt_info,
                                cal_folder="/home2/ihernand/Desktop/reach/diagnostics/",
                                outdir="/home2/ihernand/Desktop/reach/diagnostics/",
                                prefix="calibrator_quality",
                                max_flagged_frac=0.30,
                                max_bad_vis2_frac=0.10,
                                max_robust_scatter=0.25,
                                max_reduced_chi2=20.0,
                                update_tgt_info=False):
    """
    Diagnose calibrator quality using calibrated diagnostic OIFITS files.

    This version reads OI_VIS2 directly and does not assume a fixed shape for
    the baseline/wavelength arrays.
    """

    import os
    import glob
    import numpy as np
    import pandas as pd

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    cal_oifits = glob.glob(os.path.join(cal_folder, "*oidataCalibrated*.fits"))
    cal_oifits.sort()

    if len(cal_oifits) == 0:
        print("No diagnostic FITS files found in:")
        print(cal_folder)
        return pd.DataFrame()

    rows = []

    for cal_file in cal_oifits:

        filename = os.path.basename(cal_file)
        cal_name = get_calibrator_name_from_filename(filename)

        # Match name to tgt_info, using the robust function we made before
        try:
            hd_id = match_target_name(tgt_info, cal_name, verbose=False)

            if hd_id is not None:
                primary = tgt_info.loc[hd_id, "Primary"]

                if "Quality" in tgt_info.columns:
                    input_quality = tgt_info.loc[hd_id, "Quality"]
                else:
                    input_quality = None
            else:
                primary = None
                input_quality = None

        except Exception:
            hd_id = None
            primary = None
            input_quality = None

        try:
            x, vis2, e_vis2, flags = extract_vis2_flexible(cal_file)

            vis2 = np.asarray(vis2).flatten()
            e_vis2 = np.asarray(e_vis2).flatten()
            flags = np.asarray(flags).flatten().astype(bool)
            x = np.asarray(x).flatten()

            n_total = len(vis2)

            if len(e_vis2) != n_total:
                e_vis2 = np.zeros(n_total) + np.nan

            if len(flags) != n_total:
                flags = np.zeros(n_total).astype(bool)

            if len(x) != n_total:
                x = np.arange(n_total).astype(float)

            finite = (
                np.isfinite(vis2) &
                np.isfinite(e_vis2) &
                (e_vis2 > 0)
            )

            good = finite & (~flags)

            n_finite = int(np.sum(finite))
            n_good = int(np.sum(good))
            n_flagged = int(np.sum(flags))

            if n_total > 0:
                flagged_frac = float(n_flagged) / float(n_total)
            else:
                flagged_frac = np.nan

            if n_finite > 0:
                bad_vis2_mask = finite & ((vis2 < -0.10) | (vis2 > 1.30))
                bad_vis2_frac = float(np.sum(bad_vis2_mask)) / float(n_finite)
            else:
                bad_vis2_frac = np.nan

            # -----------------------------------------------------------------
            # Scatter around a smooth polynomial
            # -----------------------------------------------------------------
            if n_good >= 5:

                x_good = x[good]
                y_good = vis2[good]
                ey_good = e_vis2[good]

                # Sort by x
                order = np.argsort(x_good)

                x_good = x_good[order]
                y_good = y_good[order]
                ey_good = ey_good[order]

                x_min = np.min(x_good)
                x_max = np.max(x_good)

                if x_max > x_min:
                    x_norm = (x_good - x_min) / (x_max - x_min)
                else:
                    x_norm = x_good * 0.0

                try:
                    if n_good > 8:
                        poly_order = 2
                    else:
                        poly_order = 1

                    coeff = np.polyfit(
                        x_norm,
                        y_good,
                        poly_order,
                        w=1.0 / ey_good
                    )

                    model = np.polyval(coeff, x_norm)

                    dof = n_good - (poly_order + 1)

                    if dof <= 0:
                        dof = 1

                except Exception:
                    model = np.median(y_good) * np.ones(len(y_good))
                    dof = n_good - 1

                    if dof <= 0:
                        dof = 1

                residuals = y_good - model

                med_res = np.median(residuals)
                mad_res = np.median(np.abs(residuals - med_res))

                robust_scatter = 1.4826 * mad_res
                reduced_chi2 = np.sum((residuals / ey_good)**2) / float(dof)

                median_vis2 = np.median(y_good)
                std_vis2 = np.std(y_good)

            else:
                median_vis2 = np.nan
                std_vis2 = np.nan
                robust_scatter = np.nan
                reduced_chi2 = np.nan

            # -----------------------------------------------------------------
            # Automatic classification
            # -----------------------------------------------------------------
            reasons = []
            score = 0

            if input_quality == "BAD":
                reasons.append("Already marked as BAD in tgt_info")
                score += 1

            if n_total == 0:
                reasons.append("No V2 points")
                score += 3

            if n_good == 0:
                reasons.append("No usable unflagged finite V2 points")
                score += 3

            elif n_good < 5:
                reasons.append("Very few usable V2 points")
                score += 2

            if np.isfinite(flagged_frac) and flagged_frac > max_flagged_frac:
                reasons.append("High fraction of flagged points")
                score += 1

            if np.isfinite(bad_vis2_frac) and bad_vis2_frac > max_bad_vis2_frac:
                reasons.append("Many non-physical V2 values")
                score += 2

            if np.isfinite(robust_scatter) and robust_scatter > max_robust_scatter:
                reasons.append("Large V2 scatter around smooth curve")
                score += 1

            if np.isfinite(reduced_chi2) and reduced_chi2 > max_reduced_chi2:
                reasons.append("Large reduced chi2 around smooth curve")
                score += 1

            if score >= 3:
                diagnosis = "BAD"
            elif score >= 1:
                diagnosis = "CHECK"
            else:
                diagnosis = "OK"

            if len(reasons) == 0:
                reason = "No obvious problem detected"
            else:
                reason = "; ".join(reasons)

            rows.append({
                "filename": filename,
                "cal_name_from_file": cal_name,
                "HD_ID": hd_id,
                "Primary": primary,
                "input_Quality": input_quality,
                "n_total": n_total,
                "n_finite": n_finite,
                "n_good": n_good,
                "n_flagged": n_flagged,
                "flagged_frac": flagged_frac,
                "bad_vis2_frac": bad_vis2_frac,
                "median_vis2": median_vis2,
                "std_vis2": std_vis2,
                "robust_scatter": robust_scatter,
                "reduced_chi2": reduced_chi2,
                "diagnosis": diagnosis,
                "reason": reason
            })

        except Exception as err:

            rows.append({
                "filename": filename,
                "cal_name_from_file": cal_name,
                "HD_ID": hd_id,
                "Primary": primary,
                "input_Quality": input_quality,
                "n_total": np.nan,
                "n_finite": np.nan,
                "n_good": np.nan,
                "n_flagged": np.nan,
                "flagged_frac": np.nan,
                "bad_vis2_frac": np.nan,
                "median_vis2": np.nan,
                "std_vis2": np.nan,
                "robust_scatter": np.nan,
                "reduced_chi2": np.nan,
                "diagnosis": "READ_ERROR",
                "reason": "Could not read or process FITS file: %s" % str(err)
            })

    diag_df = pd.DataFrame(rows)

    csv_path = os.path.join(outdir, prefix + ".csv")
    txt_path = os.path.join(outdir, prefix + ".txt")

    diag_df.to_csv(csv_path, index=False)

    with open(txt_path, "w") as f:

        f.write("Calibrator quality diagnostic\n")
        f.write("=" * 70 + "\n\n")

        f.write("Input folder:\n")
        f.write("%s\n\n" % cal_folder)

        f.write("Number of diagnostic FITS files: %i\n\n" % len(cal_oifits))

        for status in ["READ_ERROR", "BAD", "CHECK", "OK"]:

            sub = diag_df[diag_df["diagnosis"] == status]

            f.write("-" * 70 + "\n")
            f.write("%s calibrators: %i\n" % (status, len(sub)))
            f.write("-" * 70 + "\n")

            for i, row in sub.iterrows():

                f.write("File: %s\n" % row["filename"])
                f.write("Calibrator name: %s\n" % row["cal_name_from_file"])
                f.write("HD_ID: %s\n" % row["HD_ID"])
                f.write("Primary: %s\n" % row["Primary"])
                f.write("Diagnosis: %s\n" % row["diagnosis"])
                f.write("Reason: %s\n" % row["reason"])
                f.write("n_good: %s\n" % str(row["n_good"]))
                f.write("flagged_frac: %s\n" % str(row["flagged_frac"]))
                f.write("bad_vis2_frac: %s\n" % str(row["bad_vis2_frac"]))
                f.write("robust_scatter: %s\n" % str(row["robust_scatter"]))
                f.write("reduced_chi2: %s\n" % str(row["reduced_chi2"]))
                f.write("\n")

            f.write("\n")

    print("Saved calibrator quality diagnostic:")
    print(csv_path)
    print(txt_path)

    print("\nDiagnosis summary:")
    print(diag_df["diagnosis"].value_counts())

    if update_tgt_info:

        bad_df = diag_df[diag_df["diagnosis"] == "BAD"]

        for i, row in bad_df.iterrows():

            hd_id = row["HD_ID"]

            if hd_id is not None and hd_id in tgt_info.index:
                tgt_info.loc[hd_id, "Quality"] = "BAD"

        updated_path = os.path.join(outdir, prefix + "_updated_tgt_info.csv")
        tgt_info.to_csv(updated_path)

        print("Updated tgt_info with BAD calibrators:")
        print(updated_path)

    return diag_df

def plot_final_calibrators(tgt_info,
                           cal_folder="/home2/ihernand/Desktop/reach/diagnostics/",
                           outdir="/home2/ihernand/Desktop/reach/diagnostics/",
                           prefix="final_calibrators",
                           quality_csv=None,
                           ncols=4):
    """
    Plot final calibrated V2 curves for all calibrators.
    """

    import os
    import glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    cal_oifits = glob.glob(os.path.join(cal_folder, "*oidataCalibrated*.fits"))
    cal_oifits.sort()

    if len(cal_oifits) == 0:
        print("No calibrator diagnostic FITS files found in:")
        print(cal_folder)
        return pd.DataFrame()

    quality_dict = {}

    if quality_csv is not None and os.path.exists(quality_csv):

        qdf = pd.read_csv(quality_csv)

        for i, row in qdf.iterrows():

            filename = row.get("filename", None)

            if filename is not None:
                quality_dict[filename] = {
                    "diagnosis": row.get("diagnosis", "not classified"),
                    "reason": row.get("reason", "")
                }

    pdf_path = os.path.join(outdir, prefix + ".pdf")
    csv_path = os.path.join(outdir, prefix + "_summary.csv")

    summary_rows = []

    nplots = len(cal_oifits)

    max_rows_per_page = 4
    plots_per_page = ncols * max_rows_per_page

    with PdfPages(pdf_path) as pdf:

        for page_start in range(0, nplots, plots_per_page):

            page_files = cal_oifits[page_start:page_start + plots_per_page]

            n_page = len(page_files)
            nrows_page = int(np.ceil(float(n_page) / float(ncols)))

            fig, axes = plt.subplots(
                nrows_page,
                ncols,
                figsize=(4.0 * ncols, 3.2 * nrows_page),
                squeeze=False
            )

            axes = axes.flatten()

            for ax_i, cal_file in enumerate(page_files):

                ax = axes[ax_i]

                filename = os.path.basename(cal_file)
                cal_name = get_calibrator_name_from_filename(filename)

                try:
                    hd_id = match_target_name(tgt_info, cal_name, verbose=False)

                    if hd_id is not None:
                        primary = tgt_info.loc[hd_id, "Primary"]

                        if "Quality" in tgt_info.columns:
                            input_quality = tgt_info.loc[hd_id, "Quality"]
                        else:
                            input_quality = None
                    else:
                        primary = cal_name
                        input_quality = None

                except Exception:
                    hd_id = None
                    primary = cal_name
                    input_quality = None

                try:
                    x, vis2, e_vis2, flags = extract_vis2_flexible(cal_file)

                    x = np.asarray(x).flatten()
                    vis2 = np.asarray(vis2).flatten()
                    e_vis2 = np.asarray(e_vis2).flatten()
                    flags = np.asarray(flags).flatten().astype(bool)

                    n_total = len(vis2)

                    if len(x) != n_total:
                        x = np.arange(n_total).astype(float)

                    if len(e_vis2) != n_total:
                        e_vis2 = np.zeros(n_total) + np.nan

                    if len(flags) != n_total:
                        flags = np.zeros(n_total).astype(bool)

                    finite = (
                        np.isfinite(x) &
                        np.isfinite(vis2) &
                        np.isfinite(e_vis2) &
                        (e_vis2 > 0)
                    )

                    good = finite & (~flags)
                    bad = finite & flags

                    if filename in quality_dict:
                        diagnosis = quality_dict[filename]["diagnosis"]
                    else:
                        diagnosis = "not classified"

                    # Plot good points
                    if np.sum(good) > 0:
                        ax.errorbar(
                            x[good],
                            vis2[good],
                            yerr=e_vis2[good],
                            fmt=".",
                            elinewidth=0.4,
                            markersize=3,
                            color="black",
                            alpha=0.8
                        )

                    # Plot flagged points
                    if np.sum(bad) > 0:
                        ax.errorbar(
                            x[bad],
                            vis2[bad],
                            yerr=e_vis2[bad],
                            fmt="x",
                            elinewidth=0.4,
                            markersize=3,
                            color="red",
                            alpha=0.6
                        )

                    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.6)
                    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.6)

                    ax.set_ylim(-0.2, 1.4)

                    if np.sum(finite) > 0:
                        x_finite = x[finite]
                        ax.set_xlim(np.nanmin(x_finite) * 0,
                                    np.nanmax(x_finite) * 2.02)

                    title = "%s\n%s" % (str(primary), str(diagnosis))

                    if diagnosis == "BAD" or input_quality == "BAD":
                        ax.set_title(title, fontsize=8, color="red")
                    elif diagnosis == "CHECK":
                        ax.set_title(title, fontsize=8, color="orange")
                    elif diagnosis == "READ_ERROR":
                        ax.set_title(title, fontsize=8, color="red")
                    else:
                        ax.set_title(title, fontsize=8, color="black")

                    ax.tick_params(axis="both", labelsize=7)

                    if ax_i % ncols == 0:
                        ax.set_ylabel(r"$V^2$", fontsize=9)

                    if ax_i >= (nrows_page - 1) * ncols:
                        ax.set_xlabel(r"$B / \lambda$", fontsize=9)

                    n_good = int(np.sum(good))
                    n_flagged = int(np.sum(flags))

                    if n_total > 0:
                        flagged_frac = float(n_flagged) / float(n_total)
                    else:
                        flagged_frac = np.nan

                    summary_rows.append({
                        "filename": filename,
                        "cal_name_from_file": cal_name,
                        "HD_ID": hd_id,
                        "Primary": primary,
                        "input_Quality": input_quality,
                        "diagnosis": diagnosis,
                        "n_total": n_total,
                        "n_good": n_good,
                        "n_flagged": n_flagged,
                        "flagged_frac": flagged_frac
                    })

                except Exception as err:

                    ax.text(
                        0.5,
                        0.5,
                        "Could not read file\n%s" % str(err),
                        ha="center",
                        va="center",
                        fontsize=8,
                        transform=ax.transAxes
                    )

                    ax.set_title(filename, fontsize=8, color="red")

                    summary_rows.append({
                        "filename": filename,
                        "cal_name_from_file": cal_name,
                        "HD_ID": hd_id,
                        "Primary": primary,
                        "input_Quality": input_quality,
                        "diagnosis": "READ_ERROR",
                        "n_total": np.nan,
                        "n_good": np.nan,
                        "n_flagged": np.nan,
                        "flagged_frac": np.nan
                    })

            for empty_i in range(len(page_files), len(axes)):
                axes[empty_i].axis("off")

            fig.suptitle("Final calibrator diagnostic: calibrated visibility curves",
                         fontsize=14)

            fig.tight_layout(rect=[0, 0, 1, 0.96])

            pdf.savefig(fig)
            plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(csv_path, index=False)

    print("Saved final calibrator plot:")
    print(pdf_path)

    print("Saved final calibrator summary:")
    print(csv_path)

    return summary_df



def run_n_bootstraps_diagnostic(sequences, complete_sequences, base_path,
                                tgt_info, n_pred_ldd, e_pred_ldd,
                                n_bootstraps, results_path,
                                diag_path=None,
                                run_local=False,
                                already_calibrated=False,
                                do_random_ifg_sampling=True,
                                stop_on_error=False):
    """
    Diagnostic wrapper for bootstrapping.

    This function does not modify rpndrs.run_n_bootstraps().
    It simply runs one bootstrap at a time using rpndrs.run_one_calibration_set()
    and saves diagnostic information for each bootstrap.
    """

    import os
    import glob
    import traceback
    import datetime
    import numpy as np
    import pandas as pd
    import reach.pndrs as rpndrs

    if diag_path is None:
        diag_path = os.path.join(results_path, "bootstrap_diagnostics")

    if not os.path.exists(diag_path):
        os.makedirs(diag_path)

    summary_rows = []

    nights = [complete_sequences[seq][0] for seq in complete_sequences.keys()]
    nights = sorted(list(set(nights)))

    print("\n", "-"*79, "\n", "\tBootstrapping diagnostic mode\n", "-"*79)

    for bs_i in np.arange(0, n_bootstraps):

        t_start = datetime.datetime.now()

        print("\n", "|"*79)
        print("\tBootstrap diagnostic %i/%i" % (bs_i + 1, n_bootstraps))
        print("|"*79)

        txt_log = os.path.join(
            diag_path,
            "bootstrap_%03i_log.txt" % bs_i
        )

        ldd_csv = os.path.join(
            diag_path,
            "bootstrap_%03i_ldd_inputs.csv" % bs_i
        )

        traceback_file = os.path.join(
            diag_path,
            "bootstrap_%03i_traceback.txt" % bs_i
        )

        status = "OK"
        error_type = ""
        error_message = ""

        n_result_files_before = 0
        n_result_files_after = 0
        n_result_files_created = 0

        n_pred_nan = 0
        n_e_nan = 0
        n_ldd_targets = 0

        try:

            # -------------------------------------------------------------
            # Count existing result files before this bootstrap
            # -------------------------------------------------------------
            if os.path.exists(results_path):
                files_before = glob.glob(
                    os.path.join(results_path, "*_%02i.fits" % bs_i)
                )
                n_result_files_before = len(files_before)

            # -------------------------------------------------------------
            # Get LDD values for this bootstrap
            # This is exactly what rpndrs.run_n_bootstraps() does internally:
            # n_pred_ldd.iloc[bs_i]
            # -------------------------------------------------------------
            pred_this_bootstrap = n_pred_ldd.iloc[bs_i]

            target_ids = list(pred_this_bootstrap.index)
            n_ldd_targets = len(target_ids)

            # -------------------------------------------------------------
            # Extract e_pred_LDD only for diagnostic purposes
            # -------------------------------------------------------------
            if isinstance(e_pred_ldd, pd.DataFrame):

                if set(target_ids).issubset(set(e_pred_ldd.columns)):
                    e_values = e_pred_ldd.loc[:, target_ids].iloc[0].values

                elif set(target_ids).issubset(set(e_pred_ldd.index)):
                    e_values = e_pred_ldd.loc[target_ids].values.flatten()

                else:
                    e_values = np.zeros(len(target_ids)) + np.nan

            else:
                try:
                    e_values = e_pred_ldd.loc[target_ids].values
                except Exception:
                    try:
                        e_values = e_pred_ldd[target_ids].values
                    except Exception:
                        e_values = np.zeros(len(target_ids)) + np.nan

            # -------------------------------------------------------------
            # Save LDD input diagnostic
            # -------------------------------------------------------------
            ldd_diag = pd.DataFrame({
                "target_id": target_ids,
                "pred_LDD": pred_this_bootstrap.values,
                "e_pred_LDD": e_values
            })

            ldd_diag["pred_LDD_isnan"] = pd.isnull(ldd_diag["pred_LDD"])
            ldd_diag["e_pred_LDD_isnan"] = pd.isnull(ldd_diag["e_pred_LDD"])

            n_pred_nan = int(ldd_diag["pred_LDD_isnan"].sum())
            n_e_nan = int(ldd_diag["e_pred_LDD_isnan"].sum())

            ldd_diag.to_csv(ldd_csv, index=False)

            # -------------------------------------------------------------
            # Write initial log
            # -------------------------------------------------------------
            with open(txt_log, "w") as f:

                f.write("Bootstrap diagnostic\n")
                f.write("=" * 70 + "\n\n")

                f.write("Bootstrap index: %i\n" % bs_i)
                f.write("Start time: %s\n" % str(t_start))
                f.write("Number of sequences: %i\n" % len(complete_sequences))
                f.write("Number of nights: %i\n" % len(nights))

                f.write("\nNights:\n")
                for night in nights:
                    f.write("  %s\n" % night)

                f.write("\n")
                f.write("n_pred_ldd shape: %s\n" % str(n_pred_ldd.shape))

                if isinstance(e_pred_ldd, pd.DataFrame):
                    f.write("e_pred_ldd shape: %s\n" % str(e_pred_ldd.shape))
                    f.write("e_pred_ldd index sample: %s\n" %
                            str(list(e_pred_ldd.index[:10])))
                    f.write("e_pred_ldd columns sample: %s\n" %
                            str(list(e_pred_ldd.columns[:10])))
                else:
                    f.write("e_pred_ldd type: %s\n" % str(type(e_pred_ldd)))

                f.write("\n")
                f.write("Number of LDD targets: %i\n" % n_ldd_targets)
                f.write("Number of NaN pred_LDD: %i\n" % n_pred_nan)
                f.write("Number of NaN e_pred_LDD: %i\n" % n_e_nan)

                f.write("\n")
                f.write("LDD diagnostic CSV:\n")
                f.write("%s\n" % ldd_csv)

            # -------------------------------------------------------------
            # Run one bootstrap
            # -------------------------------------------------------------
            rpndrs.run_one_calibration_set(
                sequences,
                complete_sequences,
                base_path,
                tgt_info,
                pred_this_bootstrap,
                e_pred_ldd,
                bs_i,
                results_path,
                run_local=run_local,
                already_calibrated=already_calibrated,
                do_random_ifg_sampling=do_random_ifg_sampling
            )

            # -------------------------------------------------------------
            # Count result files after this bootstrap
            # -------------------------------------------------------------
            if os.path.exists(results_path):
                files_after = glob.glob(
                    os.path.join(results_path, "*_%02i.fits" % bs_i)
                )
                n_result_files_after = len(files_after)

            n_result_files_created = (
                n_result_files_after - n_result_files_before
            )

        except Exception as err:

            status = "FAILED"
            error_type = type(err).__name__
            error_message = str(err)

            with open(traceback_file, "w") as f:
                f.write(traceback.format_exc())

            print("\nERROR in bootstrap %i" % bs_i)
            print("%s: %s" % (error_type, error_message))
            print("Traceback saved to:")
            print(traceback_file)

            if stop_on_error:
                raise

        t_end = datetime.datetime.now()
        duration_seconds = (t_end - t_start).total_seconds()

        # -------------------------------------------------------------
        # Complete TXT log
        # -------------------------------------------------------------
        with open(txt_log, "a") as f:

            f.write("\n")
            f.write("End time: %s\n" % str(t_end))
            f.write("Duration seconds: %.3f\n" % duration_seconds)
            f.write("Status: %s\n" % status)

            f.write("\n")
            f.write("Result files before: %i\n" % n_result_files_before)
            f.write("Result files after: %i\n" % n_result_files_after)
            f.write("Result files created: %i\n" % n_result_files_created)

            if status == "FAILED":
                f.write("\n")
                f.write("Error type: %s\n" % error_type)
                f.write("Error message: %s\n" % error_message)
                f.write("Traceback file:\n")
                f.write("%s\n" % traceback_file)

        summary_rows.append({
            "bootstrap_i": bs_i,
            "status": status,
            "start_time": str(t_start),
            "end_time": str(t_end),
            "duration_seconds": duration_seconds,
            "n_sequences": len(complete_sequences),
            "n_nights": len(nights),
            "n_ldd_targets": n_ldd_targets,
            "n_pred_LDD_nan": n_pred_nan,
            "n_e_pred_LDD_nan": n_e_nan,
            "n_result_files_before": n_result_files_before,
            "n_result_files_after": n_result_files_after,
            "n_result_files_created": n_result_files_created,
            "error_type": error_type,
            "error_message": error_message,
            "ldd_csv": ldd_csv,
            "txt_log": txt_log,
            "traceback_file": traceback_file
        })

        print("\nBootstrap %i finished with status: %s" % (bs_i, status))
        print("Diagnostic log:")
        print(txt_log)

    # ---------------------------------------------------------------------
    # Save global summary
    # ---------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)

    summary_csv = os.path.join(diag_path, "bootstrap_summary.csv")
    summary_txt = os.path.join(diag_path, "bootstrap_summary.txt")

    summary_df.to_csv(summary_csv, index=False)

    with open(summary_txt, "w") as f:

        f.write("Bootstrap summary\n")
        f.write("=" * 70 + "\n\n")

        f.write("Number of bootstraps requested: %i\n" % n_bootstraps)
        f.write("Number of bootstraps completed/logged: %i\n" % len(summary_df))

        f.write("\nStatus counts:\n")

        if len(summary_df) > 0:
            for status_name, count in summary_df["status"].value_counts().items():
                f.write("  %s: %i\n" % (status_name, count))

        failed = summary_df[summary_df["status"] == "FAILED"]

        f.write("\nFailed bootstraps:\n")

        if len(failed) == 0:
            f.write("  None\n")
        else:
            for i, row in failed.iterrows():
                f.write("  bootstrap %i: %s - %s\n" %
                        (row["bootstrap_i"],
                         row["error_type"],
                         row["error_message"]))

        f.write("\nCSV summary:\n")
        f.write("%s\n" % summary_csv)

    print("\n", "-"*79)
    print("Bootstrapping diagnostics saved:")
    print(summary_csv)
    print(summary_txt)
    print("-"*79)

    return summary_df

def extract_vis2_for_bootstrap_plot(oifits_file):
    """
    Extract calibrated V2 data from an OIFITS file.

    Returns
    -------
    x : numpy array
        Spatial frequency B/lambda if possible. Otherwise point index.

    vis2 : numpy array
        Squared visibility values.

    e_vis2 : numpy array
        Squared visibility uncertainties.

    flags : numpy array
        Boolean flags.
    """

    import numpy as np
    from astropy.io import fits

    hdul = fits.open(oifits_file)

    wavelengths = None

    # ------------------------------------------------------------
    # Read effective wavelengths
    # ------------------------------------------------------------
    for hdu in hdul:

        extname = hdu.header.get("EXTNAME", "")

        if extname == "OI_WAVELENGTH":

            if hdu.data is not None and "EFF_WAVE" in hdu.data.names:
                wavelengths = np.asarray(hdu.data["EFF_WAVE"]).astype(float)
                break

    all_x = []
    all_vis2 = []
    all_evis2 = []
    all_flags = []

    # ------------------------------------------------------------
    # Read all OI_VIS2 extensions
    # ------------------------------------------------------------
    for hdu in hdul:

        extname = hdu.header.get("EXTNAME", "")

        if extname != "OI_VIS2":
            continue

        data = hdu.data

        if data is None:
            continue

        names = data.names

        if "VIS2DATA" not in names:
            continue

        vis2 = np.asarray(data["VIS2DATA"]).astype(float)

        if "VIS2ERR" in names:
            e_vis2 = np.asarray(data["VIS2ERR"]).astype(float)
        else:
            e_vis2 = np.zeros_like(vis2) + np.nan

        if "FLAG" in names:
            flags = np.asarray(data["FLAG"]).astype(bool)
        else:
            flags = np.zeros_like(vis2).astype(bool)

        # --------------------------------------------------------
        # Baseline from UCOORD/VCOORD
        # --------------------------------------------------------
        if "UCOORD" in names and "VCOORD" in names:
            ucoord = np.asarray(data["UCOORD"]).astype(float)
            vcoord = np.asarray(data["VCOORD"]).astype(float)
            baselines = np.sqrt(ucoord**2 + vcoord**2)
        else:
            baselines = None

        vis2_flat = vis2.flatten()
        e_vis2_flat = e_vis2.flatten()
        flags_flat = flags.flatten()

        n_total = len(vis2_flat)

        x = None

        # Normal OIFITS case: rows x wavelength channels
        if baselines is not None and wavelengths is not None:

            n_rows = len(baselines)
            n_wl = len(wavelengths)

            if n_rows * n_wl == n_total:

                bl_grid = np.repeat(baselines, n_wl)
                wl_grid = np.tile(wavelengths, n_rows)

                x = bl_grid / wl_grid

        # Fallback: use baseline only
        if x is None and baselines is not None:

            baselines_flat = np.asarray(baselines).flatten()

            if len(baselines_flat) == n_total:
                x = baselines_flat

        # Final fallback: point index
        if x is None:
            x = np.arange(n_total).astype(float)

        all_x.extend(list(x))
        all_vis2.extend(list(vis2_flat))
        all_evis2.extend(list(e_vis2_flat))
        all_flags.extend(list(flags_flat))

    hdul.close()

    if len(all_vis2) == 0:
        raise ValueError("No OI_VIS2 data found in file")

    return (
        np.asarray(all_x),
        np.asarray(all_vis2),
        np.asarray(all_evis2),
        np.asarray(all_flags)
    )

def get_target_name_from_oifits_filename(filename):
    """
    Extract target name from calibrated OIFITS filename.

    Examples
    --------
    2019-11-25_SCI_HD61574_oidataCalibrated_00.fits
    returns HD61574
    """

    import os

    base = os.path.basename(filename)

    if "_SCI_" in base:
        name = base.split("_SCI_")[-1].split("_oidata")[0]
    elif "SCI" in base:
        name = base.split("SCI")[-1].split("oidata")[0]
    else:
        name = base.split("_oidata")[0]

    name = name.replace("_bad", "")
    name = name.replace(".fits", "")
    name = name.strip("_")

    return name

def plot_bootstrap_visibilities(results_path,
                                bootstrap_i,
                                outdir,
                                prefix="bootstrap_visibility",
                                ncols=4):
    """
    Plot all calibrated visibility curves for one bootstrap.

    This function searches in results_path for files ending in _XX.fits,
    where XX is the bootstrap number.

    Example
    -------
    bootstrap_i = 0 reads files like *_00.fits
    bootstrap_i = 1 reads files like *_01.fits
    """

    import os
    import glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    pattern = os.path.join(results_path, "*_%02i.fits" % bootstrap_i)
    oifits_files = glob.glob(pattern)
    oifits_files.sort()

    pdf_path = os.path.join(
        outdir,
        "%s_%03i.pdf" % (prefix, bootstrap_i)
    )

    csv_path = os.path.join(
        outdir,
        "%s_%03i_summary.csv" % (prefix, bootstrap_i)
    )

    summary_rows = []

    if len(oifits_files) == 0:

        print("No OIFITS files found for bootstrap %i:" % bootstrap_i)
        print(pattern)

        empty_df = pd.DataFrame()
        empty_df.to_csv(csv_path, index=False)

        return {
            "pdf_path": pdf_path,
            "csv_path": csv_path,
            "n_files": 0
        }

    max_rows_per_page = 4
    plots_per_page = ncols * max_rows_per_page

    with PdfPages(pdf_path) as pdf:

        for page_start in range(0, len(oifits_files), plots_per_page):

            page_files = oifits_files[page_start:page_start + plots_per_page]

            n_page = len(page_files)
            nrows_page = int(np.ceil(float(n_page) / float(ncols)))

            fig, axes = plt.subplots(
                nrows_page,
                ncols,
                figsize=(4.0 * ncols, 3.2 * nrows_page),
                squeeze=False
            )

            axes = axes.flatten()

            for ax_i, oifits_file in enumerate(page_files):

                ax = axes[ax_i]

                filename = os.path.basename(oifits_file)
                target_name = get_target_name_from_oifits_filename(filename)

                try:
                    x, vis2, e_vis2, flags = extract_vis2_for_bootstrap_plot(
                        oifits_file
                    )

                    x = np.asarray(x).flatten()
                    vis2 = np.asarray(vis2).flatten()
                    e_vis2 = np.asarray(e_vis2).flatten()
                    flags = np.asarray(flags).flatten().astype(bool)

                    n_total = len(vis2)

                    if len(x) != n_total:
                        x = np.arange(n_total).astype(float)

                    if len(e_vis2) != n_total:
                        e_vis2 = np.zeros(n_total) + np.nan

                    if len(flags) != n_total:
                        flags = np.zeros(n_total).astype(bool)

                    finite = (
                        np.isfinite(x) &
                        np.isfinite(vis2) &
                        np.isfinite(e_vis2) &
                        (e_vis2 > 0)
                    )

                    good = finite & (~flags)
                    bad = finite & flags

                    n_good = int(np.sum(good))
                    n_flagged = int(np.sum(flags))
                    n_finite = int(np.sum(finite))

                    if n_total > 0:
                        flagged_frac = float(n_flagged) / float(n_total)
                    else:
                        flagged_frac = np.nan

                    if n_finite > 0:
                        bad_vis2 = finite & ((vis2 < -0.10) | (vis2 > 1.30))
                        bad_vis2_frac = float(np.sum(bad_vis2)) / float(n_finite)
                    else:
                        bad_vis2_frac = np.nan

                    if n_good > 0:
                        median_vis2 = np.nanmedian(vis2[good])
                        std_vis2 = np.nanstd(vis2[good])
                    else:
                        median_vis2 = np.nan
                        std_vis2 = np.nan

                    # --------------------------------------------------------
                    # Plot good points
                    # --------------------------------------------------------
                    if np.sum(good) > 0:
                        ax.errorbar(
                            x[good],
                            vis2[good],
                            yerr=e_vis2[good],
                            fmt=".",
                            markersize=3,
                            elinewidth=0.4,
                            alpha=0.8
                        )

                    # --------------------------------------------------------
                    # Plot flagged points
                    # --------------------------------------------------------
                    if np.sum(bad) > 0:
                        ax.errorbar(
                            x[bad],
                            vis2[bad],
                            yerr=e_vis2[bad],
                            fmt="x",
                            markersize=3,
                            elinewidth=0.4,
                            alpha=0.6
                        )

                    ax.axhline(1.0, linestyle="--", linewidth=0.6)
                    ax.axhline(0.0, linestyle=":", linewidth=0.6)

                    ax.set_ylim(-0.2, 1.4)

                    if np.sum(finite) > 0:
                        x_finite = x[finite]

                        xmin = np.nanmin(x_finite)
                        xmax = np.nanmax(x_finite)

                        if xmax > xmin:
                            ax.set_xlim(xmin * 0.98, xmax * 1.02)

                    ax.set_title(
                        "%s\nN=%i, flagged=%.2f" %
                        (target_name, n_good, flagged_frac),
                        fontsize=8
                    )

                    ax.tick_params(axis="both", labelsize=7)

                    if ax_i % ncols == 0:
                        ax.set_ylabel(r"$V^2$", fontsize=9)

                    if ax_i >= (nrows_page - 1) * ncols:
                        ax.set_xlabel(r"$B / \lambda$", fontsize=9)

                    summary_rows.append({
                        "bootstrap_i": bootstrap_i,
                        "filename": filename,
                        "target_name": target_name,
                        "n_total": n_total,
                        "n_finite": n_finite,
                        "n_good": n_good,
                        "n_flagged": n_flagged,
                        "flagged_frac": flagged_frac,
                        "bad_vis2_frac": bad_vis2_frac,
                        "median_vis2": median_vis2,
                        "std_vis2": std_vis2,
                        "read_status": "OK",
                        "error_message": ""
                    })

                except Exception as err:

                    ax.text(
                        0.5,
                        0.5,
                        "Could not read file\n%s" % str(err),
                        ha="center",
                        va="center",
                        fontsize=8,
                        transform=ax.transAxes
                    )

                    ax.set_title(target_name, fontsize=8)

                    summary_rows.append({
                        "bootstrap_i": bootstrap_i,
                        "filename": filename,
                        "target_name": target_name,
                        "n_total": np.nan,
                        "n_finite": np.nan,
                        "n_good": np.nan,
                        "n_flagged": np.nan,
                        "flagged_frac": np.nan,
                        "bad_vis2_frac": np.nan,
                        "median_vis2": np.nan,
                        "std_vis2": np.nan,
                        "read_status": "FAILED",
                        "error_message": str(err)
                    })

            for empty_i in range(len(page_files), len(axes)):
                axes[empty_i].axis("off")

            fig.suptitle(
                "Bootstrap %03i calibrated visibilities" % bootstrap_i,
                fontsize=14
            )

            fig.tight_layout(rect=[0, 0, 1, 0.96])

            pdf.savefig(fig)
            plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(csv_path, index=False)

    print("Saved visibility plot for bootstrap %i:" % bootstrap_i)
    print(pdf_path)
    print("Saved visibility summary:")
    print(csv_path)

    return {
        "pdf_path": pdf_path,
        "csv_path": csv_path,
        "n_files": len(oifits_files)
    }

def run_n_bootstraps_diagnostic_with_visibility_plots(
        sequences,
        complete_sequences,
        base_path,
        tgt_info,
        n_pred_ldd,
        e_pred_ldd,
        n_bootstraps,
        results_path,
        diag_path=None,
        run_local=False,
        already_calibrated=False,
        do_random_ifg_sampling=True,
        stop_on_error=False):
    """
    Diagnostic wrapper for bootstrapping with visibility plots.

    This function does not modify rpndrs.run_n_bootstraps().
    It runs one bootstrap at a time, saves diagnostics, and creates
    one PDF with calibrated V2 curves for each bootstrap.
    """

    import os
    import glob
    import traceback
    import datetime
    import numpy as np
    import pandas as pd
    import reach.pndrs as rpndrs

    if diag_path is None:
        diag_path = os.path.join(results_path, "bootstrap_diagnostics")

    if not os.path.exists(diag_path):
        os.makedirs(diag_path)

    visibility_plot_dir = os.path.join(diag_path, "visibility_plots")

    if not os.path.exists(visibility_plot_dir):
        os.makedirs(visibility_plot_dir)

    summary_rows = []

    nights = [complete_sequences[seq][0] for seq in complete_sequences.keys()]
    nights = sorted(list(set(nights)))

    print("\n", "-"*79)
    print("\tBootstrapping diagnostic mode with visibility plots")
    print("-"*79)

    for bs_i in np.arange(0, n_bootstraps):

        t_start = datetime.datetime.now()

        print("\n", "|"*79)
        print("\tBootstrap diagnostic %i/%i" % (bs_i + 1, n_bootstraps))
        print("|"*79)

        txt_log = os.path.join(
            diag_path,
            "bootstrap_%03i_log.txt" % bs_i
        )

        ldd_csv = os.path.join(
            diag_path,
            "bootstrap_%03i_ldd_inputs.csv" % bs_i
        )

        traceback_file = os.path.join(
            diag_path,
            "bootstrap_%03i_traceback.txt" % bs_i
        )

        status = "OK"
        error_type = ""
        error_message = ""

        n_result_files_before = 0
        n_result_files_after = 0
        n_result_files_created = 0

        n_pred_nan = 0
        n_e_nan = 0
        n_ldd_targets = 0

        visibility_pdf = ""
        visibility_csv = ""
        n_visibility_files = 0

        try:

            # -------------------------------------------------------------
            # Count existing result files before this bootstrap
            # -------------------------------------------------------------
            if os.path.exists(results_path):
                files_before = glob.glob(
                    os.path.join(results_path, "*_%02i.fits" % bs_i)
                )
                n_result_files_before = len(files_before)

            # -------------------------------------------------------------
            # Get LDD values for this bootstrap
            # -------------------------------------------------------------
            pred_this_bootstrap = n_pred_ldd.iloc[bs_i]

            target_ids = list(pred_this_bootstrap.index)
            n_ldd_targets = len(target_ids)

            # -------------------------------------------------------------
            # Extract e_pred_LDD only for diagnostics
            # -------------------------------------------------------------
            if isinstance(e_pred_ldd, pd.DataFrame):

                if set(target_ids).issubset(set(e_pred_ldd.columns)):
                    e_values = e_pred_ldd.loc[:, target_ids].iloc[0].values

                elif set(target_ids).issubset(set(e_pred_ldd.index)):
                    e_values = e_pred_ldd.loc[target_ids].values.flatten()

                else:
                    e_values = np.zeros(len(target_ids)) + np.nan

            else:
                try:
                    e_values = e_pred_ldd.loc[target_ids].values
                except Exception:
                    try:
                        e_values = e_pred_ldd[target_ids].values
                    except Exception:
                        e_values = np.zeros(len(target_ids)) + np.nan

            # -------------------------------------------------------------
            # Save LDD input diagnostic
            # -------------------------------------------------------------
            ldd_diag = pd.DataFrame({
                "target_id": target_ids,
                "pred_LDD": pred_this_bootstrap.values,
                "e_pred_LDD": e_values
            })

            ldd_diag["pred_LDD_isnan"] = pd.isnull(ldd_diag["pred_LDD"])
            ldd_diag["e_pred_LDD_isnan"] = pd.isnull(ldd_diag["e_pred_LDD"])

            n_pred_nan = int(ldd_diag["pred_LDD_isnan"].sum())
            n_e_nan = int(ldd_diag["e_pred_LDD_isnan"].sum())

            ldd_diag.to_csv(ldd_csv, index=False)

            # -------------------------------------------------------------
            # Write initial log
            # -------------------------------------------------------------
            with open(txt_log, "w") as f:

                f.write("Bootstrap diagnostic with visibility plots\n")
                f.write("=" * 70 + "\n\n")

                f.write("Bootstrap index: %i\n" % bs_i)
                f.write("Start time: %s\n" % str(t_start))
                f.write("Number of sequences: %i\n" % len(complete_sequences))
                f.write("Number of nights: %i\n" % len(nights))

                f.write("\nNights:\n")
                for night in nights:
                    f.write("  %s\n" % night)

                f.write("\n")
                f.write("n_pred_ldd shape: %s\n" % str(n_pred_ldd.shape))

                if isinstance(e_pred_ldd, pd.DataFrame):
                    f.write("e_pred_ldd shape: %s\n" % str(e_pred_ldd.shape))
                    f.write("e_pred_ldd index sample: %s\n" %
                            str(list(e_pred_ldd.index[:10])))
                    f.write("e_pred_ldd columns sample: %s\n" %
                            str(list(e_pred_ldd.columns[:10])))
                else:
                    f.write("e_pred_ldd type: %s\n" % str(type(e_pred_ldd)))

                f.write("\n")
                f.write("Number of LDD targets: %i\n" % n_ldd_targets)
                f.write("Number of NaN pred_LDD: %i\n" % n_pred_nan)
                f.write("Number of NaN e_pred_LDD: %i\n" % n_e_nan)

                f.write("\n")
                f.write("LDD diagnostic CSV:\n")
                f.write("%s\n" % ldd_csv)

            # -------------------------------------------------------------
            # Run one bootstrap calibration
            # -------------------------------------------------------------
            rpndrs.run_one_calibration_set(
                sequences,
                complete_sequences,
                base_path,
                tgt_info,
                pred_this_bootstrap,
                e_pred_ldd,
                bs_i,
                results_path,
                run_local=run_local,
                already_calibrated=already_calibrated,
                do_random_ifg_sampling=do_random_ifg_sampling
            )

            # -------------------------------------------------------------
            # Count result files after this bootstrap
            # -------------------------------------------------------------
            if os.path.exists(results_path):
                files_after = glob.glob(
                    os.path.join(results_path, "*_%02i.fits" % bs_i)
                )
                n_result_files_after = len(files_after)

            n_result_files_created = (
                n_result_files_after - n_result_files_before
            )

            # -------------------------------------------------------------
            # Plot calibrated visibilities for this bootstrap
            # -------------------------------------------------------------
            plot_info = plot_bootstrap_visibilities(
                results_path=results_path,
                bootstrap_i=bs_i,
                outdir=visibility_plot_dir,
                prefix="bootstrap_visibility",
                ncols=4
            )

            visibility_pdf = plot_info["pdf_path"]
            visibility_csv = plot_info["csv_path"]
            n_visibility_files = plot_info["n_files"]

        except Exception as err:

            status = "FAILED"
            error_type = type(err).__name__
            error_message = str(err)

            with open(traceback_file, "w") as f:
                f.write(traceback.format_exc())

            print("\nERROR in bootstrap %i" % bs_i)
            print("%s: %s" % (error_type, error_message))
            print("Traceback saved to:")
            print(traceback_file)

            # Try to plot whatever exists, even if the bootstrap failed
            try:
                plot_info = plot_bootstrap_visibilities(
                    results_path=results_path,
                    bootstrap_i=bs_i,
                    outdir=visibility_plot_dir,
                    prefix="bootstrap_visibility",
                    ncols=4
                )

                visibility_pdf = plot_info["pdf_path"]
                visibility_csv = plot_info["csv_path"]
                n_visibility_files = plot_info["n_files"]

            except Exception as plot_err:
                print("Could not create visibility plot after failure:")
                print(str(plot_err))

            if stop_on_error:
                raise

        t_end = datetime.datetime.now()
        duration_seconds = (t_end - t_start).total_seconds()

        # -------------------------------------------------------------
        # Complete TXT log
        # -------------------------------------------------------------
        with open(txt_log, "a") as f:

            f.write("\n")
            f.write("End time: %s\n" % str(t_end))
            f.write("Duration seconds: %.3f\n" % duration_seconds)
            f.write("Status: %s\n" % status)

            f.write("\n")
            f.write("Result files before: %i\n" % n_result_files_before)
            f.write("Result files after: %i\n" % n_result_files_after)
            f.write("Result files created: %i\n" % n_result_files_created)

            f.write("\n")
            f.write("Visibility PDF:\n")
            f.write("%s\n" % visibility_pdf)
            f.write("Visibility summary CSV:\n")
            f.write("%s\n" % visibility_csv)
            f.write("Number of files plotted: %i\n" % n_visibility_files)

            if status == "FAILED":
                f.write("\n")
                f.write("Error type: %s\n" % error_type)
                f.write("Error message: %s\n" % error_message)
                f.write("Traceback file:\n")
                f.write("%s\n" % traceback_file)

        summary_rows.append({
            "bootstrap_i": bs_i,
            "status": status,
            "start_time": str(t_start),
            "end_time": str(t_end),
            "duration_seconds": duration_seconds,
            "n_sequences": len(complete_sequences),
            "n_nights": len(nights),
            "n_ldd_targets": n_ldd_targets,
            "n_pred_LDD_nan": n_pred_nan,
            "n_e_pred_LDD_nan": n_e_nan,
            "n_result_files_before": n_result_files_before,
            "n_result_files_after": n_result_files_after,
            "n_result_files_created": n_result_files_created,
            "n_visibility_files": n_visibility_files,
            "visibility_pdf": visibility_pdf,
            "visibility_csv": visibility_csv,
            "error_type": error_type,
            "error_message": error_message,
            "ldd_csv": ldd_csv,
            "txt_log": txt_log,
            "traceback_file": traceback_file
        })

        print("\nBootstrap %i finished with status: %s" % (bs_i, status))
        print("Diagnostic log:")
        print(txt_log)

        if visibility_pdf != "":
            print("Visibility plot:")
            print(visibility_pdf)

    # ---------------------------------------------------------------------
    # Save global summary
    # ---------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)

    summary_csv = os.path.join(diag_path, "bootstrap_summary.csv")
    summary_txt = os.path.join(diag_path, "bootstrap_summary.txt")

    summary_df.to_csv(summary_csv, index=False)

    with open(summary_txt, "w") as f:

        f.write("Bootstrap summary with visibility plots\n")
        f.write("=" * 70 + "\n\n")

        f.write("Number of bootstraps requested: %i\n" % n_bootstraps)
        f.write("Number of bootstraps completed/logged: %i\n" % len(summary_df))

        f.write("\nStatus counts:\n")

        if len(summary_df) > 0:
            for status_name, count in summary_df["status"].value_counts().items():
                f.write("  %s: %i\n" % (status_name, count))

        failed = summary_df[summary_df["status"] == "FAILED"]

        f.write("\nFailed bootstraps:\n")

        if len(failed) == 0:
            f.write("  None\n")
        else:
            for i, row in failed.iterrows():
                f.write("  bootstrap %i: %s - %s\n" %
                        (row["bootstrap_i"],
                         row["error_type"],
                         row["error_message"]))

        f.write("\nVisibility PDFs:\n")

        for i, row in summary_df.iterrows():
            f.write("  bootstrap %i: %s\n" %
                    (row["bootstrap_i"], row["visibility_pdf"]))

        f.write("\nCSV summary:\n")
        f.write("%s\n" % summary_csv)

    print("\n", "-"*79)
    print("Bootstrapping diagnostics with visibility plots saved:")
    print(summary_csv)
    print(summary_txt)
    print("-"*79)

    return summary_df


def diagnose_ldd_analysis_results(label,
                                  tgt_info,
                                  sampled_sci_params,
                                  bs_results,
                                  results,
                                  outdir):
    """
    Diagnostic output for the LDD fitting analysis.

    This function does not modify the fitting results.
    It only saves CSV, TXT and PDF diagnostic files.

    Parameters
    ----------
    label : str
        Name of the analysis stage, e.g. "initial_literature_teff"
        or "final_interferometric_teff".

    tgt_info : pandas DataFrame
        Target information table.

    sampled_sci_params : object
        Output from rutils.load_sampled_params() or rparam.sample_all().

    bs_results : object
        Output from rdiam.fit_ldd_for_all_bootstraps().

    results : pandas DataFrame
        Output from rdiam.summarise_results().

    outdir : str
        Directory where diagnostics will be saved.
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # -------------------------------------------------------------------------
    # Helper functions
    # -------------------------------------------------------------------------

    def make_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def safe_dataframe(obj, name="object"):
        """
        Try to convert an object into a pandas DataFrame.

        Works for:
        - DataFrame
        - Series
        - dict of DataFrames
        - dict of Series
        - dict of arrays/lists/scalars
        """

        if isinstance(obj, pd.DataFrame):
            df = obj.copy()
            df.reset_index(inplace=True)
            return df

        if isinstance(obj, pd.Series):
            df = obj.to_frame(name)
            df.reset_index(inplace=True)
            return df

        if isinstance(obj, dict):

            rows = []

            for key in obj.keys():

                value = obj[key]

                if isinstance(value, pd.DataFrame):
                    df = value.copy()
                    df.reset_index(inplace=True)
                    df.insert(0, "dict_key", key)
                    rows.append(df)

                elif isinstance(value, pd.Series):
                    df = value.to_frame("value")
                    df.reset_index(inplace=True)
                    df.insert(0, "dict_key", key)
                    rows.append(df)

                elif isinstance(value, (list, tuple, np.ndarray)):
                    arr = np.asarray(value)

                    if arr.ndim == 1:
                        df = pd.DataFrame({
                            "dict_key": [key] * len(arr),
                            "array_index": np.arange(len(arr)),
                            "value": arr
                        })
                        rows.append(df)

                    elif arr.ndim == 2:
                        df = pd.DataFrame(arr)
                        df.insert(0, "dict_key", key)
                        rows.append(df)

                    else:
                        rows.append(pd.DataFrame({
                            "dict_key": [key],
                            "value": [str(value)]
                        }))

                else:
                    rows.append(pd.DataFrame({
                        "dict_key": [key],
                        "value": [value]
                    }))

            if len(rows) > 0:
                return pd.concat(rows, ignore_index=True, sort=False)

            return pd.DataFrame()

        # Fallback
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame({
                "object_name": [name],
                "type": [str(type(obj))],
                "value": [str(obj)]
            })

    def write_object_summary(f, obj, name):
        f.write("\n%s\n" % name)
        f.write("-" * 70 + "\n")
        f.write("Type: %s\n" % str(type(obj)))

        if isinstance(obj, pd.DataFrame):
            f.write("Shape: %s\n" % str(obj.shape))
            f.write("Columns:\n")
            for col in obj.columns:
                f.write("  %s\n" % str(col))
            f.write("Index sample:\n")
            f.write("%s\n" % str(list(obj.index[:10])))

        elif isinstance(obj, pd.Series):
            f.write("Length: %i\n" % len(obj))
            f.write("Index sample:\n")
            f.write("%s\n" % str(list(obj.index[:10])))

        elif isinstance(obj, dict):
            f.write("Number of keys: %i\n" % len(obj.keys()))
            f.write("Key sample:\n")
            for key in list(obj.keys())[:10]:
                f.write("  %s : %s\n" % (str(key), str(type(obj[key]))))

        else:
            f.write("Object preview:\n")
            f.write("%s\n" % str(obj)[:1000])

    def save_nan_summary(df, csv_path):
        if df is None or len(df) == 0:
            empty = pd.DataFrame()
            empty.to_csv(csv_path, index=False)
            return empty

        rows = []

        for col in df.columns:
            try:
                n_nan = int(pd.isnull(df[col]).sum())
                n_total = int(len(df[col]))
                frac_nan = float(n_nan) / float(n_total) if n_total > 0 else np.nan
            except Exception:
                n_nan = np.nan
                n_total = np.nan
                frac_nan = np.nan

            rows.append({
                "column": col,
                "n_total": n_total,
                "n_nan": n_nan,
                "frac_nan": frac_nan
            })

        nan_df = pd.DataFrame(rows)
        nan_df.to_csv(csv_path, index=False)

        return nan_df

    def save_numeric_summary(df, csv_path):
        if df is None or len(df) == 0:
            empty = pd.DataFrame()
            empty.to_csv(csv_path, index=False)
            return empty

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        rows = []

        for col in numeric_cols:

            values = np.asarray(df[col]).astype(float)
            finite = np.isfinite(values)

            if np.sum(finite) > 0:
                rows.append({
                    "column": col,
                    "n_total": len(values),
                    "n_finite": int(np.sum(finite)),
                    "n_nan": int(np.sum(~finite)),
                    "mean": np.nanmean(values),
                    "median": np.nanmedian(values),
                    "std": np.nanstd(values),
                    "min": np.nanmin(values),
                    "max": np.nanmax(values)
                })
            else:
                rows.append({
                    "column": col,
                    "n_total": len(values),
                    "n_finite": 0,
                    "n_nan": len(values),
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan
                })

        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(csv_path, index=False)

        return summary_df

    def plot_numeric_histograms(df, pdf_path, title):
        if df is None or len(df) == 0:
            return

        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

        if len(numeric_cols) == 0:
            return

        with PdfPages(pdf_path) as pdf:

            for col in numeric_cols:

                values = np.asarray(df[col]).astype(float)
                values = values[np.isfinite(values)]

                fig, ax = plt.subplots(figsize=(6, 4))

                if len(values) > 0:
                    ax.hist(values, bins=25)
                    ax.axvline(np.nanmedian(values), linestyle="--", linewidth=1)
                    ax.set_title("%s\n%s" % (title, col))
                    ax.set_xlabel(col)
                    ax.set_ylabel("N")
                else:
                    ax.text(0.5, 0.5, "No finite values",
                            ha="center", va="center",
                            transform=ax.transAxes)
                    ax.set_title("%s\n%s" % (title, col))

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    def plot_bootstrap_distributions(bs_df, pdf_path, label):
        """
        Try to plot bootstrap distributions per target.

        This is robust: it searches for likely target/star columns and likely
        LDD/diameter columns.
        """

        if bs_df is None or len(bs_df) == 0:
            return

        # Possible target ID columns
        possible_target_cols = [
            "dict_key",
            "target",
            "Target",
            "TARGET",
            "star",
            "Star",
            "STAR",
            "HD_ID",
            "Primary"
        ]

        target_col = None

        for col in possible_target_cols:
            if col in bs_df.columns:
                target_col = col
                break

        numeric_cols = list(bs_df.select_dtypes(include=[np.number]).columns)

        if len(numeric_cols) == 0:
            return

        # Prefer columns that look like LDD/diameter/theta
        preferred_cols = []

        for col in numeric_cols:
            col_low = str(col).lower()

            if ("ldd" in col_low or
                "diam" in col_low or
                "theta" in col_low or
                "ud" in col_low):
                preferred_cols.append(col)

        if len(preferred_cols) == 0:
            preferred_cols = numeric_cols[:5]

        with PdfPages(pdf_path) as pdf:

            if target_col is not None:

                targets = list(bs_df[target_col].dropna().unique())
                targets = targets[:200]

                for value_col in preferred_cols:

                    for target in targets:

                        sub = bs_df[bs_df[target_col] == target]

                        if value_col not in sub.columns:
                            continue

                        values = np.asarray(sub[value_col]).astype(float)
                        values = values[np.isfinite(values)]

                        fig, ax = plt.subplots(figsize=(6, 4))

                        if len(values) > 0:
                            ax.hist(values, bins=20)
                            ax.axvline(np.nanmedian(values),
                                       linestyle="--", linewidth=1)
                            ax.set_xlabel(value_col)
                            ax.set_ylabel("N")
                            ax.set_title(
                                "%s\n%s: %s\nmedian=%.5f, std=%.5f" %
                                (label, target_col, str(target),
                                 np.nanmedian(values), np.nanstd(values))
                            )
                        else:
                            ax.text(0.5, 0.5, "No finite values",
                                    ha="center", va="center",
                                    transform=ax.transAxes)
                            ax.set_title("%s\n%s: %s" %
                                         (label, target_col, str(target)))

                        fig.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)

            else:
                # If no target column exists, plot global distributions
                for value_col in preferred_cols:

                    values = np.asarray(bs_df[value_col]).astype(float)
                    values = values[np.isfinite(values)]

                    fig, ax = plt.subplots(figsize=(6, 4))

                    if len(values) > 0:
                        ax.hist(values, bins=25)
                        ax.axvline(np.nanmedian(values),
                                   linestyle="--", linewidth=1)
                        ax.set_xlabel(value_col)
                        ax.set_ylabel("N")
                        ax.set_title("%s\n%s" % (label, value_col))
                    else:
                        ax.text(0.5, 0.5, "No finite values",
                                ha="center", va="center",
                                transform=ax.transAxes)
                        ax.set_title("%s\n%s" % (label, value_col))

                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

    # -------------------------------------------------------------------------
    # Create diagnostic folder
    # -------------------------------------------------------------------------

    stage_dir = os.path.join(outdir, label)
    make_dir(stage_dir)

    # -------------------------------------------------------------------------
    # Convert objects to DataFrames when possible
    # -------------------------------------------------------------------------

    results_df = safe_dataframe(results, name="results")
    bs_df = safe_dataframe(bs_results, name="bs_results")
    sampled_df = safe_dataframe(sampled_sci_params, name="sampled_sci_params")

    # -------------------------------------------------------------------------
    # Save raw-ish CSV files
    # -------------------------------------------------------------------------

    results_csv = os.path.join(stage_dir, "%s_results.csv" % label)
    bs_csv = os.path.join(stage_dir, "%s_bs_results_flat.csv" % label)
    sampled_csv = os.path.join(stage_dir, "%s_sampled_sci_params_flat.csv" % label)

    results_df.to_csv(results_csv, index=False)
    bs_df.to_csv(bs_csv, index=False)
    sampled_df.to_csv(sampled_csv, index=False)

    # -------------------------------------------------------------------------
    # Save summaries
    # -------------------------------------------------------------------------

    results_nan_csv = os.path.join(stage_dir, "%s_results_nan_summary.csv" % label)
    bs_nan_csv = os.path.join(stage_dir, "%s_bs_results_nan_summary.csv" % label)
    sampled_nan_csv = os.path.join(stage_dir, "%s_sampled_params_nan_summary.csv" % label)

    save_nan_summary(results_df, results_nan_csv)
    save_nan_summary(bs_df, bs_nan_csv)
    save_nan_summary(sampled_df, sampled_nan_csv)

    results_num_csv = os.path.join(stage_dir, "%s_results_numeric_summary.csv" % label)
    bs_num_csv = os.path.join(stage_dir, "%s_bs_results_numeric_summary.csv" % label)

    save_numeric_summary(results_df, results_num_csv)
    save_numeric_summary(bs_df, bs_num_csv)

    # -------------------------------------------------------------------------
    # TXT diagnostic summary
    # -------------------------------------------------------------------------

    txt_path = os.path.join(stage_dir, "%s_diagnostic_summary.txt" % label)

    with open(txt_path, "w") as f:

        f.write("LDD analysis diagnostic\n")
        f.write("=" * 70 + "\n\n")

        f.write("Label: %s\n" % label)
        f.write("Output directory: %s\n" % stage_dir)

        write_object_summary(f, tgt_info, "tgt_info")
        write_object_summary(f, sampled_sci_params, "sampled_sci_params")
        write_object_summary(f, bs_results, "bs_results")
        write_object_summary(f, results, "results")

        f.write("\nSaved files\n")
        f.write("-" * 70 + "\n")
        f.write("results_csv: %s\n" % results_csv)
        f.write("bs_results_csv: %s\n" % bs_csv)
        f.write("sampled_sci_params_csv: %s\n" % sampled_csv)
        f.write("results_nan_summary: %s\n" % results_nan_csv)
        f.write("bs_results_nan_summary: %s\n" % bs_nan_csv)
        f.write("sampled_params_nan_summary: %s\n" % sampled_nan_csv)
        f.write("results_numeric_summary: %s\n" % results_num_csv)
        f.write("bs_results_numeric_summary: %s\n" % bs_num_csv)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    results_hist_pdf = os.path.join(
        stage_dir,
        "%s_results_numeric_histograms.pdf" % label
    )

    bs_hist_pdf = os.path.join(
        stage_dir,
        "%s_bootstrap_distributions.pdf" % label
    )

    plot_numeric_histograms(
        results_df,
        results_hist_pdf,
        "%s final results" % label
    )

    plot_bootstrap_distributions(
        bs_df,
        bs_hist_pdf,
        "%s bootstrap results" % label
    )

    print("\nDiagnostic saved for: %s" % label)
    print("Summary TXT:")
    print(txt_path)
    print("Results CSV:")
    print(results_csv)
    print("Bootstrap CSV:")
    print(bs_csv)
    print("Results histogram PDF:")
    print(results_hist_pdf)
    print("Bootstrap distribution PDF:")
    print(bs_hist_pdf)

    return {
        "stage_dir": stage_dir,
        "summary_txt": txt_path,
        "results_csv": results_csv,
        "bs_csv": bs_csv,
        "sampled_csv": sampled_csv,
        "results_hist_pdf": results_hist_pdf,
        "bs_hist_pdf": bs_hist_pdf,
        "results_nan_csv": results_nan_csv,
        "bs_nan_csv": bs_nan_csv,
        "results_numeric_csv": results_num_csv,
        "bs_numeric_csv": bs_num_csv
    }