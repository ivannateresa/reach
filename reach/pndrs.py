"""Module to handle interacting with the PIONIER data reduction pipeline, pndrs
"""
from __future__ import division, print_function
import os
import sys
import glob
import datetime
import numpy as np
import pandas as pd
import reach.diameters as rdiam
import reach.plotting as rplt
import reach.utils as rutils
from shutil import copyfile, rmtree
from astropy.io import fits
from astropy.time import Time
from collections import OrderedDict, Counter

# -----------------------------------------------------------------------------
# pndrs Affiliated Functions
# -----------------------------------------------------------------------------
base_path = "/home2/ihernand/Desktop/reach/complete_sequences" 

def clean_target_id(x):
    """
    Clean target names for robust matching.

    Examples
    --------
    HR_2426       -> hr2426
    HD  63734     -> hd63734
    psi Vel A     -> psivela
    psi_Vel       -> psivel
    bet_Hyi_bad   -> bethyi
    """

    if pd.isnull(x):
        return ""

    x = str(x).strip()


    # --------------------------------------------------------
    # Remove suffix used for manually flagged/bad observations
    # --------------------------------------------------------

    if x.lower().endswith("_bad"):
        x = x[:-4]


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
def safe_get_unique_keys(tgt_info_input, names, label):

    ids, failed = match_target_list(
        tgt_info_input,
        names,
        label=label,
        verbose=True
    )

    return ids, failed

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

def save_nightly_ldd(sequences, complete_sequences, tgt_info,
                     pred_ldd, e_pred_ldd,
                     base_path,
                     dir_suffix="_v3.94_abcd", run_local=False):
    """
    Create one oiDiam.fits file per observing night.

    New behaviour
    -------------
    1. Read the targets actually present in the sampled OIFITS files.
    2. Match each observed target to tgt_info.
    3. Take LDD, uncertainty, magnitudes and SCI/CAL status from tgt_info.
    4. Write only one oiDiam entry per observed target.
    5. Use a target name compatible with the way PNDRS identifies the object.

    This avoids the old behaviour based on:
        HD_ID, Ref_ID_1, Ref_ID_2, Ref_ID_3

    which could produce aliases not recognised by PNDRS.
    """

    print(
        "\n",
        "-" * 79,
        "\n",
        "\tSaving Nightly oidiam files\n",
        "-" * 79
    )


    # =====================================================================
    # Helper functions
    # =====================================================================

    def get_scalar(data, target_id):
        """
        Get one scalar value from either:
            - pandas Series
            - pandas DataFrame with target IDs in columns
            - pandas DataFrame with target IDs in index
        """

        if isinstance(data, pd.DataFrame):

            if target_id in data.columns:

                value = data[target_id].values

            elif target_id in data.index:

                value = data.loc[target_id].values

            else:

                raise KeyError(
                    "Target %s not found"
                    % target_id
                )

        else:

            try:
                value = data.loc[target_id]

            except Exception:
                value = data[target_id]


        value = np.asarray(value).flatten()

        if len(value) == 0:

            raise ValueError(
                "No value for target %s"
                % target_id
            )

        return float(value[0])


    def safe_float(value, default):

        try:

            if pd.isnull(value):
                return default

            return float(value)

        except Exception:

            return default


    def safe_string(value, default=""):

        try:

            if pd.isnull(value):
                return default

        except Exception:
            pass

        return str(value).strip()


    def match_real_target(target_name):
        """
        Match a target appearing in OI_TARGET to tgt_info.

        Includes special fallback for names such as:
            bet_Hyi_bad -> bet_Hyi
        """

        matched_id = match_target_name(
            tgt_info,
            target_name,
            verbose=False
        )


        if matched_id is not None:
            return matched_id


        # Special suffix used in some reduced observations
        target_lower = str(target_name).lower()

        if target_lower.endswith("_bad"):

            base_name = str(target_name)[:-4]

            matched_id = match_target_name(
                tgt_info,
                base_name,
                verbose=False
            )


        return matched_id


    def choose_pndrs_name(oi_target,
                          matched_id,
                          sequence_name_by_id):
        """
        Choose TARGET name to place in oiDiam.

        We use the real OI_TARGET to identify the object, but account
        for PNDRS name formatting.

        Examples
        --------
        HR_2426      -> HR2426
        HR_2342      -> HR2342
        HD  63734    -> HD63734
        HD_63734     -> HD63734

        ksi_Gem      -> ksi_Gem
        rho_Pup      -> rho_Pup
        iot_Psc      -> iot_Psc

        bet_Hyi_bad  -> bet_Hyi_bad
        """

        oi_target = str(
            oi_target
        ).strip()


        # -------------------------------------------------------------
        # Keep _bad exactly as it appears in the OIFITS.
        # It is a real observed target name.
        # -------------------------------------------------------------

        if oi_target.lower().endswith("_bad"):
            return oi_target


        clean = clean_target_id(
            oi_target
        )


        # -------------------------------------------------------------
        # HR targets
        #
        # OI_TARGET may contain HR_2426, but PNDRS uses HR2426.
        # -------------------------------------------------------------

        if (
            clean.startswith("hr")
            and clean[2:].isdigit()
        ):

            return "HR%s" % clean[2:]


        # -------------------------------------------------------------
        # HD targets
        #
        # HD_63734 / HD  63734 -> HD63734
        # -------------------------------------------------------------

        if (
            clean.startswith("hd")
            and clean[2:].isdigit()
        ):

            return oi_target


        # -------------------------------------------------------------
        # Bayer/common-name targets:
        # use the name from the observing sequence when possible.
        #
        # Examples:
        # IOT_PSC -> iot_Psc
        # ksi_Gem -> ksi_Gem
        # rho_Pup -> rho_Pup
        # -------------------------------------------------------------

        if matched_id in sequence_name_by_id:

            return str(
                sequence_name_by_id[
                    matched_id
                ]
            ).strip()


        # Fallback
        return oi_target


    # =====================================================================
    # Build list of nights
    # =====================================================================

    nights = OrderedDict()


    for seq in complete_sequences:

        night = complete_sequences[
            seq
        ][0]

        sequence = [
            star
            for star in sequences[seq]
        ]


        if night not in nights:

            nights[night] = set(
                sequence
            )

        else:

            nights[night].update(
                sequence
            )


    print(
        "Writing oiDiam.fits for %i nights"
        % len(nights)
    )


    # Only nights successfully written will be returned
    nights_written = OrderedDict()

    diam_files_written = 0


    # =====================================================================
    # Loop through nights
    # =====================================================================

    for night in nights:


        print(
            "\n" + "=" * 79
        )

        print(
            "Building oiDiam for %s"
            % night
        )

        print(
            "=" * 79
        )


        # -----------------------------------------------------------------
        # Folder containing the interferograms that initialise_interferograms
        # already selected for this bootstrap.
        # -----------------------------------------------------------------

        obs_dir = os.path.join(
            base_path % night,
            night
        )


        # -----------------------------------------------------------------
        # Build physical-target ID -> sequence name mapping
        # -----------------------------------------------------------------

        sequence_name_by_id = {}


        failed_sequence_match = False


        for star in nights[night]:

            matched_id = match_target_name(
                tgt_info,
                star,
                verbose=False
            )


            if matched_id is None:

                print(
                    "ERROR: sequence target could not be matched:"
                )

                print(
                    "  night  = %s"
                    % night
                )

                print(
                    "  target = %s"
                    % star
                )

                failed_sequence_match = True

                continue


            if matched_id not in sequence_name_by_id:

                sequence_name_by_id[
                    matched_id
                ] = star


        if failed_sequence_match:

            print(
                "Skipping %s because a sequence target "
                "could not be matched."
                % night
            )

            continue


        # -----------------------------------------------------------------
        # Read REAL OI_TARGET names from sampled OIFITS
        # -----------------------------------------------------------------

        oifits_files = sorted(
            glob.glob(
                os.path.join(
                    obs_dir,
                    "PIONI*_oidata.fits"
                )
            )
        )


        print(
            "Sampled OIFITS found: %i"
            % len(oifits_files)
        )


        if len(oifits_files) == 0:

            print(
                "ERROR: no PIONI*_oidata.fits files found in:"
            )

            print(
                obs_dir
            )

            continue


        real_targets = []


        for filename in oifits_files:

            try:

                with fits.open(
                    filename
                ) as hdul:


                    if "OI_TARGET" not in hdul:

                        print(
                            "WARNING: OI_TARGET missing in:"
                        )

                        print(
                            filename
                        )

                        continue


                    target_table = hdul[
                        "OI_TARGET"
                    ].data


                    for value in target_table[
                        "TARGET"
                    ]:

                        target_name = str(
                            value
                        ).strip()


                        if (
                            target_name != ""
                            and
                            target_name not in real_targets
                        ):

                            real_targets.append(
                                target_name
                            )


            except Exception as e:

                print(
                    "ERROR reading:"
                )

                print(
                    filename
                )

                print(
                    str(e)
                )


        real_targets.sort()


        print(
            "\nReal targets found in OI_TARGET:"
        )

        for target in real_targets:

            print(
                "  %s"
                % target
            )


        if len(real_targets) == 0:

            print(
                "ERROR: no targets found for %s"
                % night
            )

            continue


        # =================================================================
        # Build oiDiam rows
        # =================================================================

        rows = []

        used_pndrs_names = set()

        night_failed = False
        expected_ids = set()
        written_ids = set()

        print(
            "\nOI_TARGET -> tgt_info -> oiDiam"
        )

        print(
            "-" * 79
        )


        for oi_target in real_targets:


            # -------------------------------------------------------------
            # Match REAL target to tgt_info
            # -------------------------------------------------------------

            matched_id = match_real_target(
                oi_target
            )


            if matched_id is None:

                print(
                    "ERROR: cannot match OI_TARGET:"
                )

                print(
                    "  %s"
                    % oi_target
                )

                night_failed = True

                continue
            
            expected_ids.add(matched_id)
            info = tgt_info.loc[
                matched_id
            ]


            # -------------------------------------------------------------
            # Name PNDRS should receive in oiDiam
            # -------------------------------------------------------------

            pndrs_target = choose_pndrs_name(
                oi_target,
                matched_id,
                sequence_name_by_id
            )


            # Avoid exact duplicate target names
            if pndrs_target in used_pndrs_names:

                print(
                    "WARNING: duplicate PNDRS target name:"
                )

                print(
                    "  %s"
                    % pndrs_target
                )

                print(
                    "Skipping duplicate."
                )

                continue


            used_pndrs_names.add(
                pndrs_target
            )


            # -------------------------------------------------------------
            # Predicted diameter
            # -------------------------------------------------------------

            try:

                diam = get_scalar(
                    pred_ldd,
                    matched_id
                )


            except Exception as e:

                print(
                    "ERROR: cannot get LDD for:"
                )

                print(
                    "  OI_TARGET  = %s"
                    % oi_target
                )

                print(
                    "  matched_id = %s"
                    % matched_id
                )

                print(
                    str(e)
                )

                night_failed = True

                continue


            # Same behaviour as old function for NaN LDD
            if np.isnan(diam):

                diam = 1.0


            # -------------------------------------------------------------
            # Diameter uncertainty
            # -------------------------------------------------------------

            try:

                diamerr = get_scalar(
                    e_pred_ldd,
                    matched_id
                )


            except Exception as e:

                print(
                    "ERROR: cannot get e_LDD for:"
                )

                print(
                    "  OI_TARGET  = %s"
                    % oi_target
                )

                print(
                    "  matched_id = %s"
                    % matched_id
                )

                print(
                    str(e)
                )

                night_failed = True

                continue


            if np.isnan(diamerr):

                diamerr = 0.1


            # -------------------------------------------------------------
            # Magnitudes
            # -------------------------------------------------------------

            hmag = safe_float(
                info["Hmag"],
                0.0
            )

            kmag = safe_float(
                info["Kmag"],
                0.0
            )

            vmag = safe_float(
                info["Vmag"],
                0.0
            )


            # -------------------------------------------------------------
            # SCIENCE -> ISCAL
            #
            # tgt_info Science=True  -> ISCAL=0
            # tgt_info Science=False -> ISCAL=1
            # -------------------------------------------------------------

            science = bool(
                info["Science"]
            )


            if science:

                iscal = 0
                role = "SCI"

            else:

                iscal = 1
                role = "CAL"


            # -------------------------------------------------------------
            # INFO
            # -------------------------------------------------------------

            if "LDD_rel" in info.index:

                info_string = safe_string(
                    info["LDD_rel"],
                    ""
                )

            else:

                info_string = ""


            # -------------------------------------------------------------
            # Diagnostic
            # -------------------------------------------------------------

            print(
                "%-18s -> %-12s -> %-18s "
                "%s ISCAL=%i"
                % (
                    oi_target,
                    matched_id,
                    pndrs_target,
                    role,
                    iscal
                )
            )


            # TARGET_ID will be assigned below
            rows.append(
                (
                    diam,
                    diamerr,
                    hmag,
                    kmag,
                    vmag,
                    iscal,
                    pndrs_target,
                    info_string
                )
            )
            
            written_ids.add(matched_id)
        # =================================================================
        # Do NOT write a partial oiDiam
        # =================================================================
        print("\nCHECK OI_TARGET -> oiDiam")
        print("-" * 60)

        print(
            "Targets expected : %i"
            % len(expected_ids)
        )

        print(
            "Targets written  : %i"
            % len(written_ids)
        )


        missing_ids = expected_ids - written_ids


        if len(missing_ids) > 0:

            print("\nERROR: targets missing from oiDiam:")

            for target_id in sorted(missing_ids):
                print("  %s" % target_id)

            raise RuntimeError(
                "Incomplete oiDiam for night %s"
                % night
            )


        print("CHECK: OK")

        if night_failed:

            print(
                "\nERROR: night %s had matching/LDD errors."
                % night
            )

            print(
                "oiDiam will NOT be written for this night."
            )

            continue


        if len(rows) == 0:

            print(
                "ERROR: no oiDiam rows generated for %s"
                % night
            )

            continue


        # =================================================================
        # Sort alphabetically by PNDRS target name
        # =================================================================

        rows.sort(
            key=lambda x: x[6]
        )


        # Add TARGET_ID
        final_rows = []


        for target_i, row in enumerate(
            rows
        ):

            final_rows.append(
                (
                    target_i + 1,
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[7]
                )
            )


        # =================================================================
        # Build record array
        # =================================================================

        max_target = max(
            [
                len(str(row[7]))
                for row in final_rows
            ]
        )


        max_info = max(
            [
                len(str(row[8]))
                for row in final_rows
            ]
        )


        if max_target < 1:
            max_target = 1

        if max_info < 1:
            max_info = 1


        formats = (
            "int16,"
            "float64,"
            "float64,"
            "float64,"
            "float64,"
            "float64,"
            "int32,"
            "a%s,"
            "a%s"
        )


        formats = formats % (
            max_target,
            max_info
        )


        names = (
            "TARGET_ID,"
            "DIAM,"
            "DIAMERR,"
            "HMAG,"
            "KMAG,"
            "VMAG,"
            "ISCAL,"
            "TARGET,"
            "INFO"
        )


        rec = np.rec.array(
            final_rows,
            names=names,
            formats=formats
        )


        # =================================================================
        # FITS table
        # =================================================================

        hdu = fits.BinTableHDU.from_columns(
            rec
        )


        hdu.header[
            "EXTNAME"
        ] = (
            "OIU_DIAM",
            "name of this binary table extension"
        )


        # =================================================================
        # Output directory
        # =================================================================

        if not run_local:

            output_dir = obs_dir

        else:

            output_dir = "test"


        if not os.path.exists(
            output_dir
        ):

            os.makedirs(
                output_dir
            )


        fname = os.path.join(
            output_dir,
            night + "_oiDiam.fits"
        )


        hdu.writeto(
            fname,
            output_verify="warn",
            overwrite=True
        )


        print(
            "\nWROTE:"
        )

        print(
            fname
        )


        print(
            "\nFINAL oiDiam:"
        )

        print(
            "-" * 60
        )

        print(
            "%-20s %-5s %s"
            % (
                "TARGET",
                "ISCAL",
                "INFO"
            )
        )

        print(
            "-" * 60
        )


        for row in final_rows:

            print(
                "%-20s %-5i %s"
                % (
                    row[7],
                    row[6],
                    row[8]
                )
            )


        nights_written[
            night
        ] = nights[night]


        diam_files_written += 1


    print(
        "\n%i oiDiam.fits files written"
        % diam_files_written
    )


    return nights_written

def load_bad_baselines_log(
        bad_baseline_file="data/bad_baselines.txt"):
    """
    Load bad baselines.

    Format:
    night  baseline  start_MJD  end_MJD

    Multiple intervals are allowed for the same night.

    Example:
    2022-02-26 AT2-AT3 59636.1164 59636.1185
    2022-02-26 AT2-AT3 59636.1280 59636.1301
    """

    import os
    import numpy as np

    if (
        not os.path.exists(bad_baseline_file)
        or os.path.getsize(bad_baseline_file) == 0
    ):

        print(
            "No bad baselines found: %s is empty or missing"
            % bad_baseline_file
        )

        return {}

    bad_baselines = np.loadtxt(
        bad_baseline_file,
        dtype=str,
        comments="#"
    )

    if bad_baselines.size == 0:
        return {}

    # Important when the file contains only one line
    bad_baselines = np.atleast_2d(
        bad_baselines
    )

    bad_baseline_dict = {}

    for row in bad_baselines:

        if len(row) != 4:

            raise ValueError(
                "Bad baseline line must contain exactly 4 columns: "
                "night baseline start_MJD end_MJD. "
                "Found %i columns: %s"
                % (
                    len(row),
                    str(row)
                )
            )

        night = str(row[0])
        station = str(row[1])
        start = float(row[2])
        end = float(row[3])

        if night not in bad_baseline_dict:
            bad_baseline_dict[night] = []

        bad_baseline_dict[night].append(
            [
                station,
                start,
                end
            ]
        )

    # Diagnostic
    print("\nBad baselines loaded:")

    for night in sorted(bad_baseline_dict.keys()):

        for station, start, end in bad_baseline_dict[night]:

            print(
                "  %s  %-10s  %.13f -> %.13f"
                % (
                    night,
                    station,
                    start,
                    end
                )
            )

    return bad_baseline_dict
def load_bad_baselines_log_old():
    """Loads in the text file recording any bad baselines, where each entry
    has the form: (Period,ID,concatenation,station,start,end)
    
    Returns
    -------
    bad_baseline_dict: dict
        Dictionary mapping keys of string nights to (station_id, mjd1, mjd2)
    """
    bad_baseline_file = "data/bad_baselines.txt"
    
    # Load the file
    bad_baselines = np.loadtxt(bad_baseline_file, str, "#", " ")
    
    # Format to a dict
    if len(bad_baselines.shape) == 1:
        bad_baseline_dict = {bad_baselines[0]: [bad_baselines[4], 
                             float(bad_baselines[5]), float(bad_baselines[6])]}
    else:
        bad_baseline_dict = {baseline_entry[0]: [baseline_entry[4],
                            float(baseline_entry[5]), float(baseline_entry[5])]
                             for baseline_entry in bad_baselines}
                             
    return bad_baseline_dict

def get_observed_target_name(tgt_info, matched_id, observed_name):
    """
    Return the target name in the format expected by PNDRS.

    Examples
    --------
    HR_2342  -> HR2342
    HR_2391  -> HR2391
    HR_2426  -> HR2426
    ksi_Gem  -> ksi_Gem
    """

    name = str(observed_name).strip()

    # PNDRS/OIFITS uses HR2342 instead of HR_2342
    if name.startswith("HR_"):
        name = name.replace("HR_", "HR", 1)

    return name

def save_nightly_pndrs_script(
        complete_sequences,
        tgt_info,
        base_path,
        dir_suffix="_v3.94_abcd",
        run_local=False,
        use_bad_baselines=True):
    """This is a function to create and save the pndrs script files referencedF
    by pndrs during calibration. Each night of observations has a single such
    file with the name formatted per YYYY-MM-DD_pndrsScript.i containing a list
    of pndrs commands to run in order to customise the calibration procedure.
    
    Important here are the following commands:
        - Ignore some observations: oiFitsFlagOiData
        - Split the night: oiFitsSplitNight

    Parameters
    ----------
    complete_sequence: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        [night, grade, [[container, OB, target, grade, ob_time, obs_log, run, 
                         ob_fits],...]
    
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
    
    base_path: str
        String filepath where the calibrated data is stored.
    
    dir_suffix: str
        String suffix on the end of each folder of calibrated data
    
    run_local: bool
        Boolean indicating whether the pipeline is being run locally, and to
        save files instead within reach/test/ for inspection.
    """
    print("\n", "-"*79, "\n", "\tSaving Nightly pndrs Scripts\n", "-"*79)
    
    # Figure out what targets share nights
    # Of the form nights[night] = [mjd1, mjd2, ..., mjdn]
    sequence_times = {}
    
    for seq in complete_sequences.keys():
        # Get the string representing the night, YYYY-MM-DD
        night = complete_sequences[seq][0]
        
        # Get the datetime objects representing the first and last observations
        # of each sequence, and add or subtract a small increment as to bracket
        # the entire sequence between the time range. Convert these to MJD.
        delta = datetime.timedelta(seconds=10)
        first_ob = Time(complete_sequences[seq][2][0][4] - delta).mjd
        last_ob = Time(complete_sequences[seq][2][-1][4] + delta).mjd
        
        if night not in sequence_times:
            sequence_times[night] = [first_ob, last_ob]
        else:
            sequence_times[night] += [first_ob, last_ob]
            sequence_times[night].sort()
    
    # These lines are written to YYYY-MM-DD_pndrsScript.i alongside the MJD
    # to split upon
    line_split_1 = 'yocoLogInfo, "Split night to isolate SCI-CAL sequences";'
    line_split_2 = 'oiFitsSplitNight, oiWave, oiVis2, oiVis, oiT3, tsplit=cc;'
    
    # These lines are written to exclude bad calibrators, with the variable
    # 'startend' being a list with an MJD range to exclude
    line_exclude_1 = 'yocoLogInfo,"Ignore bad calibrators";'
    line_exclude_2 = ('oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis,' 
                      'tlimit=startend;')
    
    # These lines are written when excluding bad baselines based on station 
    # number and MJD
    line_bad_bl_1 = 'yocoLogInfo,"Ignore bad baselines";'
    line_bad_bl_2 = ('oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis,' 
                     'base=station, tlimit=startend;')
    
    # Get the record of sequences with bad baselines
    #bad_baseline_dict = load_bad_baselines_log()
    # ============================================================
    # BAD BASELINE MODE
    # ============================================================

    if use_bad_baselines:

        print("\nBAD BASELINE MODE: ON")
        print("Bad baselines WILL be excluded during calibration.")

        bad_baseline_dict = load_bad_baselines_log()

    else:

        print("\nBAD BASELINE MODE: OFF")
        print("ALL baselines will be preserved during calibration.")

        bad_baseline_dict = {}
    pndrs_scripts_written = 0
    no_script_nights = 0
    
    # Get a list of the target durations
    durations = calculate_target_durations(complete_sequences)
    bad_durations = select_only_bad_target_durations(durations, tgt_info)
    

    
    for night in sequence_times:
        # Save the fits file to the night directory
        if not run_local:
            dir = (base_path % night) + "%s/" % ( night)
        else:
            dir = "test"
       
        # Make the directory if it does not exist
        if not os.path.exists(dir):
            os.mkdir(dir)
        # ============================================================
        # REMOVE OLD PNDRS SCRIPT
        # ============================================================
        #
        # Very important:
        # an old script may contain baseline flags from a previous run.
        # Always remove it before deciding whether a new script is needed.
        # ============================================================

        fname = os.path.join(
            dir,
            night + "_pndrsScript.i"
        )

        if os.path.exists(fname):

            print(
                "Removing old PNDRS script: %s"
                % fname
            )

            os.remove(fname)
        # It is only meaningful to write a script if we need to split the night
        # (i.e. if more than one sequence has been observed, that is there are
        # 4 or more MJD entries) or have bad calibrators/baselines to exclude 
        if (len(sequence_times[night]) <= 2 and len(bad_durations[night]) < 1
            and night not in bad_baseline_dict):
            no_script_nights += 1
            continue     
        
        # This night requires a script to be written. When splitting the night,
        # we can neglect the first and last times as there are no observations
        # before or after these times respectively, and we only need one of any
        # pair of star1 end MJD and star2 start MJD              
        fname = dir + "/" + night + "_pndrsScript.i" 
        
        with open(fname, "w") as nightly_script:
            # Split the night
            if len(sequence_times[night]) > 2:
                nightly_script.write(line_split_1 + "\n")
                cc = "cc = %s;\n" % sequence_times[night][1:-1:2]
                nightly_script.write(cc)
                nightly_script.write(line_split_2)
            
            # Rule out bad calibrators
            # Note that this currently assumes only one bad calibrator per
            # science target - fix is to use star_i in string formatting
            if len(bad_durations[night]) >= 1:
                for star_i, bad_cal in enumerate(bad_durations[night]):
                    nightly_script.write(line_exclude_1 + "\n")
                    startend = "startend = %s;\n" % bad_cal[1:]
                    nightly_script.write(startend)
                    nightly_script.write(line_exclude_2 + "\n")
                    
            # Ignore observations with bad baselines using station ID and MJD
            if night in bad_baseline_dict:

                for bad_bl in bad_baseline_dict[night]:

                    station_name = bad_bl[0]
                    start_mjd = bad_bl[1]
                    end_mjd = bad_bl[2]

                    nightly_script.write(
                        'yocoLogInfo, "Ignore bad baseline %s";\n'
                        % station_name
                    )

                    nightly_script.write(
                        "startend = [%.13f, %.13f];\n"
                        % (
                            start_mjd,
                            end_mjd
                        )
                    )

                    nightly_script.write(
                        'station = "*%s*";\n'
                        % station_name
                    )

                    nightly_script.write(
                        "oiFitsFlagOiData, oiWave, oiArray, "
                        "oiVis2, oiT3, oiVis, "
                        "base=station, tlimit=startend;\n"
                    )

                    nightly_script.write("\n")
        
        # Done, move to the next night
        print("...wrote %s, night split into %s, bad calibrators: %s" 
              % (night, len(sequence_times[night])//2, 
                 len(bad_durations[night])))
        pndrs_scripts_written += 1
            
    print("%i pndrs.i scripts written" % pndrs_scripts_written)
    print("%i no script nights" % no_script_nights)        



def calculate_target_durations(complete_sequences):
    """For each night of observations, return the start and end time of 
    *sequential* observations associated with a given target.
    
    A typical CAL1-SCI1-CAL2-SCI2-CAL3 sequence observes each target 5 times 
    before moving on to the next target in the sequence. This function gets
    the first and last times of each block for the purpose of later excluding 
    bad calibrators.
    
    Parameters
    ----------
    complete_sequences: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        [night, grade, [[container, OB, target, grade, ob_time, obs_log, run, 
                         ob_fits],...]
    
    Returns
    -------
    sequence_durations: dict
        Output from calculate_target_durations, a dict mapping nights to start
        and end times for each target: durations[night] = [target, start, end]
    """
    # Initialise results dict
    sequence_durations = {}
    
    # Time difference to go before start of first observations, or after end of
    # last observation
    delta = datetime.timedelta(seconds=10)
    
    for seq in complete_sequences.keys():
        # Get a mapping of all target IDs to their times
        times = [(ob[2], ob[4]) for ob in complete_sequences[seq][2]]
        
        durations = [[times[0][0], Time(times[0][1] - delta).mjd, 0]]
        
        night = complete_sequences[seq][0]
        
        tgt_i = 0
        
        for (tgt, time) in times:
            # Same target
            if tgt == durations[tgt_i][0]:
                # Update the end time
                durations[tgt_i][2] = Time(time + delta).mjd
            
            # We've moved on
            else:
                tgt_i += 1
                durations.append([tgt, Time(time - delta).mjd, 0])
            
        # All done
        if night in sequence_durations.keys():
            sequence_durations[night] += durations
        else:
            sequence_durations[night] = durations
        
    return sequence_durations

def select_only_bad_target_durations(sequence_durations, tgt_info):
    """Takes the output of calculate_target_durations, and compares to the 
    target quality values in tgt_info, returning only durations for targets
    marked as BAD.
    """

    bad_durations = {}

    for night in sequence_durations:
        bad_durations[night] = []

        for star in sequence_durations[night]:

            # star[0] es el nombre del target/calibrador
           
            prim_id = rutils.get_unique_key(tgt_info, star[0])
           

            # Si no encuentra el target en tgt_info, no debe romper el codigo
            if len(prim_id) == 0:
                print("WARNING: target not found in tgt_info:", star[0], "night:", night)
                continue

            # Si existe Quality y esta marcado como BAD, se excluye
            if tgt_info.loc[prim_id[0]]["Quality"] == "BAD":
                bad_durations[night].append(star)

    return bad_durations

def select_only_bad_target_durations_old(sequence_durations, tgt_info):
    """Takes the output of calculate_target_durations, and compares to the 
    target quality values in tgt_info, returning only durations for only those
    targets which we wish to exclude from the calibration process.
    
    Parameters
    ----------
    sequence_durations: dict
        Output from calculate_target_durations, a dict mapping nights to start
        and end times for each target: durations[night] = [target, start, end]
        
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
        
    Returns
    -------
    bad_durations: dict
        Dict of same form as sequence_durations, but containing only the 
        calibrators we wish to exclude.
    """
    # Initialise results dict
    bad_durations = {}
    
    for night in sequence_durations:
        bad_durations[night] = []
        
        for star in sequence_durations[night]:
            # Get the star info, making sure to check primary, bayer, and HD
            # IDs given the non-unique IDs used
            prim_id = rutils.get_unique_key(tgt_info, star[0])
            
            # Check if it is a bad calibrator, and if so add to return dict
            if tgt_info.loc[prim_id[0]]["Quality"] == "BAD":
                bad_durations[night].append(star)
                
    return bad_durations


def reduce_all_observations(base_path):
    """Removes existing reduced and calibrated data, then runs pndrsReduce.
    
    Parameters
    ----------
    base_path: string
        Base directory housing the data.
    """
    print("\n", "-"*79, "\n", "\tDeleting old reduced/calibrated data\n", 
          "-"*79)
          
    # Delete existing reduced/calibrated data
    to_delete = glob.glob(base_path + "*_*/")
    to_delete.sort()
    
    for folder in to_delete:
        os.system("rm -rf %s" % folder)
        print("Deleted %s" % folder)
        
    # Run reduction using pndrsReduce
    to_reduce = glob.glob(base_path + "*/")
    to_reduce.sort()
    
    for folder in to_reduce:
        print("\n", "-"*79, "\n", "Reducing %s\n" % folder, "-"*79)
        os.system("cd %s; pndrsReduce" % folder)
        

def calibrate_all_observations(reduced_data_folders, bootstrap_i,
                               results_path, complete_sequences=None,
                               tgt_info=None):
    """Calls the PIONIER data reduction pipeline for each folder of reduced
    data from within Python.
    
    Parameters
    ----------
    reduced_data_folders: string array
        List of folder paths to run the calibration pipeline on
    """
    # List to record times for the start and end of each night to calibrate
    times = []
    
    # Run the PIONIER calibration pipeline for every folder with reduced data
    # TODO: capture the output and inspect for errors
    for night_i, ob_folder in enumerate(reduced_data_folders):
        # Record the start time
        times.append(datetime.datetime.now())    
    
        # Navigate to the night folder and call pndrsCalibrate from terminal
        night = ob_folder.split("/")[-2].split("_")[0]
        print("\nCalibrating %s, night %i/%i..." 
              % (night, night_i+1, len(reduced_data_folders)), end="")
        sys.stdout.flush()
        os.system("(cd %s; pndrsCalibrate >> cal_log_%0.4i.txt)" 
                  % (ob_folder, bootstrap_i))
        
        # Record and the end time and print duration
        times.append(datetime.datetime.now()) 
        cal_time = (times[-1] - times[-2]).total_seconds() 
        print("calibrated in %02d:%04.1f min" 
              % (int(np.floor(cal_time/60.)), cal_time % 60.))
        
        # Move oifits files back to central location (reach/results by default)
        move_sci_oifits_old(ob_folder,results_path,bootstrap_i)
    
    # All nights finished, print summary          
    total_time = (times[-1] - times[0]).total_seconds()    
    print("\nCalibration finished, %i nights in %02d:%04.1f\n" 
          % (len(reduced_data_folders),int(np.floor(total_time/60.)), 
             total_time % 60.))
        

def move_sci_oifits_old(obs_path, results_path, bootstrap_i):
    """Used to collect the calibrated oiFits files of all science targets after
    running the PIONIER data reduction pipeline. 
    
    Parameters
    ----------
    obs_path: string
        Base directory, will move any SCI_oifits files one directory deeper.
    
    new_path: string
        Folder to move the results to.
        
    bootstrap_i: int
        Integer count for the ith bootstrapping iteration
    """
    sci_oi_fits = glob.glob(obs_path + "/*SCI*oidataCalibrated.fits")
    
    #print("\n", "-"*79, "\n", "\tCopying complete sequences\n", "-"*79)
    
    for files_copied, oifits in enumerate(sci_oi_fits):
        # Make the folder if it doesn't exist
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        print(oifits)
        # Update the filename to keep copies of all potential bootstraps
        fname = oifits.split("/")[-1].replace(".fits", 
                                              "_%02i.fits" % bootstrap_i)

        print("...copying %s as %s" % (oifits.split("/")[-1], fname))
            
        copyfile(oifits, results_path + fname)
        files_copied += 1
    
    print("%i files copied" % files_copied)
    




def build_allowed_science_names(complete_sequences, tgt_info):
    """
    Build a set of cleaned names corresponding only to the real science targets.

    The real science target is taken from complete_sequences keys:
        seq = (period, science_target, bright/faint)

    Then we add all aliases from tgt_info for that same science target.
    """

    allowed = set()

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

    for seq in complete_sequences.keys():

        sci_name = seq[1]
        sci_clean = clean_target_id(sci_name)

        if sci_clean != "":
            allowed.add(sci_clean)

        matched_id = match_target_name(tgt_info, sci_name, verbose=False)

        if matched_id is None:
            print("WARNING: science target from complete_sequences not found:")
            print("  %s" % sci_name)
            continue

        # Add dataframe index / HD ID
        allowed.add(clean_target_id(matched_id))

        # Add aliases
        for col in search_cols:
            allowed.add(clean_target_id(tgt_info.loc[matched_id, col]))

    return allowed

def force_science_prefix(fname):
    """
    Change only the CAL/SCI label in the output filename.

    Examples
    --------
    2022-01-01_CAL_iot_Psc_oidataCalibrated.fits
    -> 2022-01-01_SCI_iot_Psc_oidataCalibrated.fits

    2022-01-01_SCI_iot_Psc_oidataCalibrated.fits
    -> unchanged
    """

    if "_CAL_" in fname:
        fname = fname.replace("_CAL_", "_SCI_", 1)

    elif fname.startswith("CAL_"):
        fname = fname.replace("CAL_", "SCI_", 1)

    return fname

def get_target_from_calibrated_filename(oifits):
    """
    Extract target name from calibrated pndrs filename.

    Works for:
    2022-01-01_SCI_iot_Psc_oidataCalibrated.fits
    2022-01-01_CAL_iot_Psc_oidataCalibrated.fits
    """

    import os

    base = os.path.basename(oifits)

    if "_SCI_" in base:
        target = base.split("_SCI_")[-1].split("_oidata")[0]

    elif "_CAL_" in base:
        target = base.split("_CAL_")[-1].split("_oidata")[0]

    elif "SCI_" in base:
        target = base.split("SCI_")[-1].split("_oidata")[0]

    elif "CAL_" in base:
        target = base.split("CAL_")[-1].split("_oidata")[0]

    else:
        target = base.split("_oidata")[0]

    target = target.strip("_")

    if target.endswith("_bad"):
        target = target.replace("_bad", "")

    return target


def force_correct_science_calibrator_prefix(fname, is_science):
    """
    Force output filename to have the correct SCI/CAL prefix.

    If is_science=True:
        *_CAL_target_* -> *_SCI_target_*

    If is_science=False:
        *_SCI_target_* -> *_CAL_target_*
    """

    if is_science:
        correct_prefix = "_SCI_"
        wrong_prefix = "_CAL_"
    else:
        correct_prefix = "_CAL_"
        wrong_prefix = "_SCI_"

    # Case with date prefix: 2022-01-01_CAL_target...
    if wrong_prefix in fname:
        fname = fname.replace(wrong_prefix, correct_prefix, 1)

    # Case starting directly with CAL_target or SCI_target
    elif is_science and fname.startswith("CAL_"):
        fname = fname.replace("CAL_", "SCI_", 1)

    elif (not is_science) and fname.startswith("SCI_"):
        fname = fname.replace("SCI_", "CAL_", 1)

    # If filename has neither SCI nor CAL, insert prefix before target is harder.
    # Usually pndrs filenames already contain SCI/CAL, so we leave it unchanged.

    return fname


def move_sci_oifits(obs_path, results_path, bootstrap_i, tgt_info=None):
    """
    Copy all calibrated OIFITS files, but force the correct SCI/CAL prefix
    using tgt_info["Science"].

    Important:
    pndrs may save calibrators as SCI_* or science targets as CAL_*.
    Therefore, the filename prefix is not trusted.
    """

    import os
    import glob
    from shutil import copyfile

    # Look for all calibrated files, not only SCI.
    all_oi_fits = glob.glob(obs_path + "/*oidataCalibrated.fits")
    all_oi_fits.sort()

    files_copied = 0
    files_skipped = 0

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    if len(all_oi_fits) == 0:

        print("No calibrated OIFITS found in:")
        print(obs_path)

        all_fits = glob.glob(obs_path + "/*.fits")
        for f in all_fits[:10]:
            print("  %s" % f)

        print("%i files copied" % files_copied)
        return files_copied

    for oifits in all_oi_fits:

        original_fname = os.path.basename(oifits)
        target_raw = get_target_from_calibrated_filename(oifits)

        # ------------------------------------------------------------
        # Decide if target is science or calibrator from tgt_info
        # ------------------------------------------------------------
        if tgt_info is not None:

            matched_id = match_target_name(tgt_info, target_raw, verbose=False)

            if matched_id is None:
                print("Skipping calibrated file because target was not found in tgt_info:")
                print("  file: %s" % oifits)
                print("  target_raw: %s" % target_raw)
                files_skipped += 1
                continue

            is_science = bool(tgt_info.loc[matched_id]["Science"])

        else:
            # Fallback: if no tgt_info is provided, keep original prefix
            matched_id = "UNKNOWN"
            is_science = None

        # ------------------------------------------------------------
        # Force correct prefix in copied filename
        # ------------------------------------------------------------
        if is_science is not None:
            output_fname = force_correct_science_calibrator_prefix(
                original_fname,
                is_science
            )
        else:
            output_fname = original_fname

        fname = output_fname.replace(
            ".fits",
            "_%02i.fits" % bootstrap_i
        )

        print("...copying %s as %s" % (original_fname, fname))
        print("   target_raw: %s" % target_raw)
        print("   matched_id: %s" % matched_id)
        print("   Science: %s" % str(is_science))

        copyfile(oifits, results_path + fname)
        files_copied += 1

    print("%i files copied" % files_copied)
    print("%i files skipped" % files_skipped)

    return files_copied

def initialise_interferograms(complete_sequences, base_path, n_ifg=5,
                              do_random_ifg_sampling=True):
    """Initialises interferograms for calibration by sampling from available 
    files and moving those selected to a subdirectory where pndrsCalibrate will
    be run. This involves random sampling with repeats, and renaming of files
    to account for there now potentially being duplicates.
    
    Parameters
    ----------
    complete_sequences: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        [night, grade, [[container, OB, target, grade, ob_time, obs_log, run, 
                         ob_fits],...]
    
    base_path: str
        String filepath where the calibrated data is stored.
        
    n_ifg: int
        Number of random samples with repeats to make of each target. Defaults
        to 5.
    
    do_random_ifg_sampling: bool
        Boolean indicating whether to randomly sample from the interferograms 
        or to use all data available.
    """
    print("\n", "-"*79, "\n", "\tInitialising Interferograms\n", "-"*79)
    
    # Clean out any old files before we get into the main loop - we can't do it
    # within the main loop itself, otherwise we'll potentially be deleting 
    # sequence from the same night that have already been sampled for this 
    # iteration of the bootstrapping. To ensure any deletions don't affect our
    # ability to run in "parallel" by calibrating different sets of nights
    # separately, get the nights from complete_sequences[seq][0]
    total_old_files = 0
    
    nights = [complete_sequences[seq][0] for seq in complete_sequences.keys()]
    nights = list(set(nights))
    nights.sort()
    #nights = [night for night in nights if night != "2022-08-11"]
    #nights = [night for night in nights if night != "2021-09-05"]
    for night in nights:
        night_folder = base_path % night
        bootstrapping_folder = night_folder + "%s/" % night
        old_files = glob.glob(bootstrapping_folder + "PIONI*")

        
        print("Deleting %i files from: %s" % (len(old_files), 
                                              bootstrapping_folder))
        for old_file in old_files:
            os.remove(old_file)
            total_old_files += 1
        
    print("\nRemoved %i old files \n" % total_old_files)

    # For every sequence, perform bootstrapping at the interferogram level


    for seq in complete_sequences.keys():
       
        night = complete_sequences[seq][0]
        night_folder = base_path % night
        bootstrapping_folder = night_folder + "/%s/" % night
        
        # Collect interferograms of the same target together, select N randomly
        # with repeats from these, copy to the subdirectory and rename, then
        # proceed to the next target
    
        if not os.path.exists(bootstrapping_folder):
           os.makedirs(bootstrapping_folder)
        #ifgs = sample_interferograms(complete_sequences[seq][2], n_ifg, 
        #                             do_random_ifg_sampling)
                # ============================================================
        # Remove FRINGE observations that do not have a reduced
        # _oidata.fits file.
        #
        # Example:
        # PNDRS may skip a raw FRINGE because there is no DARK.
        # Such a file must NOT be available for bootstrap sampling.
        # ============================================================

        obs_sequence = []

        n_missing_oidata = 0


        for obs in complete_sequences[seq][2]:

            # Keep DARK, KAPPA, etc.
            # sample_interferograms() already knows how to ignore them.
            if obs[8] != "FRINGE":

                obs_sequence.append(obs)

                continue


            # Raw filename stored in complete_sequences
            raw_filename = obs[7]

            fn = raw_filename.split("/")[-1]

            oidata_name = fn.replace(
                ".fits.Z",
                "_oidata.fits"
            )

            oidata_path = os.path.join(
                night_folder,
                oidata_name
            )


            # Only allow this FRINGE into the bootstrap pool
            # if PNDRS actually produced its reduced OIFITS.
            if os.path.exists(oidata_path):

                obs_sequence.append(obs)

            else:

                n_missing_oidata += 1

                print(
                    "WARNING: skipping unreduced FRINGE:"
                )

                print(
                    "  target : %s"
                    % obs[2]
                )

                print(
                    "  file   : %s"
                    % oidata_name
                )


        if n_missing_oidata > 0:

            print(
                "Skipped %i unreduced FRINGE file/s "
                "for %s on %s"
                % (
                    n_missing_oidata,
                    seq,
                    night
                )
            )


        # Now bootstrap ONLY from usable interferograms
        ifgs = sample_interferograms(
            obs_sequence,
            n_ifg,
            do_random_ifg_sampling
        )
        
        for i_ifg, ifg in enumerate(ifgs):
            fn = ifg.split("/")[-1]
            old_fn = fn.replace(".fits.Z", "_oidata.fits")
            new_fn = old_fn.replace("_oidata", "_i%02i_oidata" % i_ifg)
            
            copyfile(night_folder + old_fn, bootstrapping_folder + new_fn)
            
        print("Moved %i new interferograms for %s on %s" % (i_ifg+1, seq,
                                                            night))
            

def sample_interferograms(obs_sequence, n_ifg=5, do_random_ifg_sampling=True,
                          validate_mode=False):
    """Samples from among the available interferograms and returns a list of
    filenames.
    
    If do_random_ifg_sampling is True, n_ifg interferograms will be selected
    for each star for each appearance in the CAL1-SCI1-CAL2-SCI2-CAL3 sequence
    at random with repeats. If false, all available data from the sequence will
    be used.
    
    Parameters
    ----------
    obs_sequence: list
        List of all *raw* observations taken for this sequence originally from
        complete_sequences. Note that raw observations include DARK and KAPPA
        (flux splitting) exposures, and that these are ignored.
        
    n_ifg: int
        Number of random samples with repeats to make of each target. Defaults
        to 5.
    
    do_random_ifg_sampling: bool
        Boolean indicating whether to randomly sample from the interferograms 
        or to use all data available.
        
    validate_mode: bool
        Boolean used for testing purposes to inspect random sampling.
        
    Returns
    -------
    selected_ifgs: list
        List of sampled *raw* files of type FRINGE.
    """
    selected_ifgs = []
    ifg_i = 0
    
    # Initialise the current target/sequence
    current_tgt = obs_sequence[0][2]
    
    night = obs_sequence[0][5].split("/")[-2]
    
    current_ifgs = []
    
    while ifg_i < len(obs_sequence):
        # Get the current target
        new_tgt = obs_sequence[ifg_i][2]
        ifg_filename = obs_sequence[ifg_i][7]
        ifg_type = obs_sequence[ifg_i][8]
        
        # See if it matches the previous target, and if so record the filename
        # for the interferogram and continue the loop
        if new_tgt == current_tgt:
            # Only add file if it is a fringe
            if ifg_type == "FRINGE" and validate_mode:
                # In validate mode, append easier to read star names/data types
                # and numbers rather than filenames
                current_ifgs.append("%s_%s_%i" % (new_tgt, ifg_type, ifg_i))
                
            if ifg_type == "FRINGE":
                current_ifgs.append(ifg_filename)
                
            ifg_i += 1
        
        # Does not match, means we've moved onto the next target in the seq.
        # Now we should sample n_ifg times, and reset
        if new_tgt != current_tgt or ifg_i == len(obs_sequence):
            # Either sample randomly with repeats, or use all data
            if do_random_ifg_sampling:
                # In some cases where sequences have not been completed as 
                # CAL1-SCI1-CAL2-SCI2-CAL3 (e.g. out of order) we may end up
                # with a block that does not contain any fringes, specifically
                # a block containing only kappa files. These should be ignored.
                if len(current_ifgs) > 0:
                    selected_ifgs.extend(np.random.choice(current_ifgs, n_ifg))
                else:
                    print("Found block of entirely non-fringe files for",
                          "%s on night %s" % (current_tgt, night)) 
            else:
                selected_ifgs.extend(current_ifgs)
            
            # Reset, but don't increment counter (will just go through the loop
            # again and hit the first if statement)
            current_tgt = new_tgt
            current_ifgs = []
    
    #for ifg_i, ifg in enumerate(selected_ifgs):
        #print("%i\t%s" % (ifg_i, ifg))
     
    return selected_ifgs
    
    
def run_one_calibration_set(sequences, complete_sequences, base_path, 
                            tgt_info, pred_ldd, e_pred_ldd, bs_i, results_path,
                            run_local=False, already_calibrated=False,
                            do_random_ifg_sampling=True):
    """Runs a single bootstrapping iteration, completing the following steps: 
        - Write YYYY-MM-DD_oiDiam.fits files for each night of observing
        - Run pndrsCalibrate for each night of observing
        - Collate vis^2 and fit angular diameters for all science targets
    
    Parameters
    ----------
    sequences: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        lists of the targets in said CAL1-SCI1-CAL2-SCI2-CAL3 sequence. 
    
    complete_sequences: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        [night, grade, [[container, OB, target, grade, ob_time, obs_log, run, 
                         ob_fits],...]
    
    base_path: str
        String filepath where the calibrated data is stored.
    
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
        
    pred_ldd: pandas dataframe
        Pandas dataframe with columns being stars, and the values being LDD for
        a given bootstrapping iteration. Only one row.
    
    e_pred_ldd: pandas dataframe
        Pandas dataframe with columns being stars, and the values being the 
        uncertainties corresponding to e_pred_ldd. Only one row.
        
    bs_i: int
        Integer count for the ith bootstrapping iteration
    
    results_path: string
        Path to store the bootstrapped oifits files.
    
    run_local: bool
        Boolean indicating whether the pipeline is being run locally, and to
        save files instead within reach/test/ for inspection.
    
    already_calibrated: bool
        Boolean to skip calibration and proceed straight to result collation
        for testing purposes when results remain.
    
    do_random_ifg_sampling: bool
        Boolean indicating whether to randomly sample from the interferograms 
        or to use all data available.
    """
    # Intialise interferograms
    # Select the reduced interferograms which should be used for calibration
    initialise_interferograms(complete_sequences, base_path, n_ifg=5, 
                              do_random_ifg_sampling=do_random_ifg_sampling)
    
    if not run_local and not already_calibrated:
        # Save oiDiam files
        nights = save_nightly_ldd(sequences, complete_sequences, tgt_info,
                                  pred_ldd, e_pred_ldd, base_path)
        
        print("\n", "-"*79, "\n", "\tCalibrating %i night/s, bootstrap %i\n" 
              % (len(nights), bs_i), "-"*79)
     # ==============================================================
        # DEBUG: inspect oiDiam BEFORE pndrsCalibrate
        # ==============================================================

        print("\n" + "=" * 79)
        print("OIDiam BEFORE pndrsCalibrate")
        print("=" * 79)

        for night in nights.keys():

            oidiam = (
                base_path % night
                + "%s/%s_oiDiam.fits"
                % (night, night)
            )

            print("\nFile:")
            print(oidiam)

            # Save an untouched copy BEFORE pndrsCalibrate
            backup = (
                base_path % night
                + "%s/%s_oiDiam_BEFORE_pndrsCalibrate.fits"
                % (night, night)
            )

            copyfile(
                oidiam,
                backup
            )

            print("Backup:")
            print(backup)

            with fits.open(oidiam) as hdul:

                data = hdul["OIU_DIAM"].data

                print("\nTARGET        ISCAL        INFO")
                print("-" * 60)

                for row in data:

                    print(
                        "%-20s %-5s %s"
                        % (
                            row["TARGET"],
                            row["ISCAL"],
                            row["INFO"]
                        )
                    )
        # Run Calibration
        obs_folders = [base_path % night + "%s/" % night for night in nights.keys()]
        calibrate_all_observations(obs_folders,bs_i,results_path,complete_sequences=complete_sequences,tgt_info=tgt_info)
    
    elif run_local and not already_calibrated:
        # Save oiDiam files for local inspection
        nights = save_nightly_ldd(sequences, complete_sequences, tgt_info, 
                                  pred_ldd, e_pred_ldd, base_path,
                                  run_local=run_local)
    
    
def run_n_bootstraps(sequences, complete_sequences, base_path, tgt_info,
                     n_pred_ldd, e_pred_ldd, n_bootstraps, results_path,
                     run_local=False, already_calibrated=False, 
                     do_random_ifg_sampling=True):
    """Runs N bootstrapping iterations, collating and return the results.
    
    Parameters
    ----------
    sequences: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        lists of the targets in said CAL1-SCI1-CAL2-SCI2-CAL3 sequence. 
    
    complete_sequences: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        [night, grade, [[container, OB, target, grade, ob_time, obs_log, run, 
                         ob_fits],...]
    
    base_path: str
        String filepath where the calibrated data is stored.
    
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
        
    n_pred_ldd: pandas dataframe
        Pandas dataframe with columns being stars, and each row being a set of
        LDD for a given bootstrapping iteration. If not doing calibrator 
        bootstrapping, each row will be the same, but otherwise the calibrator
        angular diameters are drawn from a Gaussian distribution as part of the
        bootstrapping.
    
    e_pred_ldd: pandas dataframe
        Pandas dataframe with columns being stars, and the values being the 
        uncertainties corresponding to n_pred_ldd. Only one row.
        
    n_bootstraps: int
        The number of bootstrapping iterations to run.
    
    results_path: string
        Path to store the bootstrapped oifits files.
    
    run_local: bool
        Boolean indicating whether the pipeline is being run locally, and to
        save files instead within reach/test/ for inspection.
    
    already_calibrated: bool
        Boolean to skip calibration and proceed straight to result collation
        for testing purposes when results remain.
    
    do_random_ifg_sampling: bool
        Boolean indicating whether to randomly sample from the interferograms 
        or to use all data available.
        
    """
    # Initialise data structures for results
    n_vis2 = {}
    n_baselines = {}
    n_ldd_fit = {}
    
    times = []
    
    # Bootstrap n times
    for bs_i in np.arange(0, n_bootstraps):
        times.append(datetime.datetime.now())  
        print("\n", "|"*79, "\n\tBootstrapping iteration %i\n" % bs_i, "|"*79)
        
        # Run a single calibration run
        run_one_calibration_set(sequences, complete_sequences, base_path, 
                                tgt_info, n_pred_ldd.iloc[bs_i], 
                                e_pred_ldd, bs_i, results_path, 
                                run_local=run_local, 
                                already_calibrated=already_calibrated,
                                do_random_ifg_sampling=do_random_ifg_sampling)
                
        times.append(datetime.datetime.now())  
        b_i_time = (times[-1] - times[-2]).total_seconds() 
        print("\n\nBoostrap %i done in %02d:%04.1f\n" 
              % (bs_i+1, int(np.floor(b_i_time/60.)), b_i_time % 60.))
    
    total_t = (times[-1] - times[0]).total_seconds() 
    print("\n%i bootstraps done in %02d:%04.1f\n" 
              % (n_bootstraps, int(np.floor(total_t/60.)), total_t % 60.))
                
    # All done
    print("\n", "-"*79, "\n", "\tBootstrapping Complete\n", "-"*79)