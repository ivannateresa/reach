from __future__ import print_function

import os
import glob
import numpy as np

from astropy.io import fits


# ============================================================
# Sequence to inspect
# ============================================================

night = "2022-02-26"

target_to_inspect = "ksi_Gem"

baseline_to_inspect = "G1-J2"


# ============================================================
# Folder
# ============================================================

folder = (
    "/home/ihernand/Desktop/reach/"
    "complete_sequences/%s_v3.94_abcd/%s/"
    % (night, night)
)


print("\nFolder:")
print(folder)


# ============================================================
# Extra margin around time intervals
# ============================================================

# Extra margin applied before and after the measured interval.
mjd_margin = 0.001


# ============================================================
# Helper functions
# ============================================================

def normalise_name(name):

    return (
        str(name)
        .replace("_", "")
        .replace(" ", "")
        .lower()
    )


def canonical_baseline(name):

    parts = str(name).split("-")

    return "-".join(
        sorted(parts)
    )


# Requested target
target_clean_requested = normalise_name(
    target_to_inspect
)


# Requested baseline
requested_baseline_clean = canonical_baseline(
    baseline_to_inspect
)


# ============================================================
# Find reduced OIFITS
# ============================================================

files = sorted(
    glob.glob(
        folder + "PIONI*_oidata.fits"
    )
)


print("\nFound %i files" % len(files))


# ============================================================
# Store baseline time intervals
# ============================================================

baseline_intervals = []


# ============================================================
# Global flag counters
# ============================================================

total_vis2_rows = 0

total_vis2_points = 0

total_flagged_rows = 0

total_flagged_points = 0


# ============================================================
# Inspect files
# ============================================================

for filename in files:

    try:

        with fits.open(filename) as hdul:

            # ------------------------------------------------
            # Get target name
            # ------------------------------------------------

            if "OI_TARGET" not in hdul:
                continue


            targets = hdul["OI_TARGET"].data


            target_names = []


            for row in targets:

                target = row["TARGET"]


                if isinstance(target, bytes):

                    target = target.decode(
                        "utf-8"
                    )


                target_names.append(
                    str(target).strip()
                )


            # ------------------------------------------------
            # Keep only selected target
            # ------------------------------------------------

            target_found = False


            for target in target_names:

                clean = normalise_name(
                    target
                )


                if clean == target_clean_requested:

                    target_found = True


            if not target_found:
                continue


            print("")
            print("=" * 100)

            print(
                os.path.basename(
                    filename
                )
            )

            print(
                "TARGET:",
                target_names
            )

            print("=" * 100)


            # ------------------------------------------------
            # Read OI_ARRAY
            # ------------------------------------------------

            if "OI_ARRAY" not in hdul:

                print("\nNo OI_ARRAY")

                continue


            array = hdul["OI_ARRAY"].data


            tel_map = {}

            sta_map = {}


            print("")
            print("OI_ARRAY:")
            print("-" * 100)


            for row in array:

                idx = int(
                    row["STA_INDEX"]
                )


                tel = row["TEL_NAME"]

                sta = row["STA_NAME"]


                if isinstance(tel, bytes):

                    tel = tel.decode(
                        "utf-8"
                    )


                if isinstance(sta, bytes):

                    sta = sta.decode(
                        "utf-8"
                    )


                tel = str(tel).strip()

                sta = str(sta).strip()


                tel_map[idx] = tel

                sta_map[idx] = sta


                print(
                    "STA_INDEX=%i   TEL_NAME=%s   STA_NAME=%s"
                    % (
                        idx,
                        tel,
                        sta
                    )
                )


            # ------------------------------------------------
            # Read OI_VIS2
            # ------------------------------------------------

            if "OI_VIS2" not in hdul:

                print("\nNo OI_VIS2")

                continue


            vis2 = hdul["OI_VIS2"].data


            # ============================================================
            # BASIC VIS2 INFORMATION
            # ============================================================

            print("")
            print("=" * 100)
            print("OI_VIS2 INFORMATION")
            print("=" * 100)

            print(
                "Number of OI_VIS2 rows:",
                len(vis2)
            )

            if len(vis2) > 0:

                print(
                    "VIS2 channels per row:",
                    len(
                        np.asarray(
                            vis2["VIS2DATA"][0]
                        ).ravel()
                    )
                )

            print("=" * 100)


            # ============================================================
            # DEBUG ALL VIS2 FLAGS
            # ============================================================

            print("")
            print("=" * 120)
            print("VIS2 FLAG DEBUG - ALL ROWS")
            print("=" * 120)

            print(
                "%-5s %-12s %-16s %-45s %-8s"
                % (
                    "ROW",
                    "BASELINE",
                    "MJD",
                    "FLAGS",
                    "NFLAG"
                )
            )

            print("-" * 120)


            file_flagged_rows = 0

            file_flagged_points = 0


            for row_i in range(
                    len(vis2)):

                pair = vis2[
                    "STA_INDEX"
                ][row_i]


                i1 = int(
                    pair[0]
                )

                i2 = int(
                    pair[1]
                )


                sta1 = sta_map.get(
                    i1,
                    "?"
                )

                sta2 = sta_map.get(
                    i2,
                    "?"
                )


                baseline = "%s-%s" % (
                    sta1,
                    sta2
                )


                mjd = float(
                    vis2[
                        "MJD"
                    ][row_i]
                )


                flags = np.asarray(
                    vis2[
                        "FLAG"
                    ][row_i]
                ).astype(
                    bool
                )


                flags = flags.ravel()


                n_true = int(
                    np.sum(
                        flags
                    )
                )


                print(
                    "%-5i %-12s %-16.8f %-45s %-8i"
                    % (
                        row_i,
                        baseline,
                        mjd,
                        str(flags),
                        n_true
                    )
                )


                total_vis2_rows += 1

                total_vis2_points += (
                    len(flags)
                )


                if n_true > 0:

                    file_flagged_rows += 1

                    file_flagged_points += (
                        n_true
                    )

                    total_flagged_rows += 1

                    total_flagged_points += (
                        n_true
                    )


            print("=" * 120)


            # ============================================================
            # ONLY FLAGGED ROWS
            # ============================================================

            print("")
            print("=" * 120)

            print(
                "ONLY ROWS WITH AT LEAST ONE FLAG = TRUE"
            )

            print("=" * 120)


            found_flagged = False


            for row_i in range(
                    len(vis2)):

                pair = vis2[
                    "STA_INDEX"
                ][row_i]


                i1 = int(
                    pair[0]
                )

                i2 = int(
                    pair[1]
                )


                sta1 = sta_map.get(
                    i1,
                    "?"
                )

                sta2 = sta_map.get(
                    i2,
                    "?"
                )


                baseline = "%s-%s" % (
                    sta1,
                    sta2
                )


                mjd = float(
                    vis2[
                        "MJD"
                    ][row_i]
                )


                flags = np.asarray(
                    vis2[
                        "FLAG"
                    ][row_i]
                ).astype(
                    bool
                )


                flags = flags.ravel()


                n_true = int(
                    np.sum(
                        flags
                    )
                )


                if n_true == 0:

                    continue


                found_flagged = True


                print(
                    "ROW=%i  baseline=%s  MJD=%.8f  flags=%s  Nflag=%i"
                    % (
                        row_i,
                        baseline,
                        mjd,
                        str(flags),
                        n_true
                    )
                )


            if not found_flagged:

                print(
                    "No FLAG=True points found in this file."
                )


            print("")
            print(
                "Flagged rows in file   : %i"
                % file_flagged_rows
            )

            print(
                "Flagged V2 points       : %i"
                % file_flagged_points
            )

            print("=" * 120)


            # ============================================================
            # ONLY SELECTED BASELINE FLAGS
            # ============================================================

            print("")
            print("=" * 120)

            print(
                "FLAGS FOR SELECTED BASELINE: %s"
                % baseline_to_inspect
            )

            print("=" * 120)


            selected_rows = 0

            selected_flagged_rows = 0

            selected_flagged_points = 0


            for row_i in range(
                    len(vis2)):

                pair = vis2[
                    "STA_INDEX"
                ][row_i]


                i1 = int(
                    pair[0]
                )

                i2 = int(
                    pair[1]
                )


                sta1 = sta_map.get(
                    i1,
                    "?"
                )

                sta2 = sta_map.get(
                    i2,
                    "?"
                )


                baseline = "%s-%s" % (
                    sta1,
                    sta2
                )


                baseline_clean = (
                    canonical_baseline(
                        baseline
                    )
                )


                if (
                    baseline_clean
                    !=
                    requested_baseline_clean
                ):

                    continue


                selected_rows += 1


                mjd = float(
                    vis2[
                        "MJD"
                    ][row_i]
                )


                flags = np.asarray(
                    vis2[
                        "FLAG"
                    ][row_i]
                ).astype(
                    bool
                )


                flags = flags.ravel()


                n_true = int(
                    np.sum(
                        flags
                    )
                )


                if n_true > 0:

                    selected_flagged_rows += 1

                    selected_flagged_points += (
                        n_true
                    )


                print(
                    "ROW=%i  baseline=%s  MJD=%.8f  flags=%s  Nflag=%i"
                    % (
                        row_i,
                        baseline,
                        mjd,
                        str(flags),
                        n_true
                    )
                )


            print("")

            print(
                "Rows for %s        : %i"
                % (
                    baseline_to_inspect,
                    selected_rows
                )
            )

            print(
                "Flagged rows        : %i"
                % selected_flagged_rows
            )

            print(
                "Flagged V2 points   : %i"
                % selected_flagged_points
            )

            print("=" * 120)


            # ============================================================
            # Unique baselines
            # ============================================================

            pairs = np.unique(
                vis2[
                    "STA_INDEX"
                ],
                axis=0
            )


            print("")
            print("=" * 100)
            print("BASELINES")
            print("=" * 100)


            for pair in pairs:

                i1 = int(
                    pair[0]
                )

                i2 = int(
                    pair[1]
                )


                tel1 = tel_map.get(
                    i1,
                    "?"
                )

                tel2 = tel_map.get(
                    i2,
                    "?"
                )


                sta1 = sta_map.get(
                    i1,
                    "?"
                )

                sta2 = sta_map.get(
                    i2,
                    "?"
                )


                print(
                    "%s-%s    -->    %s-%s"
                    % (
                        tel1,
                        tel2,
                        sta1,
                        sta2
                    )
                )


                # ============================================================
                # Select rows belonging to this baseline
                # ============================================================

                sta_index = vis2[
                    "STA_INDEX"
                ]


                mask = (

                    (
                        (
                            sta_index[:, 0]
                            == i1
                        )
                        &
                        (
                            sta_index[:, 1]
                            == i2
                        )
                    )

                    |

                    (
                        (
                            sta_index[:, 0]
                            == i2
                        )
                        &
                        (
                            sta_index[:, 1]
                            == i1
                        )
                    )

                )


                # ============================================================
                # Get MJD values
                # ============================================================

                mjd_values = np.asarray(
                    vis2[
                        "MJD"
                    ][mask],
                    dtype=float
                )


                mjd_values = mjd_values[
                    np.isfinite(
                        mjd_values
                    )
                ]


                if len(
                        mjd_values) == 0:

                    continue


                # ============================================================
                # Original interval
                # ============================================================

                original_start_mjd = (
                    np.min(
                        mjd_values
                    )
                )


                original_end_mjd = (
                    np.max(
                        mjd_values
                    )
                )


                # ============================================================
                # Add margin
                # ============================================================

                start_mjd = (
                    original_start_mjd
                    - mjd_margin
                )


                end_mjd = (
                    original_end_mjd
                    + mjd_margin
                )


                # ============================================================
                # Baseline name
                # ============================================================

                baseline = "%s-%s" % (
                    sta1,
                    sta2
                )


                # ============================================================
                # Store result
                # ============================================================

                baseline_intervals.append(
                    (
                        night,
                        baseline,
                        start_mjd,
                        end_mjd,
                        original_start_mjd,
                        original_end_mjd,
                        os.path.basename(
                            filename
                        )
                    )
                )


    except Exception as e:

        print("")
        print("ERROR:")

        print(
            filename
        )

        print(
            str(e)
        )


# ============================================================
# GLOBAL FLAG SUMMARY
# ============================================================

print("")
print("=" * 120)
print("GLOBAL VIS2 FLAG SUMMARY")
print("=" * 120)


print(
    "Total OI_VIS2 rows       : %i"
    % total_vis2_rows
)


print(
    "Total V2 points          : %i"
    % total_vis2_points
)


print(
    "Rows with FLAG=True      : %i"
    % total_flagged_rows
)


print(
    "Individual flagged V2    : %i"
    % total_flagged_points
)


print(
    "Individual unflagged V2  : %i"
    % (
        total_vis2_points
        -
        total_flagged_points
    )
)


print("=" * 120)


# ============================================================
# Final table: all baselines
# ============================================================

print("")

print("=" * 90)


print(
    "BASELINE TIME INTERVALS FOR %s"
    % target_to_inspect
)


print("=" * 90)


print(
    "%-13s %-10s %-18s %-18s"
    % (
        "night",
        "baseline",
        "start_MJD",
        "end_MJD"
    )
)


print("-" * 90)


for (
    obs_night,
    baseline,
    start_mjd,
    end_mjd,
    original_start_mjd,
    original_end_mjd,
    filename
) in baseline_intervals:


    print(
        "%-13s %-10s %-18.8f %-18.8f"
        % (
            obs_night,
            baseline,
            start_mjd,
            end_mjd
        )
    )


# ============================================================
# Only selected baseline
# ============================================================

print("")

print("=" * 90)


print(
    "ONLY %s FOR %s"
    % (
        baseline_to_inspect,
        target_to_inspect
    )
)


print("=" * 90)


print(
    "%-13s %-10s %-18s %-18s"
    % (
        "night",
        "baseline",
        "start_MJD",
        "end_MJD"
    )
)


print("-" * 90)


for (
    obs_night,
    baseline,
    start_mjd,
    end_mjd,
    original_start_mjd,
    original_end_mjd,
    filename
) in baseline_intervals:


    baseline_clean = (
        canonical_baseline(
            baseline
        )
    )


    if (
        baseline_clean
        !=
        requested_baseline_clean
    ):

        continue


    print(
        "%-13s %-10s %-18.8f %-18.8f"
        % (
            obs_night,
            baseline,
            start_mjd,
            end_mjd
        )
    )


# ============================================================
# Detailed selected baseline table
# ============================================================

print("")

print("=" * 125)


print(
    "DETAILED %s INTERVALS FOR %s"
    % (
        baseline_to_inspect,
        target_to_inspect
    )
)


print("=" * 125)


print(
    "%-13s %-10s %-16s %-16s %-16s %-16s %s"
    % (
        "night",
        "baseline",
        "start_original",
        "end_original",
        "start_margin",
        "end_margin",
        "file"
    )
)


print("-" * 125)


for (
    obs_night,
    baseline,
    start_mjd,
    end_mjd,
    original_start_mjd,
    original_end_mjd,
    filename
) in baseline_intervals:


    baseline_clean = (
        canonical_baseline(
            baseline
        )
    )


    if (
        baseline_clean
        !=
        requested_baseline_clean
    ):

        continue


    print(
        "%-13s %-10s %-16.8f %-16.8f %-16.8f %-16.8f %s"
        % (
            obs_night,
            baseline,
            original_start_mjd,
            original_end_mjd,
            start_mjd,
            end_mjd,
            filename
        )
    )