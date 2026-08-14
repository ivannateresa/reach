from __future__ import print_function

import os
import glob
import csv
import collections

import pandas as pd
from astropy.io import fits

import reach.utils as rutils
import reach.pndrs as rpndrs


# =============================================================================
# SETTINGS
# =============================================================================

ROOT = "/home/ihernand/Desktop/reach"

base_path = os.path.join(
    ROOT,
    "complete_sequences",
    "%s_v3.94_abcd"
) + "/"

output_csv = os.path.join(
    ROOT,
    "oi_target_audit.csv"
)


# =============================================================================
# LOAD TARGET INFORMATION + SEQUENCES
# =============================================================================

print("\n" + "=" * 100)
print("LOAD DATA")
print("=" * 100)


tgt_info = rutils.initialise_tgt_info(
    True,
    70,
    False
)


complete_sequences, sequences = rutils.load_sequence_logs()


print("Complete sequences:", len(complete_sequences))


# =============================================================================
# HELPERS
# =============================================================================

def clean_name(name):

    if name is None:
        return ""

    return (
        str(name)
        .strip()
        .replace("_", "")
        .replace(" ", "")
        .replace(".", "")
        .replace("-", "")
        .lower()
    )


def decode_string(value):

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    return str(value).strip()


def safe_value(row, col):

    if col not in row.index:
        return ""

    value = row[col]

    if pd.isnull(value):
        return ""

    return str(value).strip()


# =============================================================================
# BUILD INFORMATION ABOUT EACH NIGHT
# =============================================================================

night_sequences = collections.defaultdict(list)


for seq in complete_sequences:

    night = complete_sequences[seq][0]

    night_sequences[night].append(seq)


# =============================================================================
# RESULTS
# =============================================================================

rows = []

problem_count = 0
unmatched_count = 0


# =============================================================================
# LOOP THROUGH ALL NIGHTS
# =============================================================================

print("\n" + "=" * 100)
print("AUDIT REAL OI_TARGET NAMES")
print("=" * 100)


for night in sorted(night_sequences.keys()):

    folder = base_path % night


    print("\n" + "=" * 100)
    print("NIGHT:", night)
    print("FOLDER:", folder)
    print("=" * 100)


    # =========================================================================
    # Expected targets from sequences
    # =========================================================================

    expected_names = []

    science_names = []


    for seq in night_sequences[night]:

        science_target = seq[1]

        science_names.append(
            science_target
        )


        if seq in sequences:

            for star in sequences[seq]:

                if star not in expected_names:
                    expected_names.append(star)


    print("\nTargets in sequence logs:")

    for star in expected_names:
        print("  ", star)


    # =========================================================================
    # Find reduced files
    # =========================================================================

    files = sorted(
        glob.glob(
            os.path.join(
                folder,
                "PIONI*_oidata.fits"
            )
        )
    )


    print(
        "\nReduced OIFITS found:",
        len(files)
    )


    if len(files) == 0:

        print("WARNING: no reduced files")

        continue


    # =========================================================================
    # Collect actual OI_TARGET names
    # =========================================================================

    actual_targets = collections.defaultdict(list)


    for filename in files:

        try:

            with fits.open(filename) as hdul:

                if "OI_TARGET" not in hdul:

                    print(
                        "WARNING: OI_TARGET missing:",
                        os.path.basename(filename)
                    )

                    continue


                data = hdul["OI_TARGET"].data


                for target in data["TARGET"]:

                    target = decode_string(target)

                    actual_targets[target].append(
                        os.path.basename(filename)
                    )


        except Exception as e:

            print(
                "ERROR reading:",
                filename
            )

            print(str(e))


    # =========================================================================
    # Inspect every unique actual target
    # =========================================================================

    print("\nREAL TARGETS FOUND IN OI_TARGET")
    print("-" * 100)


    for oi_target in sorted(actual_targets.keys()):


        # ---------------------------------------------------------------------
        # Match actual OI_TARGET against tgt_info
        # ---------------------------------------------------------------------

        matched_id = rpndrs.match_target_name(
            tgt_info,
            oi_target,
            verbose=False
        )


        if matched_id is None:

            actual_role = "UNKNOWN"

            primary = ""
            hd_id = ""
            ref1 = ""
            ref2 = ""
            ref3 = ""

            unmatched_count += 1

        else:

            info = tgt_info.loc[matched_id]


            if bool(info["Science"]):
                actual_role = "SCI"
            else:
                actual_role = "CAL"


            primary = safe_value(
                info,
                "Primary"
            )

            hd_id = safe_value(
                info,
                "HD_ID"
            )

            ref1 = safe_value(
                info,
                "Ref_ID_1"
            )

            ref2 = safe_value(
                info,
                "Ref_ID_2"
            )

            ref3 = safe_value(
                info,
                "Ref_ID_3"
            )


        # ---------------------------------------------------------------------
        # Which sequence name corresponds to this OI_TARGET?
        # ---------------------------------------------------------------------

        sequence_matches = []


        for star in expected_names:

            star_id = rpndrs.match_target_name(
                tgt_info,
                star,
                verbose=False
            )


            if (
                matched_id is not None
                and
                star_id == matched_id
            ):

                sequence_matches.append(
                    star
                )


            elif clean_name(star) == clean_name(oi_target):

                sequence_matches.append(
                    star
                )


        # ---------------------------------------------------------------------
        # Determine role expected from sequence
        # ---------------------------------------------------------------------

        expected_roles = []


        for seq in night_sequences[night]:

            science_target = seq[1]

            science_id = rpndrs.match_target_name(
                tgt_info,
                science_target,
                verbose=False
            )


            if (
                matched_id is not None
                and
                science_id == matched_id
            ):

                expected_roles.append("SCI")


        # If it was not the science target of any sequence on that night,
        # it should be a calibrator.
        if len(expected_roles) == 0:
            expected_role = "CAL"

        else:
            expected_role = "SCI"


        # ---------------------------------------------------------------------
        # Names OLD save_nightly_ldd could place into oiDiam
        # ---------------------------------------------------------------------

        old_names = []

        for name in [
            hd_id,
            ref1,
            ref2,
            ref3
        ]:

            if name != "":
                old_names.append(name)


        # ---------------------------------------------------------------------
        # Does the real OI_TARGET match one of those old oiDiam names?
        # ---------------------------------------------------------------------

        old_name_match = False


        for old_name in old_names:

            if clean_name(old_name) == clean_name(oi_target):

                old_name_match = True


        # ---------------------------------------------------------------------
        # Status
        # ---------------------------------------------------------------------

        issues = []


        if matched_id is None:

            issues.append(
                "ERROR:NO_TGT_INFO_MATCH"
            )


        if expected_role != actual_role and actual_role != "UNKNOWN":

            issues.append(
                "ERROR:SCI_CAL_MISMATCH"
            )


        if not old_name_match:

            issues.append(
                "WARNING:OLD_OIDIAM_NAME_MISMATCH"
            )


        if len(sequence_matches) == 0:

            issues.append(
                "WARNING:NO_SEQUENCE_NAME_MATCH"
            )


        if len(issues) == 0:

            status = "OK"

        else:

            status = ";".join(
                issues
            )

            problem_count += 1


        # ---------------------------------------------------------------------
        # Print
        # ---------------------------------------------------------------------

        print(
            "%-18s -> %-15s "
            "expected=%-3s tgt_info=%-3s "
            "files=%-3i %s"
            % (
                oi_target,
                str(matched_id),
                expected_role,
                actual_role,
                len(actual_targets[oi_target]),
                status
            )
        )


        if len(sequence_matches) > 0:

            print(
                "    sequence name(s): %s"
                % ", ".join(sequence_matches)
            )


        if len(old_names) > 0:

            print(
                "    old oiDiam names : %s"
                % ", ".join(old_names)
            )


        print(
            "    REAL PNDRS TARGET : %s"
            % oi_target
        )


        # ---------------------------------------------------------------------
        # Save CSV row
        # ---------------------------------------------------------------------

        rows.append({

            "night":
                night,

            "oi_target":
                oi_target,

            "matched_id":
                matched_id,

            "expected_role":
                expected_role,

            "tgt_info_role":
                actual_role,

            "sequence_names":
                " | ".join(sequence_matches),

            "Primary":
                primary,

            "HD_ID":
                hd_id,

            "Ref_ID_1":
                ref1,

            "Ref_ID_2":
                ref2,

            "Ref_ID_3":
                ref3,

            "old_oidiam_names":
                " | ".join(old_names),

            "old_name_matches_real_oi_target":
                old_name_match,

            "n_files":
                len(actual_targets[oi_target]),

            "status":
                status
        })


# =============================================================================
# WRITE CSV
# =============================================================================

columns = [

    "night",
    "oi_target",
    "matched_id",

    "expected_role",
    "tgt_info_role",

    "sequence_names",

    "Primary",
    "HD_ID",
    "Ref_ID_1",
    "Ref_ID_2",
    "Ref_ID_3",

    "old_oidiam_names",
    "old_name_matches_real_oi_target",

    "n_files",

    "status"
]


with open(output_csv, "wb") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=columns
    )

    writer.writeheader()

    for row in rows:
        writer.writerow(row)


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)


print(
    "\nUnique night/target combinations:",
    len(rows)
)


print(
    "Problem entries:",
    problem_count
)


print(
    "Targets not matched to tgt_info:",
    unmatched_count
)


print("\nCSV saved to:")
print(output_csv)


print("\nTo inspect only warnings/errors:")
print(
    'grep -E "WARNING|ERROR" %s'
    % output_csv
)


