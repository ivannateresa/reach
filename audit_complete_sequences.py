from __future__ import print_function

import os
import csv
import collections

import pandas as pd

import reach.utils as rutils
import reach.pndrs as rpndrs


# =============================================================================
# SETTINGS
# =============================================================================

output_csv = "complete_sequences_audit.csv"


# =============================================================================
# LOAD TARGET INFORMATION
# =============================================================================

print("\n" + "=" * 90)
print("LOAD TARGET INFORMATION")
print("=" * 90)


tgt_info = rutils.initialise_tgt_info(
    True,       # assign_default_uncertainties
    70,         # lb_pc
    False       # use_plx_systematic
)


# =============================================================================
# LOAD SEQUENCES
# =============================================================================

complete_sequences, sequences = rutils.load_sequence_logs()


print("\nNumber of complete sequences:")
print(len(complete_sequences))


# =============================================================================
# HELPERS
# =============================================================================

def clean_name(name):

    if name is None:
        return ""

    return (
        str(name)
        .replace("_", "")
        .replace(" ", "")
        .replace(".", "")
        .replace("-", "")
        .lower()
    )


def pndrs_observed_name(name):
    """
    Convert sequence name to the form normally appearing in PNDRS.

    HR_2426 -> HR2426
    ksi_Gem -> ksi_Gem
    """

    name = str(name).strip()

    if name.startswith("HR_"):
        name = name.replace(
            "HR_",
            "HR",
            1
        )

    return name


def safe_value(row, column):

    if column not in row.index:
        return ""

    value = row[column]

    if pd.isnull(value):
        return ""

    return str(value)


# =============================================================================
# OUTPUT STRUCTURE
# =============================================================================

rows = []

n_errors = 0
n_warnings = 0


# Used later to detect aliases of same physical target
night_target_names = collections.defaultdict(set)


# =============================================================================
# LOOP OVER ALL COMPLETE SEQUENCES
# =============================================================================

print("\n" + "=" * 90)
print("AUDIT COMPLETE SEQUENCES")
print("=" * 90)


for seq in sorted(complete_sequences.keys()):

    period = seq[0]
    science_target = seq[1]
    mode = seq[2]

    night = complete_sequences[seq][0]

    try:
        grade = complete_sequences[seq][1]
    except Exception:
        grade = ""


    print("\n" + "=" * 90)

    print(
        "P%s | %-20s | %-8s | %s"
        % (
            period,
            science_target,
            mode,
            night
        )
    )

    print("=" * 90)


    if seq not in sequences:

        print("ERROR: sequence is missing from sequences dictionary")

        n_errors += 1

        continue


    seq_stars = sequences[seq]


    print("Sequence:")
    print(seq_stars)


    science_clean = clean_name(
        science_target
    )


    # =================================================================
    # Count raw observations in complete_sequences
    # =================================================================

    obs_list = complete_sequences[seq][2]

    raw_counts = collections.Counter()

    for obs in obs_list:

        try:
            raw_target = obs[2]
            raw_counts[raw_target] += 1

        except Exception:
            pass


    # =================================================================
    # Inspect every unique star
    # =================================================================

    stars_seen = []

    for star in seq_stars:

        if star in stars_seen:
            continue

        stars_seen.append(star)


        # -------------------------------------------------------------
        # Expected role from the sequence itself
        # -------------------------------------------------------------

        if clean_name(star) == science_clean:
            expected_role = "SCI"
        else:
            expected_role = "CAL"


        # -------------------------------------------------------------
        # Match to tgt_info
        # -------------------------------------------------------------

        matched_id = rpndrs.match_target_name(
            tgt_info,
            star,
            verbose=False
        )


        issues = []


        if matched_id is None:

            actual_role = "NOT_FOUND"
            science_value = ""
            primary = ""
            hd_id = ""
            ref1 = ""
            ref2 = ""
            ref3 = ""
            observed_pndrs = pndrs_observed_name(star)

            issues.append(
                "ERROR:NO_MATCH_TGT_INFO"
            )

            n_errors += 1

        else:

            info = tgt_info.loc[matched_id]


            science_value = bool(
                info["Science"]
            )


            if science_value:
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


            observed_pndrs = pndrs_observed_name(
                star
            )


            # ---------------------------------------------------------
            # Check SCI/CAL
            # ---------------------------------------------------------

            if expected_role != actual_role:

                issues.append(
                    "ERROR:SCI_CAL_MISMATCH"
                )

                n_errors += 1


            # ---------------------------------------------------------
            # Check old oiDiam naming behaviour
            #
            # Old save_nightly_ldd uses:
            # HD_ID, Ref_ID_1, Ref_ID_2, Ref_ID_3
            #
            # It does NOT use Primary.
            # ---------------------------------------------------------

            old_oidiam_names = [
                hd_id,
                ref1,
                ref2,
                ref3
            ]


            old_clean = []

            for name in old_oidiam_names:

                if name != "":
                    old_clean.append(
                        clean_name(name)
                    )


            if clean_name(observed_pndrs) not in old_clean:

                issues.append(
                    "WARNING:OIDIAM_NAME_MISMATCH"
                )

                n_warnings += 1


            # ---------------------------------------------------------
            # Check if observed name exists as Primary
            # but not in old oiDiam identifiers
            #
            # This is exactly the kind of situation HR2426 had.
            # ---------------------------------------------------------

            if (
                clean_name(primary)
                ==
                clean_name(observed_pndrs)
                and
                clean_name(observed_pndrs)
                not in old_clean
            ):

                issues.append(
                    "WARNING:PRIMARY_NOT_USED_BY_OLD_OIDIAM"
                )


            # ---------------------------------------------------------
            # Save name per physical target/night
            # ---------------------------------------------------------

            night_target_names[
                (night, matched_id)
            ].add(
                observed_pndrs
            )


        # -------------------------------------------------------------
        # Raw observation count
        # -------------------------------------------------------------

        n_raw = 0

        for raw_name, count in raw_counts.items():

            if clean_name(raw_name) == clean_name(star):
                n_raw += count


        if len(issues) == 0:
            status = "OK"
        else:
            status = ";".join(issues)


        print(
            "%-18s -> %-15s "
            "expected=%-3s actual=%-3s "
            "PNDRS=%-15s raw=%-3i %s"
            % (
                star,
                str(matched_id),
                expected_role,
                actual_role,
                observed_pndrs,
                n_raw,
                status
            )
        )


        rows.append({
            "period": period,
            "night": night,
            "grade": grade,
            "mode": mode,
            "sequence_science": science_target,
            "observed_name": star,
            "pndrs_name": observed_pndrs,
            "matched_id": matched_id,
            "expected_role": expected_role,
            "actual_role": actual_role,
            "science_value": science_value,
            "Primary": primary,
            "HD_ID": hd_id,
            "Ref_ID_1": ref1,
            "Ref_ID_2": ref2,
            "Ref_ID_3": ref3,
            "n_raw_observations": n_raw,
            "status": status
        })


# =============================================================================
# CHECK SAME PHYSICAL TARGET WITH MULTIPLE NAMES ON SAME NIGHT
# =============================================================================

print("\n" + "=" * 90)
print("CHECK MULTIPLE NAMES FOR SAME TARGET / SAME NIGHT")
print("=" * 90)


alias_problems = 0


for key in sorted(night_target_names.keys()):

    night = key[0]
    matched_id = key[1]

    names = sorted(
        night_target_names[key]
    )


    if len(names) > 1:

        alias_problems += 1

        print(
            "\nWARNING: %s on %s appears as:"
            % (
                matched_id,
                night
            )
        )

        for name in names:
            print("   %s" % name)


# =============================================================================
# WRITE CSV
# =============================================================================

columns = [
    "period",
    "night",
    "grade",
    "mode",
    "sequence_science",
    "observed_name",
    "pndrs_name",
    "matched_id",
    "expected_role",
    "actual_role",
    "science_value",
    "Primary",
    "HD_ID",
    "Ref_ID_1",
    "Ref_ID_2",
    "Ref_ID_3",
    "n_raw_observations",
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

print("\n" + "=" * 90)
print("AUDIT SUMMARY")
print("=" * 90)


print(
    "\nComplete sequences checked : %i"
    % len(complete_sequences)
)

print(
    "Target entries checked     : %i"
    % len(rows)
)

print(
    "Errors found               : %i"
    % n_errors
)

print(
    "Warnings found             : %i"
    % n_warnings
)

print(
    "Multiple-name cases        : %i"
    % alias_problems
)


print(
    "\nCSV saved as:"
)

print(
    os.path.abspath(output_csv)
)