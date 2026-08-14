from __future__ import print_function

import pandas as pd
from astropy.io import fits

import reach.utils as rutils
import reach.pndrs as rpndrs


# ============================================================
# Load tgt_info
# ============================================================

tgt_info = rutils.initialise_tgt_info(
    True,
    70,
    False
)


print("\n" + "=" * 70)
print("HR2426 IN tgt_info")
print("=" * 70)


name = "HR_2426"

matched_id = rpndrs.match_target_name(
    tgt_info,
    name,
    verbose=True
)


print("\nMatched ID:")
print(matched_id)


if matched_id is not None:

    print("\nScience ORIGINAL:")
    print(tgt_info.loc[matched_id]["Science"])

    print("\nRelevant aliases:")

    cols = [
        "Primary",
        "HD_ID",
        "Bayer_ID",
        "Ref_ID_1",
        "Ref_ID_2",
        "Ref_ID_3",
        "Science"
    ]

    for col in cols:

        if col in tgt_info.columns:

            print(
                "%-12s : %s"
                % (
                    col,
                    str(tgt_info.loc[matched_id][col])
                )
            )


# ============================================================
# Search ALL rows containing HR2426
# ============================================================

print("\n" + "=" * 70)
print("ALL ROWS CONTAINING HR2426")
print("=" * 70)


search_cols = [
    "Primary",
    "HD_ID",
    "Bayer_ID",
    "Ref_ID_1",
    "Ref_ID_2",
    "Ref_ID_3"
]


for idx, row in tgt_info.iterrows():

    for col in search_cols:

        if col not in tgt_info.columns:
            continue

        value = row[col]

        if pd.isnull(value):
            continue

        clean = (
            str(value)
            .replace("_", "")
            .replace(" ", "")
            .lower()
        )

        if clean == "hr2426":

            print(
                "index=%-15s column=%-10s value=%-15s Science=%s"
                % (
                    idx,
                    col,
                    value,
                    str(row["Science"])
                )
            )


# ============================================================
# Read oiDiam
# ============================================================

filename = (
    "/home/ihernand/Desktop/reach/"
    "complete_sequences/"
    "2022-03-01_v3.94_abcd/"
    "2022-03-01/"
    "2022-03-01_oiDiam.fits"
)


print("\n" + "=" * 70)
print("OI_DIAM")
print("=" * 70)


with fits.open(filename) as hdul:

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