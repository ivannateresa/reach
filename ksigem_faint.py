from __future__ import print_function

import os
import glob
import numpy as np

from astropy.io import fits


night = "2022-03-01"

folder = (
    "/home2/ihernand/Desktop/reach/"
    "complete_sequences/%s_v3.94_abcd/%s/"
    % (night, night)
)


# ============================================================
# 1. ORIGINAL OIDATA
# ============================================================

raw_files = sorted(
    glob.glob(
        os.path.join(
            folder,
            "PIONI*_oidata.fits"
        )
    )
)


ksi_files = []


for filename in raw_files:

    with fits.open(filename) as hdul:

        if "OI_TARGET" not in hdul:
            continue

        targets = hdul["OI_TARGET"].data

        names = []

        for row in targets:

            name = row["TARGET"]

            if isinstance(name, bytes):
                name = name.decode("utf-8")

            names.append(
                str(name)
                .replace("_", "")
                .replace(" ", "")
                .lower()
            )


        if "ksigem" in names:

            ksi_files.append(
                filename
            )


print("")
print("=" * 100)
print("BEFORE PNDRS CALIBRATION")
print("=" * 100)

print(
    "Total PIONI oidata files :",
    len(raw_files)
)

print(
    "ksi_Gem oidata files     :",
    len(ksi_files)
)


for filename in ksi_files:

    with fits.open(filename) as hdul:

        vis2 = hdul["OI_VIS2"].data

        mjds = np.unique(
            np.asarray(
                vis2["MJD"],
                dtype=float
            )
        )

        print("")
        print(
            os.path.basename(
                filename
            )
        )

        print(
            "  OI_VIS2 rows:",
            len(vis2)
        )

        print(
            "  MJD:",
            mjds
        )


# ============================================================
# 2. CALIBRATED OIDATA
# ============================================================

cal_file = os.path.join(
    folder,
    "%s_SCI_ksi_Gem_oidataCalibrated.fits"
    % night
)


print("")
print("=" * 100)
print("AFTER PNDRS CALIBRATION")
print("=" * 100)

print(
    cal_file
)


with fits.open(cal_file) as hdul:

    vis2 = hdul["OI_VIS2"].data

    array = hdul["OI_ARRAY"].data


    sta_map = {}

    for row in array:

        idx = int(
            row["STA_INDEX"]
        )

        sta = row["STA_NAME"]

        if isinstance(sta, bytes):
            sta = sta.decode("utf-8")

        sta_map[idx] = str(sta).strip()


    print("")
    print(
        "OI_VIS2 rows:",
        len(vis2)
    )

    print(
        "VIS2 shape:",
        np.asarray(
            vis2["VIS2DATA"]
        ).shape
    )


    unique_mjd = np.unique(
        np.asarray(
            vis2["MJD"],
            dtype=float
        )
    )


    print("")
    print(
        "Unique MJD:",
        unique_mjd
    )

    print(
        "Number of unique MJD:",
        len(unique_mjd)
    )


    print("")
    print(
        "%-5s %-12s %-16s %-40s"
        % (
            "ROW",
            "BASELINE",
            "MJD",
            "FLAG"
        )
    )

    print("-" * 100)


    for i in range(
            len(vis2)):

        pair = vis2[
            "STA_INDEX"
        ][i]


        baseline = "%s-%s" % (

            sta_map[
                int(pair[0])
            ],

            sta_map[
                int(pair[1])
            ]

        )


        mjd = float(
            vis2["MJD"][i]
        )


        flag = np.asarray(
            vis2["FLAG"][i]
        ).astype(bool)


        print(
            "%-5i %-12s %-16.8f %s"
            % (
                i,
                baseline,
                mjd,
                str(flag)
            )
        )