from __future__ import print_function

import os
import glob
import numpy as np

from astropy.io import fits


# ============================================================
# ALL results folder
# ============================================================

results_folder = (
    "/home2/ihernand/Desktop/reach/results/"
    "26-08-28_KSIGEM_i2_ALL_ALL_BASELINES_ALL_CALS/"
)

night = "2022-03-01"


# ============================================================
# Find all faint bootstrap FITS
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(
            results_folder,
            "*2022-03-01*ksi_Gem*Calibrated*.fits"
        )
    )
)


print("")
print("=" * 100)
print("KSI GEM FAINT - CALIBRATED FITS CHECK")
print("=" * 100)

print("Folder:")
print(results_folder)

print("")
print("Files found:", len(files))


# ============================================================
# Inspect each file
# ============================================================

for filename in files:

    print("")
    print("=" * 100)
    print(os.path.basename(filename))
    print("=" * 100)

    with fits.open(filename) as hdul:

        # ----------------------------------------------------
        # Station mapping
        # ----------------------------------------------------

        array = hdul["OI_ARRAY"].data

        sta_map = {}

        for row in array:

            idx = int(
                row["STA_INDEX"]
            )

            name = row["STA_NAME"]

            if isinstance(name, bytes):
                name = name.decode("utf-8")

            sta_map[idx] = str(name).strip()


        # ----------------------------------------------------
        # VIS2
        # ----------------------------------------------------

        vis2 = hdul["OI_VIS2"].data

        values = np.asarray(
            vis2["VIS2DATA"],
            dtype=float
        )

        errors = np.asarray(
            vis2["VIS2ERR"],
            dtype=float
        )

        flags = np.asarray(
            vis2["FLAG"],
            dtype=bool
        )


        print(
            "VIS2 shape :",
            values.shape
        )

        print(
            "MIN VIS2   : %.8f"
            % np.nanmin(values)
        )

        print(
            "MAX VIS2   : %.8f"
            % np.nanmax(values)
        )


        # ----------------------------------------------------
        # Position of minimum
        # ----------------------------------------------------

        row_i, channel_i = np.unravel_index(
            np.nanargmin(values),
            values.shape
        )


        pair = vis2[
            "STA_INDEX"
        ][row_i]


        sta1 = sta_map[
            int(pair[0])
        ]

        sta2 = sta_map[
            int(pair[1])
        ]


        baseline = "%s-%s" % (
            sta1,
            sta2
        )


        mjd = float(
            vis2["MJD"][row_i]
        )


        print("")
        print("MINIMUM POINT")
        print("-" * 60)

        print(
            "row       :",
            row_i
        )

        print(
            "channel   :",
            channel_i
        )

        print(
            "baseline  :",
            baseline
        )

        print(
            "MJD       : %.8f"
            % mjd
        )

        print(
            "VIS2      : %.8f"
            % values[
                row_i,
                channel_i
            ]
        )

        print(
            "VIS2ERR   : %.8f"
            % errors[
                row_i,
                channel_i
            ]
        )

        print(
            "FLAG      :",
            flags[
                row_i,
                channel_i
            ]
        )


        # ----------------------------------------------------
        # Print all values below 0.7
        # ----------------------------------------------------

        print("")
        print("ALL POINTS WITH VIS2 < 0.70")
        print("-" * 60)

        found = False


        for row_i in range(
                values.shape[0]):

            pair = vis2[
                "STA_INDEX"
            ][row_i]

            sta1 = sta_map[
                int(pair[0])
            ]

            sta2 = sta_map[
                int(pair[1])
            ]

            baseline = "%s-%s" % (
                sta1,
                sta2
            )

            mjd = float(
                vis2["MJD"][row_i]
            )


            for channel_i in range(
                    values.shape[1]):

                value = values[
                    row_i,
                    channel_i
                ]

                if (
                    np.isfinite(value)
                    and
                    value < 0.70
                ):

                    found = True

                    print(
                        "row=%2i  ch=%i  "
                        "baseline=%-8s  "
                        "MJD=%.8f  "
                        "V2=%.8f  "
                        "FLAG=%s"
                        % (
                            row_i,
                            channel_i,
                            baseline,
                            mjd,
                            value,
                            str(
                                flags[
                                    row_i,
                                    channel_i
                                ]
                            )
                        )
                    )


        if not found:

            print(
                "NONE"
            )


print("")
print("=" * 100)
print("FINISHED")
print("=" * 100)