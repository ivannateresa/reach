from __future__ import print_function
 
import os
import glob
import numpy as np
 
from astropy.io import fits
 
 
# ============================================================
# Night to inspect
# ============================================================
 
night = "2022-03-01"
 
folder = (
    "/home/ihernand/Desktop/reach/"
    "complete_sequences/%s_v3.94_abcd/%s/"
    % (night, night)
)
 
 
print("\nFolder:")
print(folder)
 
 
# ============================================================
# Find reduced OIFITS
# ============================================================
 
files = sorted(
    glob.glob(folder + "PIONI*_oidata.fits")
)
 
 
print("\nFound %i files" % len(files))
 
 
# ============================================================
# Inspect telescope/station mapping
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
                    target = target.decode("utf-8")
 
                target_names.append(
                    str(target).strip()
                )
 
 
            # Keep only ksi Gem files
            is_ksi = False
 
            for target in target_names:
 
                clean = (
                    target
                    .replace("_", "")
                    .replace(" ", "")
                    .lower()
                )
 
                if clean == "ksigem":
                    is_ksi = True
 
 
            if not is_ksi:
                continue
 
 
            print("\n" + "=" * 79)
            print(os.path.basename(filename))
            print("TARGET:", target_names)
            print("=" * 79)
 
 
            # ------------------------------------------------
            # Read OI_ARRAY
            # ------------------------------------------------
 
            array = hdul["OI_ARRAY"].data
 
 
            tel_map = {}
            sta_map = {}
 
 
            print("\nOI_ARRAY:")
 
            for row in array:
 
                idx = int(row["STA_INDEX"])
 
 
                tel = row["TEL_NAME"]
 
                if isinstance(tel, bytes):
                    tel = tel.decode("utf-8")
 
 
                sta = row["STA_NAME"]
 
                if isinstance(sta, bytes):
                    sta = sta.decode("utf-8")
 
 
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
            # Read baselines from OI_VIS2
            # ------------------------------------------------
 
            if "OI_VIS2" not in hdul:
                print("\nNo OI_VIS2")
                continue
 
 
            vis2 = hdul["OI_VIS2"].data
 
 
            pairs = np.unique(
                vis2["STA_INDEX"],
                axis=0
            )
 
 
            print("\nBASELINES:")
 
 
            for pair in pairs:
 
                i1 = int(pair[0])
                i2 = int(pair[1])
 
 
                tel1 = tel_map.get(i1, "?")
                tel2 = tel_map.get(i2, "?")
 
 
                sta1 = sta_map.get(i1, "?")
                sta2 = sta_map.get(i2, "?")
 
 
                print(
                    "%s-%s    -->    %s-%s"
                    % (
                        tel1,
                        tel2,
                        sta1,
                        sta2
                    )
                )
 
 
    except Exception as e:
 
        print("\nERROR:")
        print(filename)
        print(str(e))

