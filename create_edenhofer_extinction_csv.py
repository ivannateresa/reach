import os
import gc
import numpy as np
import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord

from dustmaps.config import config
from dustmaps.edenhofer2023 import Edenhofer2023Query


# ============================================================
# PATHS
# ============================================================

REACH_DIR = "/home2/ihernand/Desktop/reach"

TARGET_FILE = os.path.join(
    REACH_DIR,
    "data",
    "target_info.tsv"
)

OUTPUT_FILE = os.path.join(
    REACH_DIR,
    "data",
    "targets_extinction_edenhofer.csv"
)

DUSTMAP_DIR = os.path.join(
    REACH_DIR,
    "dustmaps_data"
)

config["data_dir"] = DUSTMAP_DIR


# ============================================================
# FUNCTIONS
# ============================================================

def clean_primary(x):

    if pd.isnull(x):
        return ""

    return (
        str(x)
        .replace(" ", "")
        .replace(".", "")
        .replace("_", "")
    )


def find_column(df, candidates):

    for col in candidates:
        if col in df.columns:
            return col

    return None


def compute_distances(df, max_rel_plx_error=0.20):

    df["Dist"] = np.nan
    df["e_Dist"] = np.nan
    df["Dist_source"] = ""

    # --------------------------------------------------------
    # Gaia corrected EDR3 parallax
    # --------------------------------------------------------

    plx = pd.to_numeric(
        df["Plx"],
        errors="coerce"
    )

    e_plx = pd.to_numeric(
        df["e_Plx"],
        errors="coerce"
    )

    rel_error = e_plx / plx

    good = (
        plx.notna()
        & e_plx.notna()
        & np.isfinite(plx)
        & np.isfinite(e_plx)
        & (plx > 0)
        & (e_plx >= 0)
        & (rel_error <= max_rel_plx_error)
    )

    df.loc[good, "Dist"] = 1000.0 / plx[good]

    df.loc[good, "e_Dist"] = (
        1000.0
        * e_plx[good]
        / plx[good]**2
    )

    df.loc[
        good,
        "Dist_source"
    ] = "Gaia EDR3 corrected parallax"

    # --------------------------------------------------------
    # Bailer-Jones rpgeo
    # --------------------------------------------------------

    rpgeo = pd.to_numeric(
        df["rpgeo"],
        errors="coerce"
    )

    b_rpgeo = pd.to_numeric(
        df["b_rpgeo"],
        errors="coerce"
    )

    B_rpgeo = pd.to_numeric(
        df["B_rpgeo"],
        errors="coerce"
    )

    use_bj = (
        df["Dist"].isna()
        & rpgeo.notna()
        & np.isfinite(rpgeo)
        & (rpgeo > 0)
    )

    df.loc[use_bj, "Dist"] = rpgeo[use_bj]

    valid_limits = (
        use_bj
        & b_rpgeo.notna()
        & B_rpgeo.notna()
        & np.isfinite(b_rpgeo)
        & np.isfinite(B_rpgeo)
    )

    df.loc[valid_limits, "e_Dist"] = (
        B_rpgeo[valid_limits]
        - b_rpgeo[valid_limits]
    ) / 2.0

    df.loc[
        use_bj,
        "Dist_source"
    ] = "Bailer-Jones rpgeo"

    # --------------------------------------------------------
    # Hipparcos
    # --------------------------------------------------------

    hip_plx = pd.to_numeric(
        df["Plx_HIP"],
        errors="coerce"
    )

    hip_e_plx = pd.to_numeric(
        df["e_plx_HIP"],
        errors="coerce"
    )

    use_hip = (
        df["Dist"].isna()
        & hip_plx.notna()
        & hip_e_plx.notna()
        & np.isfinite(hip_plx)
        & np.isfinite(hip_e_plx)
        & (hip_plx > 0)
        & (hip_e_plx >= 0)
    )

    df.loc[use_hip, "Dist"] = (
        1000.0 / hip_plx[use_hip]
    )

    df.loc[use_hip, "e_Dist"] = (
        1000.0
        * hip_e_plx[use_hip]
        / hip_plx[use_hip]**2
    )

    df.loc[
        use_hip,
        "Dist_source"
    ] = "Hipparcos parallax"

    return df


# ============================================================
# LOAD SAME TARGET TABLE USED BY REACH
# ============================================================

print("")
print("=" * 80)
print("LOADING TARGETS")
print("=" * 80)

tgt = pd.read_csv(
    TARGET_FILE,
    sep=",",
    header=1,
    index_col=6,
    skiprows=0
)

print(tgt.columns)
# Same duplicate handling as REACH
tgt = tgt[
    ~tgt.index.duplicated(keep="first")
].copy()

tgt.index.name = "HD_ID"
tgt["HD_ID"] = tgt.index

tgt["Primary"] = tgt["Primary"].apply(
    clean_primary
)

print("Number of targets:", len(tgt))


# ============================================================
# DISTANCES
# ============================================================

tgt = compute_distances(
    tgt,
    max_rel_plx_error=0.20
)

bad_dist = (
    ~np.isfinite(tgt["Dist"])
    | (tgt["Dist"] <= 0)
)

if bad_dist.any():

    print("")
    print("INVALID DISTANCES:")
    print(
        tgt.loc[
            bad_dist,
            ["Primary", "Dist", "Dist_source"]
        ].to_string()
    )

    raise RuntimeError(
        "Some targets do not have valid distances."
    )


# ============================================================
# FIND RA AND DEC
# ============================================================

ra_col = find_column(
    tgt,
    [
        "RA",
        "ra",
        "RA_deg",
        "ra_deg",
        "RA_ICRS",
        "RAJ2000",
        "RAJ2000_deg"
    ]
)

dec_col = find_column(
    tgt,
    [
        "DEC",
        "Dec",
        "dec",
        "DE",
        "DEC_deg",
        "dec_deg",
        "DE_ICRS",
        "DEJ2000",
        "DEJ2000_deg"
    ]
)

if ra_col is None or dec_col is None:

    print("")
    print("Could not automatically identify RA / DEC.")
    print("")
    print("AVAILABLE COLUMNS:")
    print(tgt.columns.tolist())

    raise RuntimeError(
        "RA/DEC columns not identified."
    )

print("")
print("RA column :", ra_col)
print("DEC column:", dec_col)


# ============================================================
# COORDINATES
# ============================================================

ra_num = pd.to_numeric(
    tgt[ra_col],
    errors="coerce"
)

dec_num = pd.to_numeric(
    tgt[dec_col],
    errors="coerce"
)

if (
    np.isfinite(ra_num).all()
    and np.isfinite(dec_num).all()
):

    print("Coordinates interpreted as decimal degrees.")

    coords = SkyCoord(
        ra=ra_num.values * u.deg,
        dec=dec_num.values * u.deg,
        distance=tgt["Dist"].values * u.pc,
        frame="icrs"
    )

else:

    print("Coordinates interpreted as RA hourangle / DEC degrees.")

    coords = SkyCoord(
        ra=tgt[ra_col].astype(str).values,
        dec=tgt[dec_col].astype(str).values,
        unit=(u.hourangle, u.deg),
        distance=tgt["Dist"].values * u.pc,
        frame="icrs"
    )


# ============================================================
# OUTPUT TABLE
# ============================================================

out = pd.DataFrame({
    "HD_ID": tgt.index.values,
    "Primary": tgt["Primary"].values,
    "RA_deg": coords.ra.deg,
    "DEC_deg": coords.dec.deg,
    "Dist": tgt["Dist"].values,
    "Dist_source": tgt["Dist_source"].values
})

out["E_ZGR_Edenhofer"] = np.nan
out["Av_Edenhofer"] = np.nan
out["Edenhofer_map"] = "none"
out["Edenhofer_status"] = "not_queried"


# ============================================================
# LOCAL ZERO ASSUMPTION
# ============================================================

local = out["Dist"] < 69.0
nonlocal_mask = ~local

out.loc[
    local,
    "Av_Edenhofer"
] = 0.0

out.loc[
    local,
    "Edenhofer_status"
] = "assumed_zero_d_lt_69pc"


print("")
print("=" * 80)
print("SAMPLE")
print("=" * 80)

print("Total targets       :", len(out))
print("d < 69 pc           :", int(local.sum()))
print("Need Edenhofer      :", int(nonlocal_mask.sum()))


# ============================================================
# EDENHOFER MAIN
# ============================================================

if nonlocal_mask.any():

    print("")
    print("=" * 80)
    print("LOADING EDENHOFER MAIN")
    print("=" * 80)

    q_main = Edenhofer2023Query(
        integrated=True,
        flavor="main",
        load_samples=False
    )

    ii = np.where(
        nonlocal_mask.values
    )[0]

    values = np.asarray(
        q_main(
            coords[ii],
            mode="mean"
        ),
        dtype=float
    )

    good = (
        np.isfinite(values)
        & (values >= 0)
    )

    jj = ii[good]

    out.loc[
        jj,
        "E_ZGR_Edenhofer"
    ] = values[good]

    out.loc[
        jj,
        "Edenhofer_map"
    ] = "main"

    out.loc[
        jj,
        "Edenhofer_status"
    ] = "queried_main"

    print(
        "Recovered with main:",
        int(good.sum())
    )

    del q_main
    gc.collect()


# ============================================================
# 2 KPC FALLBACK
# ============================================================

missing = (
    nonlocal_mask
    & ~np.isfinite(
        out["E_ZGR_Edenhofer"]
    )
)

if missing.any():

    print("")
    print("=" * 80)
    print("LOADING EDENHOFER 2 KPC FALLBACK")
    print("=" * 80)

    q_2k = Edenhofer2023Query(
        integrated=True,
        flavor="less_data_but_2kpc",
        load_samples=False
    )

    ii = np.where(
        missing.values
    )[0]

    values = np.asarray(
        q_2k(
            coords[ii],
            mode="mean"
        ),
        dtype=float
    )

    good = (
        np.isfinite(values)
        & (values >= 0)
    )

    jj = ii[good]

    out.loc[
        jj,
        "E_ZGR_Edenhofer"
    ] = values[good]

    out.loc[
        jj,
        "Edenhofer_map"
    ] = "less_data_but_2kpc"

    out.loc[
        jj,
        "Edenhofer_status"
    ] = "queried_2kpc"

    print(
        "Recovered with 2 kpc:",
        int(good.sum())
    )

    del q_2k
    gc.collect()


# ============================================================
# A_V REFERENCE
# ============================================================

has_e = np.isfinite(
    out["E_ZGR_Edenhofer"]
)

out.loc[
    has_e,
    "Av_Edenhofer"
] = (
    2.8
    * out.loc[
        has_e,
        "E_ZGR_Edenhofer"
    ]
)


# ============================================================
# FINAL CHECK
# ============================================================

missing = (
    nonlocal_mask
    & ~np.isfinite(
        out["E_ZGR_Edenhofer"]
    )
)

if missing.any():

    out.loc[
        missing,
        "Edenhofer_status"
    ] = "outside_map_or_missing"


# ============================================================
# SAVE EVEN IF SOMETHING IS MISSING
# ============================================================

out.to_csv(
    OUTPUT_FILE,
    index=False
)


print("")
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_FILE)

print("")
print("Rows:", len(out))

print("")
print("STATUS:")
print(
    out["Edenhofer_status"]
    .value_counts(dropna=False)
)

print("")
print("MAP:")
print(
    out["Edenhofer_map"]
    .value_counts(dropna=False)
)


if missing.any():

    print("")
    print("TARGETS STILL MISSING:")
    print(
        out.loc[
            missing,
            [
                "HD_ID",
                "Primary",
                "Dist",
                "RA_deg",
                "DEC_deg"
            ]
        ].to_string(index=False)
    )

    raise RuntimeError(
        "Some d >= 69 pc targets have no Edenhofer value."
    )

print("")
print("SUCCESS: all non-local targets have extinction.")
print("=" * 80)

