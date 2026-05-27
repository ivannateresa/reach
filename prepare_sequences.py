import os
import re
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Files
# ------------------------------------------------------------

adam_file = "/home2/ihernand/Desktop/reach/data/Interferometry_adam_data(Bright-Faint Sequence).csv"
summary_file = "/home2/ihernand/Desktop/reach/concatenations_summary.csv"

outdir = "data"

def clean_name(x):
    if pd.isna(x):
        return ""

    x = str(x).strip()
    x = x.replace(",", "")
    x = x.replace(" ", "")
    x = x.replace("_", "")
    x = x.lower()

    return x


def clean_bool(x):
    if pd.isna(x):
        return False

    return str(x).strip().upper() == "TRUE"


def clean_period(x):
    if pd.isna(x):
        return np.nan

    match = re.search(r"\d+", str(x))

    if match:
        return int(match.group())

    return np.nan


def split_concatenation(x):
    if pd.isna(x):
        return []

    return [clean_name(t) for t in str(x).split("->") if clean_name(t) != ""]


def sequence_has_label(x, label):
    if pd.isna(x):
        return False

    x = str(x).strip().lower()

    if label == "bright":
        return "bright" in x or "both" in x

    if label == "faint":
        return "faint" in x or "both" in x

    return False


def get_expected_targets(p, label):
    """
    Devuelve los targets esperados para una secuencia bright o faint,
    manteniendo el orden del archivo de Adam.
    """

    p_label = p[p["Sequence"].apply(lambda x: sequence_has_label(x, label))].copy()

    targets = []

    for _, row in p_label.iterrows():
        target = clean_name(row["Primary"])
        if target != "":
            targets.append(target)

    return targets


def score_match(real_concat, expected_targets):
    real_set = set(real_concat)
    expected_set = set(expected_targets)

    common = real_set.intersection(expected_set)

    return len(common)

def science_calibrators_row(science_target, real_concat):
    """
    Convert:
        cal1 -> sci -> cal2 -> sci -> cal3

    into:
        sci, cal1, cal2, cal3
    """

    science_targets = [clean_name(x) for x in str(science_target).split(";")]
    science_targets = [x for x in science_targets if x != ""]

    cals = []

    for target in real_concat:
        target = clean_name(target)

        if target == "":
            continue

        # saltar la ciencia
        if target in science_targets:
            continue

        # evitar calibradores repetidos, manteniendo el orden
        if target not in cals:
            cals.append(target)

    return [science_target] + cals
def science_first_rows(science_target, real_concat):
    """
    Convierte:
        cal1 -> sci -> cal2 -> sci -> cal3

    en:
        sci, TRUE
        cal1, FALSE
        cal2, FALSE
        cal3, FALSE
    """

    science_target = clean_name(science_target)

    rows = []

    # primero la ciencia
    rows.append((science_target, "TRUE"))
    cals = []

    for target in real_concat:
        target = clean_name(target)

        if target == "":
            continue

        if target == science_target:
            continue

        if target not in cals:
            cals.append(target)

    for cal in cals:
        rows.append((cal, "FALSE"))

    return rows

df = pd.read_csv(adam_file, sep=",")
summary = pd.read_csv(summary_file, sep=",")

# limpiar Period
df = df[df["Period"].notna()].copy()
df["Period"] = df["Period"].apply(clean_period)
df = df[df["Period"].notna()].copy()
df["Period"] = df["Period"].astype(int)

# limpiar summary period
summary = summary[summary["period"].notna()].copy()
summary["period"] = summary["period"].apply(clean_period)
summary = summary[summary["period"].notna()].copy()
summary["period"] = summary["period"].astype(int)

# limpiar nombres
df["Primary_clean"] = df["Primary"].apply(clean_name)
df["Science_bool"] = df["Science"].apply(clean_bool)

summary["science_target_clean"] = summary["science_target"].apply(clean_name)
summary["concat_list"] = summary["concatenation"].apply(split_concatenation)

periods = sorted(df["Period"].unique())


for period in periods:

    p = df[df["Period"] == period].copy()
    summary_period = summary[summary["period"] == period].copy()

    print("\n====================================")
    print("Period:", period)
    print("Rows in Adam file:", len(p))
    print("Rows in concatenation summary:", len(summary_period))
    print("====================================")

    bright_rows = []
    faint_rows = []

    for _, row in summary_period.iterrows():

        science_target = row["science_target_clean"]
        real_concat = row["concat_list"]

        if science_target == "" or science_target == "unknown":
            continue

        # Tomar solo la parte del archivo de Adam correspondiente a esa ciencia
        # Buscamos desde la fila donde Science=True y Primary=science_target
        p_sci_index = p[
            (p["Science_bool"] == True) &
            (p["Primary_clean"] == science_target)
        ].index

        if len(p_sci_index) == 0:
            print("WARNING: science target not found in Adam file:", science_target)
            continue

        # Para comparar bright/faint usamos todo el periodo, pero filtrado por label
        bright_expected = get_expected_targets(p, "bright")
        faint_expected = get_expected_targets(p, "faint")

        bright_score = score_match(real_concat, bright_expected)
        faint_score = score_match(real_concat, faint_expected)

        if bright_score > faint_score:
            matched_sequence = "bright"
        elif faint_score > bright_score:
            matched_sequence = "faint"
        else:
            matched_sequence = "unknown"

        print(
            science_target,
            "| matched:",
            matched_sequence,
            "| bright_score:",
            bright_score,
            "| faint_score:",
            faint_score,
            "| concat:",
            " -> ".join(real_concat)
        )

        row_to_save = science_first_rows(science_target, real_concat)
        if matched_sequence == "bright":
            bright_rows.extend(row_to_save)
        elif matched_sequence == "faint":
            faint_rows.extend(row_to_save)

    # escribir archivos
    bright_file = os.path.join(outdir, "p{}_bright.txt".format(period))
    faint_file = os.path.join(outdir, "p{}_faint.txt".format(period))

    with open(bright_file, "w") as f:
        for target, science_flag in bright_rows:
            f.write("{},{}\n".format(target, science_flag))
            
    with open(faint_file, "w") as f:
        for target, science_flag in faint_rows:
            f.write("{},{}\n".format(target, science_flag))

    print("Saved:", bright_file)
    print("Saved:", faint_file)