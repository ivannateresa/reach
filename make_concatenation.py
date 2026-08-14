from __future__ import division, print_function

import os
import sys
import glob
import csv
from collections import OrderedDict
from datetime import datetime
import io

# Maximum temporal separation allowed inside one concatenation.
# A PIONIER concatenation normally lasts about 30-45 minutes.
MAX_GAP_MINUTES = 90


def clean_name(name):
    """Normalize a target name only for comparisons."""
    if name is None:
        return "unknown"
    return str(name).strip().replace(" ", "").replace("_", "").lower()


def get_period(run):
    """Extract period from strings such as 0104.D-0580(A) or 106.21KX.001."""
    try:
        return int(str(run).split(".")[0])
    except (TypeError, ValueError):
        return -1


def parse_log(obs_log):
    """Read one PIONIER .NL.txt log and extract relevant information."""

    yyyymmddhhMMss = os.path.basename(obs_log)[6:25]
    ob_time = datetime.strptime(yyyymmddhhMMss, "%Y-%m-%dT%H:%M:%S")

    folder = os.path.basename(os.path.dirname(obs_log))

    grade = "?"
    target = "unknown"
    raw_target = "unknown"
    OB = "unknown"
    container = "unknown"
    run = "unknown"
    ob_type = "unknown"

    with io.open(str(obs_log), mode="r", encoding="utf-8", errors="replace") as f:
        content = [row.strip() for row in f]

    for row in content:
        if row.startswith("Grade:"):
            grade = row[-1]

        elif row.startswith("Target:"):
            raw_target = row.split("Target:", 1)[-1].strip()
            target = raw_target

        elif row.startswith("OB:"):
            OB = row.split(" ")[-1]

        elif row.startswith("Container:"):
            container = row.split(" ")[-1]

        elif row.startswith("Run:"):
            run = row.split(" ")[-1]

        elif row.startswith("PIONIER_OBS_FRINGE"):
            parts = row.split("\t")
            if len(parts) > 1 and parts[1] in obs_log:
                ob_type = "FRINGE"

        elif row.startswith("PIONIER_GEN_DARK"):
            parts = row.split("\t")
            if len(parts) > 1 and parts[1] in obs_log:
                ob_type = "DARK"

        elif row.startswith("PIONIER_GEN_KAPPA"):
            parts = row.split("\t")
            if len(parts) > 1 and parts[1] in obs_log:
                ob_type = "KAPPA"

    return {
        "folder": folder,
        "time": ob_time,
        "date": ob_time.strftime("%Y-%m-%d"),
        "target": target,
        "raw_target": raw_target,
        "grade": grade,
        "OB": OB,
        "container": container,
        "run": run,
        "type": ob_type,
        "logfile": obs_log,
        "fitsfile": obs_log.replace("NL.txt", "fits.Z"),
    }


def compact_targets(obs_list):
    """
    Remove consecutive repeated targets while preserving the original
    target spelling in the output.
    """
    compact = []
    previous_clean = None

    for ob in obs_list:
        target = str(ob["target"]).strip()
        target_clean = clean_name(target)

        if target_clean != previous_clean:
            compact.append(target)
            previous_clean = target_clean

    return compact


def infer_science_target(compact_sequence):
    """
    Infer the science target as the target repeated in the compact
    CAL-SCI-CAL-SCI-CAL sequence.
    """
    counts = OrderedDict()
    display_name = {}

    for target in compact_sequence:
        key = clean_name(target)
        counts[key] = counts.get(key, 0) + 1
        display_name.setdefault(key, target)

    repeated = [key for key, count in counts.items() if count > 1]

    if len(repeated) == 0:
        return "unknown"

    if len(repeated) == 1:
        return display_name[repeated[0]]

    return ";".join(display_name[key] for key in repeated)


def split_by_time_gap(obs_list, max_gap_minutes=MAX_GAP_MINUTES):
    """
    Split observations sharing run+container only when there is a long
    temporal gap. This keeps concatenations together when they cross midnight.
    """
    obs_sorted = sorted(obs_list, key=lambda x: x["time"])

    groups = []
    current = []

    for ob in obs_sorted:
        if not current:
            current = [ob]
            continue

        gap_minutes = (
            ob["time"] - current[-1]["time"]
        ).total_seconds() / 60.0

        if gap_minutes <= max_gap_minutes:
            current.append(ob)
        else:
            groups.append(current)
            current = [ob]

    if current:
        groups.append(current)

    return groups


def unique_in_order(values):
    return list(OrderedDict.fromkeys(values))


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python make_concatenation_csv_fixed.py /path/to/all_sequences")
        sys.exit(1)

    base_path = sys.argv[1]
    output_csv = "concatenations_summary.csv"

    all_logs = glob.glob(os.path.join(base_path, "*", "PIONI*.NL.txt"))
    all_logs.sort()

    print("Base path:", base_path)
    print("Number of logs found:", len(all_logs))

    if not all_logs:
        print("No logs found. Check the path.")
        sys.exit(1)

    # First group by observing run + ESO container.
    # Do NOT use folder/date here because one concatenation may cross midnight.
    container_pool = OrderedDict()

    for obs_log in all_logs:
        ob = parse_log(obs_log)
        key = (ob["run"], ob["container"])
        container_pool.setdefault(key, []).append(ob)

    # The same container can be reused on another night, so split only
    # when the temporal gap is larger than MAX_GAP_MINUTES.
    sequences = []

    for (run, container), all_obs in container_pool.items():
        groups = split_by_time_gap(all_obs)
        for group_index, obs_list in enumerate(groups, start=1):
            sequences.append({
                "run": run,
                "container": container,
                "group_index": group_index,
                "obs_list": obs_list,
            })

    sequences.sort(key=lambda item: item["obs_list"][0]["time"])

    print("Number of run+container groups:", len(container_pool))
    print("Number of concatenations after time splitting:", len(sequences))

    rows = []

    for sequence in sequences:
        run = sequence["run"]
        container = sequence["container"]
        obs_list = sorted(sequence["obs_list"], key=lambda x: x["time"])

        compact_sequence = compact_targets(obs_list)
        science_target = infer_science_target(compact_sequence)

        first_time = obs_list[0]["time"]
        last_time = obs_list[-1]["time"]

        folders = unique_in_order(ob["folder"] for ob in obs_list)
        folder = ";".join(folders)

        date = first_time.strftime("%Y-%m-%d")
        grades = "".join(ob["grade"] for ob in obs_list)
        concatenation = " -> ".join(compact_sequence)

        rows.append({
            "science_target": science_target,
            "date": date,
            "folder": folder,
            "run": run,
            "container": container,
            "concatenation": concatenation,
            "grades": grades,
            "n_obs": len(obs_list),
            "first_time": first_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "last_time": last_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "period": get_period(run),
        })

    with open(str(output_csv), "wb") as f:
        writer = csv.writer(f)

        writer.writerow([
            "science_target",
            "date",
            "folder",
            "run",
            "container",
            "concatenation",
            "grades",
            "n_obs",
            "first_time",
            "last_time",
            "period",
        ])

        for row in rows:
            writer.writerow([
                row["science_target"],
                row["date"],
                row["folder"],
                row["run"],
                row["container"],
                row["concatenation"],
                row["grades"],
                row["n_obs"],
                row["first_time"],
                row["last_time"],
                row["period"],
            ])

    print("Saved:", output_csv)
    print("Rows written:", len(rows))


if __name__ == "__main__":
    main()