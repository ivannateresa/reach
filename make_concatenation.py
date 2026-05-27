from __future__ import division, print_function

import os
import sys
import glob
import csv
from collections import OrderedDict
from datetime import datetime


def clean_name(name):
    return name.strip().replace(" ", "").replace("_", "").lower()


def parse_log(obs_log):
    """
    Read one PIONIER .NL.txt log and extract relevant information.
    """

    yyyymmddhhMMss = os.path.basename(obs_log)[6:25]
    ob_time = datetime.strptime(yyyymmddhhMMss, "%Y-%m-%dT%H:%M:%S")

    folder = obs_log.split("/")[-2]

    grade = "?"
    target = "unknown"
    raw_target = "unknown"
    OB = "unknown"
    container = "unknown"
    run = "unknown"
    ob_type = "unknown"

    with open(obs_log) as f:
        content = [row.strip() for row in f.readlines()]

    for row in content:

        if row.startswith("Grade:"):
            grade = row[-1]

        elif row.startswith("Target:"):
            raw_target = row.split("Target:")[-1].strip()
            target = clean_name(raw_target)

        elif row.startswith("OB:"):
            OB = row.split(" ")[-1]

        elif row.startswith("Container:"):
            container = row.split(" ")[-1]

        elif row.startswith("Run:"):
            run = row.split(" ")[-1]

        elif row.startswith("PIONIER_OBS_FRINGE"):
            parts = row.split("\t")
            if len(parts) > 1:
                obfname = parts[1]
                if obfname in obs_log:
                    ob_type = "FRINGE"

        elif row.startswith("PIONIER_GEN_DARK"):
            parts = row.split("\t")
            if len(parts) > 1:
                obfname = parts[1]
                if obfname in obs_log:
                    ob_type = "DARK"

        elif row.startswith("PIONIER_GEN_KAPPA"):
            parts = row.split("\t")
            if len(parts) > 1:
                obfname = parts[1]
                if obfname in obs_log:
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
    Remove consecutive repeated targets.

    Example:
    hr3069 hr3069 hr2998 hr2998 hd65491
    becomes:
    hr3069 -> hr2998 -> hd65491
    """

    compact = []

    for ob in obs_list:
        target = ob["target"]

        if len(compact) == 0:
            compact.append(target)
        elif compact[-1] != target:
            compact.append(target)

    return compact


def infer_science_target(compact_sequence):
    """
    Infer science target as the target that appears more than once
    in the compact concatenation.

    Example:
    cal -> sci -> cal -> sci -> cal
    science = sci
    """

    counts = OrderedDict()

    for target in compact_sequence:
        if target not in counts:
            counts[target] = 0
        counts[target] += 1

    repeated = [target for target in counts if counts[target] > 1]

    if len(repeated) == 0:
        return "unknown"

    if len(repeated) == 1:
        return repeated[0]

    return ";".join(repeated)


def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python make_concatenation_csv.py /path/to/all_sequences")
        print("")
        print("Example:")
        print("  python make_concatenation_csv.py /home2/ihernand/Desktop/reach/all_sequences")
        sys.exit(1)

    base_path = sys.argv[1]

    output_csv = "concatenations_summary.csv"

    all_logs = glob.glob(os.path.join(base_path, "*", "PIONI*.NL.txt"))
    all_logs.sort()

    print("Base path:", base_path)
    print("Number of logs found:", len(all_logs))

    if len(all_logs) == 0:
        print("No logs found. Check the path.")
        sys.exit(1)

    containers = OrderedDict()

    for obs_log in all_logs:
        ob = parse_log(obs_log)

        key = (ob["folder"], ob["container"])

        if key not in containers:
            containers[key] = []

        containers[key].append(ob)

    print("Number of containers found:", len(containers))

    rows = []

    for key in containers:

        folder, container = key
        obs_list = containers[key]
        obs_list = sorted(obs_list, key=lambda x: x["time"])

        compact_sequence = compact_targets(obs_list)
        science_target = infer_science_target(compact_sequence)

        first_time = obs_list[0]["time"]
        last_time = obs_list[-1]["time"]

        date = first_time.strftime("%Y-%m-%d")
        run = obs_list[0]["run"]
        grades = "".join([ob["grade"] for ob in obs_list])

        concatenation = " -> ".join(compact_sequence)
        if ";" in science_target:
            science_target = science_target.split(";", 1)[1].strip()
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
            "period": int(run.split('.')[0])
        })

    with open(output_csv, "w") as f:
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
            "period"
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
                row["period"] 
            ])

    print("Saved:", output_csv)
    print("Rows written:", len(rows))


if __name__ == "__main__":
    main()