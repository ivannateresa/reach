from __future__ import division, print_function

import os
import sys
import glob
from collections import OrderedDict
from datetime import datetime


def clean_name(name):
    return name.strip().replace(" ", "").replace("_", "").lower()


def same_target(a, b):
    a = clean_name(a)
    b = clean_name(b)
    return a == b or a in b or b in a


def parse_log(obs_log):
    """
    Read one PIONIER .NL.txt log and extract the relevant information.
    """

    yyyymmddhhMMss = os.path.basename(obs_log)[6:25]
    ob_time = datetime.strptime(yyyymmddhhMMss, "%Y-%m-%dT%H:%M:%S")

    night = obs_log.split("/")[-2]

    grade = "?"
    target = "unknown"
    raw_target = "unknown"
    OB = "unknown"
    container = "unknown"
    run = "unknown"
    ob_type = "unknown"
    det_mode = "unknown"
    det_ndit_dit = "unknown"

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
            if len(parts) > 6:
                obfname = parts[1]
                if obfname in obs_log:
                    ob_type = "FRINGE"
                    det_ndit_dit = parts[2]
                    det_mode = parts[6]

        elif row.startswith("PIONIER_GEN_DARK"):
            parts = row.split("\t")
            if len(parts) > 6:
                obfname = parts[1]
                if obfname in obs_log:
                    ob_type = "DARK"
                    det_ndit_dit = parts[2]
                    det_mode = parts[6]

        elif row.startswith("PIONIER_GEN_KAPPA"):
            parts = row.split("\t")
            if len(parts) > 6:
                obfname = parts[1]
                if obfname in obs_log:
                    ob_type = "KAPPA"
                    det_ndit_dit = parts[2]
                    det_mode = parts[6]

    return {
        "night": night,
        "time": ob_time,
        "target": target,
        "raw_target": raw_target,
        "grade": grade,
        "OB": OB,
        "container": container,
        "run": run,
        "type": ob_type,
        "det_mode": det_mode,
        "det_ndit_dit": det_ndit_dit,
        "logfile": obs_log,
        "fitsfile": obs_log.replace("NL.txt", "fits.Z"),
    }


def compact_targets(obs_list):
    """
    Return target sequence without repeating the same target many times.
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


def print_container(night, container, obs_list):
    obs_list = sorted(obs_list, key=lambda x: x["time"])

    targets = compact_targets(obs_list)
    grades = "".join([ob["grade"] for ob in obs_list])

    print("\n============================================================")
    print("Night/Folder :", night)
    print("Container    :", container)
    print("Run          :", obs_list[0]["run"])
    print("N obs        :", len(obs_list))
    print("Grades       :", grades)
    print("Sequence     :", " -> ".join(targets))
    print("============================================================")

    print("%-5s %-19s %-14s %-7s %-8s %-20s" %
          ("idx", "time", "target", "grade", "type", "file"))

    for i, ob in enumerate(obs_list):
        print("%-5i %-19s %-14s %-7s %-8s %-20s" %
              (
                  i,
                  ob["time"].strftime("%Y-%m-%dT%H:%M:%S"),
                  ob["target"],
                  ob["grade"],
                  ob["type"],
                  os.path.basename(ob["logfile"])
              ))


def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python check_concatenations.py /path/to/all_sequences")
        print("  python check_concatenations.py /path/to/all_sequences target")
        print("")
        print("Example:")
        print("  python check_concatenations.py /home2/ihernand/Desktop/reach/all_sequences hr2998")
        sys.exit(1)

    base_path = sys.argv[1]

    if len(sys.argv) >= 3:
        target_filter = clean_name(sys.argv[2])
    else:
        target_filter = None

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

        key = (ob["night"], ob["container"])

        if key not in containers:
            containers[key] = []

        containers[key].append(ob)

    print("Number of containers found:", len(containers))

    for key in containers:
        night, container = key
        obs_list = containers[key]

        targets = compact_targets(obs_list)

        if target_filter is not None:
            found = False

            for target in targets:
                if same_target(target_filter, target):
                    found = True

            if not found:
                continue

        print_container(night, container, obs_list)


if __name__ == "__main__":
    main()