from __future__ import division, print_function

"""
Compare one ksi_Gem faint calibrator-removal experiment against its matching
reference experiment, point by point.

The script does not recalibrate or modify any data.  It reads bootstrap 00
calibrated OIFITS files from two REACH result folders and produces:

  * a multipage step-by-step PDF;
  * a file inventory CSV;
  * a matched science-point CSV;
  * a text summary with automatic consistency checks.

Examples
--------

All baselines, remove HR2391 only from faint:

    python diagnose_ksigem_faint_step_by_step.py HR2391

Bad baselines removed in both reference and test:

    python diagnose_ksigem_faint_step_by_step.py HR2391 \
        --baseline-mode BAD_BL_REMOVED

Use two explicit result folders:

    python diagnose_ksigem_faint_step_by_step.py HR2391 \
        --reference-folder 26-09-02_KSIGEM_i1_ALL_ALL_BASELINES_ALL_CALS \
        --test-folder 26-09-02_KSIGEM_i1_NO_CAL_ALL_BASELINES_FAINT_NO_HR2391
"""

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RESULTS_ROOT_DEFAULT = "/home2/ihernand/Desktop/reach/results"
ANALYSIS_ROOT_DEFAULT = "/home2/ihernand/Desktop/reach/analysis_runs"
SCIENCE_TARGET = "ksi_Gem"
FAINT_NIGHT = "2022-03-01"

FAINT_SEQUENCE = [
    "HR2342",
    "ksi_Gem",
    "HR2391",
    "ksi_Gem",
    "HR2426",
]

VALID_FAINT_CALIBRATORS = [
    "HR2342",
    "HR2391",
    "HR2426",
]


def normalise_name(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return (
        str(value)
        .strip()
        .replace("_", "")
        .replace(" ", "")
        .lower()
    )


def clean_token(value):
    return re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()


def newest_folder(pattern):
    candidates = [
        path for path in glob.glob(pattern)
        if os.path.isdir(path)
    ]
    if not candidates:
        raise RuntimeError(
            "No result folder matches:\n%s" % pattern
        )
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def resolve_folder(root, supplied):
    if supplied is None:
        return None
    if os.path.isabs(supplied):
        path = supplied
    else:
        path = os.path.join(root, supplied)
    if not os.path.isdir(path):
        raise RuntimeError("Result folder does not exist: %s" % path)
    return os.path.abspath(path)


def find_case_folders(args, removed_calibrator):
    reference = resolve_folder(
        args.results_root,
        args.reference_folder,
    )
    test = resolve_folder(
        args.results_root,
        args.test_folder,
    )

    if args.baseline_mode == "ALL_BASELINES":
        reference_experiment = "ALL"
        test_experiment = "NO_CAL"
    else:
        reference_experiment = "NO_BL"
        test_experiment = "NO_BL_NO_CAL"

    if reference is None:
        reference_pattern = os.path.join(
            args.results_root,
            "*_KSIGEM_i%i_%s_%s_ALL_CALS"
            % (
                args.n_bootstraps,
                reference_experiment,
                args.baseline_mode,
            ),
        )
        reference = newest_folder(reference_pattern)

    if test is None:
        test_pattern = os.path.join(
            args.results_root,
            "*_KSIGEM_i%i_%s_%s_FAINT_NO_%s"
            % (
                args.n_bootstraps,
                test_experiment,
                args.baseline_mode,
                clean_token(removed_calibrator),
            ),
        )
        test = newest_folder(test_pattern)

    return reference, test


def get_wavelengths(hdul, vis_hdu):
    requested_insname = str(
        vis_hdu.header.get("INSNAME", "")
    ).strip()

    candidates = []
    for hdu in hdul:
        if str(hdu.header.get("EXTNAME", "")).strip() != "OI_WAVELENGTH":
            continue
        candidates.append(hdu)
        candidate_insname = str(
            hdu.header.get("INSNAME", "")
        ).strip()
        if requested_insname and candidate_insname == requested_insname:
            return np.asarray(hdu.data["EFF_WAVE"], dtype=float)

    if not candidates:
        raise RuntimeError("OI_WAVELENGTH extension is missing")

    return np.asarray(candidates[0].data["EFF_WAVE"], dtype=float)


def get_target_map(hdul):
    mapping = {}
    for hdu in hdul:
        if str(hdu.header.get("EXTNAME", "")).strip() != "OI_TARGET":
            continue
        for row in hdu.data:
            target = row["TARGET"]
            if isinstance(target, bytes):
                target = target.decode("utf-8")
            mapping[int(row["TARGET_ID"])] = str(target).strip()
    return mapping


def get_station_map(hdul):
    mapping = {}
    for hdu in hdul:
        if str(hdu.header.get("EXTNAME", "")).strip() != "OI_ARRAY":
            continue
        for row in hdu.data:
            try:
                station = row["STA_NAME"]
            except Exception:
                station = row["TEL_NAME"]
            if isinstance(station, bytes):
                station = station.decode("utf-8")
            mapping[int(row["STA_INDEX"])] = str(station).strip()
    return mapping


def canonical_baseline(station_1, station_2):
    return "-".join(sorted([str(station_1), str(station_2)]))


def read_result_folder(folder, bootstrap_index, night):
    from astropy.io import fits

    pattern = os.path.join(
        folder,
        "%s_*_oidataCalibrated_%02i.fits"
        % (str(night), int(bootstrap_index)),
    )
    filenames = sorted(glob.glob(pattern))

    if not filenames:
        raise RuntimeError(
            "No calibrated bootstrap files found:\n%s" % pattern
        )

    points = []
    inventory = []

    for filename in filenames:
        basename = os.path.basename(filename)
        file_points = []
        file_targets = set()

        with fits.open(filename) as hdul:
            target_map = get_target_map(hdul)
            station_map = get_station_map(hdul)

            vis_hdus = [
                hdu for hdu in hdul
                if str(hdu.header.get("EXTNAME", "")).strip() == "OI_VIS2"
            ]

            # REACH uses the first OI_VIS2 extension.  Some calibrated files
            # contain repeated OI_VIS2 extensions; reading all of them doubles
            # the inventory and creates duplicate matching keys.
            vis_hdus = vis_hdus[:1]

            for extension_i, vis_hdu in enumerate(vis_hdus):
                wavelengths = get_wavelengths(hdul, vis_hdu)
                table = vis_hdu.data

                for row_i, row in enumerate(table):
                    target_id = int(row["TARGET_ID"])
                    target = target_map.get(target_id, "UNKNOWN")
                    target_clean = normalise_name(target)
                    file_targets.add(target)

                    pair = row["STA_INDEX"]
                    station_1 = station_map.get(int(pair[0]), str(pair[0]))
                    station_2 = station_map.get(int(pair[1]), str(pair[1]))
                    baseline = canonical_baseline(station_1, station_2)

                    ucoord = float(row["UCOORD"])
                    vcoord = float(row["VCOORD"])
                    baseline_m = np.sqrt(ucoord ** 2 + vcoord ** 2)
                    mjd = float(row["MJD"])

                    values = np.asarray(row["VIS2DATA"], dtype=float).ravel()
                    errors = np.asarray(row["VIS2ERR"], dtype=float).ravel()
                    flags = np.asarray(row["FLAG"], dtype=bool).ravel()

                    n_channels = min(
                        len(wavelengths),
                        len(values),
                        len(errors),
                        len(flags),
                    )

                    for channel_i in range(n_channels):
                        wavelength = float(wavelengths[channel_i])
                        value = float(values[channel_i])
                        error = float(errors[channel_i])
                        flag = bool(flags[channel_i])
                        valid = bool(
                            (not flag)
                            and np.isfinite(value)
                            and np.isfinite(error)
                            and np.isfinite(wavelength)
                            and wavelength > 0.0
                        )

                        point = {
                            "file": basename,
                            "extension": extension_i,
                            "row": row_i,
                            "channel": channel_i,
                            "target": target,
                            "target_clean": target_clean,
                            "role": (
                                "SCI"
                                if target_clean == normalise_name(SCIENCE_TARGET)
                                else "CAL"
                            ),
                            "baseline": baseline,
                            "mjd": mjd,
                            "wavelength_m": wavelength,
                            "baseline_m": baseline_m,
                            "spatial_frequency": baseline_m / wavelength,
                            "vis2": value,
                            "e_vis2": error,
                            "flag": flag,
                            "valid": valid,
                        }
                        points.append(point)
                        file_points.append(point)

        inventory.append({
            "file": basename,
            "targets": ";".join(sorted(file_targets)),
            "filename_role": (
                "SCI" if "_SCI_" in basename
                else "CAL" if "_CAL_" in basename
                else "UNKNOWN"
            ),
            "n_points": len(file_points),
            "n_valid": sum(1 for p in file_points if p["valid"]),
            "n_flagged": sum(1 for p in file_points if p["flag"]),
        })

    return points, inventory


def science_points(points):
    return [
        point for point in points
        if point["role"] == "SCI" and point["valid"]
    ]


def point_key(point):
    return (
        point["target_clean"],
        point["baseline"],
        round(float(point["mjd"]), 8),
        int(point["channel"]),
    )


def match_science_points(reference_points, test_points):
    reference_map = {}
    test_map = {}

    for point in science_points(reference_points):
        reference_map.setdefault(point_key(point), []).append(point)
    for point in science_points(test_points):
        test_map.setdefault(point_key(point), []).append(point)

    common_keys = sorted(set(reference_map) & set(test_map))
    matched = []

    for key in common_keys:
        reference_group = reference_map[key]
        test_group = test_map[key]
        n_pairs = min(len(reference_group), len(test_group))
        for pair_i in range(n_pairs):
            reference = reference_group[pair_i]
            test = test_group[pair_i]
            delta = test["vis2"] - reference["vis2"]
            combined_error = np.sqrt(
                reference["e_vis2"] ** 2 + test["e_vis2"] ** 2
            )
            matched.append({
                "target": reference["target"],
                "baseline": reference["baseline"],
                "mjd": reference["mjd"],
                "channel": reference["channel"],
                "wavelength_m": reference["wavelength_m"],
                "spatial_frequency": reference["spatial_frequency"],
                "reference_vis2": reference["vis2"],
                "reference_e_vis2": reference["e_vis2"],
                "test_vis2": test["vis2"],
                "test_e_vis2": test["e_vis2"],
                "delta_vis2": delta,
                "relative_delta_percent": (
                    100.0 * delta / reference["vis2"]
                    if reference["vis2"] != 0.0
                    else np.nan
                ),
                "delta_sigma": (
                    delta / combined_error
                    if combined_error > 0.0
                    else np.nan
                ),
                "reference_file": reference["file"],
                "test_file": test["file"],
            })

    return matched, reference_map, test_map


def write_csv(filename, rows, columns):
    with open(filename, "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def baseline_colours(points):
    baselines = sorted(set(point["baseline"] for point in points))
    return {
        baseline: "C%i" % (index % 10)
        for index, baseline in enumerate(baselines)
    }


def add_text_page(pdf, title, lines):
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle(title, fontsize=15)
    fig.text(
        0.05,
        0.93,
        "\n".join(lines),
        va="top",
        family="monospace",
        fontsize=9,
    )
    pdf.savefig(fig)
    plt.close(fig)


def plot_science_case(pdf, points, title):
    science = science_points(points)
    colours = baseline_colours(science)
    fig, ax = plt.subplots(figsize=(11, 7))

    for baseline in sorted(colours):
        group = [p for p in science if p["baseline"] == baseline]
        ax.errorbar(
            [p["spatial_frequency"] for p in group],
            [p["vis2"] for p in group],
            yerr=[p["e_vis2"] for p in group],
            fmt="o",
            ms=4,
            alpha=0.75,
            color=colours[baseline],
            label=baseline,
        )

    ax.set_title(title)
    ax.set_xlabel(r"Spatial frequency [rad$^{-1}$]")
    ax.set_ylabel(r"Calibrated $V^2$")
    ax.set_ylim(0.0, 1.3)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_overlay(pdf, matched):
    fig, ax = plt.subplots(figsize=(11, 7))
    colours = baseline_colours(matched)

    for baseline in sorted(colours):
        group = [p for p in matched if p["baseline"] == baseline]
        ax.scatter(
            [p["spatial_frequency"] for p in group],
            [p["reference_vis2"] for p in group],
            marker="o",
            s=30,
            facecolors="none",
            edgecolors=colours[baseline],
            label="%s reference" % baseline,
        )
        ax.scatter(
            [p["spatial_frequency"] for p in group],
            [p["test_vis2"] for p in group],
            marker="x",
            s=30,
            color=colours[baseline],
            label="%s test" % baseline,
        )

    ax.set_title("Step 4 - Same ksi_Gem faint points: reference versus test")
    ax.set_xlabel(r"Spatial frequency [rad$^{-1}$]")
    ax.set_ylabel(r"Calibrated $V^2$")
    ax.set_ylim(0.0, 1.3)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_differences(pdf, matched):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    colours = baseline_colours(matched)

    for baseline in sorted(colours):
        group = [p for p in matched if p["baseline"] == baseline]
        x = [p["spatial_frequency"] for p in group]
        axes[0].scatter(
            x,
            [p["delta_vis2"] for p in group],
            s=28,
            color=colours[baseline],
            label=baseline,
        )
        axes[1].scatter(
            x,
            [p["delta_sigma"] for p in group],
            s=28,
            color=colours[baseline],
        )

    axes[0].axhline(0.0, color="black", lw=1)
    axes[1].axhline(0.0, color="black", lw=1)
    axes[0].set_ylabel(r"$V^2_{test}-V^2_{reference}$")
    axes[1].set_ylabel(r"Difference / combined error")
    axes[1].set_xlabel(r"Spatial frequency [rad$^{-1}$]")
    axes[0].set_title("Step 5 - Point-by-point calibration change")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_calibrators(pdf, reference_points, test_points):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    for ax, points, title in [
        (axes[0], reference_points, "Reference calibrated calibrators"),
        (axes[1], test_points, "Test calibrated calibrators"),
    ]:
        calibrators = [
            point for point in points
            if point["role"] == "CAL" and point["valid"]
        ]
        targets = sorted(set(point["target"] for point in calibrators))

        for target_i, target in enumerate(targets):
            group = [p for p in calibrators if p["target"] == target]
            ax.scatter(
                [p["spatial_frequency"] for p in group],
                [p["vis2"] for p in group],
                s=25,
                alpha=0.75,
                color="C%i" % (target_i % 10),
                label=target,
            )

        if not calibrators:
            ax.text(
                0.5,
                0.5,
                "No CAL products were copied\nto this results folder",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title(title)
        ax.set_xlabel(r"Spatial frequency [rad$^{-1}$]")
        ax.grid(True, alpha=0.3)
        if targets:
            ax.legend(fontsize=8)

    axes[0].set_ylabel(r"Calibrated $V^2$")
    axes[0].set_ylim(0.0, 1.3)
    fig.suptitle("Step 6 - Calibrator products available in each case")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def sequence_without(calibrator):
    return [
        target for target in FAINT_SEQUENCE
        if normalise_name(target) != normalise_name(calibrator)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose the effect of one ksi_Gem faint calibrator removal."
    )
    parser.add_argument(
        "removed_calibrator",
        help="One of HR2342, HR2391 or HR2426.",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=["ALL_BASELINES", "BAD_BL_REMOVED"],
        default="ALL_BASELINES",
    )
    parser.add_argument("--n-bootstraps", type=int, default=1)
    parser.add_argument("--bootstrap-index", type=int, default=0)
    parser.add_argument("--results-root", default=RESULTS_ROOT_DEFAULT)
    parser.add_argument("--analysis-root", default=ANALYSIS_ROOT_DEFAULT)
    parser.add_argument("--reference-folder", default=None)
    parser.add_argument("--test-folder", default=None)
    args = parser.parse_args()

    removed_calibrator = clean_token(args.removed_calibrator)
    valid_tokens = [clean_token(value) for value in VALID_FAINT_CALIBRATORS]
    if removed_calibrator not in valid_tokens:
        raise ValueError(
            "Faint calibrator must be one of: %s"
            % ", ".join(VALID_FAINT_CALIBRATORS)
        )

    reference_folder, test_folder = find_case_folders(
        args,
        removed_calibrator,
    )

    reference_name = os.path.basename(reference_folder)
    test_name = os.path.basename(test_folder)

    issues = []
    expected_faint_token = "FAINTNO%s" % removed_calibrator
    test_clean = normalise_name(test_name)

    if normalise_name(expected_faint_token) not in test_clean:
        issues.append(
            "ERROR: test folder does not contain FAINT_NO_%s."
            % removed_calibrator
        )

    if "brightno" in test_clean:
        issues.append(
            "ERROR: test folder also removes a BRIGHT calibrator; this is not "
            "a faint-only comparison."
        )

    if args.baseline_mode not in reference_name:
        issues.append("ERROR: reference baseline mode does not match request.")
    if args.baseline_mode not in test_name:
        issues.append("ERROR: test baseline mode does not match request.")

    reference_points, reference_inventory = read_result_folder(
        reference_folder,
        args.bootstrap_index,
        FAINT_NIGHT,
    )
    test_points, test_inventory = read_result_folder(
        test_folder,
        args.bootstrap_index,
        FAINT_NIGHT,
    )

    matched, reference_map, test_map = match_science_points(
        reference_points,
        test_points,
    )

    n_reference_science = len(science_points(reference_points))
    n_test_science = len(science_points(test_points))
    n_matched = len(matched)

    if n_matched == 0:
        issues.append(
            "ERROR: no science points could be matched. The two pipeline "
            "runs did not use the same SCI interferograms."
        )

    reference_keys = set(reference_map)
    test_keys = set(test_map)
    only_reference = reference_keys - test_keys
    only_test = test_keys - reference_keys

    if only_reference or only_test:
        issues.append(
            "WARNING: the two cases do not contain the same science-point set "
            "(%i only reference; %i only test)."
            % (len(only_reference), len(only_test))
        )

    delta = np.asarray(
        [row["delta_vis2"] for row in matched],
        dtype=float,
    )
    relative = np.asarray(
        [row["relative_delta_percent"] for row in matched],
        dtype=float,
    )

    if n_matched:
        median_abs_delta = float(np.nanmedian(np.abs(delta)))
        max_abs_delta = float(np.nanmax(np.abs(delta)))
        median_abs_percent = float(np.nanmedian(np.abs(relative)))
    else:
        median_abs_delta = np.nan
        max_abs_delta = np.nan
        median_abs_percent = np.nan

    output_name = "KSIGEM_FAINT_NO_%s_%s" % (
        removed_calibrator,
        args.baseline_mode,
    )
    output_directory = os.path.join(
        args.analysis_root,
        "ksi_gem_faint_step_diagnostics",
        output_name,
    )
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    inventory_rows = []
    for case, folder, rows in [
        ("reference", reference_name, reference_inventory),
        ("test", test_name, test_inventory),
    ]:
        for row in rows:
            result = dict(row)
            result["case"] = case
            result["results_folder"] = folder
            inventory_rows.append(result)

    inventory_file = os.path.join(output_directory, "file_inventory.csv")
    matched_file = os.path.join(output_directory, "matched_science_points.csv")
    summary_file = os.path.join(output_directory, "diagnostic_summary.txt")
    pdf_file = os.path.join(output_directory, "ksi_gem_faint_step_by_step.pdf")

    write_csv(
        inventory_file,
        inventory_rows,
        [
            "case", "results_folder", "file", "filename_role", "targets",
            "n_points", "n_valid", "n_flagged",
        ],
    )
    write_csv(
        matched_file,
        matched,
        [
            "target", "baseline", "mjd", "channel", "wavelength_m",
            "spatial_frequency", "reference_vis2", "reference_e_vis2",
            "test_vis2", "test_e_vis2", "delta_vis2",
            "relative_delta_percent", "delta_sigma", "reference_file",
            "test_file",
        ],
    )

    summary_lines = [
        "ksi_Gem faint calibrator-removal diagnostic",
        "=" * 70,
        "Removed faint calibrator : %s" % removed_calibrator,
        "Baseline mode            : %s" % args.baseline_mode,
        "Reference folder         : %s" % reference_name,
        "Test folder              : %s" % test_name,
        "Reference sequence       : %s" % " -> ".join(FAINT_SEQUENCE),
        "Test sequence            : %s"
        % " -> ".join(sequence_without(removed_calibrator)),
        "Reference SCI points     : %i" % n_reference_science,
        "Test SCI points          : %i" % n_test_science,
        "Matched SCI points       : %i" % n_matched,
        "Only in reference        : %i" % len(only_reference),
        "Only in test             : %i" % len(only_test),
        "Median |delta V2|        : %.8f" % median_abs_delta,
        "Maximum |delta V2|       : %.8f" % max_abs_delta,
        "Median |relative change| : %.4f %%" % median_abs_percent,
        "",
        "Consistency checks:",
    ]

    if issues:
        summary_lines.extend(["  " + issue for issue in issues])
    else:
        summary_lines.append("  OK: exact faint-only, same-baseline comparison.")

    summary_lines.extend([
        "",
        "Interpretation:",
        "  * Same point set + shifted V2: calibration/transfer-function effect.",
        "  * Different point set: selection, flags, bootstrap or folder mismatch.",
        "  * For a controlled test, set do_random_ifg_sampling = False and",
        "    do_gaussian_diam_sampling = False in both pipeline runs.",
        "  * Small OIFITS delta but large final-fit change: inspect C_SCALE/fitting.",
    ])

    with open(summary_file, "w") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    with PdfPages(pdf_file) as pdf:
        add_text_page(pdf, "Step 1 - Experiment identity", summary_lines)
        add_text_page(
            pdf,
            "Step 2 - Calibrated-file inventory",
            [
                "%-10s %-9s %-40s %7s %7s %7s %s"
                % (
                    row["case"],
                    row["filename_role"],
                    row["file"][:40],
                    row["n_points"],
                    row["n_valid"],
                    row["n_flagged"],
                    row["targets"],
                )
                for row in inventory_rows
            ],
        )
        plot_science_case(
            pdf,
            reference_points,
            "Step 3a - Reference ksi_Gem faint calibrated VIS2",
        )
        plot_science_case(
            pdf,
            test_points,
            "Step 3b - Test ksi_Gem faint calibrated VIS2",
        )
        if matched:
            plot_overlay(pdf, matched)
            plot_differences(pdf, matched)
        else:
            add_text_page(
                pdf,
                "Steps 4-5 unavailable",
                ["No matched science points were found."],
            )
        plot_calibrators(pdf, reference_points, test_points)

    print("")
    print("=" * 79)
    print("KSI GEM FAINT STEP-BY-STEP DIAGNOSTIC FINISHED")
    print("=" * 79)
    for line in summary_lines:
        print(line)
    print("")
    print("PDF      : %s" % pdf_file)
    print("Inventory: %s" % inventory_file)
    print("Matched  : %s" % matched_file)
    print("Summary  : %s" % summary_file)


if __name__ == "__main__":
    main()
