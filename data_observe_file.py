from __future__ import division, print_function

import os
import sys
import glob
import csv
import re
from collections import OrderedDict
from datetime import datetime, timedelta


# ============================================================
# CONFIGURACION
# ============================================================

OUTPUT_TSV = "dates_observed.tsv"
SUMMARY_CSV = "concatenations_summary_with_type.csv"
UNKNOWN_CSV = "unknown_sequences.csv"
MAX_SEQUENCE_GAP_MINUTES = 90


# ============================================================
# FUNCIONES GENERALES
# ============================================================

def clean_name(name):
    """
    Normaliza nombres para compararlos sin depender de espacios,
    guiones bajos, mayusculas o caracteres invisibles.

    Ejemplos:
    iot_Psc    -> iotpsc
    iot Psc    -> iotpsc
    HD_222919  -> hd222919
    HD  63734  -> hd63734
    """
    if name is None:
        return "unknown"

    value = str(name).replace("\ufeff", "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value or "unknown"


def output_star_name(name):
    """
    Nombre que se escribe en dates_observed.tsv.

    Ejemplo:
    iot_Psc -> iotPsc
    zet_Tuc -> zetTuc
    HR_2998 -> HR2998

    Si prefieres todo lowercase, cambia el return por:
    return clean_name(name)
    """
    if name is None:
        return "unknown"

    return name


def parse_bool(value):
    """
    Convierte strings tipo True/False a boolean.
    """
    return str(value).strip().lower() in ["true", "1", "yes", "y"]


def get_period(run):
    """
    Extrae el periodo desde algo como:
    104.A-9001(A) -> 104
    """
    try:
        return str(int(str(run).split(".")[0]))
    except Exception:
        return "unknown"


# ============================================================
# LEER TABLA BRIGHT/FAINT
# ============================================================

def load_bright_faint_table(sequence_csv):
    """
    Lee el archivo:
    Interferometry_adam_data(Bright-Faint Sequence)(1).csv

    Columnas esperadas:
    Period, Science, Sequence, Primary

    Devuelve:
    target_info[(period, clean_primary)] = {
        "primary": nombre original,
        "output_name": nombre para escribir,
        "science": True/False,
        "sequence": bright/faint/both
    }
    """

    target_info = {}

    with open(sequence_csv, "r") as f:
        print(f)
        reader = csv.DictReader(f)
   
        for row in reader:
            print(row)
            period = str(row["\xef\xbb\xbfPeriod"]).strip()
            primary = str(row["Primary"]).strip()
            key_name = clean_name(primary)

            science = parse_bool(row["Science"])
            sequence = str(row["Sequence"]).strip().lower()

            key = (period, key_name)

            target_info[key] = {
                "primary": primary,
                "output_name": output_star_name(primary),
                "science": science,
                "sequence": sequence,
            }

    return target_info


# ============================================================
# LEER LOGS PIONIER
# ============================================================

def parse_log(obs_log):
    """
    Lee un archivo PIONIER .NL.txt y extrae informacion relevante.
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
            target = raw_target

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


# ============================================================
# SECUENCIAS
# ============================================================

def compact_targets(obs_list):
    """
    Elimina targets consecutivos repetidos usando nombres normalizados,
    pero conserva el nombre original para escribir la concatenacion.
    """

    compact = []
    last_clean = None

    for ob in obs_list:
        target_display = str(ob["target"]).strip()
        target_clean = clean_name(target_display)

        if target_clean != last_clean:
            compact.append(target_display)
            last_clean = target_clean

    return compact


def infer_science_target(compact_sequence, period, target_info):
    """
    Detecta la estrella cientifica usando Science=True.
    Si no la encuentra, usa como fallback el target repetido.
    """

    period = str(period)

    science_found = []
    display_names = {}

    for target in compact_sequence:
        target_clean = clean_name(target)
        display_names.setdefault(target_clean, target)
        key = (period, target_clean)

        if key in target_info and target_info[key]["science"]:
            science_found.append(target_clean)

    science_found = list(OrderedDict.fromkeys(science_found))

    if len(science_found) == 1:
        sci_clean = science_found[0]
        key = (period, sci_clean)
        return sci_clean, target_info[key]["output_name"]

    if len(science_found) > 1:
        names = [
            target_info[(period, sci)]["output_name"]
            for sci in science_found
        ]
        return ";".join(science_found), ";".join(names)

    counts = OrderedDict()

    for target in compact_sequence:
        target_clean = clean_name(target)
        display_names.setdefault(target_clean, target)
        counts[target_clean] = counts.get(target_clean, 0) + 1

    repeated = [target for target in counts if counts[target] > 1]

    if len(repeated) == 0:
        return "unknown", "unknown"

    if len(repeated) == 1:
        target_clean = repeated[0]
        return target_clean, display_names[target_clean]

    first_target = clean_name(compact_sequence[0])
    last_target = clean_name(compact_sequence[-1])

    for target_clean in repeated:
        if not (target_clean == first_target and target_clean == last_target):
            return target_clean, display_names[target_clean]

    target_clean = repeated[0]
    return target_clean, display_names[target_clean]

def infer_bright_faint_from_table(compact_sequence, period, science_target, target_info):
    """
    Clasifica una concatenacion como Bright o Faint usando el archivo CSV.

    Usa calibradores con:
    Sequence = Bright
    Sequence = Faint

    Ignora:
    Sequence = Both
    Science = True
    """

    period = str(period)

    found = []
    evidence = []

    for target in compact_sequence:

        target_clean = clean_name(target)
        key = (period, target_clean)

        if key not in target_info:
            evidence.append(target_clean + ":not_in_table")
            continue

        info = target_info[key]

        seq = info["sequence"]
        is_science = info["science"]
        primary = info["primary"]

        if is_science:
            evidence.append(primary + ":Science")
            continue

        if seq == "bright":
            found.append("b")
            evidence.append(primary + ":Bright")

        elif seq == "faint":
            found.append("f")
            evidence.append(primary + ":Faint")

        elif seq == "both":
            evidence.append(primary + ":Both")

        else:
            evidence.append(primary + ":" + seq)

    found_unique = sorted(set(found))

    if len(found_unique) == 1:
        return found_unique[0], "; ".join(evidence)

    elif len(found_unique) == 0:
        return "unknown", "; ".join(evidence)

    else:
        # Encontro Bright y Faint en la misma secuencia.
        # Conviene revisarlo manualmente.
        return "mixed", "; ".join(evidence)


def add_sequence_to_dates(dates_dict, star, period, seq_type, date):
    """
    Guarda fecha bright/faint para cada star + period.
    """

    key = (star, period)

    if key not in dates_dict:
        dates_dict[key] = {
            "star": star,
            "period": period,
            "b_dates": [],
            "f_dates": [],
        }

    if seq_type == "b":
        if date not in dates_dict[key]["b_dates"]:
            dates_dict[key]["b_dates"].append(date)

    elif seq_type == "f":
        if date not in dates_dict[key]["f_dates"]:
            dates_dict[key]["f_dates"].append(date)


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python make_dates_observed.py /path/to/all_sequences /path/to/Bright-Faint.csv")
        print("")
        print("Example:")
        print('  python make_dates_observed.py /home2/ihernand/Desktop/reach/all_sequences "/home2/ihernand/Desktop/reach/Interferometry_adam_data(Bright-Faint Sequence)(1).csv"')
        sys.exit(1)

    base_path = sys.argv[1]
    sequence_csv = sys.argv[2]

    print("Base path:", base_path)
    print("Bright/Faint table:", sequence_csv)

    target_info = load_bright_faint_table(sequence_csv)

    print("Number of targets in Bright/Faint table:", len(target_info))

    # Busca logs dentro de subcarpetas
    all_logs = glob.glob(os.path.join(base_path, "*", "PIONI*.NL.txt"))
    all_logs.sort()

    print("Number of logs found:", len(all_logs))

    if len(all_logs) == 0:
        print("No logs found. Check the path.")
        sys.exit(1)

    # ========================================================
    # Agrupar por run + container y separar por pausas largas.
    #
    # Esto permite unir una concatenacion que cruza medianoche,
    # aunque los logs esten en dos carpetas de fecha distintas.
    # Al mismo tiempo evita unir reutilizaciones del mismo
    # container en noches separadas.
    # ========================================================

    observations_by_container = OrderedDict()

    for obs_log in all_logs:
        ob = parse_log(obs_log)

        if ob["container"] == "unknown":
            base_key = (
                ob["run"],
                "unknown",
                ob["folder"],
                ob["OB"],
            )
        else:
            base_key = (ob["run"], ob["container"])

        observations_by_container.setdefault(base_key, []).append(ob)

    containers = OrderedDict()
    max_gap = timedelta(minutes=MAX_SEQUENCE_GAP_MINUTES)

    for base_key, observations in observations_by_container.items():
        observations = sorted(observations, key=lambda item: item["time"])

        chunk = []
        chunk_number = 0

        for ob in observations:
            if chunk and (ob["time"] - chunk[-1]["time"]) > max_gap:
                containers[(base_key, chunk_number)] = chunk
                chunk = []
                chunk_number += 1

            chunk.append(ob)

        if chunk:
            containers[(base_key, chunk_number)] = chunk

    print("Number of concatenations found:", len(containers))

    # ========================================================
    # Crear resumen de secuencias
    # ========================================================

    sequence_rows = []

    for key in containers:

        base_key, chunk_number = key
        obs_list = sorted(containers[key], key=lambda x: x["time"])

        container = obs_list[0]["container"]
        folders = list(OrderedDict.fromkeys(
            ob["folder"] for ob in obs_list
        ))
        folder = "|".join(folders)

        first_time = obs_list[0]["time"]
        last_time = obs_list[-1]["time"]

        date = first_time.strftime("%Y-%m-%d")
        run = obs_list[0]["run"]
        period = get_period(run)

        compact_sequence = compact_targets(obs_list)

        science_clean, science_output = infer_science_target(
            compact_sequence,
            period,
            target_info
        )

        seq_type, seq_evidence = infer_bright_faint_from_table(
            compact_sequence,
            period,
            science_clean,
            target_info
        )

        grades = "".join([ob["grade"] for ob in obs_list])
        concatenation = " -> ".join(compact_sequence)

        row = {
            "science_target": science_output,
            "science_clean": science_clean,
            "date": date,
            "folder": folder,
            "run": run,
            "period": period,
            "container": container,
            "OB": obs_list[0]["OB"],
            "logfile": obs_list[0]["logfile"],
            "seq_type": seq_type,
            "seq_evidence": seq_evidence,
            "concatenation": concatenation,
            "grades": grades,
            "n_obs": len(obs_list),
            "first_time": first_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "last_time": last_time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        sequence_rows.append(row)

    # ========================================================
    # Guardar resumen para revisar
    # ========================================================

    with open(SUMMARY_CSV, "w") as f:
        writer = csv.writer(f)

        writer.writerow([
            "science_target",
            "science_clean",
            "date",
            "folder",
            "run",
            "period",
            "container",
            "seq_type",
            "seq_evidence",
            "concatenation",
            "grades",
            "n_obs",
            "first_time",
            "last_time",
        ])

        for row in sequence_rows:
            writer.writerow([
                row["science_target"],
                row["science_clean"],
                row["date"],
                row["folder"],
                row["run"],
                row["period"],
                row["container"],
                row["seq_type"],
                row["seq_evidence"],
                row["concatenation"],
                row["grades"],
                row["n_obs"],
                row["first_time"],
                row["last_time"],
            ])

    print("Saved:", SUMMARY_CSV)

    # ========================================================
    # Crear dates_observed.tsv
    # ========================================================

    dates_dict = OrderedDict()
    unknown_rows = []

    for row in sequence_rows:

        star = row["science_target"]
        period = row["period"]
        date = row["date"]
        seq_type = row["seq_type"]

        if star == "unknown" or period == "unknown":
            unknown_rows.append(row)
            continue

        if seq_type not in ["b", "f"]:
            unknown_rows.append(row)
            continue

        add_sequence_to_dates(
            dates_dict,
            star,
            period,
            seq_type,
            date
        )

    # ========================================================
    # Preparar filas finales
    # ========================================================

    output_rows = []

    for key in dates_dict:

        item = dates_dict[key]

        star = item["star"]
        period = item["period"]

        b_dates = sorted(item["b_dates"])
        f_dates = sorted(item["f_dates"])

        nmax = max(len(b_dates), len(f_dates))

        if nmax == 0:
            continue

        for i in range(nmax):

            if i < len(b_dates):
                b_night = b_dates[i]
            else:
                b_night = ""

            if i < len(f_dates):
                f_night = f_dates[i]
            else:
                f_night = ""

            output_rows.append({
                "star": star,
                "period": period,
                "b_night": b_night,
                "b_order": i,
                "f_night": f_night,
                "f_order": i,
            })

    def sort_key(row):
        try:
            p = int(row["period"])
        except Exception:
            p = 999999

        return (p, row["star"], row["b_order"], row["f_order"])

    output_rows = sorted(output_rows, key=sort_key)

    # ========================================================
    # Escribir dates_observed.tsv
    # ========================================================

    with open(OUTPUT_TSV, "w") as f:
        writer = csv.writer(f, delimiter="\t")

        writer.writerow([
            "star",
            "period",
            "b_night",
            "b_order",
            "f_night",
            "f_order",
        ])

        for row in output_rows:
            writer.writerow([
                row["star"],
                row["period"],
                row["b_night"],
                row["b_order"],
                row["f_night"],
                row["f_order"],
            ])

    print("Saved:", OUTPUT_TSV)
    print("Rows written:", len(output_rows))

    # ========================================================
    # Guardar secuencias problematicas
    # ========================================================

    if len(unknown_rows) > 0:

        with open(UNKNOWN_CSV, "w") as f:
            writer = csv.writer(f)

            writer.writerow([
                "science_target",
                "science_clean",
                "date",
                "folder",
                "run",
                "period",
                "container",
                "seq_type",
                "seq_evidence",
                "concatenation",
            ])

            for row in unknown_rows:
                writer.writerow([
                    row["science_target"],
                    row["science_clean"],
                    row["date"],
                    row["folder"],
                    row["run"],
                    row["period"],
                    row["container"],
                    row["seq_type"],
                    row["seq_evidence"],
                    row["concatenation"],
                ])

        print("")
        print("WARNING:")
        print("Some sequences could not be classified as Bright/Faint.")
        print("Saved:", UNKNOWN_CSV)
        print("Check this file manually.")


if __name__ == "__main__":
    main()