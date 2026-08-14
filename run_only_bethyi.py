from __future__ import print_function

import os
import traceback
import pandas as pd

import reach.utils as rutils
import reach.pndrs as rpndrs


# ============================================================
# Configuracion
# ============================================================

key = (106, "bet_Hyi", "bright")

base_path = (
    "/home2/ihernand/Desktop/reach/complete_sequences/%s_v3.94_abcd/"
)

tgt_info_path = (
    "/home2/ihernand/Desktop/reach/data/tgt_info.csv"
)


# ============================================================
# 1. Cargar logs
# ============================================================

print("\nCargando logs...")

complete_sequences, sequences = rutils.load_sequence_logs()

print("Logs cargados.")


# ============================================================
# 2. Seleccionar solamente bet_Hyi
# ============================================================

if key not in sequences:
    raise KeyError(
        "No existe %r en sequences" % (key,)
    )

if key not in complete_sequences:
    raise KeyError(
        "No existe %r en complete_sequences" % (key,)
    )

bet_sequences = {
    key: sequences[key]
}

bet_complete_sequences = {
    key: complete_sequences[key]
}

print("\nSecuencia seleccionada:")
print(bet_sequences[key])

print("\nNoche:")
print(bet_complete_sequences[key][0])
# ============================================================
# Seleccionar todas las secuencias de la noche de bet_Hyi
# ============================================================

night = complete_sequences[key][0]

night_complete_sequences = {
    sequence_key: sequence_data
    for sequence_key, sequence_data in complete_sequences.items()
    if sequence_data[0] == night
}

night_sequences = {
    sequence_key: sequences[sequence_key]
    for sequence_key in night_complete_sequences
    if sequence_key in sequences
}


print("\nNoche seleccionada:")
print(night)

print("\nNumero de secuencias de esa noche:")
print(len(night_complete_sequences))

print("\nSecuencias encontradas:")

for sequence_key in sorted(
        night_complete_sequences.keys(),
        key=str
    ):

    print(
        "  %s -> %s"
        % (
            sequence_key,
            night_sequences.get(
                sequence_key,
                "NO EST EN sequences"
            )
        )
    )

# ============================================================
# 3. Cargar tgt_info guardado
# No ejecuta extinction ni bolometric corrections
# ============================================================

print("\nBuscando tgt_info.csv:")
print(tgt_info_path)

if not os.path.isfile(tgt_info_path):
    raise IOError(
        "No existe tgt_info.csv. "
        "Debes copiar el archivo generado en el servidor."
    )

tgt_info = pd.read_csv(
    tgt_info_path,
    index_col=0
)

print("\ntgt_info cargado:")
print(tgt_info.shape)


# ============================================================
# 4. Comprobar las estrellas necesarias
# ============================================================

required_stars = [
    "sig_Pav",
    "bet_Hyi",
    "HD_4332",
    "lam_Tuc"
]

print("\nRevisando estrellas en tgt_info:")

print("\nRevisando estrellas en la columna Primary:")
# ============================================================
# Revisar nombres normalizando espacios y guiones bajos
# ============================================================

required_stars = [
    "sig_Pav",
    "bet_Hyi",
    "HD_4332",
    "lam_Tuc"
]


def normalize_star_name(name):
    """
    Convierte, por ejemplo:
        bet_Hyi -> bethyi
        bet Hyi -> bethyi
        betHyi  -> bethyi
    """
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
    )


primary_normalized = (
    tgt_info["Primary"]
    .map(normalize_star_name)
)

print("\nRevisando estrellas en tgt_info:")

for star in required_stars:

    star_normalized = normalize_star_name(star)

    matches = tgt_info[
        primary_normalized == star_normalized
    ]

    if matches.empty:
        print("  %-12s NO EXISTE" % star)

    else:
        print(
            "  %-12s OK -> indice: %s, Primary: %s"
            % (
                star,
                list(matches.index),
                list(matches["Primary"])
            )
        )
print("\nProbando rutils.get_unique_key:")

for star in required_stars:

    try:
        unique_key = rutils.get_unique_key(
            tgt_info,
            star
        )

        print(
            "  %-12s -> %s"
            % (star, unique_key)
        )

    except Exception as error:

        print(
            "  %-12s ERROR -> %s"
            % (star, error)
        )


import pprint


# ============================================================
# Interceptar select_only_bad_target_durations
# ============================================================

original_select_only_bad_target_durations = (
    rpndrs.select_only_bad_target_durations
)


def debug_select_only_bad_target_durations(
        durations,
        tgt_info
    ):

    print("\n" + "=" * 70)
    print("DURATIONS ANTES DEL FILTRO")
    print("=" * 70)

    print("Tipo:", type(durations))

    try:
        print("Numero de elementos:", len(durations))
    except TypeError:
        print("No se pudo calcular la longitud")

    pprint.pprint(durations)

    result = original_select_only_bad_target_durations(
        durations,
        tgt_info
    )

    print("\n" + "=" * 70)
    print("BAD DURATIONS DEVUELTAS")
    print("=" * 70)

    print("Tipo:", type(result))

    try:
        print("Numero de elementos:", len(result))
    except TypeError:
        print("No se pudo calcular la longitud")

    pprint.pprint(result)

    return result


rpndrs.select_only_bad_target_durations = (
    debug_select_only_bad_target_durations
)
# ============================================================
# 5. Ejecutar solamente creacion del script PNDRS
# ============================================================

print("\n" + "=" * 70)
print("Probando save_nightly_pndrs_script para bet_Hyi")
print("=" * 70)

try:

    rpndrs.save_nightly_pndrs_script(
        bet_complete_sequences,
        tgt_info,
        base_path,
        run_local=True
    )

except Exception:

    print("\nERROR REAL AL PROCESAR bet_Hyi:\n")
    traceback.print_exc()

    print("\nLa excepcion ocurrio dentro de:")
    print("rpndrs.save_nightly_pndrs_script")

    raise

else:

    print("\nLa creacion del script PNDRS termino correctamente.")


# ============================================================
# Seleccionar bet_Hyi y su noche completa
# ============================================================

key = (106, "bet_Hyi", "bright")

if key not in complete_sequences:
    raise KeyError(
        "No existe %r en complete_sequences"
        % (key,)
    )

night = complete_sequences[key][0]


# ============================================================
# Conservar todas las secuencias de la misma noche
# ============================================================

night_complete_sequences = {
    sequence_key: sequence_data
    for sequence_key, sequence_data
    in complete_sequences.items()
    if sequence_data[0] == night
}

night_sequences = {
    sequence_key: sequences[sequence_key]
    for sequence_key in night_complete_sequences
    if sequence_key in sequences
}


print("\n" + "=" * 70)
print("SECUENCIAS DE LA NOCHE %s" % night)
print("=" * 70)

print(
    "Cantidad:",
    len(night_complete_sequences)
)

for sequence_key in sorted(
        night_complete_sequences,
        key=str
    ):

    print("\nClave:")
    print(sequence_key)

    print("Secuencia:")
    print(
        night_sequences.get(
            sequence_key,
            "No encontrada"
        )
    )


# ============================================================
# Ejecutar solamente la noche de bet_Hyi
# ============================================================

print("\n" + "=" * 70)
print("CREANDO PNDRS SCRIPT PARA LA NOCHE COMPLETA")
print("=" * 70)

try:

    rpndrs.save_nightly_pndrs_script(
        night_complete_sequences,
        tgt_info,
        base_path,
        run_local=True
    )

except Exception:

    print(
        "\nERROR AL PROCESAR LA NOCHE %s:\n"
        % night
    )

    traceback.print_exc()
    raise

else:

    print(
        "\nLa prueba de la noche %s termino."
        % night
    )