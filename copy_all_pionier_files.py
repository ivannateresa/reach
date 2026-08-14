import os
import re
import shutil
from pathlib import Path


SOURCE_ROOT = Path(
    "/home2/ihernand/Desktop/reach/all_data"
)

DESTINATION_ROOT = Path(
    "/home2/ihernand/Desktop/reach/all_sequences"
)


def is_pionier_file(filename):
    """
    Select PIONIER night logs and RAW FITS files.
    The comparison is case-insensitive.
    """
    lower_name = filename.lower()

    return (
        filename.startswith("PIONI")
        and lower_name.endswith(
            (
                ".nl.txt",
                ".fits",
                ".fits.z",
                ".fits.fz",
                ".fits.gz",
            )
        )
    )


n_copied = 0
n_existing = 0
n_skipped = 0

# rglob searches recursively and also includes files located
# directly inside SOURCE_ROOT.
for old_file in SOURCE_ROOT.rglob("*"):

    if not old_file.is_file():
        continue

    filename = old_file.name

    if not is_pionier_file(filename):
        continue

    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)

    if match is None:
        print("Sin fecha reconocible:", old_file)
        n_skipped += 1
        continue

    fecha = match.group(0)

    destination_folder = DESTINATION_ROOT / fecha
    if not destination_folder.exists():
        destination_folder.mkdir(parents=True)

    new_file = destination_folder / filename

    if new_file.exists():
        # Comprobar si aparentemente es el mismo archivo.
        if new_file.stat().st_size == old_file.stat().st_size:
            n_existing += 1
            continue

        print("WARNING: existe con tamanso diferente:")
        print("  origen :", old_file)
        print("  destino:", new_file)

    shutil.copy2(old_file, new_file)
    n_copied += 1

    print("Copiado:")
    print(" ", old_file)
    print(" ->", new_file)


print("")
print("Resumen")
print("-------")
print("Archivos copiados:", n_copied)
print("Ya existentes:", n_existing)
print("Omitidos:", n_skipped)