from pathlib import Path
import re
import shutil


SOURCE_ROOT = Path(
    "/home2/ihernand/Desktop/reach/all_data"
)

DESTINATION_ROOT = Path(
    "/home2/ihernand/Desktop/reach/all_sequences"
)


def is_raw_fits(path):
    """Return True for supported PIONIER RAW FITS files."""
    name_lower = path.name.lower()

    return (
        path.name.startswith("PIONI")
        and name_lower.endswith(
            (
                ".fits.z",
                ".fits",
                ".fits.fz",
                ".fits.gz",
            )
        )
    )


n_found = 0
n_copied = 0
n_existing = 0
n_replaced = 0
n_without_date = 0


for source_file in SOURCE_ROOT.rglob("*"):

    if not source_file.is_file():
        continue

    if not is_raw_fits(source_file):
        continue

    n_found += 1

    match = re.search(
        r"\d{4}-\d{2}-\d{2}",
        source_file.name,
    )

    if match is None:
        print("Sin fecha:", source_file)
        n_without_date += 1
        continue

    date = match.group(0)

    destination_folder = DESTINATION_ROOT / date
 
    if not destination_folder.exists():
        destination_folder.mkdir(parents=True)

    destination_file = destination_folder / source_file.name

    if destination_file.exists():

        source_size = source_file.stat().st_size
        destination_size = destination_file.stat().st_size

        if source_size == destination_size:
            n_existing += 1
            continue

        print("Reemplazando archivo con tamano diferente:")
        print("  Origen :", source_file)
        print("  Destino:", destination_file)

        shutil.copy2(
            source_file,
            destination_file,
        )

        n_replaced += 1
        continue

    shutil.copy2(
        source_file,
        destination_file,
    )

    n_copied += 1

    print(
        "Copiado:",
        source_file.name,
        "->",
        destination_folder,
    )


print("")
print("Resumen")
print("-------")
print("RAW encontrados en all_data:", n_found)
print("RAW nuevos copiados:", n_copied)
print("RAW ya existentes:", n_existing)
print("RAW reemplazados:", n_replaced)
print("Archivos sin fecha:", n_without_date)