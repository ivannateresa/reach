import csv
import glob
import os
import pickle
import numpy as np
from sys import exit
from shutil import copyfile
from collections import OrderedDict
from datetime import datetime
import re
import shutil



base_path = "/home2/ihernand/Desktop/reach/all_data/*"
all_logs = glob.glob(base_path)

folders = [f for f in all_logs if os.path.isdir(f)]

print(folders)

for path in folders:
    print "Carpeta:", path

    if not os.path.isdir(path):
        continue

    for filename in os.listdir(path):
        print "Archivo:", filename

        if filename.startswith("PIONI") and (
            filename.endswith(".NL.txt") or
            filename.endswith(".fits.Z") or
            filename.endswith(".fits.Z.1") or
            filename.endswith(".NL.txt.1")):

            match = re.search(r"\d{4}-\d{2}-\d{2}", filename)

            if match:
                fecha = match.group()

                folder_path = os.path.join("../Desktop/reach/all_sequences/", fecha)

                # equivalente a exist_ok=True
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                old_file = os.path.join(path, filename)
                new_file = os.path.join(folder_path, filename)

                shutil.move(old_file, new_file)
                print "Movido: {} -> {}".format(filename, folder_path)
