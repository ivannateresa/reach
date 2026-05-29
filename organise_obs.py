"""
Script to parse the ESO run summary and produce a set of complete bright/faint
concatenations, which are then copied to a new directory structure for only
**complete** sequences for simplicity during the following reduction steps.
Sequences are ideally matched automatically, but peculiarities in the way
several sequences were observed necessitate manual intervention (e.g. when
the science and calibrators are out of order, or marked with grades that don't
correspond to their quality per what the logs say.

Each observation has a night log text file describing the details of the
observation, which is parsed to build a summary of all logs. Associated with
this is the raw data, and both are copied to the new directory structure when
sequence matching is complete.

TODO: also extract any log notes, and save the sequence dictionaries assembled
here as pickles for easy reference later during data reduction and calibration.
"""
from __future__ import division, print_function
import csv
import glob
import os
import pickle
import numpy as np
from sys import exit
from shutil import copyfile
from collections import OrderedDict
from datetime import datetime

def clean_name(name):
    return name.strip().replace(" ", "").replace("_", "").lower()

# -----------------------------------------------------------------------------
# Import and separate observation log into nights
# -----------------------------------------------------------------------------
# Find of the text logsa
all_logs = glob.glob("/home2/ihernand/Desktop/reach/all_sequences/*/PIONI*.NL.txt")
all_logs.sort()

# Initialise dictionary to store observations
night_log = OrderedDict()

ref_ids = set()

# Generate an observational log from these text files
for obs_log in all_logs:
    # Get the time of the observation (truncated to seconds)
    # Example filename: PIONI.2017-09-06T08:36:53.372.NL.txt
    yyyymmddhhMMss = obs_log.split("/")[-1][6:25]
    ob_time = datetime.strptime(yyyymmddhhMMss, "%Y-%m-%dT%H:%M:%S")
    #ob_time = datetime.strptime(yyyymmddhhMMss, "%Y-%m-%dT%I_%M_%S")
    
    # Read everything in the file
    with open(obs_log) as file:
        content = file.readlines()

    # Remove newline characters from the 
    content = [row.strip() for row in content]
    
    # Get the night of observation by way of subfolder
    night = obs_log.split("/")[-2]
    
    # 
    ob_type = None
    
    # Construct a list of the station used
    station_ids = ""
    
    # Extract the relevant information
    for row in content:
        # Night grade
        if row[:6] == "Grade:":
            grade = row[-1]
        # Target name (uniform) AND target name (unaltered)
        elif row[:7] == "Target:":
            target = clean_name(row.split("Target:")[-1])
            
            ref_ids.add(row.split(" ")[-1])
            
        # OB - ESO observation ID
        elif row[:3] == "OB:":
            OB = row.split(" ")[-1]
        # Container - ESO concatenation ID
        elif row[:10] == "Container:":
            container = row.split(" ")[-1]
        # Run - the ESO period
        elif row[:4] == "Run:":
            run = row.split(" ")[-1]
        # ---------------------------------------
        # The following are to extract whether the data is of type fringe, 
        # dark, or kappa matrix (i.e. flux splitting ratio). There will be 
        # multiple lines for fringe and kappa in the file, as the text log
        # records on the level of the sequence/target, rather than just listing
        # what it specifically is. As such, we then need to compare the file
        # name and assign a label accordingly
        elif row[:18] == "PIONIER_OBS_FRINGE":
            obfname = row.split("\t")[1]
            
            if obfname in obs_log:
                ob_type = "FRINGE"
                det_mode = row.split("\t")[6]
                det_ndit_dit = row.split("\t")[2]
        
        elif row[:16] == "PIONIER_GEN_DARK":
            obfname = row.split("\t")[1]
            
            if obfname in obs_log:
                ob_type = "DARK"
                det_mode = row.split("\t")[6]
                det_ndit_dit = row.split("\t")[2]
        
        elif row[:17] == "PIONIER_GEN_KAPPA":
            obfname = row.split("\t")[1]
            
            if obfname in obs_log:
                ob_type = "KAPPA"
                det_mode = row.split("\t")[6]
                det_ndit_dit = row.split("\t")[2]
        
        # Extract the station information        
        elif row[:7] == "STATION":
            station_ids += row.split(" ")[-1] + "-"
            
            
    # In addition to the text file, grab the fits file of the data itself, 
    # which has the same path bar ".fits.Z" instead of ".NL.txt"
    ob_fits = obs_log.replace("NL.txt", "fits.Z")
    
    # Strip the final dash of the station IDs
    station_ids = station_ids[:-1]
    
    # Select details of observation to save
    ob_details = [container, OB, target, grade, ob_time, obs_log, run, ob_fits,
                  ob_type, station_ids, det_mode, det_ndit_dit]
    
    # Now store the observation details in the dictionary
    # If night entry exists, append observation
    if night in night_log.keys():
        night_log[night].append(ob_details)
    # Night entry  does not exist, create it
    else:
        night_log[night] = [ob_details]
    
    file.close()

np.savetxt("ref_ids.csv", list(ref_ids), fmt="%s")

# Can check that two datetime objects are close in time by subtracting them   
# datetime1 - datetime2 --> datetime.timedelta(...)
# datetime.timedelta(...).secondsa
# -----------------------------------------------------------------------------
# Read in and separate out the bright and faint sci-cal sequences
# -----------------------------------------------------------------------------
# Read in each sequence
#bright_list_files = ["data/p99_bright.txt", "data/p101_bright.txt"]
#faint_list_files = ["data/p99_faint.txt", "data/p101_faint.txt"]

#bright_list_files = sorted(glob.glob("data/p106_bright.txt"))
#faint_list_files = sorted(glob.glob("data/p106_faint.txt"))

#right_list = []
#faint_list = []

#for bright_list_file in bright_list_files:
#    with open(bright_list_file) as csv_file:
#        for line in csv.reader(csv_file):
#            bright_list.append(line[0].replace(" ", "_"))

#for faint_list_file in faint_list_files:        
#    with open(faint_list_file) as csv_file:
#        for line in csv.reader(csv_file):
#            faint_list.append(line[0].replace(" ", "_"))

# For consistency, remove any underscores
#bright_list = [tgt.replace("_", "") for tgt in bright_list]
#faint_list = [tgt.replace("_", "") for tgt in faint_list]


# -----------------------------------------------------------------------------
# Crear una funcion donde una secuencia puede tener 3 o MENOS calibradores, y elimina los que solo 
# tiene la estrella, es decir, que no tenga calibradores.

 
bright_list_files = ['data/p104_bright.txt']

faint_list_files  = ['data/p104_faint.txt' ]


def get_period_from_filename(filename):
    """
    Extract period from filenames like:
    data/p106_bright.txt
    """
    base = os.path.basename(filename)
    period = base.split("_")[0].replace("p", "")
    return int(period)


 
def build_sequences_for_pickle(sequence_dict, list_files, sequence_label):
    """
    Build the format expected by pipeline later.

    Output:
    {
        (106, 'alfCmi', 'bright'): ['HR2729', 'alfCmi', 'HD63241', 'alfCmi', '27Mon']
    }
    """

    sequences_pickle = OrderedDict()

    for list_file in list_files:
        period = get_period_from_filename(list_file)

        for sci in sequence_dict:
            sequences_pickle[(period, sci, sequence_label)] = sequence_dict[sci]

    return sequences_pickle



def make_sequence(current_science, current_cals):
    seq = []

    for i, cal in enumerate(current_cals):
        if i == len(current_cals) - 1:
            seq.append(cal)
        else:
            seq.append(cal)
            seq.append(current_science)

    return seq


def build_sequences(list_files):

    sequences = {}

    for list_file in list_files:

        current_science = None
        current_cals = []

        with open(list_file) as csv_file:
            reader = csv.reader(csv_file)

            for line in reader:

                if len(line) < 2:
                    continue

                name = clean_name(line[0])
                is_science = line[1].strip().upper()

                if is_science == "TRUE":

                    if current_science is not None and len(current_cals) > 0:
                        sequences[current_science] = make_sequence(
                            current_science,
                            current_cals
                        )

                    current_science = name
                    current_cals = []

                else:
                    current_cals.append(name)
        if current_science is not None and len(current_cals) > 0:
            sequences[current_science] = make_sequence(
                current_science,
                current_cals
            )

    return sequences
 

bright_sequences = build_sequences(bright_list_files)
faint_sequences  = build_sequences(faint_list_files)
# -----------------------------------------------------------------------------


# Order each sequence
#bright_sequences = {}
#faint_sequences = {}



#for i in xrange(0, len(bright_list), 4):
#    bright_sequences[bright_list[i]] = [bright_list[i+1], bright_list[i],
#                                        bright_list[i+2], bright_list[i],
#                                        bright_list[i+3]]
#for i in xrange(0, len(faint_list), 4):    
#    faint_sequences[faint_list[i]] = [faint_list[i+1], faint_list[i],
#                                      faint_list[i+2], faint_list[i],
#                                      faint_list[i+3]]

all_sequences = [bright_sequences, faint_sequences]
seq_label = ["bright", "faint"]

missing_sequences = [(key, "bright") for key in bright_sequences]
missing_sequences.extend([(key, "faint") for key in faint_sequences])
missing_sequences = set(missing_sequences)

# -----------------------------------------------------------------------------
# Crear una funcion donde una secuencia puede tener 3 o MENOS calibradores, y elimina los que solo 
# tiene la estrella, es decir, que no tenga calibradores.

# -----------------------------------------------------------------------------
# Check night by night to see which sequences were completed
# -----------------------------------------------------------------------------
# Observation grades as follow:
# - A: fully within constraints
# - B: mostly within constraints (~10% violation)
# - C: out of constraints, will be repeated
# - D: out of constraints, will not be repeated
# - X: aborted
# - ? or "-": error, will be repeated
# Given this, A and B observations will be accepted
def is_good_grade(new_grade):
    """Tests the grade
    """
    is_good_grade = True
    
    bad_grades = ["C", "D", "-", "?", "X"]
    
    for grade in bad_grades:
        if grade in new_grade:
            is_good_grade = False
            
    return is_good_grade


complete_sequences = {}

# For every night of observations...
for night in night_log.keys():
    print("\n\n---------------------------------------------------------")
    print(night, "\n---------------------------------------------------------")
    # For every sequence (i.e. bright, faint)...

    for seq_i, sequence in enumerate(all_sequences):
        print("-------", seq_label[seq_i], "-------")
        # For every science target...

        for sci in sequence:
            print("")
            print(sci, end="   ")
            # Step through the nightly observations attempting to match
            # the CAL-SCI-CAL-SCI-CAL sequences where each CAL or SCI has many
            # different files (N exposures, darks, kappa matrices)
            # Do this by building up the list of observations until you either
            # reach the end of the final calibrator (i.e. complete sequence) or 
            # something else happens (i.e. broken sequence)
            ob_i = 0            # The ith observation that night
            tgt_i = 0           # The ith target in the CAL-SCI sequence
            concatenation = []  # Current list of obs from CAL-SCI sequence
            print(sci)
            # For every observation in the night...
            while ob_i < len(night_log[night]):
                # Get the grade of the observation to be considered
                grade = night_log[night][ob_i][3]
                obs_tar = night_log[night][ob_i][2] 

                
                sequence_added_to = True
                
                # First element of sequence
                # - Add to concatenation and increment to next observation
                if (len(concatenation) == 0 and sequence[sci][tgt_i] in obs_tar
                    and is_good_grade(grade)):
                    concatenation.append(night_log[night][ob_i])
                    ob_i += 1
                    print("0", end="")
                
                # Continuation of current target
                # - Add to concatenation and increment to next observation
                elif (len(concatenation) > 0 
                    and sequence[sci][tgt_i] in obs_tar
                    and is_good_grade(grade)):
                    concatenation.append(night_log[night][ob_i])
                    ob_i += 1
                    print("1", end="")
                    
                # No more obs for current target, check next in sequence
                # - If not at last ob, add to concatenation and increment ob
                elif (len(concatenation) > 0 and tgt_i + 1 < len(sequence[sci])
                    and sequence[sci][tgt_i+1] in obs_tar
                    and is_good_grade(grade)):
                    concatenation.append(night_log[night][ob_i])
                    ob_i += 1
                    tgt_i += 1
                    print(2, end="")
                    
                else:
                    sequence_added_to = False
                    
                # Do a check here to see if we're now on the last ob/target
                # Without this, we would have previously incremented ob_i and
                # head straight into the next if statement, exiting before we
                # can properly handle the final observation of the night 
                # (typically a dark)
                if (len(concatenation) > 0 
                    and sequence[sci][tgt_i] in obs_tar
                    and is_good_grade(grade)
                    and ob_i + 1 == len(night_log[night])):
                    # No need to increment here, as we'll exit straight away
                    # once we enter the next if statement
                    concatenation.append(night_log[night][ob_i])
                    print(3, end="")
                    
                # Now we should either have finished a sequence, or found a
                # broken sequence. Two ways to complete a sequence, otherwise
                # we consider it broken and move on:
                #  1 - On last target of sequence, next is unrelated
                #  2 - On last target of sequence, reached end of night
                # If the last target in the sequence, and we either did not add
                # to anything or we are at the end of the night
                if (tgt_i + 1 == len(sequence[sci]) 
                    and (not sequence_added_to
                    or ob_i + 1 == len(night_log[night]))):
                    # Note that there is the case where the grade is bad on the
                    # final target
                    print(4, end="")
                    
                    all_grades = [ob[3] for ob in concatenation]
                    grade = "".join(all_grades)
                    
                    # Record the period of the sequence
                    period = int(concatenation[0][6].split(".")[0])
                    
                    end_time = concatenation[-1][4].isoformat()
                    key = (period, sci, seq_label[seq_i])
                    complete_sequences[key] = (night, grade, concatenation)
                    print(" [DONE, %s, # Obs: %i]" % (grade, len(grade)))
                    
                    if (sci, seq_label[seq_i]) in missing_sequences:
                        missing_sequences.remove((sci, seq_label[seq_i]))
                    
                    tgt_i = 0
                    
                    # Only add to the counter if we added to the sequence
                    if not sequence_added_to:
                        ob_i += 1
                    
                    concatenation = [] 
                    
                # We didn't add to the sequence, and we're not on the last 
                # target of the sequence - consider the concatenation broken
                # and move on
                elif not sequence_added_to:
                    
                    if len(concatenation) > 0:
                        print(
            "\nFAILED:",
            "night =", night,
            "| sci =", sci,
            "| seq =", seq_label[seq_i],
            "| expected =", sequence[sci][tgt_i],
            "| found =", night_log[night][ob_i][2],
            "| grade =", grade,
            "| good_grade =", is_good_grade(grade),
            "| ob_i =", ob_i,
            "| tgt_i =", tgt_i,
            "| full_expected_seq =", sequence[sci]
            )
                    
                    # Here is where we decide whether to move to the next 
                    # observation. We should *not* move on if:
                    # - The observation is an earlier member of the 
                    #   concatenation currently in progress *and* has a good
                    #   grade
                    # - In all other cases we want to increment the ob_i
                    #   counter: it is only when you have adjacent broken
                    #   sequences that things get messy
                    if not is_good_grade(grade):
                        ob_i += 1
                        tgt_i = 0
                        
                        if len(concatenation) > 0:
                            print("A")
                    elif (len(concatenation) > 0 
                        and night_log[night][ob_i][2] in sequence[sci]):
                        tgt_i = 0
                        
                        if len(concatenation) > 0:
                            print("B")
                        
                    else:
                        ob_i += 1
                        tgt_i = 0
                        
                        if len(concatenation) > 0:
                            print("C")
                    
                    concatenation = []             
                    
        print("\n") 

# -----------------------------------------------------------------------------
# Manually account for any out of sequence, but complete, sequences
# -----------------------------------------------------------------------------
# Note: this code should be in a imported file (ideally)

# First star: del Pav. This was the only sequence observed on 27/08/17, and 
# seems to have a mismatch between what the logs *say* are good observations,
# and what are marked as good per the ESO grades (i.e. A or B). The HR7732 data
# is fine per the logs, but marked as C. Ignore the first saturated del Pav
# sequence, but take everything else, mindful that the guiding was lost during
# the final del Pav exposure and we have to remove those observations where it
# was lost for all telescopes/baselines

# First calibrator, HR7732 --> keep
#concatenation = night_log["2017-08-27"][:6]

# Take kappa matrix and darks of first del Pav exposure, but not the science
# Then take everything else except seven exposures where the guiding was lost
# (np.mean(vis2.flatten()) == 0), and everything in between
#concatenation.extend(night_log["2017-08-27"][11:40])
#concatenation.extend(night_log["2017-08-27"][47:])

# Determine grades, period, and end time, then save to dict as usual
#grade = "".join([observation[3] for observation in concatenation])
#period = int(concatenation[0][6].split(".")[0])
#end_time = concatenation[-1][4].isoformat()

#key = (period, "delPav", "faint")
#print(key in complete_sequences)
#complete_sequences[key] = ("2017-08-27", grade, concatenation)

#missing_sequences.remove(("delPav", "faint"))

# Second star: lam Sgr. First calibrator observed within requirements, but 1st
# science was not. Everything else per the log seems to be okay, with HR6838
# observed again to bracket the science observations.
#concatenation = night_log["2017-08-26"][:6]
#concatenation.extend(night_log["2017-08-26"][17:51])

# Determine grades, period, and end time, then save to dict as usual
#grade = "".join([observation[3] for observation in concatenation])
#period = int(concatenation[0][6].split(".")[0])
#end_time = concatenation[-1][4].isoformat()

#key = (period, "lamSgr", "faint")
#complete_sequences[key] = ("2017-08-26", grade, concatenation)

#missing_sequences.remove(("lamSgr", "faint"))

# Third star: Tau Ceti (bright) in period 99, which is missed because of the
# assumption that each star has a single unique bright and faint sequence, 
# which breaks down for Tau Cet given we removed a bad calibrator from it. It's
# easier to account for this here than changing the data structures used above.
#concatenation = night_log["2017-08-26"][67:101]

# Determine grades, period, and end time, then save to dict as usual

#grade = "".join([observation[3] for observation in concatenation])
#period = int(concatenation[0][6].split(".")[0])
#end_time = concatenation[-1][4].isoformat()

#key = (period, "TauCet", "bright")
#complete_sequences[key] = ("2017-08-26", grade, concatenation)

#missing_sequences.remove(("TauCet", "bright"))

# -----------------------------------------------------------------------------
# P102 Stars
# -----------------------------------------------------------------------------
# del Eri (bright)
# All obs are fine, save first which had grade of "_"
#concatenation = night_log["2018-11-25"]

#grade = "".join([observation[3] for observation in concatenation])
#period = int(concatenation[0][6].split(".")[0])
#end_time = concatenation[-1][4].isoformat()

#key = (period, "delEri", "bright")
#complete_sequences[key] = ("2018-11-25", grade, concatenation)

# del Eri (faint)
# Observations of first cal (HD16970) are bad (grade "X"), and the first delEri
# observations is "_". Everything after this is good, and the first cal is done
# again successfully at the end
#concatenation = night_log["2018-11-26"][7:41]

#grade = "".join([observation[3] for observation in concatenation])
#period = int(concatenation[0][6].split(".")[0])
#end_time = concatenation[-1][4].isoformat()

#key = (period, "delEri", "faint")
#complete_sequences[key] = ("2018-11-26", grade, concatenation)

# omi2 Eri (faint)
# All obs are fine
#concatenation = night_log["2018-11-26"][41:75]

#grade = "".join([observation[3] for observation in concatenation])
#period = int(concatenation[0][6].split(".")[0])
#end_time = concatenation[-1][4].isoformat()

#key = (period, "omi2Eri", "faint")
#complete_sequences[key] = ("2018-11-26", grade, concatenation)

# omi2 Eri (bright)
# All obs are fine
#concatenation = night_log["2018-11-26"][75:]

#grade = "".join([observation[3] for observation in concatenation])
#period = int(concatenation[0][6].split(".")[0])
#end_time = concatenation[-1][4].isoformat()

#key = (period, "omi2Eri", "bright")
#complete_sequences[key] = ("2018-11-26", grade, concatenation)

# -----------------------------------------------------------------------------
# Summarise keys for easy inspection
# -----------------------------------------------------------------------------
obs_keys = complete_sequences.keys()
obs_keys.sort()

print("\n\n-------------------------\nSummary\n-------------------------")
print("%i/%i Unique Complete Sequences\n" % (len(obs_keys), 
                                           (len(bright_sequences) + 
                                            len(faint_sequences))))
print("%-16s%-12s%-12s%-22s%-10s\n" % ("Period", "Target", "Sequence", "Time", 
                                       "Grade")) 

for ob in obs_keys:
    print("%6i%-12s%-12s%-22s%-10s" % (ob[0], ob[1], ob[2], 
                                       complete_sequences[ob][0], 
                                       complete_sequences[ob][1])) 
   

print("\n\n----------------------\nMissing Sequences\n----------------------")  
print("%i Missing Sequences\n" % len(missing_sequences))
    
for sequence in missing_sequences:
    print("%-12s%-12s" % (sequence[0], sequence[1]))

# -----------------------------------------------------------------------------
# Copy the completed sequences to a new directory structure
# -----------------------------------------------------------------------------
print("\n\n------------------------\nCopying Files\n------------------------")
new_path = "/home2/ihernand/Desktop/reach/complete_sequences"

#new_path = "/home2/ihernand/Desktop/test_reach_code/reach/complete_sequences"


# Create new night subfolders if not already in existence
all_nights = glob.glob("/home2/ihernand/Desktop/reach/all_sequences/*/")
all_nights = [night.replace("all_sequences", "complete_sequences") 
              for night in all_nights]

for night in all_nights:
    if not os.path.exists(night):
        os.makedirs(night)
    
    
# For every observation in complete_sequences, copy it to the corresponding 
# folder in all_sequences if it isn't already there.
n_files_copied = 0
bytes_copied = 0

all_complete_obs = set()
duplicates = []

for sequence in complete_sequences:
    print("Copying data for: %s, %s, %s, %s" % (sequence[0], sequence[1],
                                            sequence[2], 
                                            complete_sequences[sequence][0]))
                                            
    for observation in complete_sequences[sequence][2]:
        # Create the new file paths for the logs and data
        new_nl = observation[5].replace("all_sequences", "complete_sequences")
        new_fits = observation[7].replace("all_sequences", 
                                          "complete_sequences")
        
        if new_nl in all_complete_obs or new_fits in all_complete_obs:
            duplicates.append( (sequence[:4]) )
        all_complete_obs.add(new_nl)
        all_complete_obs.add(new_fits)
        
        if not os.path.exists(new_nl):
            copyfile(observation[5], new_nl)
            n_files_copied += 1
            bytes_copied += os.path.getsize(new_nl)
        
        if not os.path.exists(new_fits):
            copyfile(observation[7], new_fits)
            n_files_copied += 1
            bytes_copied += os.path.getsize(new_fits)

print("Finished! %i files (%0.2f GB) copied" % (n_files_copied, 
                                                bytes_copied/1024**3))
                                                
# -----------------------------------------------------------------------------
# Save details
# -----------------------------------------------------------------------------
pkl_nightlog = open("data/pionier_night_log.pkl", "wb")
pickle.dump(night_log, pkl_nightlog)
pkl_nightlog.close()

pkl_obslog = open("data/pionier_observing_log.pkl", "wb")
pickle.dump(complete_sequences, pkl_obslog)
pkl_obslog.close()

sequences = OrderedDict()
sequences.update(build_sequences_for_pickle(bright_sequences, bright_list_files, "bright"))
sequences.update(build_sequences_for_pickle(faint_sequences, faint_list_files, "faint"))

pkl_sequences = open("data/sequences.pkl", "wb")
pickle.dump(sequences, pkl_sequences)
pkl_sequences.close()

