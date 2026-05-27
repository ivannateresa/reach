"""Module to handle interacting with the PIONIER data reduction pipeline, pndrs
"""
from __future__ import division, print_function
import os
import sys
import glob
import datetime
import numpy as np
import pandas as pd
import reach.diameters as rdiam
import reach.plotting as rplt
import reach.utils as rutils
from shutil import copyfile, rmtree
from astropy.io import fits
from astropy.time import Time
from collections import OrderedDict, Counter

# -----------------------------------------------------------------------------
# pndrs Affiliated Functions
# -----------------------------------------------------------------------------
base_path = "/home2/ihernand/Desktop/reach/all_sequence"

def clean_name_for_match(name):
    name = str(name).strip()
    name = name.replace(" ", "")
    name = name.replace("_", "")
    name = name.lower()
    return name

def save_nightly_ldd(sequences, complete_sequences, tgt_info, 
                pred_ldd, e_pred_ldd,
                base_path,
                dir_suffix="_v3.94_abcd", run_local=False):
    """This is a function to create and save the oiDiam.fits files referenced
    by pndrs during calibration. Each night of observations has a single such
    file with the name formatted per YYYY-MM-DD_oiDiam.fits containing an
    empty primary HDU, and a fits table with LDD and e_LDD for each star listed
    alphabetically.
    
    Parameters
    ----------
    sequences: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        lists of the targets in said CAL1-SCI1-CAL2-SCI2-CAL3 sequence. 
    
    complete_sequences: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        [night, grade, [[container, OB, target, grade, ob_time, obs_log, run, 
                         ob_fits],...]
    
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
    
    base_path: str
        String filepath where the calibrated data is stored.
    
    dir_suffix: str
        String suffix on the end of each folder of calibrated data
    
    run_local: bool
        Boolean indicating whether the pipeline is being run locally, and to
        save files instead within reach/test/ for inspection.
    
    ldd_col: str
        Column of predicted LDD from tgt_info to use.
    
    e_ldd_col:
        Column of predicted e_LDD from tgt_info to use.
    """
    print("\n", "-"*79, "\n", "\tSaving Nightly oidiam files\n", "-"*79)
    nights = OrderedDict()
    
    # Get nightly sets of what targets have been observed
    for seq in complete_sequences:
        night = complete_sequences[seq][0]
        
        sequence = [star.replace("_", "").replace(".","").replace(" ", "") 
                    for star in sequences[seq]]
        
        if night not in nights:
            nights[night] = set(sequence)
        else:
            nights[night].update(sequence)
    
    print("Writing oiDiam.fits for %i nights" % len(nights))
    
    diam_files_written = 0
    
    # For every night, construct a record array/fits file of target diameters
    # This record array takes the form:
    # TARGET_ID, DIAM, DIAMERR, HMAG, KMAG, VMAG, ISCAL, TARGET, INFO
    #   >i2      >f8     >f8    >f8   >f8   >f8    >i4    s8     s18
    # Where TARGET_ID is simply an integer index (one indexed), ISCAL is a 
    # boolean value of either 0 or 1, and TARGET is the name of the target. 
    # The targets are ordered by name, but sorted in ascii order (i.e. all 
    # numbers, then all capital letters, then all lower case letters). Unclear 
    # how significant this is. Only Hmags have been populated for some stars, 
    # though it is unclear what impact this has on the calibration.
    for night in nights:
        
        failed = False
        
        ids = []
        # Grab the primary IDs
        # Note that several stars are observed multiple times under different
        # primary IDs, so we need to check HD and Bayer IDs as well
        for star in nights[night]:
    
            prim_id = tgt_info[tgt_info["Primary"]==star].index
            
            if len(prim_id)==0:
                prim_id = tgt_info[tgt_info["Bayer_ID"]==star].index
                
            if len(prim_id)==0:
                prim_id = tgt_info[tgt_info["HD_ID"]==star].index
            
            try:
                assert len(prim_id) > 0
            except:
                print("...failed on %s, %s" % (night, star))
                failed = True
                break
            ids.append(prim_id[0])    
            
        if failed:
            continue
            
        # Sort the IDs
        ids.sort()   
        
        # We need to compile entries for multiple targets with the same name
        # due to the non-unique/inconsistent IDs initially sent to ESO. These
        # are stored in the following columns of the input table.
        ref_ids = ["Ref_ID_1", "Ref_ID_2", "Ref_ID_3"]
        
        recs = []
        
        # For each non-null reference ID, collate magnitude, LDD, and sci/cal
        # Rename the reference ID column in the pandas dataframe, then stack
        for ref_id in ref_ids:
            
            rec = tgt_info.loc[ids][tgt_info.loc[ids][ref_id].notnull()]
            rec = rec[["Hmag", "Kmag", "Vmag", "Science", ref_id, "LDD_rel"]]
            
            # Insert the diameters - these are now coming from a separate data
            # structure to facilitate potential bootstrapping. The variable
            # appropriate_ids are only those IDs found to have the given ref_id
            # since the ldd data structures don't know about this
            appropriate_ids = rec.index.values

            rec.insert(0,"pred_LDD", pred_ldd[appropriate_ids].values)    
            rec.insert(1,"e_pred_LDD", e_pred_ldd[appropriate_ids].values[0])     
                       
            rec.rename(columns={ref_id:"Ref_ID"}, inplace=True)
            rec.rename(columns={"LDD_rel":"INFO"}, inplace=True)
            
            if len(rec) > 0:
                recs.append(rec.copy(deep=True))

        rec = pd.concat(recs)
        
        # Replace any nans with zeroes to keep pndrs from throwing a
        # Floating point interrupt (SIGFPE) error
        rec["pred_LDD"].where(~np.isnan(rec["pred_LDD"].values), 1, inplace=True)
        rec["e_pred_LDD"].where(~np.isnan(rec["e_pred_LDD"].values), 0.1, inplace=True)
        
        cols_to_check = ["Hmag", "Kmag", "Vmag"]
        for col in cols_to_check:
            rec[col].where(~np.isnan(rec[col].values), 0, inplace=True)

        # Invert, as column is for calibrator status
        rec.Science =  np.abs(rec.Science - 1)
        #rec["INFO"] = np.repeat("(V-W3) diameter from Boyajian et al. 2014",
                                #len(rec))
        
        rec.insert(0,"TARGET_ID", np.arange(1,len(rec)+1))
        
        max_id = np.max([len(id) for id in rec["Ref_ID"]])
        max_info = np.max([len(str(info)) for info in rec["INFO"]])
        
        formats = "int16,float64,float64,float64,float64,float64,int32,a%s,a%s"
        formats = formats % (max_id, max_info)
        
        names = "TARGET_ID,DIAM,DIAMERR,HMAG,KMAG,VMAG,ISCAL,TARGET,INFO"
        rec = np.rec.array(rec.values.tolist(), names=names, formats=formats)
        
        # Construct a fits/astopy table in this form
        hdu = fits.BinTableHDU.from_columns(rec)
        
        hdu.header["EXTNAME"] = ("OIU_DIAM", 
                                 "name of this binary table extension")
    
        # Save the fits file to the night directory
        if not run_local:
            dir = base_path + "%s%s/%s/" % (night, dir_suffix, night)
        else:
            dir = "test/"
        
        if os.path.exists(dir):
            fname = dir + "/" + night + "_oiDiam.fits" 
            hdu.writeto(fname, output_verify="warn", overwrite=True)
            
            # Done, move to the next night
            print("...wrote %s, %s" % (night, nights[night]))
            diam_files_written += 1
        else:
            # The directory does not exist, flag
            print("...directory '%s' does not exist" % dir)
    print("%i oiDiam.fits files written" % diam_files_written)    
    return nights


def load_bad_baselines_log():
    """Loads in the text file recording any bad baselines, where each entry
    has the form: (Period,ID,concatenation,station,start,end)
    
    Returns
    -------
    bad_baseline_dict: dict
        Dictionary mapping keys of string nights to (station_id, mjd1, mjd2)
    """
    bad_baseline_file = "data/bad_baselines.txt"
    
    # Load the file
    bad_baselines = np.loadtxt(bad_baseline_file, str, "#", " ")
    
    # Format to a dict
    if len(bad_baselines.shape) == 1:
        bad_baseline_dict = {bad_baselines[0]: [bad_baselines[4], 
                             float(bad_baselines[5]), float(bad_baselines[6])]}
    else:
        bad_baseline_dict = {baseline_entry[0]: [baseline_entry[4],
                            float(baseline_entry[5]), float(baseline_entry[5])]
                             for baseline_entry in bad_baselines}
                             
    return bad_baseline_dict


def save_nightly_pndrs_script(complete_sequences, tgt_info, 
            base_path,
            dir_suffix="_v3.94_abcd", run_local=False):
    """This is a function to create and save the pndrs script files referencedF
    by pndrs during calibration. Each night of observations has a single such
    file with the name formatted per YYYY-MM-DD_pndrsScript.i containing a list
    of pndrs commands to run in order to customise the calibration procedure.
    
    Important here are the following commands:
        - Ignore some observations: oiFitsFlagOiData
        - Split the night: oiFitsSplitNight

    Parameters
    ----------
    complete_sequence: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        [night, grade, [[container, OB, target, grade, ob_time, obs_log, run, 
                         ob_fits],...]
    
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
    
    base_path: str
        String filepath where the calibrated data is stored.
    
    dir_suffix: str
        String suffix on the end of each folder of calibrated data
    
    run_local: bool
        Boolean indicating whether the pipeline is being run locally, and to
        save files instead within reach/test/ for inspection.
    """
    print("\n", "-"*79, "\n", "\tSaving Nightly pndrs Scripts\n", "-"*79)
    
    # Figure out what targets share nights
    # Of the form nights[night] = [mjd1, mjd2, ..., mjdn]
    sequence_times = {}
    
    for seq in complete_sequences.keys():
        # Get the string representing the night, YYYY-MM-DD
        night = complete_sequences[seq][0]
        
        # Get the datetime objects representing the first and last observations
        # of each sequence, and add or subtract a small increment as to bracket
        # the entire sequence between the time range. Convert these to MJD.
        delta = datetime.timedelta(seconds=10)
        first_ob = Time(complete_sequences[seq][2][0][4] - delta).mjd
        last_ob = Time(complete_sequences[seq][2][-1][4] + delta).mjd
        
        if night not in sequence_times:
            sequence_times[night] = [first_ob, last_ob]
        else:
            sequence_times[night] += [first_ob, last_ob]
            sequence_times[night].sort()
    
    # These lines are written to YYYY-MM-DD_pndrsScript.i alongside the MJD
    # to split upon
    line_split_1 = 'yocoLogInfo, "Split night to isolate SCI-CAL sequences";'
    line_split_2 = 'oiFitsSplitNight, oiWave, oiVis2, oiVis, oiT3, tsplit=cc;'
    
    # These lines are written to exclude bad calibrators, with the variable
    # 'startend' being a list with an MJD range to exclude
    line_exclude_1 = 'yocoLogInfo,"Ignore bad calibrators";'
    line_exclude_2 = ('oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis,' 
                      'tlimit=startend;')
    
    # These lines are written when excluding bad baselines based on station 
    # number and MJD
    line_bad_bl_1 = 'yocoLogInfo,"Ignore bad baselines";'
    line_bad_bl_2 = ('oiFitsFlagOiData, oiWave, oiArray, oiVis2, oiT3, oiVis,' 
                     'base=station, tlimit=startend;')
    
    # Get the record of sequences with bad baselines
    bad_baseline_dict = load_bad_baselines_log()
    
    pndrs_scripts_written = 0
    no_script_nights = 0
    
    # Get a list of the target durations
    durations = calculate_target_durations(complete_sequences)
    bad_durations = select_only_bad_target_durations(durations, tgt_info)
    
    for night in sequence_times:
        # Save the fits file to the night directory
        if not run_local:
            dir = base_path + "%s%s/%s/" % (night, dir_suffix, night)
        else:
            dir = "test"
        
        # Make the directory if it does not exist
        if not os.path.exists(dir):
            os.mkdir(dir)
            
        # It is only meaningful to write a script if we need to split the night
        # (i.e. if more than one sequence has been observed, that is there are
        # 4 or more MJD entries) or have bad calibrators/baselines to exclude 
        if (len(sequence_times[night]) <= 2 and len(bad_durations[night]) < 1
            and night not in bad_baseline_dict):
            no_script_nights += 1
            continue     
        
        # This night requires a script to be written. When splitting the night,
        # we can neglect the first and last times as there are no observations
        # before or after these times respectively, and we only need one of any
        # pair of star1 end MJD and star2 start MJD              
        fname = dir + "/" + night + "_pndrsScript.i" 
        
        with open(fname, "w") as nightly_script:
            # Split the night
            if len(sequence_times[night]) > 2:
                nightly_script.write(line_split_1 + "\n")
                cc = "cc = %s;\n" % sequence_times[night][1:-1:2]
                nightly_script.write(cc)
                nightly_script.write(line_split_2)
            
            # Rule out bad calibrators
            # Note that this currently assumes only one bad calibrator per
            # science target - fix is to use star_i in string formatting
            if len(bad_durations[night]) >= 1:
                for star_i, bad_cal in enumerate(bad_durations[night]):
                    nightly_script.write(line_exclude_1 + "\n")
                    startend = "startend = %s;\n" % bad_cal[1:]
                    nightly_script.write(startend)
                    nightly_script.write(line_exclude_2 + "\n")
                    
            # Ignore observations with bad baselines using station ID and MJD
            if night in bad_baseline_dict:
                nightly_script.write(line_bad_bl_1 + "\n")
                startend = "startend = %s;\n" % bad_baseline_dict[night][1:]
                nightly_script.write(startend)
                station = 'station = "*%s*";\n' % bad_baseline_dict[night][0]
                nightly_script.write(station)
                nightly_script.write(line_bad_bl_2 + "\n")
        
        # Done, move to the next night
        print("...wrote %s, night split into %s, bad calibrators: %s" 
              % (night, len(sequence_times[night])//2, 
                 len(bad_durations[night])))
        pndrs_scripts_written += 1
            
    print("%i pndrs.i scripts written" % pndrs_scripts_written)
    print("%i no script nights" % no_script_nights)        



def calculate_target_durations(complete_sequences):
    """For each night of observations, return the start and end time of 
    *sequential* observations associated with a given target.
    
    A typical CAL1-SCI1-CAL2-SCI2-CAL3 sequence observes each target 5 times 
    before moving on to the next target in the sequence. This function gets
    the first and last times of each block for the purpose of later excluding 
    bad calibrators.
    
    Parameters
    ----------
    complete_sequences: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        [night, grade, [[container, OB, target, grade, ob_time, obs_log, run, 
                         ob_fits],...]
    
    Returns
    -------
    sequence_durations: dict
        Output from calculate_target_durations, a dict mapping nights to start
        and end times for each target: durations[night] = [target, start, end]
    """
    # Initialise results dict
    sequence_durations = {}
    
    # Time difference to go before start of first observations, or after end of
    # last observation
    delta = datetime.timedelta(seconds=10)
    
    for seq in complete_sequences.keys():
        # Get a mapping of all target IDs to their times
        times = [(ob[2], ob[4]) for ob in complete_sequences[seq][2]]
        
        durations = [[times[0][0], Time(times[0][1] - delta).mjd, 0]]
        
        night = complete_sequences[seq][0]
        
        tgt_i = 0
        
        for (tgt, time) in times:
            # Same target
            if tgt == durations[tgt_i][0]:
                # Update the end time
                durations[tgt_i][2] = Time(time + delta).mjd
            
            # We've moved on
            else:
                tgt_i += 1
                durations.append([tgt, Time(time - delta).mjd, 0])
            
        # All done
        if night in sequence_durations.keys():
            sequence_durations[night] += durations
        else:
            sequence_durations[night] = durations
        
    return sequence_durations


def select_only_bad_target_durations(sequence_durations, tgt_info):
    """Takes the output of calculate_target_durations, and compares to the 
    target quality values in tgt_info, returning only durations for only those
    targets which we wish to exclude from the calibration process.
    
    Parameters
    ----------
    sequence_durations: dict
        Output from calculate_target_durations, a dict mapping nights to start
        and end times for each target: durations[night] = [target, start, end]
        
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
        
    Returns
    -------
    bad_durations: dict
        Dict of same form as sequence_durations, but containing only the 
        calibrators we wish to exclude.
    """
    # Initialise results dict
    bad_durations = {}
    
    for night in sequence_durations:
        bad_durations[night] = []
        
        for star in sequence_durations[night]:
            # Get the star info, making sure to check primary, bayer, and HD
            # IDs given the non-unique IDs used
            prim_id = rutils.get_unique_key(tgt_info, star[0])
            
            # Check if it is a bad calibrator, and if so add to return dict
            if tgt_info.loc[prim_id[0]]["Quality"] == "BAD":
                bad_durations[night].append(star)
                
    return bad_durations


def reduce_all_observations(base_path):
    """Removes existing reduced and calibrated data, then runs pndrsReduce.
    
    Parameters
    ----------
    base_path: string
        Base directory housing the data.
    """
    print("\n", "-"*79, "\n", "\tDeleting old reduced/calibrated data\n", 
          "-"*79)
          
    # Delete existing reduced/calibrated data
    to_delete = glob.glob(base_path + "*_*/")
    to_delete.sort()
    
    for folder in to_delete:
        os.system("rm -rf %s" % folder)
        print("Deleted %s" % folder)
        
    # Run reduction using pndrsReduce
    to_reduce = glob.glob(base_path + "*/")
    to_reduce.sort()
    
    for folder in to_reduce:
        print("\n", "-"*79, "\n", "Reducing %s\n" % folder, "-"*79)
        os.system("cd %s; pndrsReduce" % folder)
        

def calibrate_all_observations(reduced_data_folders, bootstrap_i, 
                               results_path):
    """Calls the PIONIER data reduction pipeline for each folder of reduced
    data from within Python.
    
    Parameters
    ----------
    reduced_data_folders: string array
        List of folder paths to run the calibration pipeline on
    """
    # List to record times for the start and end of each night to calibrate
    times = []
    
    # Run the PIONIER calibration pipeline for every folder with reduced data
    # TODO: capture the output and inspect for errors
    for night_i, ob_folder in enumerate(reduced_data_folders):
        # Record the start time
        times.append(datetime.datetime.now())    
    
        # Navigate to the night folder and call pndrsCalibrate from terminal
        night = ob_folder.split("/")[-2].split("_")[0]
        print("\nCalibrating %s, night %i/%i..." 
              % (night, night_i+1, len(reduced_data_folders)), end="")
        sys.stdout.flush()
        os.system("(cd %s; pndrsCalibrate >> cal_log_%0.4i.txt)" 
                  % (ob_folder, bootstrap_i))
        
        # Record and the end time and print duration
        times.append(datetime.datetime.now()) 
        cal_time = (times[-1] - times[-2]).total_seconds() 
        print("calibrated in %02d:%04.1f min" 
              % (int(np.floor(cal_time/60.)), cal_time % 60.))
        
        # Move oifits files back to central location (reach/results by default)
        move_sci_oifits(ob_folder, results_path, bootstrap_i)
    
    # All nights finished, print summary          
    total_time = (times[-1] - times[0]).total_seconds()    
    print("\nCalibration finished, %i nights in %02d:%04.1f\n" 
          % (len(reduced_data_folders),int(np.floor(total_time/60.)), 
             total_time % 60.))
        

def move_sci_oifits(obs_path, results_path, bootstrap_i):
    """Used to collect the calibrated oiFits files of all science targets after
    running the PIONIER data reduction pipeline. 
    
    Parameters
    ----------
    obs_path: string
        Base directory, will move any SCI_oifits files one directory deeper.
    
    new_path: string
        Folder to move the results to.
        
    bootstrap_i: int
        Integer count for the ith bootstrapping iteration
    """
    sci_oi_fits = glob.glob(obs_path + "/*SCI*oidataCalibrated.fits")
    
    #print("\n", "-"*79, "\n", "\tCopying complete sequences\n", "-"*79)
    
    for files_copied, oifits in enumerate(sci_oi_fits):
        # Make the folder if it doesn't exist
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        
        # Update the filename to keep copies of all potential bootstraps
        fname = oifits.split("/")[-1].replace(".fits", 
                                              "_%02i.fits" % bootstrap_i)

        print("...copying %s as %s" % (oifits.split("/")[-1], fname))
            
        copyfile(oifits, results_path + fname)
        files_copied += 1
    
    print("%i files copied" % files_copied)
    

def initialise_interferograms(complete_sequences, base_path, n_ifg=5,
                              do_random_ifg_sampling=True):
    """Initialises interferograms for calibration by sampling from available 
    files and moving those selected to a subdirectory where pndrsCalibrate will
    be run. This involves random sampling with repeats, and renaming of files
    to account for there now potentially being duplicates.
    
    Parameters
    ----------
    complete_sequences: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        [night, grade, [[container, OB, target, grade, ob_time, obs_log, run, 
                         ob_fits],...]
    
    base_path: str
        String filepath where the calibrated data is stored.
        
    n_ifg: int
        Number of random samples with repeats to make of each target. Defaults
        to 5.
    
    do_random_ifg_sampling: bool
        Boolean indicating whether to randomly sample from the interferograms 
        or to use all data available.
    """
    print("\n", "-"*79, "\n", "\tInitialising Interferograms\n", "-"*79)
    
    # Clean out any old files before we get into the main loop - we can't do it
    # within the main loop itself, otherwise we'll potentially be deleting 
    # sequence from the same night that have already been sampled for this 
    # iteration of the bootstrapping. To ensure any deletions don't affect our
    # ability to run in "parallel" by calibrating different sets of nights
    # separately, get the nights from complete_sequences[seq][0]
    total_old_files = 0
    
    nights = [complete_sequences[seq][0] for seq in complete_sequences.keys()]
    nights = list(set(nights))
    nights.sort()
    
    for night in nights:
        night_folder = base_path % night
        bootstrapping_folder = night_folder + "%s/" % night
        
        old_files = glob.glob(bootstrapping_folder + "PIONI*")
        
        print("Deleting %i files from: %s" % (len(old_files), 
                                              bootstrapping_folder))
        for old_file in old_files:
            os.remove(old_file)
            total_old_files += 1
        
    print("\nRemoved %i old files \n" % total_old_files)

    # For every sequence, perform bootstrapping at the interferogram level
    for seq in complete_sequences.keys():
        night = complete_sequences[seq][0]
        night_folder = base_path % night
        bootstrapping_folder = night_folder + "/%s/" % night
        
        # Collect interferograms of the same target together, select N randomly
        # with repeats from these, copy to the subdirectory and rename, then
        # proceed to the next target
        print(bootstrapping_folder)
        #if not os.path.exists(bootstrapping_folder):
        #    os.makedirs(bootstrapping_folder)
        ifgs = sample_interferograms(complete_sequences[seq][2], n_ifg, 
                                     do_random_ifg_sampling)
        
        for i_ifg, ifg in enumerate(ifgs):
            fn = ifg.split("/")[-1]
            old_fn = fn.replace(".fits.Z", "_oidata.fits")
            new_fn = old_fn.replace("_oidata", "_i%02i_oidata" % i_ifg)
            
            copyfile(night_folder + old_fn, bootstrapping_folder + new_fn)
            
        print("Moved %i new interferograms for %s on %s" % (i_ifg+1, seq,
                                                            night))
            

def sample_interferograms(obs_sequence, n_ifg=5, do_random_ifg_sampling=True,
                          validate_mode=False):
    """Samples from among the available interferograms and returns a list of
    filenames.
    
    If do_random_ifg_sampling is True, n_ifg interferograms will be selected
    for each star for each appearance in the CAL1-SCI1-CAL2-SCI2-CAL3 sequence
    at random with repeats. If false, all available data from the sequence will
    be used.
    
    Parameters
    ----------
    obs_sequence: list
        List of all *raw* observations taken for this sequence originally from
        complete_sequences. Note that raw observations include DARK and KAPPA
        (flux splitting) exposures, and that these are ignored.
        
    n_ifg: int
        Number of random samples with repeats to make of each target. Defaults
        to 5.
    
    do_random_ifg_sampling: bool
        Boolean indicating whether to randomly sample from the interferograms 
        or to use all data available.
        
    validate_mode: bool
        Boolean used for testing purposes to inspect random sampling.
        
    Returns
    -------
    selected_ifgs: list
        List of sampled *raw* files of type FRINGE.
    """
    selected_ifgs = []
    ifg_i = 0
    
    # Initialise the current target/sequence
    current_tgt = obs_sequence[0][2]
    
    night = obs_sequence[0][5].split("/")[-2]
    
    current_ifgs = []
    
    while ifg_i < len(obs_sequence):
        # Get the current target
        new_tgt = obs_sequence[ifg_i][2]
        ifg_filename = obs_sequence[ifg_i][7]
        ifg_type = obs_sequence[ifg_i][8]
        
        # See if it matches the previous target, and if so record the filename
        # for the interferogram and continue the loop
        if new_tgt == current_tgt:
            # Only add file if it is a fringe
            if ifg_type == "FRINGE" and validate_mode:
                # In validate mode, append easier to read star names/data types
                # and numbers rather than filenames
                current_ifgs.append("%s_%s_%i" % (new_tgt, ifg_type, ifg_i))
                
            if ifg_type == "FRINGE":
                current_ifgs.append(ifg_filename)
                
            ifg_i += 1
        
        # Does not match, means we've moved onto the next target in the seq.
        # Now we should sample n_ifg times, and reset
        if new_tgt != current_tgt or ifg_i == len(obs_sequence):
            # Either sample randomly with repeats, or use all data
            if do_random_ifg_sampling:
                # In some cases where sequences have not been completed as 
                # CAL1-SCI1-CAL2-SCI2-CAL3 (e.g. out of order) we may end up
                # with a block that does not contain any fringes, specifically
                # a block containing only kappa files. These should be ignored.
                if len(current_ifgs) > 0:
                    selected_ifgs.extend(np.random.choice(current_ifgs, n_ifg))
                else:
                    print("Found block of entirely non-fringe files for",
                          "%s on night %s" % (current_tgt, night)) 
            else:
                selected_ifgs.extend(current_ifgs)
            
            # Reset, but don't increment counter (will just go through the loop
            # again and hit the first if statement)
            current_tgt = new_tgt
            current_ifgs = []
    
    #for ifg_i, ifg in enumerate(selected_ifgs):
        #print("%i\t%s" % (ifg_i, ifg))
     
    return selected_ifgs
    
    
def run_one_calibration_set(sequences, complete_sequences, base_path, 
                            tgt_info, pred_ldd, e_pred_ldd, bs_i, results_path,
                            run_local=False, already_calibrated=False,
                            do_random_ifg_sampling=True):
    """Runs a single bootstrapping iteration, completing the following steps: 
        - Write YYYY-MM-DD_oiDiam.fits files for each night of observing
        - Run pndrsCalibrate for each night of observing
        - Collate vis^2 and fit angular diameters for all science targets
    
    Parameters
    ----------
    sequences: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        lists of the targets in said CAL1-SCI1-CAL2-SCI2-CAL3 sequence. 
    
    complete_sequences: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        [night, grade, [[container, OB, target, grade, ob_time, obs_log, run, 
                         ob_fits],...]
    
    base_path: str
        String filepath where the calibrated data is stored.
    
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
        
    pred_ldd: pandas dataframe
        Pandas dataframe with columns being stars, and the values being LDD for
        a given bootstrapping iteration. Only one row.
    
    e_pred_ldd: pandas dataframe
        Pandas dataframe with columns being stars, and the values being the 
        uncertainties corresponding to e_pred_ldd. Only one row.
        
    bs_i: int
        Integer count for the ith bootstrapping iteration
    
    results_path: string
        Path to store the bootstrapped oifits files.
    
    run_local: bool
        Boolean indicating whether the pipeline is being run locally, and to
        save files instead within reach/test/ for inspection.
    
    already_calibrated: bool
        Boolean to skip calibration and proceed straight to result collation
        for testing purposes when results remain.
    
    do_random_ifg_sampling: bool
        Boolean indicating whether to randomly sample from the interferograms 
        or to use all data available.
    """
    # Intialise interferograms
    # Select the reduced interferograms which should be used for calibration
    initialise_interferograms(complete_sequences, base_path, n_ifg=5, 
                              do_random_ifg_sampling=do_random_ifg_sampling)
    
    if not run_local and not already_calibrated:
        # Save oiDiam files
        print(list(tgt_info.columns))
        nights = save_nightly_ldd(sequences, complete_sequences, tgt_info,
                                  pred_ldd, e_pred_ldd, base_path)
        
        print("\n", "-"*79, "\n", "\tCalibrating %i night/s, bootstrap %i\n" 
              % (len(nights), bs_i), "-"*79)
        
        # Run Calibration
        obs_folders = [base_path % night + "%s/" % night 
                       for night in nights.keys()]
        calibrate_all_observations(obs_folders, bs_i, results_path)
    
    elif run_local and not already_calibrated:
        # Save oiDiam files for local inspection
        nights = save_nightly_ldd(sequences, complete_sequences, tgt_info,
                                  pred_ldd, e_pred_ldd,base_path, 
                                  run_local=run_local)
    
    
def run_n_bootstraps(sequences, complete_sequences, base_path, tgt_info,
                     n_pred_ldd, e_pred_ldd, n_bootstraps, results_path,
                     run_local=False, already_calibrated=False, 
                     do_random_ifg_sampling=True):
    """Runs N bootstrapping iterations, collating and return the results.
    
    Parameters
    ----------
    sequences: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        lists of the targets in said CAL1-SCI1-CAL2-SCI2-CAL3 sequence. 
    
    complete_sequences: dict
        Dictionary mapping sequences (period, science target, bright/faint) to
        [night, grade, [[container, OB, target, grade, ob_time, obs_log, run, 
                         ob_fits],...]
    
    base_path: str
        String filepath where the calibrated data is stored.
    
    tgt_info: pandas dataframe
        Pandas dataframe of all target info
        
    n_pred_ldd: pandas dataframe
        Pandas dataframe with columns being stars, and each row being a set of
        LDD for a given bootstrapping iteration. If not doing calibrator 
        bootstrapping, each row will be the same, but otherwise the calibrator
        angular diameters are drawn from a Gaussian distribution as part of the
        bootstrapping.
    
    e_pred_ldd: pandas dataframe
        Pandas dataframe with columns being stars, and the values being the 
        uncertainties corresponding to n_pred_ldd. Only one row.
        
    n_bootstraps: int
        The number of bootstrapping iterations to run.
    
    results_path: string
        Path to store the bootstrapped oifits files.
    
    run_local: bool
        Boolean indicating whether the pipeline is being run locally, and to
        save files instead within reach/test/ for inspection.
    
    already_calibrated: bool
        Boolean to skip calibration and proceed straight to result collation
        for testing purposes when results remain.
    
    do_random_ifg_sampling: bool
        Boolean indicating whether to randomly sample from the interferograms 
        or to use all data available.
        
    """
    # Initialise data structures for results
    n_vis2 = {}
    n_baselines = {}
    n_ldd_fit = {}
    
    times = []
    
    # Bootstrap n times
    for bs_i in np.arange(0, n_bootstraps):
        times.append(datetime.datetime.now())  
        print("\n", "|"*79, "\n\tBootstrapping iteration %i\n" % bs_i, "|"*79)
        
        # Run a single calibration run
    
        run_one_calibration_set(sequences, complete_sequences, base_path, 
                                tgt_info, n_pred_ldd.iloc[bs_i], 
                                e_pred_ldd, bs_i, results_path, 
                                run_local=run_local, 
                                already_calibrated=already_calibrated,
                                do_random_ifg_sampling=do_random_ifg_sampling)
                
        times.append(datetime.datetime.now())  
        b_i_time = (times[-1] - times[-2]).total_seconds() 
        print("\n\nBoostrap %i done in %02d:%04.1f\n" 
              % (bs_i+1, int(np.floor(b_i_time/60.)), b_i_time % 60.))
    
    total_t = (times[-1] - times[0]).total_seconds() 
    print("\n%i bootstraps done in %02d:%04.1f\n" 
              % (n_bootstraps, int(np.floor(total_t/60.)), total_t % 60.))
                
    # All done
    print("\n", "-"*79, "\n", "\tBootstrapping Complete\n", "-"*79)