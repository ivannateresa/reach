"""
"""
from __future__ import division, print_function

import os
import glob
import numpy as np
import pandas as pd
import reach.pndrs as rpndrs
import reach.utils as rutils
import reach.diameters as rdiam
import matplotlib.pylab as plt
from decimal import Decimal
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages

def clean_target_id(x):
    """
    Clean target names for robust matching.

    Examples:
    HD_79810  -> hd79810
    HD  63734 -> hd63734
    psi Vel A -> psivela
    psi_Vel   -> psivel
    HR_3729   -> hr3729
    """

    import pandas as pd

    if pd.isnull(x):
        return ""

    x = str(x)

    x = x.replace("_", "")
    x = x.replace(" ", "")
    x = x.replace(".", "")
    x = x.replace("-", "")
    x = x.replace("\t", "")
    x = x.lower()

    return x


def match_target_name(tgt_info, name, verbose=False):
    """
    Match one target name to the tgt_info index using cleaned identifiers.
    """

    import pandas as pd

    name_clean = clean_target_id(name)

    search_cols = [
        "Primary",
        "Bayer_ID",
        "Ref_ID_1",
        "Ref_ID_2",
        "Ref_ID_3",
        "HD_ID",
        "HP"
    ]

    search_cols = [col for col in search_cols if col in tgt_info.columns]

    matches = []

    # Search inside columns
    for col in search_cols:

        col_clean = tgt_info[col].apply(clean_target_id)
        this_match = tgt_info.index[col_clean == name_clean]

        if len(this_match) > 0:
            matches.extend(list(this_match))

    # Search also in dataframe index
    index_clean = []

    for idx in tgt_info.index:
        index_clean.append(clean_target_id(idx))

    for i, idx_clean in enumerate(index_clean):
        if idx_clean == name_clean:
            matches.append(tgt_info.index[i])

    # Remove duplicates
    matches_unique = []

    for m in matches:
        if m not in matches_unique:
            matches_unique.append(m)

    if len(matches_unique) == 0:

        if verbose:
            print("NO MATCH: %s  cleaned as  %s" % (name, name_clean))

        return None

    if len(matches_unique) > 1:
        print("WARNING: multiple matches for %s:" % name)
        print(matches_unique)
        print("Using first match: %s" % matches_unique[0])

    return matches_unique[0]


def match_target_list(tgt_info, names, label="target", verbose=True):
    """
    Match a list of target names to tgt_info indices.
    """

    ids = []
    failed = []

    for name in names:

        this_id = match_target_name(tgt_info, name, verbose=False)

        if this_id is None:
            failed.append(name)

            if verbose:
                print("WARNING: could not match %s name: %s" % (label, name))

        else:
            ids.append(this_id)

    return ids, failed


def safe_get_unique_keys(tgt_info_input, names, label):

    ids, failed = match_target_list(
        tgt_info_input,
        names,
        label=label,
        verbose=True
    )

    return ids, failed

def run_vis2_diagnostics(complete_sequences, base_path):
    """Inspect vis2 as a function of time for baseline dropouts.
    """
    # For each sequence, go through the fringe files and plot vis2 over time 
    # on a per baseline basis. This should give ~37 plots.
    
    # Construct a data cube of dimensions [n_seq, n_bl, n_vis2] 
    
    #n_seq = len(complete_sequences)
    #n_bl = 6
    #n_vis2 = 60
    
    #seq_bl_vis2 = np.zeros([n_seq, n_bl, n_vis, n_vis2])
    
    vis2_per_bl = {}
    
    for seq_i, seq in enumerate(complete_sequences.keys()):
        print("Sequence %i --> %s" % (seq_i, seq))
        # Get the files
        fringe_files = [data[7] for data in complete_sequences[seq][2] if data[8]=="FRINGE"]
        fringe_files = [ff.replace(".fits.Z", "_oidata.fits") for ff in fringe_files]
        fringe_files = [ff.replace("all_sequences", "complete_sequences") for ff in fringe_files]
        fringe_files = [ff.replace("/PIONI", "_v3.73_abcd/PIONI") for ff in fringe_files]
        fringe_files.sort()                     
        
        # Create empty dataframe, and pre-allocate memory
        cols = ["MJD", "AT1-AT2", "BL 1-2", "AT1-AT3", "BL 1-3", 
                "AT1-AT4", "BL 1-4", "AT2-AT3", "BL 2-3", "AT2-AT4", "BL 2-4",
                "AT3-AT4", "BL 3-4"] 
        seq_recs = pd.DataFrame(index=np.arange(0, len(fringe_files)), columns=cols)
        
        # Extract from each one
        for ff_i, oi_fits_file in enumerate(fringe_files):
            #oi_fits_file = ff.replace(".fits.Z", "_oidata.fits")
            
            # Open file and populate dataframe
            with fits.open(oi_fits_file, memmap=False) as oifits:
                # Get the telescope/station data
                tels = oifits[3].data
                
                # Step through the vis2 data
                for bs_i, baseline_data in enumerate(oifits[4].data):
                    # Save MJD data
                    seq_recs.loc[ff_i, "MJD"] = baseline_data[2]
                    
                    # Get column name, and save vis2 data
                    tel_index = baseline_data[8] - [1,1]
                    
                    tel_col = "%s-%s" % (tels[tel_index[0]][0], 
                                         tels[tel_index[1]][0])
                   
                    seq_recs.loc[ff_i, tel_col] = baseline_data[4]
                    
                    # Save the UV baseline data
                    uv = np.sqrt(baseline_data[6]**2 + baseline_data[7]**2)
                    bl_col = "BL %i-%i" % (baseline_data[8][0], baseline_data[8][1])
                    seq_recs.loc[ff_i, bl_col] = uv
                    
        # Save the pandas array and move on
        vis2_per_bl[seq] = seq_recs
        
    return vis2_per_bl
  
            
def plot_vis2_diagnostics(vis2_per_bl):
    """
    """
    plt.close("all")
    fig, axes = plt.subplots(6, 7)
    axes = axes.flatten()
    
    vis2_cols = ["AT1-AT2", "AT1-AT3", "AT1-AT4", "AT2-AT3", "AT2-AT4", 
                "AT3-AT4"] 
    bl_cols = ["BL 1-2", "BL 1-3", "BL 1-4", "BL 2-3", "BL 2-4", "BL 3-4"] 
    
    for seq_i, seq in enumerate(vis2_per_bl.keys()):
        for vis2_col, bl_col in zip(vis2_cols, bl_cols):
            # Average the vis2 across the wavelength dimension
            mean_vis2 = np.mean(np.vstack(vis2_per_bl[seq][vis2_col].values), 
                                axis=1)
            
            # Average the baseline data to get a rough idea of relative lengths
            mean_bl = np.mean(vis2_per_bl[seq][bl_col].values)
            
            # Construct the label for the legend
            label = vis2_col + "(%i m)" % mean_bl
            
            axes[seq_i].plot(vis2_per_bl[seq]["MJD"], mean_vis2, ".-", 
                             label=label)

        axes[seq_i].set_title(seq)
        
        axes[seq_i].set_ylim([0,1])
        axes[seq_i].legend(loc="best")
        
    fig.suptitle("vis^2 vs MJD")
    #fig.tight_layout()
    plt.gcf().set_size_inches(32, 32)
    plt.savefig("plots/vis2_vs_time.pdf")
    
    
    
def plot_vis2_diagnostics_time_wl(vis2_per_bl):
    """
    """
    plt.close("all")
    
    vis2_cols = ["AT1-AT2", "AT1-AT3", "AT1-AT4", "AT2-AT3", "AT2-AT4", 
                "AT3-AT4"] 
    bl_cols = ["BL 1-2", "BL 1-3", "BL 1-4", "BL 2-3", "BL 2-4", "BL 3-4"] 
    
    wavelengths = [1.533e-06, 1.581e-06, 1.629e-06, 1.677e-06, 1.725e-06, 1.773e-06]
    
    with PdfPages("plots/vis2_vs_time_vs_baseline.pdf") as pdf:
        for seq_i, seq in enumerate(vis2_per_bl.keys()):
            # Initialise subplot
            fig, axes = plt.subplots(3, 3)
            axes = axes.flatten()
    
            for bl_i, (vis2_col, bl_col) in enumerate(zip(vis2_cols, bl_cols)):
                # Retrive all the vis2 values for this baseline
                vis2 = np.vstack(vis2_per_bl[seq][vis2_col].values)
                
                # For each wavelength *column*, plot
                for wl_i in np.arange(0, 6):
                    axes[bl_i].plot(vis2_per_bl[seq]["MJD"], vis2[:, wl_i], ".-", 
                                 label="%0.3E m" % Decimal(wavelengths[wl_i]))

            
                # Average the baseline data to get a rough idea of relative lengths
                mean_bl = np.mean(vis2_per_bl[seq][bl_col].values)
            
                # Construct the label for the legend
                title = vis2_col + "(%i m)" % mean_bl
            
                #axes[seq_i].plot(vis2_per_bl[seq]["MJD"], mean_vis2, ".-", 
                #                 label=label)
                axes[bl_i].set_title(title)
                axes[bl_i].set_ylim([0,1])
                axes[bl_i].legend(loc="top", fontsize="xx-small")
            
            fig.suptitle(seq)
            #fig.suptitle("vis^2 vs MJD")
            fig.tight_layout()
            plt.gcf().set_size_inches(16, 9)
            #plt.savefig("plots/vis2_vs_time.pdf")
            pdf.savefig()
            plt.close("all")
            

def calibrate_calibrators(sequences, complete_sequences, base_path, tgt_info,
                          n_pred_ldd, e_pred_ldd, test_all_cals=False):
    """
    Diagnostic calibration of calibrators.

    This version does not assume that all sequences have exactly three
    calibrators. It works with sequences of length 5:

        calibrator - science - calibrator - science - calibrator

    and also with sequences of length 3:

        calibrator - science - calibrator

    The function treats calibrators as science targets, one calibrator position
    at a time, in order to test whether the calibrators are internally
    consistent.
    """

    import os
    import glob
    import numpy as np
    import pandas as pd
    from collections import OrderedDict

    import reach.utils as rutils
    import reach.pndrs as rpndrs

    run_local = False
    already_calibrated = False
    do_random_ifg_sampling = False
    n_bootstraps = 1

    results_path = "/home2/ihernand/Desktop/reach/diagnostics/"

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Work on a copy, so the original tgt_info is not permanently modified
    tgt_info_cal = tgt_info.copy()

    # Reset any currently BAD quality targets if requested
    if test_all_cals:
        tgt_info_cal["Quality"] = [None] * len(tgt_info_cal)

    # Remove existing diagnostic FITS files
    existing_files = glob.glob(results_path + "*.fits")

    for fits_file in existing_files:
        os.remove(fits_file)

    # -------------------------------------------------------------------------
    # First inspect the sequence structure
    # -------------------------------------------------------------------------
    sequence_rows = []

    all_science_names = []
    max_n_calibrators = 0

    for seq_key, seq_targets in sequences.items():

        seq_targets = list(seq_targets)

        # Calibrators are in positions 0, 2, 4, ...
        # Science targets are in positions 1, 3, ...
        calibrators_this_seq = seq_targets[::2]
        science_this_seq = seq_targets[1::2]

        if len(calibrators_this_seq) > max_n_calibrators:
            max_n_calibrators = len(calibrators_this_seq)

        for sci in science_this_seq:
            all_science_names.append(sci)

        row = {
            "Period": seq_key[0],
            "Science_target_key": seq_key[1],
            "Sequence_type": seq_key[2],
            "N_targets": len(seq_targets),
            "N_calibrators": len(calibrators_this_seq),
            "N_science_positions": len(science_this_seq),
            "Full_sequence": " - ".join(seq_targets)
        }

        for i, target in enumerate(seq_targets):
            row["target_%02d" % (i + 1)] = target

        for i, cal in enumerate(calibrators_this_seq):
            row["calibrator_%02d" % (i + 1)] = cal

        for i, sci in enumerate(science_this_seq):
            row["science_%02d" % (i + 1)] = sci

        sequence_rows.append(row)

    sequence_df = pd.DataFrame(sequence_rows)

    sequence_csv = os.path.join(results_path, "calibrator_sequence_structure.csv")
    sequence_df.to_csv(sequence_csv, index=False)

    print("Saved sequence structure diagnostic:")
    print(sequence_csv)

    print("\nSequence length summary:")
    print(sequence_df["N_targets"].value_counts())

    short_sequences = sequence_df[sequence_df["N_targets"] != 5]

    if len(short_sequences) > 0:
        print("\nWARNING: Some sequences do not have 5 targets.")
        print("These are not necessarily wrong, but they have fewer calibrators.")
        print(short_sequences[[
            "Period",
            "Science_target_key",
            "Sequence_type",
            "N_targets",
            "N_calibrators",
            "Full_sequence"
        ]])


    # -------------------------------------------------------------------------
    # Mark all real science targets as BAD, so they are ignored in this diagnostic
    # -------------------------------------------------------------------------
    all_science_names_unique = []

    for name in all_science_names:
        if name not in all_science_names_unique:
            all_science_names_unique.append(name)

    science_ids, failed_science = safe_get_unique_keys(
        tgt_info_cal,
        all_science_names_unique,
        "science"
    )

    if len(science_ids) > 0:
        tgt_info_cal.loc[science_ids, "Quality"] = ["BAD"] * len(science_ids)

    print("\nNumber of real science targets marked as BAD:")
    print(len(science_ids))

    # -------------------------------------------------------------------------
    # Run calibrator diagnostic position by position
    # -------------------------------------------------------------------------
    # Example:
    # cal_i = 0 --> first calibrator in each sequence
    # cal_i = 1 --> second calibrator in each sequence
    # cal_i = 2 --> third calibrator, only for sequences that actually have one
    # -------------------------------------------------------------------------

    for cal_i in np.arange(0, max_n_calibrators):

        print("\n" + "-" * 79)
        print("Now running calibrator diagnostic for calibrator position %i" % cal_i)
        print("-" * 79)

        # Reset all Science flags
        tgt_info_cal["Science"] = [False] * len(tgt_info_cal)

        # Build subset of sequences that actually have this calibrator position
        sequences_this_run = OrderedDict()
        complete_sequences_this_run = OrderedDict()

        cal_names_this_run = []

        for seq_key, seq_targets in sequences.items():

            seq_targets = list(seq_targets)
            calibrators_this_seq = seq_targets[::2]

            # Skip sequences that do not have this calibrator position
            if len(calibrators_this_seq) <= cal_i:
                continue

            cal_name = calibrators_this_seq[cal_i]
            cal_names_this_run.append(cal_name)

            sequences_this_run[seq_key] = seq_targets

            if seq_key in complete_sequences:
                complete_sequences_this_run[seq_key] = complete_sequences[seq_key]

        # Remove duplicate calibrator names while preserving order
        cal_names_unique = []

        for name in cal_names_this_run:
            if name not in cal_names_unique:
                cal_names_unique.append(name)

        cal_ids, failed_cals = safe_get_unique_keys(
            tgt_info_cal,
            cal_names_unique,
            "calibrator"
        )

        print("Number of sequences used in this run:")
        print(len(sequences_this_run))

        print("Number of calibrators treated as science:")
        print(len(cal_ids))

        # Save which calibrators are being tested in this run
        cal_set_rows = []

        for input_name, cal_id in zip(cal_names_unique, cal_ids):

            cal_set_rows.append({
                "calibrator_position": cal_i,
                "input_name": input_name,
                "HD_ID": cal_id,
                "Primary": tgt_info_cal.loc[cal_id, "Primary"],
                "Quality": tgt_info_cal.loc[cal_id, "Quality"],
                "Science": True
            })

        cal_set_df = pd.DataFrame(cal_set_rows)

        cal_set_csv = os.path.join(
            results_path,
            "calibrator_position_%i.csv" % cal_i
        )

        cal_set_df.to_csv(cal_set_csv, index=False)

        print("Saved calibrator set:")
        print(cal_set_csv)

        # Skip if nothing was matched
        if len(cal_ids) == 0:
            print("No calibrators matched for position %i. Skipping." % cal_i)
            continue

        # Treat these calibrators as science targets
        tgt_info_cal.loc[cal_ids, "Science"] = [True] * len(cal_ids)

        # Run calibration only on the sequences that contain this calibrator position
        rpndrs.run_n_bootstraps(
            sequences_this_run,
            complete_sequences_this_run,
            base_path,
            tgt_info_cal,
            n_pred_ldd,
            e_pred_ldd,
            n_bootstraps,
            run_local=run_local,
            already_calibrated=already_calibrated,
            do_random_ifg_sampling=do_random_ifg_sampling,
            results_path=results_path
        )

    print("\nFinished calibrator diagnostic calibration")
def calibrate_calibrators_old(sequences, complete_sequences, base_path, tgt_info,
                          n_pred_ldd, e_pred_ldd, test_all_cals=False):
    """
    Assume that every sequence has three calibrators - we can simply do three
    loops of the calibration routine, turning the science off each time, and 
    flipping each calibrator to science in order going through.
    
    To do this, we'll need to flip all the science targets to "BAD" so that
    they're ignored. Hopefully this doesn't affect the kappa matrices. Should
    probably also flip them to calibrator status in case the pipeline complains
    that an entire science target is being ignored.
    """
    run_local = False
    already_calibrated = False
    do_random_ifg_sampling = False
    n_bootstraps = 1
    base_path = "/home2/ihernand/Desktop/reach/complete_sequences/%s_v3.94_abcd/"
    results_path = "/home2/ihernand/Desktop/reach/diagnostics/"
    
    # Get a list of the calibrators, in order, turn one calibrator per sequence
    # to science per calibration run - should need three runs
   
    calibrators = np.vstack(sequences.values())[:,::2]
    
    # Reset any currently "BAD" quality targets
    if test_all_cals:
        tgt_info["Quality"] = [None] * len(tgt_info)
    
    # Remove existing diagnostic files
    existing_files = glob.glob(results_path + "*.fits")
    
    for fits_file in existing_files:
        os.remove(fits_file) 
    
    # Set all science targets to "BAD" so that we ignore them when calibrating
    science = rutils.get_unique_key(tgt_info, np.vstack(sequences.values())[:,1])
    tgt_info.loc[science, "Quality"] = ["BAD"] * len(science)
    
    # Write nightly pndrs.i scripts
    for cal_i in np.arange(0,3):
        print("Now running for set number %i of calibrators" % cal_i)
        
        # Set all science targets to "BAD" so that we ignore them when calibrating
        tgt_info["Science"] = [False] * len(tgt_info)
        
        # Get the cal_i-th column of calibrators, and get their unique IDs
        cal_ids = rutils.get_unique_key(tgt_info, calibrators[:,cal_i])
        
        # Flip each of the calibrators in cals_to_sci to be a science target
        tgt_info.loc[cal_ids, "Science"] = [True] * len(cal_ids)
        
        # Run calibration (no bootstrapping)
        rpndrs.run_n_bootstraps(sequences, complete_sequences, base_path, 
                                    tgt_info,  n_pred_ldd, e_pred_ldd, 
                                    n_bootstraps, run_local=run_local, 
                                    already_calibrated=already_calibrated,
                                    do_random_ifg_sampling=do_random_ifg_sampling,
                                    results_path=results_path)
                                    
        # Compile results and look at visibility curves
        #oifiles = glob.glob("results/*oidata


def plot_calibrator_vis2(tgt_info, cal_folder="diagnostics/"):
    """
    """
    cal_oifits = glob.glob(cal_folder + "*fits")
    cal_oifits.sort()
    xy = np.ceil(len(cal_oifits)**0.5).astype(int)
    
    plt.close("all")
    with PdfPages("plots/calibrator_vis2.pdf") as pdf:
        # Initialise subplot
        fig, axes = plt.subplots(xy, xy)
        axes = axes.flatten()
        
        for cal_i, cal in enumerate(cal_oifits):
            cal_id = cal.split("SCI")[-1].split("oidata")[0].replace("_","")
            date = cal.split("/")[-1].split("_")[0]
            #rplt.plot_vis2(cal, cal_id)
            
            # Check if we've previously flagged this calibrator as bad
            cal_hd = rutils.get_unique_key(tgt_info, [cal_id])[0]
            is_bad = tgt_info.loc[cal_hd]["Quality"] == "BAD"
            
            if is_bad:
                colour = "red"
            else:
                colour="black"
            
            mjds, pairs, vis2, e_vis2, flags, baselines, wavelengths = rdiam.extract_vis2(cal)
    
            n_bl = len(baselines)
            n_wl = len(wavelengths)
            bl_grid = np.tile(baselines, n_wl).reshape([n_wl, n_bl]).T
            wl_grid = np.tile(wavelengths, n_bl).reshape([n_bl, n_wl])
            
            b_on_lambda = (bl_grid / wl_grid).flatten()
    
            axes[cal_i].errorbar(b_on_lambda, vis2.flatten(), 
                                 yerr=e_vis2.flatten(), fmt=".", 
                                 elinewidth=0.1, markersize=0.25)
            
            #axes[cal_i].set_xlabel(r"Spatial Frequency (rad$^{-1})$")
            #axes[cal_i].set_ylabel(r"Visibility$^2$")
            axes[cal_i].set_title(r"%s (%s)" % (cal_id, date), 
                                  color=colour, fontsize="small")
            #plt.legend(loc="best")
            axes[cal_i].set_xlim([0.0,25E7])
            axes[cal_i].set_ylim([0.0,1.0])
            axes[cal_i].tick_params(axis="both", labelsize="xx-small")
            axes[cal_i].xaxis.get_offset_text().set_fontsize("xx-small")
            axes[cal_i].grid()
            
        #plt.tight_layout()
        plt.gcf().set_size_inches(32, 32)
        plt.subplots_adjust(hspace=0.3)
        pdf.savefig() #bbox_inches="tight"
        #plt.close("all")
            
            
def inspect_bootstrap_iterations(oifits_files):
    """
    """
    times = []
    baselines = []
    wavelengths = []
    stations = []
    
    for file_i, file in enumerate(oifits_files):
        with fits.open(file, memmap=False) as oifits:
            wavelengths.append(tuple(oifits[2].data["EFF_WAVE"]))
            times.append(tuple(oifits[4].data["MJD"]))
            
            bl = np.sqrt(oifits[4].data["UCOORD"]**2 + oifits[4].data["VCOORD"]**2)
            bl.sort()
            
            baselines.append(tuple(bl))
            
            sta = tuple([tuple(pair) for pair in oifits[4].data["STA_INDEX"]])
            
            stations.append(sta)
            
    return set(times), set(baselines), set(wavelengths), set(stations)
    