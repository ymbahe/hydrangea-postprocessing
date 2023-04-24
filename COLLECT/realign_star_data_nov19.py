"""
Script to collect the luminosity data, and align SHI data for stars 
from ESP to snapshot

By comparing IDs, new data sets are written that are aligned to the snapshot
outputs and can therefore be directly combined with them. Currently only 
S29 is implemented.

!! NOTE !! As discovered in April 2019, for a number of outputs the information
           used to calculate the luminosities is corrupted / mixed up in the 
           ESP files, so that this script assigns *incorrect* luminosities. 
           This does not affect the SubHaloIndex data set. This affects 
           CE-14, CE-16, and CE-18.

The output is written to a single file per output:
[freya]/highlev/StellarMagnitudes_029_TEST.hdf5

    <> PartType4/SubHaloIndex (SHI for each particle, -1 if unbound)
    <> PartType4/[u/g/r/i/z]-Magnitude (particle absolute magnitude, 1000 if
                                        not in ESP catalogue).

Started 04-Apr-19
Adapted for full-snapshot data on 21-Nov-19
"""

import sys
sys.path.insert(0, '/u/ybahe/ANALYSIS/sim-utils/')

import hydrangea as hy
import numpy as np
import time
import os
import glob
from shutil import copyfile

from pdb import set_trace

rootdir = '/virgo/simulations/Hydrangea/C-EAGLE/'
simtype = 'HYDRO'
n_snap = 30
n_sim = 30

for isim in [18, 19, 28]:

    print("")
    print("==============================")
    print("Starting CE-{:d}..." .format(isim))
    print("==============================")    
    print("")


    rundir = rootdir + 'CE-' + str(isim) + '/' + simtype + '/'
    if not os.path.isdir(rundir): continue

    for isnap in range(0, 30):

        print("Processing snapshot {:d}..." .format(isnap))

        snapdir = hy.form_files(rundir, isnap, 'snap')
        
        outdir = ('/'.join(snapdir.split('/')[:9]) + '/stars_extra/' +
                  (snapdir.split('/'))[9] + '/')

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        magdir = ("/virgo/scratch/ybahe/PARTICLE_MAGS/dataRelease/"
                  "CE-{:d}/HYDRO/data/" .format(isim))

        lum_files = glob.glob(magdir + 'partMags_EMILES_PDXX_DUST_CH_{:03d}_*'
                              .format(isnap))

        if len(lum_files) == 0:
            print("No files found for snapshot {:d}, sim {:d}..."
                  .format(isnap, isim))
            continue
            
        for ifile in lum_files:
            filename = (ifile.split('/'))[-1]
            print("File: {:s}..." .format(filename))

            if os.path.exists(outdir + filename):
                print("File already exists, NOT copying...")
            else:
                copyfile(ifile, outdir + filename) 
        
print("Done!")
