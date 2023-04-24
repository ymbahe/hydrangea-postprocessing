"""
Script to re-align the luminosity & SHI data for Stars from ESP to snapshot

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
"""

import sim_tools as st
import yb_utils as yb
import numpy as np
from pdb import set_trace
import monk
import time
import hydrangea_tools as ht
from scipy.spatial import cKDTree
import os

rootdir = '/virgo/simulations/Hydrangea/10r200/'
simtype = 'HYDRO'
n_snap = 30
n_sim = 30

for isim in range(14, 15):

    print("")
    print("==============================")
    print("Starting CE-{:d}..." .format(isim))
    print("==============================")    
    print("")

    simstime = time.time()

    rundir = rootdir + 'CE-' + str(isim) + '/' + simtype + '/'

    if not os.path.isdir(rundir): continue

    hldir = rundir + '/highlev/'
    outloc = ht.clone_dir(hldir) + '/StellarMagnitudes_029_TEST.hdf5'

    for isnap in range(29, 30):
        
        subdir, snapdir, partdir = st.form_files(rundir, isnap, 'sub snap subpart')
        snap_ids = st.eagleread(snapdir, 'PartType4/ParticleIDs', astro = False)
        esp_ids = st.eagleread(partdir, 'PartType4/ParticleIDs', astro = False)
        magdir = "/virgo/scratch/ybahe/PARTICLE_MAGS/CE-" + str(isim) + "/HYDRO/data/partMags_EMILES_PDXX_DUST_CH_029_z000p000.0.hdf5"

        mag_u = st.eagleread(magdir, "u-Magnitude", astro = False)
        mag_g = st.eagleread(magdir, "g-Magnitude", astro = False)
        mag_r = st.eagleread(magdir, "r-Magnitude", astro = False)
        mag_i = st.eagleread(magdir, "i-Magnitude", astro = False)
        mag_z = st.eagleread(magdir, "z-Magnitude", astro = False)

        subids = st.eagleread(subdir, 'IDs/ParticleID', astro = False)
        suboff = st.eagleread(subdir, 'Subhalo/SubOffset', astro = False)
        sublen = st.eagleread(subdir, 'Subhalo/SubLength', astro = False)

        nsh = len(sublen)
        
        gate_esp_snap = st.Gate(esp_ids, snap_ids)

        set_trace()
        
        nSnapPart = len(snap_ids)
        snap_magU = np.zeros(nSnapPart, dtype = np.float32)+1000
        snap_magG = np.zeros(nSnapPart, dtype = np.float32)+1000
        snap_magR = np.zeros(nSnapPart, dtype = np.float32)+1000
        snap_magI = np.zeros(nSnapPart, dtype = np.float32)+1000
        snap_magZ = np.zeros(nSnapPart, dtype = np.float32)+1000
        
        snap_magU[gate_esp_snap.in2()] = mag_u
        snap_magG[gate_esp_snap.in2()] = mag_g
        snap_magR[gate_esp_snap.in2()] = mag_r
        snap_magI[gate_esp_snap.in2()] = mag_i
        snap_magZ[gate_esp_snap.in2()] = mag_z
        
        snap_SHI = np.zeros(nSnapPart, dtype = np.int32)-1
        gate_ids_snap = st.Gate(subids, snap_ids)
        snap_SHI[gate_ids_snap.in2()] = st.ind_to_sh(np.arange(len(subids)), suboff, sublen)
        
    
        yb.write_hdf5(snap_SHI, outloc, 'PartType4/SubHaloIndex', new = True, comment = "Subhalo index to which each particle belongs (-1 if it is not in a subhalo).")
        yb.write_hdf5(snap_magU, outloc, 'PartType4/u-Magnitude', comment = 'u-band magnitude of particle (1000 if not in eagle_subfind_particle list)')
        yb.write_hdf5(snap_magG, outloc, 'PartType4/g-Magnitude', comment = 'g-band magnitude of particle (1000 if not in eagle_subfind_particle list)')
        yb.write_hdf5(snap_magR, outloc, 'PartType4/r-Magnitude', comment = 'r-band magnitude of particle (1000 if not in eagle_subfind_particle list)')
        yb.write_hdf5(snap_magI, outloc, 'PartType4/i-Magnitude', comment = 'i-band magnitude of particle (1000 if not in eagle_subfind_particle list)')
        yb.write_hdf5(snap_magZ, outloc, 'PartType4/z-Magnitude', comment = 'z-band magnitude of particle (1000 if not in eagle_subfind_particle list)')



print("Done!")
