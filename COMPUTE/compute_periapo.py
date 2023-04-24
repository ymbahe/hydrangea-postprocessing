"""
Compute the peri-/apocentric times and distances.
Based on interpolated position catalogues.

!! 08-May-2019: Unclear whether cross-simulation combination is done correctly, need to check and potentially re-run. !!

Started 08-Mar-2018
"""

import time
import os
import numpy as np
import scipy.spatial
from pdb import set_trace
import sim_tools as st
import h5py as h5
import yb_utils as yb
from mpi4py import MPI
from scipy.signal import argrelextrema
from astropy.io import ascii
from astropy.cosmology import Planck13
import scipy.interpolate
import hydrangea_tools as ht

rundir_base = '/virgo/simulations/Hydrangea/10r200/'

catloc_acc = '/freya/ptmp/mpa/ybahe/HYDRANGEA/DISRUPTION/AccretionCatalogue_1Oct18_HYDRO.hdf5'

outloc = '/freya/ptmp/mpa/ybahe/HYDRANGEA/DISRUPTION/PeriApo_HYDRO_21Nov18.hdf5' 

n_halo = 30
n_snap = 30

flag_redo = True    # Set to true to force re-computation of already existing files

snapAexpLoc = '/freya/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/hydrangea_snepshots_allsnaps.dat'
snap_aexp = np.array(ascii.read(snapAexpLoc, format = 'no_header', guess = False)['col1'])
snap_zred = 1/snap_aexp - 1
snap_time = Planck13.age(snap_zred).value

comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()

gal_cat = yb.read_hdf5(catloc_acc, 'Galaxy')
host_cat = yb.read_hdf5(catloc_acc, 'KeyHost')
sim_cat = yb.read_hdf5(catloc_acc, 'Sim')
time_skb_cat = yb.read_hdf5(catloc_acc, "TimeFirstSatInKeyBranch")
lastSnap_cat = yb.read_hdf5(catloc_acc, "LastSnap")
time_lastSnap = snap_time[lastSnap_cat]

# Set up output arrays:

ngal = len(gal_cat)

nPeriAll = np.zeros((ngal, 2), dtype = np.int16)-1  # 0 = all, 1 = alive
nApoAll = np.zeros((ngal, 2), dtype = np.int16)-1  # 0 = all, 1 = alive

radPeriAll = np.zeros((ngal, 2, 2), dtype = np.float32)-1  # igal | /r200? | # 
radApoAll = np.zeros((ngal, 2, 2), dtype = np.float32)-1  # igal | /r200? | # 
radApoFinAll = np.zeros((ngal, 2, 2), dtype = np.float32)-1  # igal | /r200? | # 

deltaTApo = np.zeros((ngal, 3), dtype = np.float32)-1  # 0:1-2, 1:alive, 2:all
deltaTPeri = np.zeros((ngal, 3), dtype = np.float32)-1  # 0:1-2, 1:alive, 2:all

tApo = np.zeros((ngal, 2), dtype = np.float32)-1  # 0: first, 1: second
tPeri = np.zeros((ngal, 2), dtype = np.float32)-1  # 0: first, 1: second


for ihalo in range(0, 30):

    # Skip this one if we are multi-threading and it's not for this task to worry about
    if not ihalo % numtasks == rank:
        continue
    
    hstime = time.time()

    print("")
    print("**************************")
    print("Now processing halo CE-{:d}" .format(ihalo))
    print("**************************")
    print("")

    ind_thissim = np.nonzero(sim_cat == ihalo)[0]
    n_thissim = len(ind_thissim)

    if n_thissim == 0:
        continue
     
    rundir = rundir_base + '/CE-' + str(ihalo) + '/DM'
 
    if not os.path.exists(rundir):
        set_trace()

    fgtloc = rundir + '/highlev/FullGalaxyTables.hdf5'
    posloc = ht.clone_dir(rundir, loc = 'virgo') + '/highlev/GalaxyCoordinates10Myr_fromSnapsOnly.hdf5'
    
    pos_all = yb.read_hdf5(posloc, 'InterpolatedPositions')
    intIndex_all = yb.read_hdf5(posloc, 'GalaxyRevIndex') 
    intTime = yb.read_hdf5(posloc, 'InterpolationTimes')

    SHI = yb.read_hdf5(fgtloc, 'SHI')
    r200All = yb.read_hdf5(fgtloc, 'R200')

    for iigal, igal in enumerate(ind_thissim):

        curr_gal = gal_cat[igal]
        curr_host = host_cat[igal]
        
        if curr_host < 0: continue   # Not attached to any host

        ind_in = np.nonzero(intTime >= time_skb_cat[igal])[0]
        if len(ind_in) == 0: continue

        subind_in_alive = np.nonzero(intTime[ind_in] <= time_lastSnap[igal])[0]

        ind_host_ok = np.nonzero(SHI[curr_host, :] >= 0)[0]

        if igal % 100 == 0:
            print("igal={:d}" .format(igal))
        
        gotR200 = True
        if len(ind_host_ok) < 2:
            gotR200 = False
            
        if len(ind_host_ok) < 4:
            kind = 'linear'
        else:
            kind = 'cubic'

        if gotR200:
            csi_r200 = scipy.interpolate.interp1d(snap_time[ind_host_ok], r200All[curr_host, ind_host_ok], kind = kind, assume_sorted = True, fill_value = "extrapolate")

        ind_curr = intIndex_all[curr_gal]
        ind_host = intIndex_all[curr_host]

        if ind_curr < 0 or ind_host < 0: continue  # might be contaminated...

        pos_curr = pos_all[ind_curr, :, :]
        pos_host = pos_all[ind_host, :, :]

        # N.B.: Coordinate index is the middle one in pos_all
        rad = np.linalg.norm(pos_curr-pos_host, axis = 0)
        rad_in = rad[ind_in]

        #if np.min(time[ind_in]) < np.min(snap_time[ind_host_ok]) or np.max(time[ind_in]) > np.max(snap_time[ind_host_ok]):
        #    set_trace()
            
        if gotR200:
            r200Int_in = csi_r200(intTime[ind_in])

        subind_apo = argrelextrema(rad_in, np.greater)[0]
        n_apo = len(subind_apo)
        subind_peri = argrelextrema(rad_in, np.less)[0]
        n_peri = len(subind_peri)


        time_apo_all = intTime[ind_in[subind_apo]]
        time_peri_all = intTime[ind_in[subind_peri]]

        # subind_xxx are indices INTO APO/PERI events
        subind_apo_alive = np.nonzero(time_apo_all <= time_lastSnap[igal])[0]
        subind_peri_alive = np.nonzero(time_peri_all <= time_lastSnap[igal])[0]

        nPeriAll[igal, 0] = n_peri
        nPeriAll[igal, 1] = len(subind_peri_alive)

        nApoAll[igal, 0] = n_apo
        nApoAll[igal, 1] = len(subind_apo_alive)

        rad_apo_all = rad_in[subind_apo]
        rad_peri_all = rad_in[subind_peri]

        if n_peri > 0: 
            tPeri[igal, 0] = time_peri_all[0]
            radPeriAll[igal, 0, 0] = rad_peri_all[0]
            if gotR200:
                radPeriAll[igal, 1, 0] = rad_peri_all[0]/r200Int_in[subind_peri[0]]

        if n_peri > 1: 
            tPeri[igal, 1] = time_peri_all[1]
            radPeriAll[igal, 0, 1] = rad_peri_all[1]
            if gotR200:
                radPeriAll[igal, 1, 1] = rad_peri_all[1]/r200Int_in[subind_peri[1]]

            deltaTPeri[igal, 0] = time_peri_all[1]-time_peri_all[0]
            deltaTPeri[igal, 2] = (time_peri_all[-1]-time_peri_all[0])/(n_peri-1)
        if nPeriAll[igal, 1] > 1:
            deltaTPeri[igal, 1] = (time_peri_all[subind_peri_alive[-1]]-time_peri_all[0])/(nPeriAll[igal, 1]-1)


        if n_apo > 0: 
            tApo[igal, 0] = time_apo_all[0]
            radApoAll[igal, 0, 0] = rad_apo_all[0]

            if len(subind_apo_alive) > 0:
                radApoFinAll[igal, 0, 0] = rad_apo_all[subind_apo_alive[-1]]

            if gotR200:
                radApoAll[igal, 1, 0] = rad_apo_all[0]/r200Int_in[subind_apo[0]]
                if len(subind_apo_alive) > 0:
                    radApoFinAll[igal, 1, 0] = rad_apo_all[subind_apo_alive[-1]]/r200Int_in[subind_apo[subind_apo_alive[-1]]]

        if n_apo > 1: 
            tApo[igal, 1] = time_apo_all[1]
            radApoAll[igal, 0, 1] = rad_apo_all[1]

            if len(subind_apo_alive) > 1:
                radApoFinAll[igal, 0, 1] = rad_apo_all[subind_apo_alive[-2]]

            if gotR200:
                radApoAll[igal, 1, 1] = rad_apo_all[1]/r200Int_in[subind_apo[1]]
                if len(subind_apo_alive) > 1:
                    radApoFinAll[igal, 1, 1] = rad_apo_all[subind_apo_alive[-2]]/r200Int_in[subind_apo[subind_apo_alive[-2]]]

            deltaTApo[igal, 0] = time_apo_all[1]-time_apo_all[0]
            deltaTApo[igal, 2] = (time_apo_all[-1]-time_apo_all[0])/(n_apo-1)

        if nApoAll[igal, 1] > 1:
            deltaTApo[igal, 1] = (time_apo_all[subind_apo_alive[-1]]-time_apo_all[0])/(nApoAll[igal, 1]-1)


    # Ends loop through this sims gals

    yb.write_hdf5(sim_cat, outloc, 'Sim', new = True)
    yb.write_hdf5(gal_cat, outloc, 'Galaxy')
    yb.write_hdf5(nPeriAll, outloc, 'NumPeri')
    yb.write_hdf5(nApoAll, outloc, 'NumApo')
    yb.write_hdf5(radPeriAll, outloc, 'RadPeri')
    yb.write_hdf5(radApoAll, outloc, 'RadApo')
    yb.write_hdf5(radApoFinAll, outloc, 'RadApoFin')
    yb.write_hdf5(deltaTApo, outloc, 'DeltaTApo')
    yb.write_hdf5(deltaTPeri, outloc, 'DeltaTPeri')
    yb.write_hdf5(tApo, outloc, 'TimeApo')
    yb.write_hdf5(tPeri, outloc, 'TimePeri')
    


print("Done!")
