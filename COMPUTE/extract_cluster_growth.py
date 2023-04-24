# Program to extract the mass (m200) growth history of each cluster 
# Started 21 Sep 2016

rundir_base = '/virgo/simulations/Hydrangea/C-EAGLE/'

outloc = '/freya/ptmp/mpa/ybahe/HYDRANGEA/RESULTS/cluster_growth_table_withrad_Jun18_HYDRO.hdf5'

import hydrangea_tools as ht
import sim_tools as st
import eagle_routines as er
import numpy as np
from pdb import set_trace
from scipy.optimize import curve_fit
from astropy.io import ascii
import glob
import image_routines as im
import time
import os.path
import yb_utils as yb
import scipy.interpolate
from astropy.cosmology import Planck13

snapAexpLoc = '/freya/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/hydrangea_snepshots_allsnaps.dat'

full_m200 = np.zeros((30, 30))-1
full_m500 = np.zeros((30, 30))-1
full_r200 = np.zeros((30,30))-1
full_r500 = np.zeros((30,30))-1

full_pos = np.zeros((30,30,3))-1
full_shi = np.zeros((30,30),dtype=int)-1000

halo_list = np.arange(30, dtype = np.int)

#halo_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,21,22,24,25,28,29]
#halo_list = [0,1]
#halo_list = [8]

#halo_list = np.array(halo_list)
snap_list = np.arange(30, dtype = np.int)

#snap_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
#snap_list = [29]
#snap_list = np.array(snap_list)

snap_aexp = np.array(ascii.read(snapAexpLoc, format = 'no_header', guess = False)['col1'])
snap_zred = 1/snap_aexp - 1
snap_time = Planck13.age(snap_zred).value

for ihalo, halo in enumerate(halo_list):
    
    print("Now processing halo {:d}..." .format(halo))

    rundir = rundir_base + '/CE-' + str(halo) + '/HYDRO/'

    if not os.path.exists(rundir):
        continue

    spiderloc = rundir + '/highlev/SpiderwebTables.hdf5'
    pathloc = rundir + '/highlev/GalaxyPaths.hdf5'
    sw = st.Spiderweb(rundir, filename = 'SpiderwebTables.hdf5', highlev = False)
    
    # Look up galaxy number of central cluster
    galnr = sw.sh_to_gal(0, 29)

    print("Central cluster has galaxy number {:d}..." .format(galnr))
    
    for isnap, snap in enumerate(snap_list):
        
        print("   Snap {:d}..." .format(snap))
        
        sh_curr = sw.gal_to_sh(galnr, snap)

        full_shi[halo,isnap] = sh_curr
        if sh_curr < 0:
            continue

        print("      [SH={:d}]" .format(sh_curr))

        subdir = st.form_files(rundir, isnap=snap, types = 'sub', stype = 'snap')

        print("RUNDIR=", rundir)
        print("SUBDIR=", subdir)
        
        curr_gn = np.abs(st.eagleread(subdir, 'Subhalo/GroupNumber', astro = False))-1
        curr_m200 = st.eagleread(subdir, 'FOF/Group_M_Crit200', astro = True)[0]
        curr_r200 = st.eagleread(subdir, 'FOF/Group_R_Crit200', astro = True)[0]
        
        curr_m500 = st.eagleread(subdir, 'FOF/Group_M_Crit500', astro = True)[0]
        curr_r500 = st.eagleread(subdir, 'FOF/Group_R_Crit500', astro = True)[0]

        full_m500[halo,isnap] = curr_m500[curr_gn[sh_curr]]
        full_r500[halo,isnap] = curr_r500[curr_gn[sh_curr]]

        full_m200[halo,isnap] = curr_m200[curr_gn[sh_curr]]
        full_r200[halo,isnap] = curr_r200[curr_gn[sh_curr]]
        
        curr_pos = st.eagleread(subdir, 'Subhalo/CentreOfPotential', astro = True)[0]
        full_pos[halo,isnap,:] = curr_pos[sh_curr, :]

    ind_hidden = np.nonzero(full_shi[halo,:] == -9)[0]
    ind_ok = np.nonzero(full_shi[halo, :] >= 0)[0]

    if len(ind_hidden) > 0:

        print("-- Need to interpolate properties at snaps", ind_hidden, "--")

        csi_m200 = scipy.interpolate.interp1d(snap_time[ind_ok], full_m200[halo, ind_ok], kind = 'cubic', assume_sorted = True)
        csi_m500 = scipy.interpolate.interp1d(snap_time[ind_ok], full_m500[halo, ind_ok], kind = 'cubic', assume_sorted = True)
        csi_r200 = scipy.interpolate.interp1d(snap_time[ind_ok], full_r200[halo, ind_ok], kind = 'cubic', assume_sorted = True)
        csi_r500 = scipy.interpolate.interp1d(snap_time[ind_ok], full_r500[halo, ind_ok], kind = 'cubic', assume_sorted = True)
    
        full_m200[halo, ind_hidden] = csi_m200(snap_time[ind_hidden])
        full_m500[halo, ind_hidden] = csi_m500(snap_time[ind_hidden])
        full_r200[halo, ind_hidden] = csi_r200(snap_time[ind_hidden])       
        full_r500[halo, ind_hidden] = csi_r500(snap_time[ind_hidden])       

        if os.path.exists(pathloc):

            sneps = yb.read_hdf5(pathloc, "RootIndex/Allsnaps")
            for ihidden in ind_hidden:

                isnep = sneps[ihidden]
                pos_curr_all = yb.read_hdf5(pathloc, "Snepshot_" + str(isnep).zfill(4) + "/Coordinates")/0.6777*snap_aexp[ihidden]
                full_pos[halo, ihidden] = pos_curr_all[galnr, :]


    yb.write_hdf5(full_m200, outloc, 'M200c', new = True)
    yb.write_hdf5(full_r200, outloc, 'R200c')

    yb.write_hdf5(full_m500, outloc, 'M500c')
    yb.write_hdf5(full_r500, outloc, 'R500c')

    yb.write_hdf5(full_shi, outloc, 'SHI')
    yb.write_hdf5(full_pos, outloc, 'Position')



print("Done!")
