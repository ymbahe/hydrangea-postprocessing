"""
Compute stellar mass deficit (i.e. Mstar_ini, max - mstar_ini, z=0)

Started 19 Jan 2017
"""

import numpy as np
import h5py as h5
import yb_utils as yb
import os
from pdb import set_trace
import hydrangea_tools as ht
import sys
import sim_tools as st
from astropy.io import fits

basedir = '/virgo/simulations/Hydrangea/10r200/'
peakname = 'GalaxyPeakQuantsAll.hdf5'

catloc_match = '/ptmp/mpa/ybahe/HYDRANGEA/RESULTS/CombinedBasicGalaxyCatalogue_20JAN17_S29_noradcut.fit'
outloc = '/ptmp/mpa/ybahe/HYDRANGEA/RESULTS/StellarMassDeficit_20Jan17_S29.hdf5'

hdulist = fits.open(catloc_match)
cat_match = hdulist[1].data

sim_match = cat_match['CLUSTER_INDEX']
sim_match_ind = np.array([int(cn[1:]) for cn in sim_match])
shi_match = cat_match['SHI']

smini_def_full = np.zeros(len(sim_match_ind),dtype=float)-1000
sm_def_full = np.zeros(len(sim_match_ind),dtype=float)-1000

for isim in range(41):

    ind_full_thissim = np.nonzero(sim_match_ind == isim)[0]
    print("This simulation contributes {:d} subhaloes to catalogue..." .format(len(ind_full_thissim)))

    if len(ind_full_thissim) == 0:
        continue

    rundir = basedir + 'HaloF' + str(isim) + '/HYDRO/'
    if not os.path.exists(rundir):
        continue

    print("")
    print("Processing simulation F" + str(isim))
    print("", flush = True)

    # General plan: First form mass deficit for each z=0 subhalo, then fill cat-aligned list

    tracingfile = rundir + 'TracingTable.hdf5'
    peakfile = ht.clone_dir(rundir) + peakname

    subdir_z0 = st.form_files(rundir, 29, 'sub')
    smini_z0 = st.eagleread(subdir_z0, 'Subhalo/StellarInitialMass', astro = True, zoom = 0)[0]
    smini_z0 = np.log10(smini_z0)+10.0

    sm_z0 = st.eagleread(subdir_z0, 'Subhalo/MassType', astro = True, zoom = 0)[0]
    sm_z0 = np.log10(sm_z0[:,4])+10.0

    smini_peak = yb.read_hdf5(peakfile, 'MaxMstarIni')
    sm_peak = yb.read_hdf5(peakfile, 'MaxMassType')
    sm_peak = sm_peak[:,4]

    gal_shi = yb.read_hdf5(tracingfile, 'Snapshot_029/SubHaloIndexAll')
    
    gal_shi_rev = st.create_reverse_list(gal_shi, cut = True)

    if len(gal_shi_rev) != len(smini_z0):
        print("Inconsistent lengths encountered, please investigate!")
        set_trace()

    smini_def = smini_peak[gal_shi_rev] - smini_z0
    sm_def = sm_peak[gal_shi_rev] - sm_z0
    
    # Now fill entries in cumulative table...

    shi_thissim = shi_match[ind_full_thissim]

    if len(shi_thissim) != len(np.unique(shi_thissim)):
        print("Duplicate entries detected in this sim's catalogue shi...")
        set_trace()

    if shi_thissim.min() < 0 or shi_thissim.max() >= len(smini_z0):
        print("Inconsistent entries detected in sim's catalogue shi...")
        set_trace()

    smini_def_full[ind_full_thissim] = smini_def[shi_thissim]
    sm_def_full[ind_full_thissim] = sm_def[shi_thissim]

yb.write_hdf5(smini_def_full, outloc, 'StellarInitialMassDeficit', new = True)
yb.write_hdf5_attribute(outloc, 'StellarInitialMassDeficit', 'CatalogueFile', np.string_('/ptmp/mpa/ybahe/HYDRANGEA/RESULTS/CombinedBasicGalaxyCatalogue_20JAN17_S29_noradcut.fit'), group = False)

yb.write_hdf5(sm_def_full, outloc, 'StellarMassDeficit', new = False)
yb.write_hdf5_attribute(outloc, 'StellarMassDeficit', 'Comment', np.string_('Difference in actual stellar mass, including from stellar wind losses'), group = False)

yb.write_hdf5(sim_match_ind, outloc, 'SimIndex', new = False)
yb.write_hdf5_attribute(outloc, 'SimIndex', 'Comment', np.string_('Indices of corresponding simulations, for later cross-checking'), group = False)

yb.write_hdf5(shi_match, outloc, 'SHI', new = False)
yb.write_hdf5_attribute(outloc, 'SHI', 'Comment', np.string_('Corresponding subfind subhalo indices, for later cross-checking'), group = False)



print("Done!")
