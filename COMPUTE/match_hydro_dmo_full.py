"""
Match subhaloes between hydro and DM-only runs

Started 20 Jan 2017
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
import time
import calendar
from mpi4py import MPI

n_tracers = 50

comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()


def match_subhaloes(ind_sh_a, ids_a, offset_a, length_a, offset_b, length_b, inds_in_b):

    match_in_b = np.zeros(len(ind_sh_a), dtype = int)-1
    
    for ish, sh in enumerate(ind_sh_a):

        if ish % 1000 == 0:
            print("...reached subhalo {:d}/{:d}..." .format(ish, len(ind_sh_a)))
            
        curr_off_a = offset_a[sh]
        curr_len_a = length_a[sh]
        
        ind_dm_parts_curr = np.nonzero(ids_a[curr_off_a:(curr_off_a+curr_len_a)] % 2 == 0)[0]

        n_tracers = 50
        # Limit search to the 50 most bound particles in the Hydro subhalo:

        if len(ind_dm_parts_curr) > n_tracers:
            ind_dm_parts_curr = ind_dm_parts_curr[:n_tracers]
        else:
            continue  # Don't trace DM-poor subhaloes

        ind_dm_parts_curr += curr_off_a
        indices_in_b = inds_in_b[ind_dm_parts_curr]

        inds_in_b_subhalo = np.nonzero(indices_in_b >= 0)[0]
        indices_in_b = indices_in_b[inds_in_b_subhalo]

        # Find the potential matching DMO subhalo. 
        # 'side = right' is there to make sure that if this is the first particle in the DMO-list
        # and there are multiple entries with this offset, the highest is chosen 
        # (i.e. the one with length > 0)

        b_shs_trial = np.searchsorted(offset_b, indices_in_b, side = 'right')-1

        # Now check which of the particles are within LENGTH of the DMO-match OFFSET...
        
        b_gap = indices_in_b - offset_b[b_shs_trial] 
        ind_confirmed = np.nonzero(b_gap < length_b[b_shs_trial])[0]
            
        if len(ind_confirmed) > n_tracers/2:
            b_shs_confirmed = b_shs_trial[ind_confirmed]
                
            match_hist = np.bincount(b_shs_confirmed)
            best_match = np.argmax(match_hist)
            
            if match_hist[best_match] >= n_tracers/2:
                match_in_b[ish] = best_match


    return match_in_b


basedir = '/virgo/simulations/Hydrangea/10r200/'
match_to_fof = False
return_central = False

is_restart = True

for isim in range(29, 30):
    
    # Skip this one if we are multi-threading and it's not for this task to worry about
    #if not isim % numtasks == rank:
    #    continue

    rundir_hy = basedir + '/HaloF' + str(isim) + '/HYDRO/'
    rundir_dm = basedir + '/HaloF' + str(isim) + '/DM/'

    outloc_hy = rundir_hy + '/highlev/MatchInDM.hdf5'
    outloc_dm = rundir_dm + '/highlev/MatchInHydro.hdf5'

    if not os.path.exists(rundir_hy):
        print("No hydro rundir...")
        continue

    if not os.path.exists(rundir_dm):
        print("No DM rundir...") 
        continue

    print("")
    print("=============================")
    print("Processing simulation F" + str(isim))
    print("=============================")
    print("", flush = True)

    if not is_restart:
        catID = calendar.timegm(time.gmtime())
        yb.write_hdf5_attribute(outloc_dm, "Header", "CatalogueID", catID, group = True, new = True)
        yb.write_hdf5_attribute(outloc_dm, "Header", "Simulation", isim, group = True)
        yb.write_hdf5_attribute(outloc_dm, "Header", "NumTracers", n_tracers, group = True) 


        yb.write_hdf5_attribute(outloc_hy, "Header", "CatalogueID", catID, group = True, new = True)
        yb.write_hdf5_attribute(outloc_hy, "Header", "Simulation", isim, group = True)
        yb.write_hdf5_attribute(outloc_hy, "Header", "NumTracers", n_tracers, group = True) 

    for isnap in range(26, 30):
        if not isnap % numtasks == rank:
            continue

        
        sstime = time.time()

        print("")
        print("----------------------")
        print("Snapshot {:d} (F{:d})..." .format(isnap, isim))
        print("----------------------")
        print("")

        subdir_hy = st.form_files(rundir_hy, isnap, 'sub')
        subdir_dm = st.form_files(rundir_dm, isnap, 'sub')

        if not os.path.exists(subdir_hy) or not os.path.exists(subdir_dm):
            print("Why does a simulation output not have subfind?")
            set_trace()

        ids_hy = st.eagleread(subdir_hy, 'IDs/ParticleID', astro = False, zoom = 0)

        if match_to_fof:
            offset_hy = st.eagleread(subdir_hy, 'FOF/GroupOffset', astro = False, zoom = 0)
            length_hy = st.eagleread(subdir_hy, 'FOF/GroupLength', astro = False, zoom = 0)
            if return_central_sh:
                fsh_hy = st.eagleread(subdir_hy, 'FOF/FirstSubHaloID', astro = False, zoom = 0) 
                
        else:
            offset_hy = st.eagleread(subdir_hy, 'Subhalo/SubOffset', astro = False, zoom = 0)
            length_hy = st.eagleread(subdir_hy, 'Subhalo/SubLength', astro = False, zoom = 0)
            

        # ------ Now load DM IDs and offset list... ------------

        ids_dm = st.eagleread(subdir_dm, 'IDs/ParticleID', astro = False, zoom = 0)

        if match_to_fof:
            offset_dm = st.eagleread(subdir_dm, 'FOF/GroupOffset', astro = False, zoom = 0)
            length_dm = st.eagleread(subdir_dm, 'FOF/GroupLength', astro = False, zoom = 0)
            if return_central_sh:
                fsh_dm = st.eagleread(subdir_dm, 'FOF/FirstSubHaloID', astro = False, zoom = 0) 

        else:    
            offset_dm = st.eagleread(subdir_dm, 'Subhalo/SubOffset', astro = False, zoom = 0)
            length_dm = st.eagleread(subdir_dm, 'Subhalo/SubLength', astro = False, zoom = 0)
        

        # -------- Locate IDs of one sim in another (in bulk) ---------

        revid_dm = st.create_reverse_list(ids_dm, maxval = np.max([ids_dm.max(), ids_hy.max()]))
        inds_in_dm = revid_dm[ids_hy]

        revind_hy = st.create_reverse_list(ids_hy, maxval = np.max([ids_hy.max(), ids_dm.max()]))

        inds_in_hy = revind_hy[ids_dm]
        

        # ------- Find matches IN DMO ('outbound match') ---------

        # Set up 'Hydro-SH-list' to 'everything'
        ind_sh_hy = np.arange(len(length_hy), dtype = np.int32)

        print("Now matching HYDRO-->DMO...")
        mstime_a = time.time()

        match_in_dmo = match_subhaloes(ind_sh_hy, ids_hy, offset_hy, 
                                       length_hy, offset_dm, length_dm, 
                                       inds_in_dm)
        
        
        print(" ... done (took {:.3f} sec.)" .format(time.time()-mstime_a))

        match_in_hy = np.zeros_like(match_in_dmo, dtype = np.int32)-1
        ind_good_dmo = np.nonzero(match_in_dmo >= 0)[0]
        n_good_dmo = len(ind_good_dmo)

        print("Found {:d} tentative matches in DMO (={:.2f}%), now performing reverse matching..." .format(n_good_dmo, n_good_dmo/len(match_in_dmo)*100))

        mstime_b = time.time()

        match_in_hy[ind_good_dmo] = match_subhaloes(match_in_dmo[ind_good_dmo], ids_dm, offset_dm, length_dm, offset_hy, length_hy, inds_in_hy)

        print(" ... done (took {:.3f})" .format(time.time()-mstime_b))

        ind_succ = np.nonzero(match_in_hy == ind_sh_hy)[0]
        n_succ = len(ind_succ)

        print("Successfully matched {:d} subhaloes (= {:.2f}% of Hydro, {:.2f}% of DMO)" .format(n_succ, n_succ/len(ind_sh_hy)*100, n_succ/len(length_dm)*100))

        # Final step: Build equivalent list for DM-->HYDRO match

        match_in_hydro = np.zeros(len(length_hy), dtype = np.int32)-1
        match_in_hydro[match_in_dmo[ind_succ]] = ind_succ 
        
        yb.write_hdf5(match_in_dmo, outloc_hy, "Snapshot_" + str(isnap).zfill(3), comment = "Matched subhalo indices in DMO runs (-1: no good match exists)")
        yb.write_hdf5(match_in_hydro, outloc_dm, "Snapshot_" + str(isnap).zfill(3), comment = "Matched subbhalo indices in Hydro runs (-1: no good match exists)")
    
        print("Finished snapshot in {:.3f} sec."
              .format(time.time()-sstime))



print("Done!")
