"""
Re-assign BH particles that are not near the centre of their SH to a better-
matching SH. Also computes the mass of BH particles within min(3kpc, SHMR).

Started 18-May-2018
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

rundir_base = '/virgo/simulations/Hydrangea/C-EAGLE/'
outname = 'BlackHoleMasses.hdf5'

n_halo = 30
n_snap = 30

flag_redo = True    # Set to true to force re-computation of already existing files

comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()

for isim in range(0, 29):
    
    hstime = time.time()

    print("")
    print("**************************")
    print("Now processing halo F{:d}" .format(isim))
    print("**************************")
    print("")
     
    rundir = rundir_base + '/HaloF' + str(isim) + '/HYDRO'
    if not os.path.exists(rundir):
        continue

    for isnap in range(n_snap):


        # Skip this one if we are multi-threading and it's not for this task to worry about
        if not isnap % numtasks == rank:
            continue

        snap_stime = time.time()
        
        snapdir, subdir = st.form_files(rundir, isnap=isnap, types = 'snap sub', stype = 'snap')
        
        print(" --- Snapshot {:d} ---" .format(isnap))

        if snapdir is None or subdir is None:
            continue
                
        if not os.path.exists(snapdir):
            print("   ... Snapshot not found, skipping S{:d}... " .format(isnap))
            continue

        if not os.path.exists(subdir):
            print("   ... Subfind output not found, skipping S{:d}... " .format(isnap))
            continue

        outloc = yb.dir(subdir) + outname
        if not flag_redo:
            if os.path.exists(outloc):
                continue

        nbh = yb.read_hdf5_attribute(snapdir, 'Header', 'NumPart_Total')[5]
        if nbh == 0:
            print("No black hole particles, skipping...")
            continue

        bh_mass = st.eagleread(snapdir, 'PartType5/Mass', astro = True)[0]
        bh_sgmass = st.eagleread(snapdir, 'PartType5/BH_Mass', astro = True)[0]
        bh_id = st.eagleread(snapdir, 'PartType5/ParticleIDs', astro = False)
        bh_pos = st.eagleread(snapdir, 'PartType5/Coordinates', astro = True)[0]

        # Find subhalo index of each particle

        sh_len = st.eagleread(subdir, 'Subhalo/SubLength', astro = False)
        sh_off = st.eagleread(subdir, 'Subhalo/SubOffset', astro = False)
        sh_ids = st.eagleread(subdir, 'IDs/ParticleID', astro = False)

        maxid_bh = np.max(bh_id)
        maxid_sh = np.max(sh_ids)
        maxid = max(maxid_bh, maxid_sh)

        revid_bh = st.create_reverse_list(bh_id, maxval = maxid+1)
        revid_sh = st.create_reverse_list(sh_ids, maxval = maxid+1)
        
        nsh = sh_len.shape[0]

        sh_pos = st.eagleread(subdir, 'Subhalo/CentreOfPotential', astro = True)[0]
        sh_mstar = st.eagleread(subdir, 'Subhalo/MassType', astro = False)[:, 4]
        sh_shmr = (st.eagleread(subdir, 'Subhalo/HalfMassRad', astro = True)[0])[:, 4]*1000  # in pkpc
        
        # Find original SHI for each BH particle:
        ind_bh = revid_sh[bh_id]
        shi_bh_guess = np.searchsorted(sh_off, ind_bh, side = 'right')-1
        ind_bh_in_sh = np.nonzero(ind_bh < sh_off[shi_bh_guess]+sh_len[shi_bh_guess])[0]

        n_bh_in_sh = len(ind_bh_in_sh)
        n_bh = len(bh_id)
        print("Could locate {:d}/{:d} (={:.2f}%) of BH particles in a subhalo.".format(n_bh_in_sh, n_bh, n_bh_in_sh/n_bh*100))

        shi_bh_orig = np.zeros(n_bh, dtype = np.int32)-1
        shi_bh_orig[ind_bh_in_sh] = shi_bh_guess[ind_bh_in_sh]
        shi_bh_reass = np.copy(shi_bh_orig)

        # Compute distance of BHs from their SH centre
        bh_rad = np.zeros(n_bh)+np.inf
        bh_deltapos_sh = bh_pos[ind_bh_in_sh] - sh_pos[shi_bh_orig[ind_bh_in_sh]]
        bh_rad[ind_bh_in_sh] = np.linalg.norm(bh_deltapos_sh, axis = 1)*1000

        # Next line works because BHs outside SH are always included through
        # first clause (so nonsense value in second does not matter)
        ind_far_bh = np.nonzero((bh_rad > 3.0) | (bh_rad > sh_shmr[shi_bh_orig]))[0]

        n_far_bh = len(ind_far_bh)
        print("{:d}/{:d} BHs (={:.2f}%) are non-central..."
              .format(n_far_bh, n_bh, n_far_bh/n_bh*100))
        
        if n_far_bh > 0:
            time_0 = time.time()
            print("Build BH tree...")
            # Build trees of subhalo positions and 'far' BHs:
            tree_bh = scipy.spatial.cKDTree(bh_pos[ind_far_bh,:])
            time_a = time.time()
            print("done ({:.3f} sec.)! Build SH tree..." .format(time_a-time_0))
            tree_sh = scipy.spatial.cKDTree(sh_pos)
            time_b = time.time()
            
            print("done ({:.3f} sec.)! Finding SHs near BHs..." .format(time_b-time_a))

            #matches = []
            #next_point = len(ind_far_bh)/50
            #for iibh, ibh in enumerate(ind_far_bh):

                #if iibh > next_point:
                #    next_point += len(ind_far_bh)/50
                #    print(".", end = '', flush = True)
                
                #matches_this = tree_sh.query_ball_point(bh_pos[ibh, :], 3.0)
                #matches.append(matches_this)

            #print("")

            matches = tree_bh.query_ball_tree(tree_sh, 0.003)
            time_c = time.time()
            print("done ({:.3f} sec.)!" .format(time_c-time_b))

            next_point = len(ind_far_bh)/50
            for iibh, ibh in enumerate(ind_far_bh):

                if iibh > next_point:
                    next_point += len(ind_far_bh)/50
                    print(".", end = '', flush = True)

                n_this = len(matches[iibh])
                if n_this == 0: 
                    continue

                sh_cand = np.array(matches[iibh], dtype = int)
                bh_pos_this = bh_pos[ibh]
                bh_deltapos_this = sh_pos[sh_cand] - bh_pos_this[None, :]
                bh_rad_this = np.linalg.norm(bh_deltapos_this, axis = 1)*1000

                if n_this == 1: 
                
                    if bh_rad_this[0] <= sh_shmr[sh_cand[0]]:
                        shi_bh_reass[ibh] = sh_cand[0]

                    continue
                    
                # If we get here, there are multiple SHs to choose from.
                ind_in_shmr = np.nonzero(sh_shmr[sh_cand] >= bh_rad_this)[0]
                if len(ind_in_shmr) == 0: continue
                
                mstar_in_shmr = sh_mstar[sh_cand[ind_in_shmr]]
                subind_match = np.argmax(mstar_in_shmr)
                shi_bh_reass[ibh] = sh_cand[ind_in_shmr[subind_match]]
            
            # Ends loop through 'far' BHs, done with re-assignment!
            ind_changed = np.nonzero(shi_bh_reass != shi_bh_orig)[0]
            n_changed = len(ind_changed)
            print("")
            print("Finished re-assigning BHs in {:.3f} sec., changed {:d}/{:d} far BHs ({:.2f}%)" .format(time.time()-time_c, n_changed, n_far_bh, n_changed/n_far_bh*100))

        # We're on the home run now. Compute per-subhalo masses: 
        
        mbh_part = np.zeros(nsh, dtype = np.float32)   # Plain particle mass
        mbh_sg = np.zeros(nsh, dtype = np.float32)   # Plain subgrid mass
        
        mbh_reass = np.zeros(nsh, dtype = np.float32)   # Total reassigned mass
        mbh_sg_reass = np.zeros(nsh, dtype = np.float32)   # Total reassigned SG mass

        mbh_orig_cen = np.zeros(nsh, dtype = np.float32)   # Total orig mass within centre
        mbh_sg_orig_cen = np.zeros(nsh, dtype = np.float32)   # Total orig SG mass within centre

        mbh_reass_cen = np.zeros(nsh, dtype = np.float32)   # Total reassigned mass within centre
        mbh_sg_reass_cen = np.zeros(nsh, dtype = np.float32)   # Total reassigned SG mass within centre
        
        # Loop through black hole particles and add them appropriately
        print("")
        print("Compute total per-subhalo masses...")


        sum_stime = time.time()
        next_point = n_bh/50        
        for ibh in range(n_bh):

            if ibh > next_point:
                next_point += n_bh/50
                print(".", end = '', flush = True)


            mbh_this = bh_mass[ibh]
            mbh_sg_this = bh_sgmass[ibh]

            sh_this_orig = shi_bh_orig[ibh]  
            sh_this_reass = shi_bh_reass[ibh]  

            rad_orig = np.linalg.norm(bh_pos[ibh, :] - sh_pos[sh_this_orig])*1000
            rad_reass = np.linalg.norm(bh_pos[ibh, :] - sh_pos[sh_this_reass])*1000
            
            shmr_orig = sh_shmr[sh_this_orig]
            shmr_reass = sh_shmr[sh_this_reass]
            
            mbh_part[sh_this_orig] += mbh_this
            mbh_sg[sh_this_orig] += mbh_sg_this
            
            mbh_reass[sh_this_reass] += mbh_this
            mbh_sg_reass[sh_this_reass] += mbh_sg_this

            if rad_reass < min(3.0, shmr_reass):
                mbh_reass_cen[sh_this_reass] += mbh_this
                mbh_sg_reass_cen[sh_this_reass] += mbh_sg_this

            if rad_orig < min(3.0, shmr_orig):
                mbh_orig_cen[sh_this_orig] += mbh_this
                mbh_sg_orig_cen[sh_this_orig] += mbh_sg_this
                
        # Ends loop through individual BHs
        # We're done! Just write output

        print("")
        print("Finished computing subhalo sums in {:.3f} sec."
              .format(time.time()-sum_stime))
        
        yb.write_hdf5(mbh_part, outloc, "BHMass", new = True, comment = "Sum of BH *particle* masses of all BH particles in this subhalo. Should be identical to Subhalo/MassType[:, 5] in subfind tables *in astro units*, but may not be.")
        
        yb.write_hdf5(mbh_sg, outloc, "BHSubgridMass", comment = "Sum of BH *subgrid* masses of all BH particles in this subhalo. Should be identical to Subhalo/BlackHoleMass in subfind tables *in astro units*, but may not be.")

        yb.write_hdf5(mbh_reass, outloc, "ReassignedBHMass", comment = "Sum of BH *particle* masses of all BH particles in this subhalo *after reassignment*, in astro units.")
        yb.write_hdf5(mbh_sg_reass, outloc, "ReassignedBHSubgridMass", comment = "Sum of BH *subgrid* masses of all BH particles in this subhalo *after reassignment*, in astro units.")

        yb.write_hdf5(mbh_orig_cen, outloc, "BHCentralMass", comment = "Sum of BH *particle* masses of central BH particles in this subhalo, in astro units.")
        yb.write_hdf5(mbh_sg_orig_cen, outloc, "BHCentralSubgridMass", comment = "Sum of BH *subgrid* masses of central BH particles in this subhalo, in astro units.")

        yb.write_hdf5(mbh_reass_cen, outloc, "ReassignedBHCentralMass", comment = "Sum of BH *particle* masses of central BH particles in this subhalo *after reassignment*, in astro units.")
        yb.write_hdf5(mbh_sg_reass_cen, outloc, "ReassignedBHCentralSubgridMass", comment = "Sum of BH *subgrid* masses of central BH particles in this subhalo *after reassignment*, in astro units.")

        yb.write_hdf5_attribute(outloc, "BHMass", "aexp-scale-exponent", 0)
        yb.write_hdf5_attribute(outloc, "BHMass", "h-scale-exponent", -1.0)
    
        print("")
        print("Finished snapshot CE-{:d}.{:d} in {:.3f} sec." 
              .format(isim, isnap, time.time()-snap_stime))
        print("")

    # ends loop through snapshots

# ends loop through simulations


print("Done!")
