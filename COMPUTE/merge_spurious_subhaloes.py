"""
Identify artificial subhaloes that lie very close to others.

This program checks each subhalo in an output for other subhaloes that lie 
within an exclusion radius that is min(3 pkpc, R_half^star) and belong to
the same FOF group. 

If one subhalo is 'claimed' by multiple other (more massive) subhaloes,
it is assigned to the closest one, and then passed on if this is itself
claimed by another.

Started 13 JAN 2017 

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
import eagle_routines as er

simname = 'EAGLE'
simtype = 'HYDRO'
datestamp = '26Oct17'

if simname == 'Hydrangea':
    basedir = '/virgo/simulations/Hydrangea/10r200/'
    nsnap = 30
    nsim = 30

elif simname == 'EAGLE':
    basedir = '/virgo/simulations/Eagle/L0100N1504/REFERENCE/'
    nsnap = 29
    nsim = 1

outname = 'SubhaloMergerFlag.hdf5'

flag_redo = True    # Set to true to force re-computation of already existing files

max_xcl_rad = 3.0 / 1000   # Maximum exclusion radius, in pMpc

comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()


for ihalo in range(nsim):
    
    if simname == 'Hydrangea':
        rundir = basedir + 'HaloF' + str(isim) + '/' + simtype

    elif simname == 'EAGLE':
        rundir = basedir

    if not os.path.exists(rundir):
        continue

    hstime = time.time()

    print("")
    print("**************************")
    print("Now processing halo F{:d}" .format(ihalo))
    print("**************************")
    print("")

    for isnap in range(nsnap):

        # Skip this one if we are multi-threading and it's not for this task to worry about
        if not isnap % numtasks == rank:
            #print("Skipping frame", iframe, "because it's not ours")
            continue

        subdir = st.form_files(rundir, isnap=isnap, types = 'sub', stype = 'snap')
        
        print(" --- Snapshot {:d} ---" .format(isnap))

        if simname == 'Hydrangea':
            smfdir = yb.dir(subdir)
        elif simname == 'EAGLE':
            smfdir = er.clone_dir(yb.dir(subdir))
            if not os.path.exists(smfdir):
                os.mkdir(smfdir)

        outloc = smfdir + '/' + outname

        if not flag_redo:
            if os.path.exists(outloc):
                continue
                
        if not os.path.exists(subdir):
            print("   ... Subfind output not found, skipping S{:d}... " .format(isnap))
            continue
            
        sh_pos, conv_astro, aexp  = st.eagleread(subdir, 'Subhalo/CentreOfPotential', astro = True)
        sh_mass = st.eagleread(subdir, 'Subhalo/Mass', astro = True)[0]
        sh_rhalf = st.eagleread(subdir, 'Subhalo/HalfMassRad', astro = True)[0]    # in pMpc
        sh_rhalf = sh_rhalf[:,4] # Select stellar half-mass radius
        sh_fof = st.eagleread(subdir, 'Subhalo/GroupNumber', astro = False)-1
        
        nsh = sh_pos.shape[0]

        print("There are {:d} subhaloes to check..." .format(nsh))

        # First step: build a tree from all subhalo positions
        tree = scipy.spatial.cKDTree(sh_pos)

        print("Finished building the tree...")

        flaglist = [list([]) for _ in range(nsh)]

        # Loop through each subhalo and flag potential spurious subhaloes near it:
        for ish in range(nsh):

            xcl_rad = np.min([sh_rhalf[ish], max_xcl_rad])
            ngblist = tree.query_ball_point(sh_pos[ish,:], xcl_rad)
            
            ngblist = np.array(ngblist)

            ind_fake = np.nonzero((ngblist != ish) &   # Explicitly exclude subhalo as neighbour of itself...
                                  (sh_mass[ngblist] < sh_mass[ish]) &    # Only exclude smaller things
                                  (sh_fof[ngblist] == sh_fof[ish]))[0]   # Only exclude things in same FOF group
            
            for ifake in ngblist[ind_fake]:
                flaglist[ifake].append(ish)

        nparent = np.array([len(list_) for list_ in flaglist])

        ind_bad = np.nonzero(nparent > 0)[0]
        nbad = len(ind_bad)

        revlist_bad = np.zeros(nsh+1,dtype=int)-1
        revlist_bad[ind_bad] = np.arange(nbad)

        print("There are {:d} artificial subhaloes ({:.2f} per cent)" .format(nbad, nbad/nsh*100))
                        
        parentlist = np.zeros(nbad, dtype = int)-1


        # The simple case: only one parent claiming a bad subhalo:
        ind_simple = np.nonzero(nparent[ind_bad] == 1)[0]
        
        flaglist = np.array(flaglist)
        parentlist[ind_simple] = [list_[0] for list_ in flaglist[ind_bad[ind_simple]]]

        # The not so simple case: several parents arguing over custody:
        ind_complicated = np.nonzero(nparent[ind_bad] > 1)[0]

        if len(ind_complicated) > 0:
            
            for icompl in ind_complicated:

                n_parent_candidates = nparent[ind_bad[icompl]]
                distlist = np.zeros(n_parent_candidates)

                for iparent in range(n_parent_candidates):
                    dist_sq = np.sum((sh_pos[ind_bad[icompl],:]-sh_pos[flaglist[ind_bad[icompl]][iparent]])**2)
                    distlist[iparent] = dist_sq

                ind_best_parent = np.argmin(distlist)
                parentlist[icompl] = flaglist[ind_bad[icompl]][ind_best_parent]
               
 
        # Finished computing initial parent list
        # Now, we need to check that no parents are junk themselves...

        iround = 0

        while True:
            
            iround += 1
            print("Performing parent correction round {:d}..." .format(iround))
    
            ind_badparent = np.nonzero(nparent[parentlist] > 0)[0]
            n_badparent = len(ind_badparent)

            print("... {:d} parent entries need correcting..." .format(n_badparent))

            if n_badparent == 0:
                break

            # At this point, we have some bad parents that must be dealt with...

            badparent_sh = np.unique(parentlist[ind_badparent]) # List of all bad parent SHs

            revlist_badparent_sh = np.zeros(nsh+1, dtype = int)-1
            revlist_badparent_sh[badparent_sh] = np.arange(len(badparent_sh))

            better_parent_list = np.zeros(len(badparent_sh))

            for iibadpar, ibadpar in enumerate(badparent_sh):
                ind_in_bad = revlist_bad[ibadpar]    # Index in ind_bad of the current trouble-making parent
                better_parent_list[iibadpar] = parentlist[ind_in_bad]    # The (grand-)parent of the troublemaker
            
            # Now go through all to-be corrected entries
            # and correct their parent info:
    
            for ii, iparent in enumerate(parentlist[ind_badparent]):
                parentlist[ind_badparent[ii]] = better_parent_list[revlist_badparent_sh[iparent]]
        
        
        # Need to write output
        yb.write_hdf5(ind_bad, outloc, 'Spurious', new=True)
        yb.write_hdf5(parentlist, outloc, 'Parents')
        yb.write_hdf5_attribute(outloc, 'Header', 'DateStamp', np.string_(datestamp), new = False, group = True)
        yb.write_hdf5_attribute(outloc, 'Header', 'MaxLimit', np.string_('3.0 pkpc'))
        yb.write_hdf5_attribute(outloc, 'Header', 'Comment', np.string_('Children assigned to closest subhalo as parent'))



print("Done!")
