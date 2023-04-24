"""
Determine which subhalos lie in the resolved high-resolution region

This program loops through haloes and snapshots, and checks whether
each of the subhaloes lie closer than a given threshold value
from a low-res boundary particle. The resulting flag-list is then
written out into the subfind groups_xxx directory as a 
single-dataset HDF5 file.

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


rundir_base = '/virgo/simulations/Hydrangea/C-EAGLE/'
outname = 'BoundaryFlag.hdf5'

n_halo = 30
n_snap = 30

flag_redo = True    # Set to true to force re-computation of already existing files

datestamp = '8-May-2018'

gap_lax = 1.0
gap_std = 2.0   # Standard distance from boundary particles, in cMpc
gap_strict = 5.0
gap_cons = 8.0  # Conservative distance from "      "      , "   "

comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()

for ihalo in [17, 19, 20, 23, 26, 27]:
    
    hstime = time.time()

    print("")
    print("**************************")
    print("Now processing halo F{:d}" .format(ihalo))
    print("**************************")
    print("")
     
    rundir = rundir_base + '/HaloF' + str(ihalo) + '/HYDRO'
    if not os.path.exists(rundir):
        continue

    for isnap in range(30):

        # Skip this one if we are multi-threading and it's not for this task to worry about
        if not isnap % numtasks == rank:
            #print("Skipping frame", iframe, "because it's not ours")
            continue

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

            
        sh_pos, conv_astro, aexp  = st.eagleread(subdir, 'Subhalo/CentreOfPotential', astro = True)
        sh_pos /= aexp  # Converts positions to cMpc, but without h

        nsh = sh_pos.shape[0]

        t2_pos = st.eagleread(snapdir, 'PartType2/Coordinates', astro = True)[0] / aexp
        t3_pos = st.eagleread(snapdir, 'PartType3/Coordinates', astro = True)[0] / aexp

        tpos = np.concatenate((t2_pos, t3_pos))

        # Now select those boundary particles 'near' subhaloes

        xmin,ymin,zmin = sh_pos.min(axis=0)-gap_cons-0.5
        xmax,ymax,zmax = sh_pos.max(axis=0)+gap_cons+0.5

        ind_sel = np.nonzero((tpos[:,0] >= xmin) & (tpos[:,0] <= xmax) &
                             (tpos[:,1] >= ymin) & (tpos[:,1] <= ymax) &
                             (tpos[:,2] >= zmin) & (tpos[:,2] <= zmax))[0]

        print("   ... found {:d} subhaloes and {:d} boundary particles within [{:.1f}, {:.1f}, {:.1f}] cMpc region... "
              .format(nsh, len(ind_sel), (xmax-xmin), (ymax-ymin), (zmax-zmin)))


        tree = scipy.spatial.cKDTree(tpos[ind_sel,:])    # Note we do not include periodic wrapping here - zooms
        tree_sh = scipy.spatial.cKDTree(sh_pos)
        
        ngbs_lax = tree_sh.query_ball_tree(tree, gap_lax)
        ngbs_std = tree_sh.query_ball_tree(tree, gap_std)
        ngbs_strict = tree_sh.query_ball_tree(tree, gap_strict)
        ngbs_cons = tree_sh.query_ball_tree(tree, gap_cons)

        nngb_lax = np.array([len(_list) for _list in ngbs_lax])
        nngb_std = np.array([len(_list) for _list in ngbs_std])
        nngb_strict = np.array([len(_list) for _list in ngbs_strict])
        nngb_cons = np.array([len(_list) for _list in ngbs_cons])

        flaglist = np.zeros(nsh,dtype=np.byte)
        flaglist_multi = np.zeros(nsh, dtype = np.byte)

        ind_cont_lax = np.nonzero(nngb_lax > 0)[0]
        ind_cont_std = np.nonzero(nngb_std > 0)[0]
        ind_cont_strict = np.nonzero(nngb_strict > 0)[0]
        ind_cont_cons = np.nonzero(nngb_cons > 0)[0]
        
        print("   ... {:d} ({:d}, {:d}, {:d}) subhaloes contaminated with {:.1f} ({:.1f}, {:.1f}, {:.1f}) cMpc exclusion radius ... "
              .format(len(ind_cont_std), len(ind_cont_lax), len(ind_cont_strict), len(ind_cont_cons), 
                      gap_std, gap_lax, gap_strict, gap_cons))

        flaglist[ind_cont_cons] = 1
        flaglist[ind_cont_std] = 2

        flaglist_multi[ind_cont_cons] = 1
        flaglist_multi[ind_cont_strict] = 2
        flaglist_multi[ind_cont_std] = 3
        flaglist_multi[ind_cont_lax] = 4

        yb.write_hdf5(flaglist, outloc, 'ContaminationFlag', new=True)
        yb.write_hdf5_attribute(outloc, 'ContaminationFlag', 'StandardLimit', 2.0, new = False, group = False)
        yb.write_hdf5_attribute(outloc, 'ContaminationFlag', 'ConservativeLimit', 8.0, new = False, group = False)
        yb.write_hdf5_attribute(outloc, 'Header', 'DateStamp', np.string_(datestamp), new = False, group = True)

        yb.write_hdf5(flaglist_multi, outloc, 'ContaminationFlagMulti', new=False)        
        yb.write_hdf5_attribute(outloc, 'ContaminationFlagMulti', 'StandardLimit', 2.0, new = False, group = False)
        yb.write_hdf5_attribute(outloc, 'ContaminationFlagMulti', 'LaxLimit', 1.0, new = False, group = False)
        yb.write_hdf5_attribute(outloc, 'ContaminationFlagMulti', 'StrictLimit', 5.0, new = False, group = False)
        yb.write_hdf5_attribute(outloc, 'ContaminationFlagMulti', 'ConservativeLimit', 8.0, new = False, group = False)
        
        
print("Done!")
