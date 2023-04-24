"""
Re-compute central/satellite status of galaxies by considering their full
evolution over all snapshots. The aim is that, for each pair of galaxies that
are every occupying the same FOF group, the same one is always the central, so
that central/satellite swaps are eliminated. 

The result is stored in a single HDF5 file per simulation, located in 
[freya]/highlev/[outname]:

    <> 'CenGal_Regularized': pointer to the central *galaxy* of each galaxy 
       [first index] in each snapshot [second index], after the regularization
       procedure. For galaxies classed as centrals in a given snapshot, this 
       will point to themselves.

    <> 'SatFlag_Regularized': variable analogous to CenGal_Regularized, that
       is 0 if the galaxy is a central, 1 if it is a satellite, and 
       255 if it does not exist.

    <> 'Snapshot_[sss]/SatFlag': flag indicating whether subhalo [i] in this 
       snapshot is a central (0) or satellite (1), after regularization.

    <> 'Snapshot_[sss]/CentralSubhalo': pointer to the central subhalo of 
       subhalo [i] in this snapshot, after regularization.

 -- Started 12-Apr-2019
 -- Some tidying and extending (mergers) done 15-May-2019

"""

import sim_tools as st
import yb_utils as yb
import numpy as np
from pdb import set_trace
import monk
import time
import eagle_routines as er
from scipy.spatial import cKDTree
import os

rootdir = '/virgo/simulations/Eagle/L00100N1504/REFERENCE/'
simtype = 'HYDRO'
nsnap = 29
nsim = 1
verbose = False
outname = 'RegularizedCentrals.hdf5'

# Weight each snapshot for a potential central by the mass of its halo?
mass_weighting = True

# If multiple galaxies have equal weight, should the last-central be taken
# as the regularized central (True) or no change be made (False)?
enforce_strict_regularization = True    

def main():

    for isim in range(0, nsim):

        stime = time.time()

        rundir = rootdir# + 'CE-{:d}/' .format(isim) + simtype + '/'
        print("Analysing simulation '" + rundir + "'...")
        if not os.path.isdir(rundir): continue
            
        regularize_sim(isim, rundir)
        
        etime = time.time()
        print("Regularization of simulation {:d} took {:.3f} sec."
              .format(isim, etime-stime))

    print("Done!")


def regularize_sim(isim, rundir):
    """Regularize one (full) simulation."""

    global spiderloc, satFlag, cenGal, m200, cenGalNew, mergeList

    hldir = er.clone_dir(rundir) + 'highlev/'
    fgtloc = hldir + 'FullGalaxyTables.hdf5'
    spiderloc = hldir + 'SpiderwebTables.hdf5'

    satFlag = yb.read_hdf5(fgtloc, 'SatFlag')
    cenGal = yb.read_hdf5(fgtloc, 'CenGal')
    m200 = yb.read_hdf5(fgtloc, 'M200')
    mergeList = yb.read_hdf5(spiderloc, 'MergeList')

    outloc_sim = hldir + outname
    
    cenGalNew = np.copy(cenGal)

    n_swaps = 0
    n_deadHeat = 0

    for isnap in range(nsnap):
        nInc_swaps, nInc_deadHeat = regularize_snap(isim, isnap)
        n_swaps += nInc_swaps
        n_deadHeat += nInc_deadHeat

    maxChange = np.max(np.abs(cenGalNew-cenGal), axis = 1)
    ind_someChange = np.nonzero(maxChange > 0)[0]
    n_someChange = len(ind_someChange)

    print("")
    print("{:d}/{:d} galaxies had their central changed at some point." 
          .format(n_someChange, cenGal.shape[0]))
    print("{:d} changes took place in total, {:d} dead heats..." 
          .format(n_swaps, n_deadHeat))
    print("")

    write_regularized_result(cenGalNew, outloc_sim)
    print("Finished processing simulation {:d}" .format(isim))


def write_regularized_result(cenGalNew, outloc_sim):
    """Write output to file."""

    yb.write_hdf5(cenGalNew, outloc_sim, 'CenGal_Regularized', new = True, 
                  comment = "Pointer to the central *galaxy* of each galaxy "
                  "[first index] in each snapshot [second index], after the "
                  "regularization procedure. For galaxies classed as centrals "
                  "in a given snapshot, this will point to themselves.")

    # Also create a new satflag (combined for all snapshots):
    satFlag = np.zeros(cenGalNew.shape, dtype = np.int8) + 255
    shi = yb.read_hdf5(spiderloc, 'SubHaloIndex')

    for isnap in range(nsnap):
        gal_sh = yb.read_hdf5(spiderloc, 'Subhalo/Snapshot_' + 
                              str(isnap).zfill(3) + '/Galaxy')

        # Use gal_sh as index, so we only hit alive galaxies:
        ind_sat = np.nonzero(cenGalNew[gal_sh, isnap] != gal_sh)[0]
        satFlag[gal_sh[ind_sat], isnap] = 1

        ind_cen = np.nonzero(cenGalNew[gal_sh, isnap] == gal_sh)[0]
        satFlag[gal_sh[ind_cen], isnap] = 0
    
        # Array pointing to the central subhalo of each subhalo
        cenSH = shi[cenGalNew[gal_sh, isnap], isnap]

        yb.write_hdf5(satFlag[gal_sh, isnap], outloc_sim, 
                      'Snapshot_{:03d}/SatFlag' .format(isnap), 
                      comment = "Flag indicating whether subhalo [i] in this "
                      "snapshot is a central (0) or satellite (1), after "
                      "regularization.")
        yb.write_hdf5(cenSH, outloc_sim, 
                      'Snapshot_{:03d}/CentralSubhalo' .format(isnap), 
                      comment = "Pointer to the central subhalo of subhalo "
                      "[i] in this snapshot, after regularization.")

    yb.write_hdf5(satFlag, outloc_sim, 'SatFlag_Regularized',
                  comment = "Flag that is 0 if galaxy i [first index] is "
                  "a central in snapshot j [second index], 1 if it is a "
                  "satellite, and 255 if it does not exist.")

def regularize_snap(isim, isnap):
    """
    Regularize a specific snapshot.

    The result is recorded in the (global) variable cenGalNew.

    Parameters:
    -----------

    isim : int
        The simulation index to process
    isnap : int
        The snapshot index to process

    Returns:
    --------

    n_changed : int
        The number of central changes that were made.
    n_fail : int
        The number of dead heats encountered.
    """

    print("")
    print("============================")
    print("Processing snapshot {:d}.{:d}..." .format(isim, isnap))
    print("============================")    
    print("")

    ind_cen_thisSnap = np.nonzero(satFlag[:, isnap] == 0)[0]
    n_cen_thisSnap = len(ind_cen_thisSnap)

    print("There are {:d} centrals in snap {:d}..." 
          .format(n_cen_thisSnap, isnap))

    n_update = 0
    n_fail = 0
    for icen in ind_cen_thisSnap:
        nInc_update, nInc_fail = update_central(icen, isnap)
        n_update += nInc_update
        n_fail += nInc_fail

    print("")
    print("------------------------------------------")
    print("In snap {:d}, we updated {:d} centrals..." 
          .format(isnap, n_update))
    print("------------------------------------------")
    print("")

    # Still need step at the end where all galaxies are updated to their
    # new centrals (i.e., the new central of their old central):
    ind_alive = np.nonzero(cenGal[:, isnap] >= 0)[0]
    cenGalNew[ind_alive, isnap] = cenGalNew[cenGalNew[ind_alive, isnap], isnap]

    ind_changed = np.nonzero(cenGalNew[:, isnap] != cenGal[:, isnap])[0]
    n_changed = len(ind_changed)

    print("In total, {:d} central marks were changed in snap {:d}." 
          .format(n_changed, isnap))

    return n_changed, n_fail


def find_last_carrier(igal, candidates, mergeList):
    """
    Find the last carrier of a galaxy out of a set of candidates.

    If none of the candidates ever carries the galaxy, None is returned.
    """

    # First test: does the galaxy merge at all? If not, quit:
    if mergeList[igal, -1] == igal:
        return None

    # For each candidate, test in which snap (if any) it was the last
    # carrier, and update the 'global best' if appropriate:
    lastCarrierSnap = -1
    lastCarrierGal = None
    for icand in candidates:
        allCarrierSnaps = np.nonzero(mergeList[igal, :] == icand)[0]
        if len(allCarrierSnaps) > 0:
            lastCarrierSnap_curr = allCarrierSnaps[-1]
        if lastCarrierSnap_curr > lastCarrierSnap:
            lastCarrierGal = icand
            lastCarrierSnap = lastCarrierSnap_curr
        
    return lastCarrierGal

def find_merger_survivors(galaxies, mergeList):
    """
    Find galaxies that do not merge with each other. 

    For each galaxy in a given input set ('galaxies' --> their IDs), 
    test whether it merges with another one in the set. Return the IDs
    of those that do not merge in this way (there must be at least one!)
    """

    survivors = []   # Output list --> append as appropriate
    for igal in galaxies:
        carriers = mergeList[igal, :]
        nMerge = [np.count_nonzero(
            (carriers == _g) & (carriers != igal)) for _g in galaxies]
        if max(nMerge) == 0:
            survivors.append(igal)

    if len(survivors) == 0:
        print("Why is there not at least one survivor...??")
        set_trace()

    return np.array(survivors)

def find_highest_weighted(gal_shortlist, cenGalsThis):
    """
    Find the galaxy that has the strongest claim to being the central.
    
    For each galaxy in a given input set ('galaxies' --> their IDs),
    test how often it is a central (optionally weighted) hosting the 
    current test galaxy; the highest-valued galaxy is returned. If there
    is a tie, either find the galaxy that is a central the latest, or 
    return None to indicate that no change should be made.

    Parameters:
    -----------
    
    gal_shortlist:
        The input list of galaxy IDs.
    cenGalsThis:
        The centrals of the test galaxy in each snapshot.

    Returns:
    --------

    best_galaxy:
        The ID of the best galaxy (None if none could be found)
    n_deadHeat:
        1 if there was a dead heat between galaxies, 0 otherwise.
    """
    
    weights_shortlist = np.zeros(len(gal_shortlist))

    for iishort, ishort in enumerate(gal_shortlist):
        ind_short_cen = np.nonzero(cenGalsThis == ishort)[0]
        
        # In 'mass_weighting' mode, we sum the halo masses in all 
        # the snapshots in which the current shortlist entry was
        # a central. This implicitly favours galaxies that were a
        # central later in time, when haloes were more massive.
        # Otherwise, just count the snapshot numbers:

        if mass_weighting:
            weights_shortlist[iishort] = np.sum(
                10.0**m200[ishort, ind_short_cen])
        else:
            weights_shortlist[iishort] = len(ind_short_cen)

    # In general, pick the shortlist entry with the highest weight.
    # If there are more than one ('dead heat'), find the one that
    # is a central the latest:

    maxWeight = np.max(weights_shortlist)
    ind_max = np.nonzero(weights_shortlist == maxWeight)[0]
    if len(ind_max) == 0:
        print("?!? How can no galaxy be at the maximum weight?")
        set_trace()

    if len(ind_max) == 1:
        return gal_shortlist[ind_max[0]], 0   # No dead heat

    # If we get here, there are multiple highest-weighted candidates...
    print("Dead heat between {:d} galaxies..." .format(len(ind_max)))

    if enforce_strict_regularization: 
        lastCenSnap_all = -1
        for iimax in ind_max:
            lastCenSnap = np.nonzero(
                cenGalsThis == gal_shortlist[iimax])[0][-1]
            if lastCenSnap > lastCenSnap_all:
                lastCenSnap_all = lastCenSnap
                bestMax = iimax
        if bestMax is None:
            raise Exception("Could not find best max partner??")
        return gal_shortlist[ind_max[bestMax]], 1  # Decision after deat heat
            
    else:
        return None, 1   # Dead heat
 
        
def update_central(icen, isnap):
    """
    Test and (if necessary) update a specific central.

    This is the core function of the program. The result is recorded in the
    (global) variable cenGalNew.

    Parameters:
    -----------
 
    icen : int
        GalaxyID of the central galaxy to test ('test galaxy').
    isnap : int
        Snapshot in which the test galaxy is a central ('current snap').

    Returns:
    --------
    
    num_changes : int
        0 if no change was made, 1 otherwise.
    num_of_dead_heats : int
        0 if there was no dead heat, 1 otherwise.
    """

    n_deadHeat = 0

    # Find all snapshots in which the test galaxy did a cen/sat swap
    # with one of its current satellites, and the involved galaxies.
    cenGalsThis = cenGalNew[icen, :]    # Cens of test gal in all snaps
    ind_swapSnap = np.nonzero(
        (cenGalsThis >= 0)                     # Test gal must have existed...
        & (cenGalsThis != icen)                # ... been a sat...
        & (cenGal[cenGalsThis, isnap] == icen) # ... and its (then) cen is 
    )[0]                                       #     [ now its satellite. 

    # Simple case: no snapshots have swaps --> can exit immediately.
    if len(ind_swapSnap) == 0: return 0, 0   # 0 changes, 0 dead heats

    # Ok, there *are* swaps between test galaxy and its current sats...
    if verbose:
        print("Galaxy {:d} has cen/sat swaps..." .format(icen))

    # Identify the (unique) galaxies it has done swaps with
    # (the same one may show up in multiple snapshots):
    gal_partner = np.unique(cenGalsThis[ind_swapSnap])

    """
    We want to determine whether any of the 'partner' galaxies should
    also be chosen as the central of the test galaxy in the current snap.
    By doing this symmetrically, we will get consistent results when
    testing the partner galaxies in the snaps where they are centrals.
    """

    gal_shortlist = find_merger_survivors(
        np.append(gal_partner, icen), mergeList)

    # If there is only one galaxy on the shortlist, we're done already.
    # Otherwise, need to find out which of the survivors is cen 'most often':
    if len(gal_shortlist) == 1:
        gal_newCen = gal_shortlist[0]
    
    else:
        gal_newCen, n_deadHeat = find_highest_weighted(
            gal_shortlist, cenGalsThis)
        if gal_newCen is None:
            return 0, n_deadHeat      # 0 changes, 1 dead heat
    
    if gal_newCen == icen:
        return 0, n_deadHeat          # Don't need to change anything
    
    # At this point, we do need to change the central in current snap.
    if verbose:
        print("Update central from {:d} to {:d}..." 
              .format(icen, gal_newCen))

    cenGalNew[icen, isnap] = gal_newCen
    cenGalNew[gal_newCen, isnap] = gal_newCen
    return 1, n_deadHeat


if __name__ == "__main__":
    main()




