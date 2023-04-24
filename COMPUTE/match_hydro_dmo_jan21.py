"""
Match subhaloes between hydro and DM-only runs

Started 20 Jan 2017
Updated and restructured 22 Jan 2021
"""

import numpy as np
import hydrangea as hy
from pdb import set_trace
import time
import calendar
import os
from tqdm import tqdm
from tools import TimeStamp
#from mpi4py import MPI

# Number of particles to use for matching. Also the minimum number of
# DM particles a subhalo must have for being considered.
n_tracers = 50

# Range of simulations and snapshots to process (first to beyond-last)
sim_range = [12, 30]
snap_range = [0, 30]

# Are we matching 'FOF' groups or 'Subhaloes'?
match_type = 'Subhalo'

# Set up MPI system, for acceleration
#comm = MPI.COMM_WORLD
#numtasks = comm.Get_size()
#rank = comm.Get_rank()

def main():
    for isim in range(sim_range[0], sim_range[1]):

        # Do not split across MPI tasks, since we will split on a
        # snap-by-snap basis
        process_sim(isim)
        

def process_sim(isim):
    """Process one particular simulation."""
    
    # Set up the simulation objects and output files
    sim_hy = hy.Simulation(isim)
    sim_dm = hy.Simulation(isim, sim_type='DM')

    # There is little point continuing unless both DM and HYDRO sims exist...
    if not os.path.exists(sim_hy.run_dir):
        print("No HYDRO rundir...")
        return
    if not os.path.exists(sim_dm.run_dir):
        print("No DM rundir...") 
        return

    print("")
    print("=============================")
    print(f"Processing simulation CE-{isim}")
    print("=============================")
    print("", flush = True)

    for isnap in range(snap_range[0], snap_range[1]):
        # Skip this one if we are multi-threading and it's not for this task
        #if isnap % numtasks == rank:
        process_snap(isim, isnap)


def process_snap(isim, isnap):
    """Core function that processes one snapshot in one simulation."""

    ts = TimeStamp()
    
    # Set up the simulation objects and output files
    # (yes, we could pass them from process_sim(), but this takes no time)
    sim_hy = hy.Simulation(isim)
    sim_dm = hy.Simulation(isim, sim_type='DM')
    outloc_hy = sim_hy.high_level_dir + '/SubhaloExtra.hdf5'
    outloc_dm = sim_dm.high_level_dir + '/SubhaloExtra.hdf5'
    
    print("")
    print("----------------------")
    print(f"Snapshot {isnap} (CE-{isim})...")
    print("----------------------")
    print("")

    subdir_hy = sim_hy.get_subfind_file(isnap)
    subdir_dm = sim_dm.get_subfind_file(isnap)
    
    # To be on the safe side, verify that SUBFIND catalogues exist...
    if (not os.path.exists(subdir_hy)) or (not os.path.exists(subdir_dm)):
        print("Why does a simulation output not have SUBFIND?")
        set_trace()

    ts.set_time('Initialization')
        
    # Load required data from subfind catalogues
    hydro = load_data(subdir_hy)
    dmo = load_data(subdir_dm)

    ts.set_time('Data reading')
    
    print("Set up ID gate...", end='', flush=True)
    id_gate = hy.crossref.Gate(hydro['IDs'], dmo['IDs'])
    ts.set_time('ID gate setup')
    print(f" done ({ts.get_time():.2f} sec.)")
    
    # ------- Find matches IN DMO ('outbound match') ---------

    # Create list of hydro subhaloes to match (everything)
    hydro['ToMatch'] = np.arange(hydro['NumSH'], dtype = np.int32)
    
    # Do the particle-level matching HYDRO --> DMO...
    print("Now matching HYDRO --> DMO...")
    match_time_a = time.time()
    match_in_dmo = match_subhaloes(hydro, dmo, id_gate)

    # ... and see what we got
    ind_matched_from_hy_to_dmo = np.nonzero(match_in_dmo >= 0)[0]
    n_good_in_dmo = len(ind_matched_from_hy_to_dmo)
    match_fraction = n_good_in_dmo / hydro['NumSH']
    ts.set_time('Hydro --> DMO match')
    print(f" ... done (took {ts.get_time():.2f} sec.)")
    print(f"Found {n_good_in_dmo} tentative matches in DMO "
          f"(= {match_fraction * 100:.2f}%), now doing reverse matching...")

    # ------- Find matches IN HYDRO ('inbound match') ---------

    # Create list of DMO subhaloes to match (only the ones that were selected
    # as targets of at least one HYDRO subhalo!). And don't check targets
    # of more than one HYDRO subhalo more than once, it's a waste of time.
    dmo_targets = match_in_dmo[ind_matched_from_hy_to_dmo]
    dmo['ToMatch'] = np.unique(dmo_targets)

    # And again, do the particle-level matching (now DMO --> HYDRO)...
    print("Now matching DMO --> HYDRO...")
    match_in_hy = match_subhaloes(dmo, hydro, id_gate, use_gate_forward=False)

    # ... and see what we got
    ind_matched_from_dmo_to_hy = np.nonzero(match_in_hy >= 0)[0]
    good_hy_targets = match_in_hy[ind_matched_from_dmo_to_hy]
    n_good_in_hydro = len(ind_matched_from_dmo_to_hy)
    match_fraction = n_good_in_hydro / len(dmo['ToMatch'])

    ts.set_time('DMO --> Hydro match')
    print(f" ... done (took {ts.get_time():.2f} sec.)")
    print(f"Found {n_good_in_hydro} tentative matches in HYDRO "
          f"(= {match_fraction * 100:.2f}%).")

    # `good_hy_targets` is now a list of HYDRO subhaloes that are the
    # match targets of at least one DMO subhalo. We can exclude all other
    # HYDRO ones, but still need to check which of these themselves link back
    # to their DMO starting point (i.e. the successes out of dmo['ToMatch']):
    dmo_subind_bijective = np.nonzero(
        match_in_dmo[good_hy_targets] ==
        dmo['ToMatch'][ind_matched_from_dmo_to_hy])[0]

    n_successful = len(dmo_subind_bijective)
    print(f"Successfully matched {n_successful} subhaloes "
          f"(= {n_successful / hydro['NumSH'] * 100:.2f}% of Hydro, "
          f"{n_successful / dmo['NumSH'] * 100:.2f}% of DMO).")
    
    # Finally, translate this to a direct match between Hydro and DMO
    # indices. Note from above that `good_hy_targets` is aligned to
    # `ind_matched_from_dmo_to_hy`, so both can be indexed directly with
    # `dmo_subind_bijective`.
    dmo_ind_bijective = (
        dmo['ToMatch'][ind_matched_from_dmo_to_hy[dmo_subind_bijective]])
    hydro_ind_bijective = good_hy_targets[dmo_subind_bijective]
    dmo['BijectiveMatch'][dmo_ind_bijective] = hydro_ind_bijective
    hydro['BijectiveMatch'][hydro_ind_bijective] = dmo_ind_bijective

    ts.set_time('Find bijective matches')
    
    # Do some sanity checks (add 1 to avoid illegal negative values...)
    bc = np.bincount(dmo['BijectiveMatch'] + 1)
    if np.max(bc[1:]) > 1:
        print("Hydro subhaloes multiply assigned to DMO!")
        set_trace()

    bc = np.bincount(hydro['BijectiveMatch'] + 1)
    if np.max(bc[1:]) > 1:
        print("DMO subhaloes multiply assigned to Hydro!")
        set_trace()

    ts.set_time('Sanity checks')
        
    # Write output
    hy.hdf5.write_data(
        outloc_hy, f"Snapshot_{isnap:03d}/MatchInDM", hydro['BijectiveMatch'],
        comment="Index of the subhalo in the DM-only simulation that is "
        "bijectively matched to subhalo [i]. A value of -1 means that [i] "
        "has no bijective match in DM-only.")
    hy.hdf5.write_data(
        outloc_dm, f"Snapshot_{isnap:03d}/MatchInHydro", dmo['BijectiveMatch'],
        comment="Index of the subhalo in the Hydro simulation that is "
        "bijectively matched to subhalo [i]. A value of -1 means that [i] "
        "has no bijective match in Hydro.")
    hy.hdf5.write_attribute(outloc_dm, f"Snapshot_{isnap:03d}/MatchInHydro",
                            "NumTracers", n_tracers, group=False)
    hy.hdf5.write_attribute(outloc_hy, f"Snapshot_{isnap:03d}/MatchInDM",
                            "NumTracers", n_tracers, group=False)

    ts.set_time('Output writing')
    ts.print_time_usage(f"Finished processing snapshot {isnap} (CE-{isim})",
                        minutes=True)
    
    
def load_data(subdir):
    """Load the required data for a particular snapshot.

    Parameters
    ----------
    subdir : string
        The (full) path to the (first) file of the Subfind catalogue to load.

    Returns
    -------
    catalogue : dict of arrays
        The data required from this particular subfind catalogue.
    """

    # We always need the Subfind particle IDs
    catalogue = {'IDs': hy.SplitFile(subdir, 'IDs').ParticleID}

    # Offset and length depend on whether we match FOF groups or subhaloes
    if match_type.lower() == 'fof':
        fof = hy.SplitFile(subdir, 'FOF')
        catalogue['Offsets'] = fof.GroupOffset
        catalogue['Lengths'] = fof.GroupLength
    else:
        subhalo = hy.SplitFile(subdir, 'Subhalo')
        catalogue['Offsets'] = subhalo.SubOffset
        catalogue['Lengths'] = subhalo.SubLength

    # Set up some things for later use
    catalogue['NumSH'] = len(catalogue['Lengths'])
    catalogue['BijectiveMatch'] = -np.ones(catalogue['NumSH'], dtype=np.int32)
        
    return catalogue
    

def match_subhaloes(sim_a, sim_b, id_gate, use_gate_forward=True):
    """Find matching subhaloes for simulation A in simulation B.
    
    Parameters
    ----------
    sim_a : dict
        Dict containing the subfind data arrays for simulation A
    sim_b : dict
        Dict containing the subfind data arrays for simulation B
    """

    ts = TimeStamp()
    # Output array, initialised to -1 (placeholder for "not matched")
    match_in_b = np.zeros(len(sim_a['ToMatch']), dtype = int) - 1
    ts.set_time('Initialize output')

    print("Transcribe Sim-B subhalo list...", end='', flush=True)
    subhaloes_b = np.zeros(len(sim_b['IDs']), dtype=np.int32) - 1
    for ish in range(sim_b['NumSH']):
        subhaloes_b[sim_b['Offsets'][ish] :
                    sim_b['Offsets'][ish] + sim_b['Lengths'][ish]] = ish
    ts.set_time('Transcribe Sim-B subhalo list')
    print(f" done ({ts.get_time():.2f} sec.)")
    
    # Locate IDs of sim A in sim B:
    #print(f"  ... matching {len(sim_a['IDs'])} --> {len(sim_b['IDs'])} "
    #      "IDs... ", end='', flush=True)
    #match_time = time.time()
    #a_inds_in_b, a_matched = hy.crossref.find_id_indices(
    #    sim_a['IDs'], sim_b['IDs'])
    #print(f"done ({time.time() - match_time:.3f} sec.)")
    #print(f"Matching {len(sim_a['ToMatch'])} subhaloes...")


    ts_inds = ts.add_counters(['Find DM', 'Locate', 'Link to Subhaloes',
                               'Find in Subhaloes', 'Subhalo histogram',
                               'Find target'])
    
    # Main loop over (sim-A) subhaloes
    for ish, sh in enumerate(tqdm(sim_a['ToMatch'])):

        ts.start_time()
        
        #if ish % 1000 == 0:
        #    print(f"...reached subhalo {ish}/{sim_a['NumSH']}...")

        # Find (indices of) DM particles in current subhalo in full list
        curr_off_a, curr_len_a = sim_a['Offsets'][sh], sim_a['Lengths'][sh]
        ind_dm_parts_curr = (
            np.nonzero(sim_a['IDs'][curr_off_a:(curr_off_a+curr_len_a)] % 2
                       == 0)[0] + curr_off_a)
        
        # Limit particle list to max number of tracers. Keep in mind that
        # particles are sorted in order of binding energy, so this directly
        # selects the most bound ones.
        if len(ind_dm_parts_curr) > n_tracers:
            ind_dm_parts_curr = ind_dm_parts_curr[:n_tracers]
            ts.increase_time(index=ts_inds[0])
        else:
            ts.increase_time(index=ts_inds[0])
            continue  # Don't trace DM-poor subhaloes
        
        # Locate these particle IDs in sim-B, rejecting unmatched particles
        if use_gate_forward:
            indices_in_b, inds_in_b_in_subfind = id_gate.in_int(
                ind_dm_parts_curr)
        else:
            indices_in_b, inds_in_b_in_subfind = id_gate.in_ext(
                ind_dm_parts_curr)

        #test = a_inds_in_b[ind_dm_parts_curr]
        #inds_in_b_in_subfind = np.nonzero(indices_in_b >= 0)[0]
        indices_in_b = indices_in_b[inds_in_b_in_subfind]

        ts.increase_time(index=ts_inds[1])
        
        # Find the potential matching B subhalo for each particle, rejecting
        # particles that are not in any subhalo
        #sh_b_per_particle = hy.tools.ind_to_block(
        #    indices_in_b, sim_b['Offsets'], sim_b['Lengths'])
        sh_b_per_particle = subhaloes_b[indices_in_b]
        
        ts.increase_time(index=ts_inds[2])

        inds_in_subhalo = np.nonzero(sh_b_per_particle >= 0)[0]
        sh_b_per_particle = sh_b_per_particle[inds_in_subhalo]

        ts.increase_time(index=ts_inds[3])
        
        # Find the (sim-B) subhalo, if any, that contains > 50% of the
        # tracer particles. The default of -1 is already set.
        if len(sh_b_per_particle) > n_tracers / 2:

            unique_sh, sh_counts = np.unique(sh_b_per_particle,
                                             return_counts=True)
            #match_hist = np.bincount(sh_b_per_particle)
            ts.increase_time(index=ts_inds[4])

            if np.max(sh_counts) > n_tracers / 2:
                match_in_b[ish] = unique_sh[np.argmax(sh_counts)]
                ts.increase_time(index=ts_inds[5])


    ts.print_time_usage("Finished subhalo matching", minutes=False)
        
    return match_in_b


if __name__ == "__main__":
    main()
    print("All done!")
