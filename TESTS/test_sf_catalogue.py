"""Test whether Subfind catalogue is consistent with snapshots."""

import hydrangea as hy
import numpy as np
from pdb import set_trace
import sim_tools as st

BASE_DIR = '/virgo/simulations/Hydrangea/'
sim = hy.objects.Simulation(index=0, base_dir=BASE_DIR)

sf_file = sim.get_subfind_file(29)
snap_file = sim.get_snap_file(29)
#snap_file = hy.form_files(sim.run_dir, 29, 'subpart')

subhalo = hy.SplitFile(sf_file, group_name='Subhalo', verbose=0)
sf_ids = hy.SplitFile(sf_file, group_name='IDs')
stars = hy.SplitFile(snap_file, part_type=4)
stars.subfind_file = sim.get_subfind_file(29)

for ii in range(1):
    ind_ii = np.nonzero(stars.SubhaloIndex == ii)[0]
    print("SH-M_init=", subhalo.StellarInitialMass[ii])
    print("snap-M_init=", np.sum(stars.InitialMass[ind_ii]))

print("Done!")
