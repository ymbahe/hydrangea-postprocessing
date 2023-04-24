import numpy as np
import yb_utils as yb
import sim_tools as st
import hydrangea_tools as ht
from pdb import set_trace

rundir = '/virgo/simulations/Hydrangea/10r200/CE-0/HYDRO/'
hldir = rundir + 'highlev/'

fgtloc = hldir + 'FullGalaxyTables.hdf5'
posloc = hldir + 'GalaxyPositionsSnap.hdf5'

galID = 0
isnap = 0
readRad = 0.03
ptype = 1

pos_gal_all = yb.read_hdf5(posloc, 'Centre')[:, isnap, :]
galpos = pos_gal_all[galID, :]

snapdir = st.form_files(rundir, isnap, 'snap')
aexp_factor = st.snap_age(snapdir, type = 'aexp')

conv_astro_pos = aexp_factor/0.6777

readRad_sim = readRad/conv_astro_pos
galpos_sim = galpos/conv_astro_pos


readReg = ht.ReadRegion(snapdir, ptype, [*galpos_sim, readRad_sim])

ids = readReg.read_data("ParticleIDs", astro = False)

cantorloc = (ht.clone_dir(rundir) 
             + '/highlev/CantorCatalogue_19Jun19_noCen.hdf5')

snap_pre = 'Snapshot_' + str(isnap).zfill(3) + '/'

shi = yb.read_hdf5(cantorloc, 'SubhaloIndex')[galID, isnap]

ref_off = yb.read_hdf5(cantorloc, snap_pre + 'Subhalo/Offset')
ref_len = yb.read_hdf5(cantorloc, snap_pre + 'Subhalo/Length')
ref_offType = yb.read_hdf5(cantorloc, snap_pre + 'Subhalo/OffsetType')

ref_ids_all = yb.read_hdf5(cantorloc, snap_pre + 'IDs')
ref_ids = ref_ids_all[ref_off[shi]:ref_off[shi]+ref_len[shi]]
ref_ids_ptype = ref_ids_all[ref_offType[shi, ptype]:ref_offType[shi, ptype+1]]

gate = st.Gate(ids, ref_ids)
ref_index = gate.in2()
ind_in_ref = np.nonzero(ref_index >= 0)[0]

gate2 = st.Gate(ids, ref_ids_ptype)
ref_index_ptype = gate2.in2()
ind_in_ref_ptype = np.nonzero(ref_index_ptype >= 0)[0]

set_trace()
