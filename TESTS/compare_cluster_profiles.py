"""
Extract cluster's radial profiles to compare definitions.

Started 24-Jun-19
"""

import sim_tools as st
import yb_utils as yb
import numpy as np
import hydrangea_tools as ht
from pdb import set_trace

simdir = '/virgo/simulations/Hydrangea/10r200/CE-0/HYDRO/'
hldir = simdir + 'highlev/'

outloc = '/virgo/scratch/ybahe/HYDRANGEA/RESULTS/CantorProfiles_24Jun19.hdf5'

galID = 760
maxRad = 1.0
isnap = 29

posloc = hldir + 'GalaxyPositionsSnap.hdf5'
galpos_all = yb.read_hdf5(posloc, 'Centre')[:, isnap, :]
cl_pos = galpos_all[galID, :]
cl_pos_sim = cl_pos * 0.6777
readRad_sim = maxRad * 0.6777

fgtloc = hldir + 'FullGalaxyTables.hdf5'
shi_all = yb.read_hdf5(fgtloc, 'SHI')[:, isnap]
shi_cl = shi_all[galID]

ptype = 4

subdir, snapdir = st.form_files(simdir, 29, 'sub snap')
readReg = ht.ReadRegion(snapdir, ptype, [*cl_pos_sim, readRad_sim])

pos = readReg.read_data("Coordinates", astro = True)
mass = readReg.read_data("Mass", astro = True)
ids = readReg.read_data("ParticleIDs", astro = False)
revID = yb.create_reverse_list(ids, maxval = int(1e9))
rad = np.linalg.norm(pos - cl_pos[None, :], axis = 1)

binEdges = np.logspace(-3, 0, 101, endpoint=True)
def extract_profile(inds):
    
    profile = np.zeros(100)
    sorter = np.argsort(rad[inds])
    splits = np.searchsorted(rad[inds], binEdges, sorter=sorter)

    for ibin in range(len(binEdges)-1):
        ind_curr = inds[sorter[splits[ibin]:splits[ibin+1]]]
        profile[ibin] = np.sum(mass[ind_curr])

    return profile



# SF first
ids = st.eagleread(subdir, 'IDs/ParticleID', astro = False)
offset = st.eagleread(subdir, 'Subhalo/SubOffset', astro = False)[shi_cl]
length = st.eagleread(subdir, 'Subhalo/SubLength', astro = False)[shi_cl]
ids_cl = ids[offset:offset+length]
ind_cl = revID[ids_cl]

del ids
del offset
del length

profile = np.zeros((7, 100))

profile[0, :] = extract_profile(np.arange(len(rad)))
profile[1, :] = extract_profile(ind_cl)

# Now various Cantors:

CantorFiles = ['noCen']#, 'noSF', 'noSats', 'noMerge', 'noRef', 'noPrior']

names = np.array(CantorFiles)
names = np.insert(names, 0, 'Subfind')

for iic, cantor in enumerate(CantorFiles):

    cantorloc = ht.clone_dir(hldir) + 'CantorCatalogue_19Jun19_' + cantor + '.hdf5'
    cid = yb.read_hdf5(cantorloc, 'SubhaloIndex')[galID, isnap]

    set_trace()
    ids_all = yb.read_hdf5(cantorloc, 'Snapshot_029/IDs')
    off_all = yb.read_hdf5(cantorloc, 'Snapshot_029/Subhalo/Offset')
    len_all = yb.read_hdf5(cantorloc, 'Snapshot_029/Subhalo/Length')
    #ids_cl = ids_all[off_all[cid, ptype]:off_all[cid, ptype+1]]
    ids_cl = ids_all[off_all[cid]:
                     off_all[cid]+len_all[cid]]
    ind_cl = revID[ids_cl]

    profile[2, :] = extract_profile(ind_cl)


yb.write_hdf5(profile, outloc, 'Profiles', new = True)
yb.write_hdf5(binEdges, outloc, 'BinEdges')
#yb.write_hdf5(names, outloc, 'Names')



print("Done!")
