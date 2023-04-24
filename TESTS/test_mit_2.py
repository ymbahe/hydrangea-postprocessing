import numpy as np
import yb_utils as yb
from pdb import set_trace
import hydrangea_tools as ht
import sim_tools as st

rundir = '/virgo/simulations/Hydrangea/10r200/CE-0/HYDRO/'


cantorloc = ht.clone_dir(rundir) + '/highlev/CantorCatalogue_24Jun19_TEST.hdf5'
igal = 24
isnap = 12
ptype = 4

cid = yb.read_hdf5(cantorloc, 'SubhaloIndex')[igal, isnap]
pre = 'Snapshot_' + str(isnap).zfill(3) 
eid = yb.read_hdf5(cantorloc, pre + '/Subhalo/Extra/ExtraIDs')[cid]

print("Galaxy {:d}, snap {:d}, cantorID={:d}, extraID={:d}."
      .format(igal, isnap, cid, eid))

readRad = 0.3
ap = 4
axInd = 2

aexp = yb.read_hdf5_attribute(cantorloc, pre, 'aExp') 
readRad_sim = readRad*0.6777/aexp

galpos = yb.read_hdf5(cantorloc, pre + '/Subhalo/CentreOfPotential')[cid, :]
galpos_sim = galpos*0.6777/aexp

snapdir = st.form_files(rundir, isnap, 'snap')

readReg = ht.ReadRegion(snapdir, ptype, [*galpos_sim, readRad_sim])

mass = readReg.read_data('Mass', astro = True)
pos = readReg.read_data('Coordinates', astro = True) - galpos[None, :]
ids = readReg.read_data('ParticleIDs')

ref_ids_all = yb.read_hdf5(cantorloc, pre + '/IDs')
ref_off = yb.read_hdf5(cantorloc, pre + '/Subhalo/OffsetType')

if ap < 4:
    ref_off_ap = yb.read_hdf5(
        cantorloc, pre + '/Subhalo/Extra/OffsetTypeApertures')
    ref_ids = ref_ids_all[ref_off[cid, ptype]:ref_off_ap[eid, ptype, ap]]
else:
    ref_ids = ref_ids_all[ref_off[cid, ptype]:ref_off[cid, ptype+1]]

refInds, ind = yb.find_id_indices(ids, ref_ids)

#rad = np.linalg.norm(pos, axis = 1)
#ind = np.nonzero(rad < readRad)[0]

ptOffset = yb.read_hdf5(
    cantorloc, pre + '/Subhalo/OffsetType')[cid, :]
#apOffset = yb.read_hdf5(
#    cantorloc, pre + '/Subhalo/Extra/OffsetTypeApertures')[eid, ptype, :]

#if len(ind) != apOffset[ap]-ptOffset: set_trace()
if len(ind) != ptOffset[ptype+1]-ptOffset[ptype]: set_trace()

mit = np.zeros((3, 3))
for xx in range(3):
    for yy in range(3):
        mit[xx, yy] = np.sum(pos[ind, xx]*pos[ind, yy]*mass[ind])

# Now diagonalise MIT:
eigVal, eigVec = np.linalg.eig(mit)

# Find which eigenvector is the major, intermediate, and minor axis:
sorter = np.argsort(eigVal)
ax_a = sorter[0]  # Minor axis
ax_b = sorter[1]  # Intermediate axis
ax_c = sorter[2]  # Major axis

# Arrange eigenvectors appropriately for output:
axisVecs = np.zeros((3, 3))
axisVecs[:, 0] = eigVec[:, ax_a]
axisVecs[:, 1] = eigVec[:, ax_b]
axisVecs[:, 2] = eigVec[:, ax_c]
        
# Calculate axis ratios int/major, minor/major:
axisRatios = np.zeros(2)
axisRatios[0] = abs(eigVal[ax_a]/eigVal[ax_c])
axisRatios[1] = abs(eigVal[ax_b]/eigVal[ax_c])

print("Minor: ", eigVec[:, ax_a], "\nIntermediate: ", eigVec[:, ax_b], 
      "\nMajor: ", eigVec[:, ax_c])
print("Ratios: {:.3f}, {:.3f}" .format(*axisRatios))

# Extract values from Cantor:
axes = yb.read_hdf5(cantorloc, pre + '/Subhalo/Extra/Stars/Axes')[eid, axInd, :, :]
axRat = yb.read_hdf5(cantorloc, pre + '/Subhalo/Extra/Stars/AxisRatios')[
    eid, axInd, :]

print("Cantor:")
print("Minor: ", axes[0, :], "\nIntermediate: ", axes[1, :], 
      "\nMajor: ", axes[2, :])
print("Ratios: {:.3f}, {:.3f}" .format(*axRat))


set_trace()
