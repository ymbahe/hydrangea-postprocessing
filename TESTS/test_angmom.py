import numpy as np
import yb_utils as yb
from pdb import set_trace
import hydrangea_tools as ht
import sim_tools as st

rundir = '/virgo/simulations/Hydrangea/10r200/CE-0/HYDRO/'

cantorloc = ht.clone_dir(rundir) + '/highlev/CantorCatalogue_24Jun19.hdf5'
igal = 24
isnap = 29
ptype = 4

cid = yb.read_hdf5(cantorloc, 'SubhaloIndex')[igal, isnap]
pre = 'Snapshot_' + str(isnap).zfill(3) 
eid = yb.read_hdf5(cantorloc, pre + '/Subhalo/Extra/ExtraIDs')[cid]

print("Galaxy {:d}, snap {:d}, cantorID={:d}, extraID={:d}."
      .format(igal, isnap, cid, eid))

rMax = yb.read_hdf5(cantorloc, pre + '/Subhalo/MaxRadiusType')[cid, ptype]

readRad = rMax
ap = 1

if ap == 1:
    eind = 0
elif ap == 2:
    eind = 1

aexp = yb.read_hdf5_attribute(cantorloc, pre, 'aExp') 
readRad_sim = readRad*0.6777/aexp

galpos = yb.read_hdf5(cantorloc, pre + '/Subhalo/CentreOfPotential')[cid, :]
galpos_sim = galpos*0.6777/aexp

snapdir = st.form_files(rundir, isnap, 'snap')

readReg = ht.ReadRegion(snapdir, ptype, [*galpos_sim, readRad_sim])

mass = readReg.read_data('Mass', astro = True)
vel = readReg.read_data('Velocity', astro = True)
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

if ap == 4:
    if len(ind) != ptOffset[ptype+1]-ptOffset[ptype]: set_trace()
else:
    apOffset = yb.read_hdf5(
        cantorloc, pre + '/Subhalo/Extra/OffsetTypeApertures')[eid, ptype, :]
    if len(ind) != apOffset[ap]-ptOffset[ptype]: set_trace()

# First step: find average velocity
vZMF = np.average(vel[ind, :], weights = mass[ind], axis = 0)
#vZMF = yb.read_hdf5(cantorloc, pre + '/Subhalo/Velocity')[cid, :]
vRel = vel[ind, :] - vZMF[None, :]
vRelMag = np.linalg.norm(vRel, axis = 1)

# Then: find total angular momentum
jInd = np.cross(pos[ind, :], vRel)
jTot = np.sum(jInd*mass[ind, None], axis = 0)

# Find z-component of AM per particle
zax = jTot/np.linalg.norm(jTot)
jz = np.sum(jInd * zax[None, :], axis = 1)

print("jTot: ", jTot)

# Decompose position into 'along j-z' and 'in R plane':
rad = np.linalg.norm(pos[ind, :], axis = 1)
zProj = np.sum(pos[ind, :] * zax[None, :], axis = 1)
rProj = np.sqrt(rad**2 - zProj**2)

v_r = jz/rProj

ind_co = np.nonzero((jz >= 0) & (rProj > 0))[0]

k_rot = np.sum(mass[ind[ind_co]] * v_r[ind_co]**2)
k_tot = np.sum(mass[ind[ind_co]] * vRelMag[ind_co]**2)

print("kappa_co: ", k_rot/k_tot)


# Extract values from Cantor:
if ap == 4:
    j_ca = yb.read_hdf5(
        cantorloc, pre + '/Subhalo/AngularMomentum_Stars')[cid, :]
else:
    j_ca = yb.read_hdf5(
        cantorloc, pre + '/Subhalo/Extra/Stars/AngularMomentum')[eid, eind, :]

kco_all = yb.read_hdf5(
    cantorloc, pre + '/Subhalo/Extra/Stars/KappaCo')
cid_all = yb.read_hdf5(
    cantorloc, pre + '/Subhalo/Extra/SubhaloIndex')
gal_all = yb.read_hdf5(
    cantorloc, pre + '/Subhalo/Galaxy')[cid_all]

kco = kco_all[eid, eind]

print("Cantor:")
print("jTot: ", j_ca)
print("kappaCo: ", kco)


set_trace()
