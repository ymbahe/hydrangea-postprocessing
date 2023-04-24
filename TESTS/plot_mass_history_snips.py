"""
Simple program to extract the mass evolution of one galaxy,
from Subfind and Cantor.
"""

import numpy as np
import sim_tools as st
import yb_utils as yb
import hydrangea_tools as ht
from pdb import set_trace
from astropy.cosmology import Planck13
import os

rundir = '/virgo/simulations/Hydrangea/10r200/CE-11/HYDRO/'
hldir = rundir + 'highlev/'
cantorloc = (ht.clone_dir(hldir) 
             + 'CantorCatalogue_28Jun19.hdf5')

cantorloc_snips = ht.clone_dir(hldir) + '/Snipshots/Cantor_'

fgtloc = hldir + 'FullGalaxyTables.hdf5'
posloc = hldir + 'GalaxyPositionsSnap.hdf5'
fineloc = ht.clone_dir(hldir) + 'GalaxyCoordinates10Myr_May18.hdf5'

outloc = '/virgo/scratch/ybahe/TESTS/GalaxyMassEvol_CE-11_G-1455_CC28_SNIPS.hdf5'

igal = 1455

shi_cantor = yb.read_hdf5(cantorloc, 'SubhaloIndex')[igal, :]

# Load snepshot list
sIndex, sAexp, sType, sNum = ht.get_snepshot_indices(rundir, 
                                                     list = 'short_movie')

nSnep = len(sIndex)
print("There are {:d} snepshots..." .format(nSnep))

mass_cantor = np.zeros(nSnep)
mdm_cantor = np.zeros(nSnep)
mstar_cantor = np.zeros(nSnep)
mgas_cantor = np.zeros(nSnep)
mass30_cantor = np.zeros(nSnep)
mdm30_cantor = np.zeros(nSnep)
mstar30_cantor = np.zeros(nSnep)
mgas30_cantor = np.zeros(nSnep)

sTime = Planck13.age(1/sAexp-1).value

kappa_co10 = np.zeros(nSnep)
kappa_co30 = np.zeros(nSnep)
phi = np.zeros(nSnep)
phiDM = np.zeros(nSnep)

bTeklu = np.zeros(nSnep)
jMag10 = np.zeros(nSnep)
jMag30 = np.zeros(nSnep)
jMagFull = np.zeros(nSnep)

jMagDM = np.zeros(nSnep)

vd_stars_cantor = np.zeros(nSnep)
vd_dm_cantor = np.zeros(nSnep)

ind_good = np.zeros(nSnep, dtype = np.int8)

for isnep in range(nSnep):

    # Case I: snapshot --> load from 'main' cantor catalogue
    if sType[isnep] == 'snap':
        catloc = cantorloc
        cantorID = shi_cantor[sNum[isnep]]
        if cantorID < 0: 
            print("No cantorID?!")
            continue
        pre = 'Snapshot_' + str(sNum[isnep]).zfill(3) + '/'
    else:
        catloc = cantorloc_snips + str(sIndex[isnep]).zfill(4) + '.hdf5'
        if not os.path.exists(catloc): continue
        galaxy = yb.read_hdf5(catloc, 'Subhalo/Galaxy')
        cantorID = np.nonzero(galaxy == igal)[0]
        if len(cantorID) == 0: 
            print("No Soprano-ID in isnep={:d}?" .format(isnep))
            continue
        cantorID = cantorID[0]  # Convert 1-element-array to scalar
        pre = ''

    ind_good[isnep] = 1

    print(cantorID)

    mass_cantor[isnep] = np.log10(yb.read_hdf5(catloc, 
                                               pre + 'Subhalo/Mass')
                                  [cantorID]) + 10.0

    massType = np.log10(yb.read_hdf5(catloc, pre + 'Subhalo/MassType')
                        [cantorID, :]) + 10.0
    
    mstar_cantor[isnep] = massType[4]
    mdm_cantor[isnep] = massType[1]
    mgas_cantor[isnep] = massType[0]

    angMom = yb.read_hdf5(
        catloc, pre + 'Subhalo/AngularMomentum_Stars')[cantorID, :]
    angMomDM = yb.read_hdf5(
        catloc, pre + 'Subhalo/AngularMomentum_DM')[cantorID, :]

    jMagFull[isnep] = np.linalg.norm(angMom)/10.0**(mstar_cantor[isnep]-10)*1e3
    jMagDM[isnep] = np.linalg.norm(angMomDM)/10.0**(mdm_cantor[isnep]-10)*1e3

    extraID = yb.read_hdf5(catloc, pre + 'Subhalo/Extra/ExtraIDs')[cantorID]
    if extraID >= 0:
        mdm30_cantor[isnep] = np.log10(yb.read_hdf5(
            catloc, pre + 'Subhalo/Extra/DM/ApertureMasses')[extraID, 2])+10.0
        mstar30_cantor[isnep] = np.log10(yb.read_hdf5(
            catloc, pre + 'Subhalo/Extra/Stars/ApertureMasses')[extraID, 2])+10.0
        mstar10 = yb.read_hdf5(
            catloc, pre + 'Subhalo/Extra/Stars/ApertureMasses')[extraID, 1]

        mgas30_cantor[isnep] = np.log10(yb.read_hdf5(
            catloc, pre + 'Subhalo/Extra/Gas/ApertureMasses')[extraID, 2])+10.0

        kappa_co = yb.read_hdf5(catloc, pre + 
                                '/Subhalo/Extra/Stars/KappaCo')[extraID, :]
        kappa_co10[isnep] = kappa_co[0]
        kappa_co30[isnep] = kappa_co[1]
        phi[isnep] = yb.read_hdf5(catloc, pre + 
                             '/Subhalo/Extra/Stars/AxisRatios')[extraID, 1, 0]
        phiDM[isnep] = yb.read_hdf5(catloc, pre + 
                             '/Subhalo/Extra/DM/AxisRatios')[extraID, 1, 0]

        angMom = yb.read_hdf5(catloc, pre + 
                              '/Subhalo/Extra/Stars/AngularMomentum')[extraID,
                                                                      1, :]
        jAng = np.linalg.norm(angMom)/10.0**(mstar30_cantor[isnep]-10) * 1e3
        jMag30[isnep] = jAng

        angMom10 = yb.read_hdf5(catloc, pre + 
                              '/Subhalo/Extra/Stars/AngularMomentum')[extraID,
                                                                      0, :]
        jAng10 = np.linalg.norm(angMom10)/mstar10 * 1e3
        jMag10[isnep] = jAng10
        
        bTeklu[isnep] = np.log10(jAng) - 2/3*(np.log10(mstar30_cantor[isnep])
                                              +10.0)
        
        

    else:
        mdm30_cantor[isnep] = mdm_cantor[isnep]
        mstar30_cantor[isnep] = mstar_cantor[isnep]
        mgas30_cantor[isnep] = mgas_cantor[isnep]


    vd_stars_cantor[isnep] = yb.read_hdf5(
        catloc, pre + '/Subhalo/VelocityDispersion_Stars')[cantorID]
    vd_dm_cantor[isnep] = yb.read_hdf5(
        catloc, pre + '/Subhalo/VelocityDispersion_DM')[cantorID]

ind_write = np.nonzero(ind_good == 1)[0]

yb.write_hdf5(mass_cantor[ind_write], outloc, 'MassCantor', new = True)
yb.write_hdf5(mstar_cantor[ind_write], outloc, 'MstarCantor')
yb.write_hdf5(mdm_cantor[ind_write], outloc, 'MDMCantor')
yb.write_hdf5(mgas_cantor[ind_write], outloc, 'MgasCantor')
yb.write_hdf5(mass30_cantor[ind_write], outloc, 'MassCantor30')
yb.write_hdf5(mstar30_cantor[ind_write], outloc, 'MstarCantor30')
yb.write_hdf5(mdm30_cantor[ind_write], outloc, 'MDMCantor30')
yb.write_hdf5(mgas30_cantor[ind_write], outloc, 'MgasCantor30')

yb.write_hdf5(vd_stars_cantor[ind_write], outloc, 'VelDisp_Stars')
yb.write_hdf5(vd_dm_cantor[ind_write], outloc, 'VelDisp_DM')
yb.write_hdf5(kappa_co10[ind_write], outloc, 'KappaCo10')
yb.write_hdf5(kappa_co30[ind_write], outloc, 'KappaCo30')
yb.write_hdf5(phi[ind_write], outloc, 'Phi30')
yb.write_hdf5(phiDM[ind_write], outloc, 'PhiDM30')
yb.write_hdf5(bTeklu[ind_write], outloc, 'TekluB30')
yb.write_hdf5(jMagFull[ind_write], outloc, 'jAngFull')
yb.write_hdf5(jMagDM[ind_write], outloc, 'jAngDM')
yb.write_hdf5(jMag30[ind_write], outloc, 'jAng30')
yb.write_hdf5(jMag10[ind_write], outloc, 'jAng10')


yb.write_hdf5(sTime[ind_write], outloc, 'Time')


