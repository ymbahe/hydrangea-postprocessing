"""
Script to collect the various additional subhalo properties into one file.

Started 20-Dec-2018
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
import datetime
import calendar

rundir_base = '/virgo/simulations/Hydrangea/C-EAGLE/'
outname = 'SubhaloExtra.hdf5'

n_sim = 30
n_snap = 30

#datestamp = '20-Dec-2018'
now = datetime.datetime.now()
datestamp = str(now)

comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()

simType = 'HYDRO'
otherType = 'DM'

for isim in range(17, n_sim):

    # Skip this one if we are multi-threading and it's not for this task to worry about
    if not isim % numtasks == rank: continue
        
    hstime = time.time()

    print("")
    print("*******************************")
    print("Now processing simulation CE-{:d}" .format(isim))
    print("*******************************")
    print("")

    rundir = rundir_base + '/CE-' + str(isim) + '/' + simType + '/' 
    if not os.path.exists(rundir): continue

    hldir = rundir + '/highlev/'
    loc_match = hldir + '/MatchIn' + otherType + '.hdf5'
    loc_fgt = hldir + 'FullGalaxyTables.hdf5'
    loc_spiderweb = hldir + 'SpiderwebTables.hdf5'
    shi = yb.read_hdf5(loc_spiderweb, 'SubHaloIndex')

    outloc = hldir + outname
    if os.path.isfile(outloc):
        os.rename(outloc, outloc+'.old')
 
    if isim not in [17, 19, 20, 23, 26, 27]:
        catalogueID = yb.read_hdf5_attribute(loc_match, 'Header', 'CatalogueID')
        numTracers = yb.read_hdf5_attribute(loc_match, 'Header', 'NumTracers')
    else:
        catalogueID = calendar.timegm(time.gmtime())


    yb.write_hdf5_attribute(outloc, 'Header', 'DateStamp', np.string_(datestamp))
    yb.write_hdf5_attribute(outloc, 'Header', 'CatalogueID', catalogueID)
    yb.write_hdf5_attribute(outloc, 'Header', 'Simulation', isim)
    yb.write_hdf5_attribute(outloc, 'Header', 'SimType', np.string_(simType))

    for isnap in range(30):

        print("   Processing snapshot {:d}..." .format(isnap))

        subfile = st.form_files(rundir, isnap=isnap)
        subdir = yb.dir(subfile)
        
        loc_BH = subdir + '/BlackHoleMasses.hdf5'
        loc_mergers = subdir + '/SubhaloMergerFlag.hdf5'
        loc_boundary = subdir + '/BoundaryFlag.hdf5'
        
        galID = yb.read_hdf5(loc_spiderweb, 'Subhalo/Snapshot_' + str(isnap).zfill(3) + '/Galaxy')

        nSubhalo = len(galID)
        snapPre = 'Snapshot_' + str(isnap).zfill(3)

        # Write number of subhaloes as attribute
        yb.write_hdf5_attribute(outloc, snapPre, 'NSubhalo', nSubhalo)
        if nSubhalo == 0: continue

        print("   ... match info...")
        if isim not in [17, 19, 20, 23, 26, 27]:
            # Copy subhalo match information
            matchInOther = yb.read_hdf5(loc_match, snapPre)
            yb.write_hdf5(matchInOther, outloc, snapPre + '/MatchIn' + otherType, comment = 'Index of the subhalo in the ' + otherType + ' simulation that matches subhalo [i]. -1 means that [i] has no match in ' + otherType + '.')
            yb.write_hdf5_attribute(outloc, snapPre + '/MatchIn' + otherType, 'NumTracers', numTracers)

        print("   ... boundary info...")
        # Copy boundary flag information
        contFlag = yb.read_hdf5(loc_boundary, 'ContaminationFlagMulti')
        yb.write_hdf5(contFlag, outloc, snapPre + '/BoundaryFlag', comment = 'Flag to indicate whether subhalo [i] is close to the edge of the high resolution region. The value encodes the distance to the nearest boundary particle: 0 --> > 8 cMpc; 1 --> 5-8 cMpc; 2 --> 2-5 cMpc; 3 --> 1-2 cMpc; 4 --> < 1 cMpc')
        yb.write_hdf5_attribute(outloc, snapPre + '/BoundaryFlag', 'DistanceThresholds', np.array((8, 5, 2, 1)))

        print("   ... close pair info...")
        # Copy merger flag information

        if not os.path.isfile(loc_mergers):
            yb.write_hdf5_attribute(outloc, snapPre + '/ClosePairs', 'NClosePairs', 0)
            
        else:
            yb.write_hdf5_attribute(outloc, snapPre + '/ClosePairs', 'MaxSeparation', 3.0)
            ind_close = yb.read_hdf5(loc_mergers, 'Spurious')
            yb.write_hdf5(ind_close, outloc, snapPre + '/ClosePairs/Spurious', comment = 'Indices of subhaloes that lie within min(r_1/2_star, r_max) of a more massive subhalo (by total mass)')
            yb.write_hdf5_attribute(outloc, snapPre + '/ClosePairs', 'NClosePairs', len(ind_close))
 
            close_parents = yb.read_hdf5(loc_mergers, 'Parents')
            yb.write_hdf5(close_parents, outloc, snapPre + '/ClosePairs/Parents', comment = 'For each entry in [Spurious], the index (in the subfind catalogue) of the more massive subhalo that is located close to it.') 

        print("   ... BH masses...")
        # Copy BH masses
        if os.path.isfile(loc_BH):
            snapPreBH = snapPre + '/BH_Masses'
            yb.write_hdf5_attribute(outloc, snapPreBH, 'MaxCenRadius', 3.0)

            mbh_part = np.log10(yb.read_hdf5(loc_BH, 'BHMass'))+10.0
            yb.write_hdf5(mbh_part, outloc, snapPreBH + '/PartMass', comment = "Sum of BH *particle* masses of all BH particles in this subhalo. Should be identical to Subhalo/MassType[:, 5] in subfind tables *in log astro units*, but may not be.")

            mbh_sg = np.log10(yb.read_hdf5(loc_BH, 'BHSubgridMass'))+10.0
            yb.write_hdf5(mbh_sg, outloc, snapPreBH + "/SubgridMass", comment = "Sum of BH *subgrid* masses of all BH particles in this subhalo. Should be identical to Subhalo/BlackHoleMass in subfind tables *in log astro units*, but may not be.")

            mbh_reass = np.log10(yb.read_hdf5(loc_BH, 'ReassignedBHMass'))+10.0
            yb.write_hdf5(mbh_reass, outloc, snapPreBH + "/PartMass_RA", comment = "Sum of BH *particle* masses of all BH particles in this subhalo *after reassignment*, in log astro units.")
 
            mbh_sg_reass = np.log10(yb.read_hdf5(loc_BH, 'ReassignedBHSubgridMass'))+10.0
            yb.write_hdf5(mbh_sg_reass, outloc, snapPreBH + "/SubgridMass_RA", comment = "Sum of BH *subgrid* masses of all BH particles in this subhalo *after reassignment*, in log astro units.")

            mbh_orig_cen = np.log10(yb.read_hdf5(loc_BH, 'BHCentralMass'))+10.0
            yb.write_hdf5(mbh_orig_cen, outloc, snapPreBH + "/CentralPartMass", comment = "Sum of BH *particle* masses of central BH particles in this subhalo, in log astro units.")

            mbh_sg_orig_cen = np.log10(yb.read_hdf5(loc_BH, 'BHCentralSubgridMass'))+10.0
            yb.write_hdf5(mbh_sg_orig_cen, outloc, snapPreBH + "/CentralSubgridMass", comment = "Sum of BH *subgrid* masses of central BH particles in this subhalo, in log astro units.")

            mbh_reass_cen = np.log10(yb.read_hdf5(loc_BH, 'ReassignedBHCentralMass'))+10.0
            yb.write_hdf5(mbh_reass_cen, outloc, snapPreBH + "/CentralPartMass_RA", comment = "Sum of BH *particle* masses of central BH particles in this subhalo *after reassignment*, in log astro units.")

            mbh_sg_reass_cen = np.log10(yb.read_hdf5(loc_BH, 'ReassignedBHCentralSubgridMass'))+10.0
            yb.write_hdf5(mbh_sg_reass_cen, outloc, snapPreBH + "/CentralSubgridMass_RA", comment = "Sum of BH *subgrid* masses of central BH particles in this subhalo *after reassignment*, in log astro units.")

        # Extract spectre information
        print("   ... spectre info...")
        galSpectreFlag = yb.read_hdf5(loc_fgt, 'Full/SpectreFlag')
        shSpectreFlag = galSpectreFlag[galID]   # Need to re-order into SHI sequence

        galSpectreParentGal = yb.read_hdf5(loc_fgt, 'Full/SpectreParents')
        galSpectreParentSH = shi[galSpectreParentGal, isnap]
        shSpectreParentSH = galSpectreParentSH[galID]  # Need to re-order into SHI sequence

        yb.write_hdf5(shSpectreFlag, outloc, snapPre + '/Spectre/Flag', comment = 'Indicates whether the subhalo represents a spectre galaxy (1) or not (0)')

        yb.write_hdf5(shSpectreParentSH, outloc, snapPre + '/Spectre/Parent', comment = 'The subhalo representing the galaxy from whose material a spectre subhalo was (predominantly) formed; <= -5 if this does not exist. If the subhalo is not a spectre, it points to itself.')
        
print("Done!")
