"""
Test a single snapshot (of a single simulation) for consistency between
Subfind and Snapshots.

Started 07-May-2019
"""

import hydrangea_tools as ht
import sim_tools as st
import numpy as np
from pdb import set_trace
import os.path
import yb_utils as yb

def check_dataset(snapdir, subpartdir, ptype, dataSet, ind_snap_in_sub, ind_matched):
    """
    Check one single data set for consistency between snap- and subpartdir.

    `gate' is a Gate instance FROM subpart TO snap
    """

    dataSetName = 'PartType' + str(ptype) + '/' + dataSet
    try:
        data_snap = st.eagleread(snapdir, dataSetName, astro = False)
    except:
        print("Data set {:s} does not exist." .format(dataSet))
        return 0

    try:
        data_sub = st.eagleread(subpartdir, dataSetName, astro = False)
    except:
        print("Data set {:s} does not exist in Subfind." .format(dataSet))
        return 0
        

    diff = data_snap[ind_snap_in_sub[ind_matched]] - data_sub
    n_diff = np.count_nonzero(np.abs(diff) > 0)
    f = open(logfile, "a")
    
    if n_diff > 0:

        medDiff = np.median(np.abs(diff/data_snap[ind_snap_in_sub[ind_matched]]))*100

        print("")
        print("==========================================")
        print("Found {:d} differences for data set {:s} (out of {:d}). Median diff={:.3f} \%." 
              .format(n_diff, dataSetName, data_sub.shape[0], medDiff))
        f.write("Found {:d} differences for data set {:s} (out of {:d}). Median diff={:.3f} \%.\n" 
                .format(n_diff, dataSetName, data_sub.shape[0], medDiff))
        print("==========================================")
        print("")
        f.close()
        return -1

    else:
        print("")
        print("Data set {:s} is consistent between snap and subfind."
              .format(dataSetName))
        print("")

        f.write("Data set {:s} is consistent between snap and subfind.\n"
                .format(dataSetName))
        f.close()
        return 1


def check_ptype(ptype, dataSetList):
    """
    Check all elements of dataSetList for ptype, if they exist.
    """

    idSetName = 'PartType' + str(ptype) + '/ParticleIDs'

    numPartSnap = yb.read_hdf5_attribute(snapdir, 'Header', 'NumPart_Total')[ptype]
    numPartSub = yb.read_hdf5_attribute(subpartdir, 'Header', 'NumPart_Total')[ptype]

    print("Expecting {:d} particles in Snap and {:d} in Sub..."  
          .format(numPartSnap, numPartSub))

    if numPartSnap == 0 or numPartSub == 0:
        return

    f = open(logfile, "a")
    f.write("\n")
    f.close()

    ids_snap = st.eagleread(snapdir, idSetName, astro = False)
    ids_sub = st.eagleread(subpartdir, idSetName, astro = False)

    gate = st.Gate(ids_sub, ids_snap)

    ind_snap_in_sub = gate.in2()
    ind_matched = np.nonzero(ind_snap_in_sub >= 0)[0]

    if np.max(np.abs(ids_snap[ind_snap_in_sub[ind_matched]]-ids_sub)) > 0:
        print("")
        print("Inconsistent IDs detected for ptype={:d}..." .format(ptype))
        print("")
        return None

    for dataSet in dataSetList:
        retVal = check_dataset(snapdir, subpartdir, ptype, dataSet, ind_snap_in_sub, ind_matched)

    
    
simtype = 'HYDRO'
isim = 14
isnap = 29
basedir = '/virgo/simulations/Hydrangea/C-EAGLE/'
rundir = basedir + 'CE-' + str(isim) + '/' + simtype + '/'
snapdir, subpartdir = st.form_files(rundir, isnap, 'snap subpart')
logfile = 'SubfindDifferences_PT2_Full_CE-14_S-29.log'

dataSetList = ['Coordinates', 'GroupNumber', 'Mass', 'ParticleIDs', 'Velocity']

#dataSetList = ['Mass', 'Coordinates', 'Velocity', 'Density', 'Temperature', 'Entropy', 'InternalEnergy', 'StarFormationRate', 'BirthDensity', 'StellarInitialMass', 'StellarFormationTime', 'SmoothedMetallicity', 'BH_Mass']

#dataSetList = ['AExpMaximumTemperature', 'Coordinates', 'Density', 'ElementAbundance/Carbon', 'Entropy', 'HostHalo_TVir_Mass', 'InternalEnergy', 'IronMassFracFromSNIa', 'Mass', 'MaximumTemperature', 'MetalMassFracFromAGB', 'MetalMassFracFromSNII', 'MetalMassFracFromSNIa', 'MetalMassWeightedRedshift', 'Metallicity', 'OnEquationOfState', 'SmoothedElementAbundance/Carbon', 'SmoothedIronMassFracFromSNIa', 'SmoothedMetallicity', 'SmoothingLength', 'StarFormationRate', 'Temperature', 'TotalMassFromAGB', 'TotalMassFromSNII', 'TotalMassFromSNIa', 'Velocity']

#dataSetList = ['AExpMaximumTemperature', 'BirthDensity', 'Coordinates', 'ElementAbundance/Carbon', 'Feedback_EnergyFraction', 'InitialMass', 'IronMassFracFromSNIa', 'Mass', 'MaximumTemperature', 'MetalMassFracFromAGB', 'MetalMassFracFromSNII', 'MetalMassFracFromSNIa', 'MetalMassWeightedRedshift', 'Metallicity', 'PreviousStellarEnrichment', 'SmoothedElementAbundance/Carbon', 'SmoothedIronMassFracFromSNIa', 'SmoothedMetallicity', 'SmoothingLength', 'StellarEnrichmentCounter', 'StellarFormationTime', 'TotalMassFromAGB', 'TotalMassFromSNII', 'TotalMassFromSNIa', 'Velocity']

#dataSetList = ['BH_AccretionLength', 'BH_CumlAccrMass', 'BH_CumlNumSeeds', 'BH_Density', 'BH_EnergyReservoir', 'BH_FormationTime', 'BH_Mass', 'BH_Mdot', 'BH_MostMassiveProgenitorID', 'BH_Pressure', 'BH_SoundSpeed', 'BH_SurroundingGasVel', 'BH_TimeLastMerger', 'BH_WeightedDensity', 'Coordinates', 'HostHalo_TVir_Mass', 'Mass', 'SmoothingLength', 'Velocity']

for ptype in [2]:
    check_ptype(ptype, dataSetList)

print("Done!")
