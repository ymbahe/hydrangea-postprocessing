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
        print("Data set {:s} does not exist in {:d}.{:d}." .format(dataSet, isim, isnap))
        return 0

    try:
        data_sub = st.eagleread(subpartdir, dataSetName, astro = False)
    except:
        print("Data set {:s} does not exist in Subfind in {:d}.{:d}." .format(dataSet, isim, isnap))
        return 0
        

    diff = data_snap[ind_snap_in_sub[ind_matched]] - data_sub
    n_diff = np.count_nonzero(np.abs(diff) > 0)


    f = open(logfile, "a")
    
    if n_diff > 0:
        print("")
        print("==========================================")
        print("Found {:d} differences for data set {:s} in {:d}.{:d} (out of {:d})" 
              .format(n_diff, dataSetName, isim, isnap, data_sub.shape[0]))
        f.write("Found {:d} differences for data set {:s} in {:d}.{:d} (out of {:d})\n" 
                .format(n_diff, dataSetName, isim, isnap, data_sub.shape[0]))
        print("==========================================")
        print("")
        f.close()
        return -1

    else:
        print("")
        print("Data set {:s} is consistent between snap and subfind in {:d}.{:d}."
              .format(dataSetName, isim, isnap))
        print("")

        f.write("Data set {:s} is consistent between snap and subfind in {:d}.{:d}.\n"
                .format(dataSetName, isim, isnap))
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
basedir = '/virgo/simulations/Eagle/L0100N1504/REFERENCE/'#Hydrangea/C-EAGLE/'
logfile = 'SubfindDifferences-EagleL0100_StellarFormationTime.log'

#dataSetList = ['Mass', 'Coordinates', 'Velocity', 'Density', 'Temperature', 'Entropy', 'InternalEnergy', 'StarFormationRate', 'BirthDensity', 'StellarInitialMass', 'StellarFormationTime', 'SmoothedMetallicity', 'BH_Mass']
dataSetList = ['Density']

for isim in range(0, 1):

    rundir = basedir# + 'CE-' + str(isim) + '/' + simtype + '/'

    for isnap in range(0, 29):

        snapdir, subpartdir = st.form_files(rundir, isnap, 'snap subpart')
        check_ptype(4, ['StellarFormationTime'])

print("Done!")
