"""
Script to collect information relevant for stellar luminosity calculation.

Started 7-Aug-2019
"""

import numpy as np
import yb_utils as yb
import sim_tools as st
from pdb import set_trace
import os
import h5py as h5

basedir = '/virgo/simulations/Hydrangea/10r200/'
outdir = '/virgo/scratch/ybahe/HYDRANGEA/STAR_CATALOGUES/SPLIT/'

extra_fields = True
individual_files = True

def extract_dataset(snapfile, outloc, dset):

    data = yb.read_hdf5(snapfile, 'PartType4/' + dset)

    cgsf = yb.read_hdf5_attribute(snapfile, 'PartType4/' + dset, 
                                  'CGSConversionFactor')
    vd = yb.read_hdf5_attribute(snapfile, 'PartType4/' + dset, 
                                'VarDescription')
    ase = yb.read_hdf5_attribute(snapfile, 'PartType4/' + dset, 
                                 'aexp-scale-exponent')
    hse = yb.read_hdf5_attribute(snapfile, 'PartType4/' + dset, 
                                 'h-scale-exponent')
    yb.write_hdf5(data, outloc, 'PartType4/' + dset, compression = 'lzf')
    yb.write_hdf5_attribute(outloc, 'PartType4/' + dset, 
                            'CGSConversionFactor', cgsf)
    yb.write_hdf5_attribute(outloc, 'PartType4/' + dset, 
                            'VarDescription', vd)
    yb.write_hdf5_attribute(outloc, 'PartType4/' + dset, 
                            'aexp-scale-exponent', ase)
    yb.write_hdf5_attribute(outloc, 'PartType4/' + dset, 
                            'h-scale-exponent', hse)

def process_snapshot(isnap):
    """
    Process one snapshot and extract its information.
    """

    print("")
    print("Processing snapshot {:d} of simulation {:d}..."
          .format(isnap, isim))
    print("")

    snapdir = st.form_files(rundir, isnap, 'snap')
    nfiles = yb.read_hdf5_attribute(snapdir, 'Header', 'NumFilesPerSnapshot')

    fileparts = snapdir.split(".")
    for seqnr in range(nfiles):
        
        print(str(seqnr)+" ", end = "",flush=True)
        
        fileparts[-2] = str(seqnr)
        snapfile = ".".join(fileparts)
        
        # Test whether file exists:
        if not os.path.isfile(snapfile):
            print("\nEnd of files found at " + str(seqnr-1))
            print("\nThis should NOT happen!", flush=True)
            sys.exit(111)

        if extra_fields:
            outloc = outdir + ('/CE-{:d}/CE-{:d}_snap_{:03d}.{:d}.hdf5' 
                               .format(isim, isim, isnap, seqnr))
        else:
            outloc = outdir + ('/CE-{:d}/CE-{:d}_snap_{:03d}_core.{:d}.hdf5'
                               .format(isim, isim, isnap, seqnr))

        if not os.path.isdir(yb.dir(outloc)):
            os.makedirs(yb.dir(outloc))

        if os.path.exists(outloc):
            os.rename(outloc, outloc + '.old')

        # Copy required headers from snapshot file:
        f = h5.File(outloc)
        f2 = h5.File(snapfile, 'r')
        f2.copy('Header', f)
        f2.copy('Constants', f)
        f2.copy('Units', f)
        f2.copy('RuntimePars', f)
        
        f.close()
        f2.close()

        if extra_fields:
            for dset in ['StellarFormationTime', 'SmoothedMetallicity',
                         'InitialMass', 'Mass', 'Coordinates', 'Velocity', 
                         'BirthDensity', 'ParticleIDs']:
                extract_dataset(snapfile, outloc, dset)
        else:
            for dset in ['StellarFormationTime', 'SmoothedMetallicity',
                         'InitialMass', 'BirthDensity']:
                extract_dataset(snapfile, outloc, dset)


    return


# ==========================
# Actual program begins here
# ==========================

for isim in range(16, 17):
    
    rundir = basedir + 'CE-{:d}' .format(isim) + '/HYDRO/'
    if not os.path.isdir(rundir):
        print("Simulation {:d} not found...")
        continue

    for isnap in range(29, 30):
        process_snapshot(isnap)
        

print("Done!")
