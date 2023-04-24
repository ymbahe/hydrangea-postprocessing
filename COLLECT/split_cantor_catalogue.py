"""
Script to split full Cantor catalogue into sub-catalogues per snapshot,
and 'meta-data' for full evolution.

Started 30-Sep-2019
"""

import numpy as np
import yb_utils as yb
import sim_tools as st
from pdb import set_trace
import os
import h5py as h5

basedir = '/virgo/simulations/Hydrangea/10r200/'
outdir = 'Cantor/'

def copy_att(source, dest, ssnap, name):
    att = yb.read_hdf5_attribute(source, ssnap, name)
    yb.write_hdf5_attribute(dest, 'Header', name, att)

for isim in range(1, 30):

    if isim == 22: continue

    rootdir = basedir + 'CE-{:d}/HYDRO/' .format(isim)
    if not os.path.isdir(rootdir): continue

    cantorloc = rootdir + 'highlev/CantorCatalogue.hdf5'

    full_outloc = rootdir + 'highlev/' + outdir + 'GalaxyTables.hdf5'
    if not os.path.isdir(yb.dir(full_outloc)):
        os.makedirs(yb.dir(full_outloc))

    f = h5.File(full_outloc, 'w')
    f2 = h5.File(cantorloc, 'r')
    f2.copy('Header', f)
    f2.copy('SubhaloIndex', f)
    f2.copy('CenGalExtended', f)

    f2.close()    
    f['CentralGalaxy'] = f['CenGalExtended']
    del f['CenGalExtended']

    f.close()
        
    for isnap in range(30):

        print("Now processing snapshot {:d}..." .format(isnap))

        outloc = (rootdir + 'highlev/' + outdir + 
                  'Cantor_{:03d}.hdf5' .format(isnap))
        idloc = (rootdir + 'highlev/' + outdir + 
                  'Cantor_{:03d}_IDs.hdf5' .format(isnap))
        radloc = (rootdir + 'highlev/' + outdir + 
                  'Cantor_{:03d}_Radii.hdf5' .format(isnap))


        ssnap = 'Snapshot_{:03d}' .format(isnap)

        # Copy required headers from snapshot file:
        f = h5.File(outloc, 'w')
        f2 = h5.File(cantorloc, 'r')

        print(" ... copy header...")
        f2.copy('Header', f)
        print(" ... copy Subhalo...")
        f2.copy(ssnap + '/Subhalo', f)
        print(" ... copy FOF...")
        f2.copy(ssnap + '/FOF', f)
        print(" ... copy IDs...")

        f.close()
        f2.close()

        ids = yb.read_hdf5(cantorloc, ssnap + '/IDs')
        yb.write_hdf5(ids, idloc, 'IDs', compression = 'lzf')
        comment = yb.read_hdf5_attribute(cantorloc, ssnap + '/IDs', 'Comment')
        yb.write_hdf5_attribute(idloc, 'IDs', 'Comment', comment)

        print(" ... copy Radius...")
        rad = yb.read_hdf5(cantorloc, ssnap + '/Radius')
        yb.write_hdf5(rad, radloc, 'Radius', compression = 'lzf')
        comment = yb.read_hdf5_attribute(cantorloc, ssnap + '/Radius', 
                                         'Comment')
        yb.write_hdf5_attribute(radloc, 'Radius', 'Comment', comment)
        
        copy_att(cantorloc, outloc, ssnap, 'NumFOF')
        copy_att(cantorloc, outloc, ssnap, 'NumSubhalo')
        copy_att(cantorloc, outloc, ssnap, 'Redshift')
        copy_att(cantorloc, outloc, ssnap, 'aExp')

print("Done!")
