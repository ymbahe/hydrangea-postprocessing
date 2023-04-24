"""
Small helper program to combine origin tables produced by different MPI tasks.

Started 04-Jul-2019
"""

import yb_utils as yb
import numpy as np
import os
from shutil import copyfile

combine_sims = True

dataloc = ('/virgo/scratch/ybahe/HYDRANGEA/RESULTS/ICL/' +
          #'OriginMasses_08Jul19_HostMhalo5_RootMsub24_HostRad7_firstroot')
           'OriginMasses_08Jul19_HostMhalo5_HostRad7_firstroot_xD')

nfiles = 1

copyfile(dataloc + '.0.hdf5', dataloc + '.hdf5')

mbins_stars = yb.read_hdf5(dataloc + '.hdf5', 'BinnedMasses_Stars')

for ii in range(1, nfiles):
    filename = dataloc + '.' + str(ii) + '.hdf5'
    mbins_stars_part = yb.read_hdf5(filename, 'BinnedMasses_Stars')
    mbins_stars += mbins_stars_part

if combine_sims:
    mbins_stars = np.sum(mbins_stars, axis = 0)

yb.write_hdf5(mbins_stars, dataloc + '.hdf5', 'BinnedMasses_Stars')

print("Done!")
