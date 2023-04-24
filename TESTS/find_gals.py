import hydrangea_tools as ht
import sim_tools as st
import eagle_routines as er
import numpy as np
from pdb import set_trace
from scipy.optimize import curve_fit
from astropy.io import fits
import glob
import image_routines as im
import time
import os.path
import yb_utils as yb
from mpi4py import MPI
import copy


basedir = '/virgo/simulations/Hydrangea/C-EAGLE/'
nsnap = 30
nsim = 30

simtype = 'HYDRO'
snaplist = np.arange(nsnap, dtype=int)

isim = 0

rundir = basedir + 'CE-' + str(isim) + '/' + simtype
fgtloc = rundir + '/highlev/FullGalaxyTables.hdf5'

mstar = yb.read_hdf5(fgtloc, 'Mstar')[:, -1]
satFlag = yb.read_hdf5(fgtloc, 'SatFlag')[:, -1]
contFlag = yb.read_hdf5(fgtloc, 'ContFlag')[:, -1]

ind = np.nonzero((mstar > 10.4) & (mstar < 10.7) & (satFlag == 0) & (contFlag == 0))[0]

print(ind)

    
