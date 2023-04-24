"""
Checks whether simulations have been snipshot catalogued

Started 20 MAY 2016
"""

import os 
import numpy as np
import time
from pdb import set_trace
import sim_tools as sim
import h5py as h5
import yb_utils as yb

n_halo   = 43    # Number of haloes

rundir_base = '/virgo/simulations/Hydrangea/10r200/'

stime = time.time()

for ihalo in range(n_halo):
    
    hstime = time.time()

    print("")
    print("**************************")
    print("Now processing halo F{:d}" .format(ihalo))
    print("**************************")
    print("")
    
    rundir = rundir_base + '/HaloF' + str(ihalo) + '/HYDRO'
    if not os.path.exists(rundir):
        continue

    flag_sniplist = os.path.exists(rundir + "/snipshot_times.txt")
    flag_sneplist_basic = os.path.exists(rundir + "/sneplist_for_basic.txt")
    flag_sneplist_default = os.path.exists(rundir + "/sneplist_for_default.txt")
    flag_sneplist_full_movie = os.path.exists(rundir + "/sneplist_for_full_movie.txt")

    print("Sniplist:     ", flag_sniplist)
    print("Snep/Basic:   ", flag_sneplist_basic)
    print("Snep/Default: ", flag_sneplist_default)
    print("Snep/Movie:   ", flag_sneplist_full_movie)


print("Done!")
