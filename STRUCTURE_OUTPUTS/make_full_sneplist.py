"""
This program creates a complete list of ALL snapshot AND snipshots in a simulation
"""

import argparse
import sys
from astropy.io import ascii
import os.path
import numpy as np
import subprocess

outlistdir = '/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/'

parser = argparse.ArgumentParser()
parser.add_argument("dir", help = "The directory in which the simulation is sitting")

args = parser.parse_args()
rootloc = args.dir

outloc = rootloc + '/sneplists/full.dat'


# Sanity checks:
sniplistloc = rootloc + '/sneplists/snipshot_times.dat'
if not os.path.exists(sniplistloc):
    print("The current simulation '" + rootloc + "' does not (yet) have a snipshot_times.txt file.\nCreating it now with list_snipshot_times.py....")
    subprocess.call(["/u/ybahe/anaconda3/bin/python3.4", "/u/ybahe/ANALYSIS/list_snipshot_times.py", rootloc])
    print("...done!")
    
snaplistloc = outlistdir + 'hydrangea_snapshots_plus.dat'

 
# Step 1: Load the desired snepshot list:

snipdata = ascii.read(sniplistloc, guess=False, format='no_header')
ind_snip = snipdata['col1']
aexp_snip = snipdata['col2']
n_snip = len(aexp_snip)

snapdata = ascii.read(snaplistloc, guess=False, format='no_header')
aexp_snap = snapdata['col1']
n_snap = len(aexp_snap)

n_snep = n_snip+n_snap

# Set up the output lists:
list_snepind = np.empty(n_snep, dtype=int)-1
list_sneptype = np.empty(n_snep, dtype='<U5')
list_sourceind = np.empty(n_snep, dtype=int)-1


# Step 2: Loop through snepshot entries:
for i in range(n_snep):
    list_snepind[i] = i
    
list_sneptype[:n_snap] = "snap"
list_sneptype[n_snap:] = "snip"

list_sourceind[:n_snap] = range(n_snap)
list_sourceind[n_snap:] = ind_snip

ascii.write(np.vstack((list_snepind, list_sneptype, list_sourceind)).T, outloc, format='no_header')


print("Done!")
