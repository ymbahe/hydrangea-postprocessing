"""
This program creates a list of all snipshot times (expansion factors) in a simulation.

Doing this is necessary because snipshot numbering varies erratically between runs, for reasons 
that are (at present = 1 Oct 2015) 'not entirely understood' (no clue).

(The above statement is still accurate as of 8 Mar 2016).
"""

import os
import argparse
import glob
import h5py as h5
import numpy as np
from astropy.io import ascii
import eagle_routines as er


parser = argparse.ArgumentParser()
parser.add_argument("dir", help = "The directory in which the simulation is sitting")
parser.add_argument("-e", "--eagle", help = "Switch for running on original Eagle", action = "store_true")


args = parser.parse_args()
rootloc = args.dir

# Step 1: Find number of snipshots

sniplist = sorted(glob.glob(rootloc + '/data/snipshot_*_z*p*'))
n_snip = len(sniplist)

print("There are %d snipshots in total..." % n_snip)

# Step 2: Loop through all snipshots... [Fun]
aexplist = np.zeros(n_snip)
index = np.zeros(n_snip, dtype = '<U5')

for isnip, snipdir in enumerate(sniplist):
#    print("This is snipshot %d [" % isnip + snipdir + "]")

    if (isnip % 10) == 0:
        print("Reached snip %d" % isnip)

    # Step 2.a: Need to form the exact filename of the .0 file
    filename = sorted(glob.glob(snipdir + '/snip*'))[0]

    # Step 2.b: Need to load the attribute 'ExpansionFactor' from the Header group...
    f = h5.File(filename, "r")
    header = f["/Header"]
    aexp_curr = header.attrs["ExpansionFactor"]
                
    # Step 2.c: Write this into full list... 
    aexplist[isnip] = aexp_curr

    # Step 3: Find actual snipshot index (!!)
    filenameparts = filename.split('/')
    actual_file_name = filenameparts[-1]
    filepart = actual_file_name.split('_')
    index[isnip] = filepart[1]

    
# Step 3: Write output

if args.eagle:
    outdir = er.clone_dir(rootloc)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
        
else:
    outdir = rootloc

outloc = outdir + "/sneplists/snipshot_times.dat"

ascii.write(np.vstack((index, aexplist)).T, outloc, format='no_header')


print("Done!")

