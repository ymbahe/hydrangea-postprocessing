"""
This program creates a sub-list of snip and snapshots to match pre-defined snepshot lists
"""

import eagle_routines as er
import argparse
import sys
from astropy.io import ascii
import os
import numpy as np
from pdb import set_trace

outlistdir = '/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/'

parser = argparse.ArgumentParser()
parser.add_argument("dir", help = "The directory in which the simulation is sitting")
parser.add_argument("-l", "--list", help = "The snepshot list to be matched")
parser.add_argument("-t", "--tolerance", help = "Maximum allowed deviation in aexp", type = float)
parser.add_argument("-e", "--eagle", help = "Set this to run on original Eagle", action = "store_true")

args = parser.parse_args()
rootloc = args.dir

if args.list == None:
    print("Please be so kind to provide a snepshot list!")
    sys.exit()
else:
    if args.eagle:
        outdir = er.clone_dir(rootloc)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        sneplistloc = outlistdir + args.list + '.txt'
        outloc = outdir + '/sneplist_for_' + args.list + '.dat'

    else:
        outdir = rootloc
        sneplistloc = outlistdir + 'hydrangea_snepshots_' + args.list + '.dat'
        outloc = outdir + '/sneplist_for_' + args.list + '.dat'

if args.tolerance == None:
    tol = 1e-4
else:
    tol = args.tolerance




# Sanity checks:
sniplistloc = outdir + '/snipshot_times.txt'
if not os.path.exists(sniplistloc):
    print("The current simulation '" + rootloc + "' does not (yet) have a snipshot_times.txt file.\nPlease create one by running list_snipshot_times.py.")
    sys.exit()

if args.eagle:
    snaplistloc = outlistdir + 'eagle_outputs_new.txt'
else:
    snaplistloc = outlistdir + 'hydrangea_snapshots_plus.dat'

 
# Step 1: Load the desired snepshot list:
snepdata = ascii.read(sneplistloc, guess=False, format='no_header')
aexp_snep = snepdata['col1']
n_snep = len(aexp_snep)

snipdata = ascii.read(sniplistloc, guess=False, format='no_header')
aexp_snip = snipdata['col2']

snapdata = ascii.read(snaplistloc, guess=False, format='no_header')
aexp_snap = snapdata['col1']

# Set up the output lists:
list_snepind = np.empty(n_snep, dtype=int)-1
list_sneptype = np.empty(n_snep, dtype='<U5')
list_sourceind = np.empty(n_snep, dtype=int)-1

# Step 2: Loop through snepshot entries:
for i, aexp_curr in enumerate(aexp_snep):

    if i % 10 == 0:
        print("Matching snep", i, "/", len(aexp_snep))

    ind_match_snip = (np.nonzero(abs(aexp_snip-aexp_curr) < tol))[0]
    n_match = len(ind_match_snip)
    if n_match > 1:
        print("   Weird fact: more than one matching snipshot (n=%d)..." % n_match)
        print("      i=%s, aexp=%.4f" % (i, aexp_curr))
        match_devs = aexp_snip[ind_match_snip]-aexp_curr
        for imatch, curr_match in enumerate(ind_match_snip):
            print("         %d: Delta = %f" % (curr_match, match_devs[imatch]))
            
        ind_bestmatch = np.argmin(np.abs(match_devs))
        print("      Best match: %d with |Delta| = %f" % (ind_match_snip[ind_bestmatch], np.abs(match_devs[ind_bestmatch])))
        currtype = 'snip'
        currind = ind_match_snip[ind_bestmatch]      

    elif n_match == 1:
        currtype = 'snip'
        currind = ind_match_snip[0]

    elif len(ind_match_snip) == 0:
        
        ind_match_snap = (np.nonzero(abs(aexp_snap-aexp_curr) < tol))[0]
        if len(ind_match_snap) > 1:
            print("Really? More than one matching SNAP???")
            print("Ok, this happened at i=%s, aexp_curr=%.4f" % (i, aexp_curr))
            print("This should not happen, please investigate!")
            sys.exit()

        elif len(ind_match_snap) == 1:
            currtype = 'snap'
            currind = ind_match_snap[0]
        else:
            print("No matching output found for snep %i at aexp=%.4f..." % (i, aexp_curr))    
            print("If you think this should not happen, please investigate...")
            set_trace()
            sys.exit()

    else:
        print("How can this be reached???")
        sys.exit()

    list_snepind[i] = i
    list_sneptype[i] = currtype
    list_sourceind[i] = currind

ascii.write(np.vstack((list_snepind, list_sneptype, list_sourceind)).T, outloc, format='no_header')


print("Done!")
