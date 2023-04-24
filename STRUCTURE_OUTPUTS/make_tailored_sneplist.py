"""
This program creates a sub-list of snip and snapshots to match pre-defined snepshot lists
"""

import eagle_routines as er
import os
import argparse
import sys
from astropy.io import ascii
from astropy.table import Table
import numpy as np
from pdb import set_trace

outlistdir = '/freya/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/'

parser = argparse.ArgumentParser()
parser.add_argument("dir", help = "The directory in which the simulation is sitting")
parser.add_argument("-l", "--list", help = "The snepshot list to be matched")
parser.add_argument("-r", "--root", help = "Switch to create root list", action = "store_true")
parser.add_argument("-t", "--tolerance", help = "Maximum allowed deviation in aexp", type = float)
parser.add_argument("-x", "--exclude", help = "Optional list of snapshots to be excluded")
parser.add_argument("-e", "--eagle", help = "Set this to run on original Eagle", action = "store_true")
parser.add_argument("-n", "--nomatch_root", help = "Do not match root list", action = "store_true") 

args = parser.parse_args()
rootloc = args.dir


if args.list == None:
    print("Please be so kind to provide a snepshot list!")
    sys.exit()

if args.eagle:
    outdir = er.clone_dir(rootloc) + '/sneplists/'
else:
    outdir = rootloc + '/sneplists/'
    
if not os.path.isdir(outdir):
        os.makedirs(outdir)

if args.root:
    lists = args.list.split()
    
    outloc = outdir + 'root.dat'
else:
    lists = [args.list]
    outloc = outdir + args.list + '.dat'

if args.tolerance == None:
    tol = 1e-5
else:
    tol = args.tolerance

if args.exclude == None:
    list_exclude = []
else:
    list_exclude = []
    excl_list_string = args.exclude.split()    
    for iexcl in excl_list_string:
        list_exclude.append(int(iexcl))

sniplistloc = outdir + '/snipshot_times.dat'
rootlistloc = outdir + '/root.dat'

if not os.path.exists(sniplistloc):
    print("The current simulation '" + rootloc + "' does not (yet) have a snipshot_times.dat file.\nPlease create one by running list_snipshot_times.py.")
    print("For now, we are ignoring snipshots.")
    sniplist_exists = False
else:
    sniplist_exists = True

if args.eagle:
    snaplistloc = outlistdir + 'eagle_outputs_new.txt'
else:
    snaplistloc = outlistdir + 'hydrangea_snapshots_plus.dat'


# Step 1: Load the desired snepshot list(s):
aexp_snep = np.zeros(0)
source_snep = np.zeros(0, dtype = np.int8)

list_rank = np.array(["regsnaps", "allsnaps", "z0_only", "basic", 
                      "default_long", "full_movie", "short_movie"])

for inlist in lists:

    listcode = np.nonzero(list_rank == inlist)[0]
    if len(listcode) != 1:
        print("Unrecognized list!")
        set_trace()
    
    sneplistloc = outlistdir + 'hydrangea_snepshots_' + inlist + '.dat'
    if not os.path.exists(sneplistloc):
        print("Specified sneplist '" + inlist + "' does not exist...")
        set_trace()

    snepdata = ascii.read(sneplistloc, guess=False, format='no_header')
    aexp_snep = np.concatenate((aexp_snep, snepdata['col1']))
    source_snep = np.concatenate((source_snep, np.zeros(len(snepdata['col1']), dtype = np.int8)+listcode))

aexp_snep, ind = np.unique(aexp_snep, return_index = True)
aexp_diff = aexp_snep[1:]-aexp_snep[:-1]
ind_dupl = np.nonzero(aexp_diff < tol)[0]


if len(ind_dupl) > 0:
    print("WARNING: found {:d} duplicates in input lists!" 
          .format(len(ind_dupl)))
    
    mask_keep = np.zeros(len(aexp_diff)+1, dtype = np.int8)+1
    mask_keep[ind_dupl] = 0
    mask_keep[ind_dupl+1] = 0

    for idupl in ind_dupl:
        input_this = np.nonzero(np.abs(aexp_snep-aexp_snep[idupl]) < 1e-4)[0]
        if len(input_this) != 2:
            set_trace()

        ind_keep = np.argmin(source_snep[ind[input_this]])
        mask_keep[input_this[ind_keep]] = 1
            
    aexp_snep = aexp_snep[np.nonzero(mask_keep == 1)[0]]

n_snep = len(aexp_snep)

# Step 1b: load the snip/snapshot data of simulation OR root list

if not args.root and not args.nomatch_root:
    rootdata = ascii.read(rootlistloc)

    index_root = rootdata['rootIndex']
    aexp_root = rootdata['aexp']
    type_root = rootdata['sourceType']
    num_root = rootdata['sourceNum']

else:
    
    if sniplist_exists:
        snipdata = ascii.read(sniplistloc, guess=False, format='no_header')
        aexp_snip = snipdata['col2']
        index_snip = snipdata['col1']
    else:
        aexp_snip = np.zeros(0)
        index_snip = np.zeros(0, dtype = int)

    snapdata = ascii.read(snaplistloc, guess=False, format='no_header')
    aexp_snap = snapdata['col1']


# Set up the output lists:
list_rootind = np.empty(n_snep, dtype=int)-1
list_sourcetype = np.empty(n_snep, dtype='<U5')
list_sourcenum = np.empty(n_snep, dtype=int)-1

# Step 2: Loop through snepshot entries:
for i, aexp_curr in enumerate(aexp_snep):

    if i % 10 == 0:
        print("Matching snep", i, "/", len(aexp_snep))
        
    # If root exists and is to be matched, only select from there: 
    if not args.root and not args.nomatch_root:
        ind_match_root = (np.nonzero(abs(aexp_root-aexp_curr) < tol))[0]
        if len(ind_match_root) != 1:
            print("Could not (uniquely) match snep {:d}..."
                  .format(i))
            set_trace()

        list_rootind[i] = ind_match_root[0]
        list_sourcetype[i] = type_root[ind_match_root[0]]
        list_sourcenum[i] = num_root[ind_match_root[0]] 
        
    else:
        # "Normal" case where we have to load from sn[a/i]p lists...

        # Try to find matching SNAPSHOT first:

        ind_match_snap = (np.nonzero(abs(aexp_snap-aexp_curr) < tol))[0]
        if len(ind_match_snap) > 1:
            print("Really? More than one matching SNAP???")
            print("Ok, this happened at i=%s, aexp_curr=%.4f" % (i, aexp_curr))
            print("This should not happen, please investigate!")
            sys.exit()
        
        elif len(ind_match_snap) == 1:
            if ind_match_snap[0] not in list_exclude:
                list_sourcetype[i] = 'snap'
                list_sourcenum[i] = ind_match_snap[0]
                continue
        
        # If we get here, we could not find a matching snapshot, so...
        # ... try finding a snipshot instead

        ind_match_snip = (np.nonzero(abs(aexp_snip-aexp_curr) < tol))[0]
        n_match = len(ind_match_snip)

        # First case: multiple matches
        if n_match > 1:
            print("   Weird fact: more than one matching snipshot (n=%d)..." % n_match)
            print("      i=%s, aexp=%.4f" % (i, aexp_curr))

            match_devs = aexp_snip[ind_match_snip]-aexp_curr
            for imatch, curr_match in enumerate(ind_match_snip):
                print("         %d: Delta = %f" % (curr_match, match_devs[imatch]))
            
            ind_bestmatch = np.argmin(np.abs(match_devs))
            print("      Best match: %d with |Delta| = %f" % (ind_match_snip[ind_bestmatch], np.abs(match_devs[ind_bestmatch])))

            list_sourcetype[i] = 'snip'
            list_sourcenum[i] = int(index_snip[ind_match_snip[ind_bestmatch]])

        # Second case: exactly one match
        elif n_match == 1:
            list_sourcetype[i] = 'snip'
            list_sourcenum[i] = int(index_snip[ind_match_snip[0]])

        # Third case: no match...
        elif len(ind_match_snip) == 0:
            print("No matching output found for snep %i at aexp=%.4f..." % (i, aexp_curr))    
            print("If you think this should not happen, please investigate...")
            sys.exit()

        else:
            print("How can this be reached???")
            sys.exit()


list_snepind = np.arange(n_snep, dtype = int)
if args.root or args.nomatch_root:
    list_rootind = list_snepind
    
data = Table([list_snepind, list_rootind, aexp_snep, list_sourcetype, list_sourcenum], names = ['index', 'rootIndex', 'aexp', 'sourceType', 'sourceNum'])
ascii.write(data, outloc, format = 'fixed_width', delimiter = ' ', overwrite = True)


#ascii.write(np.vstack((list_snepind, list_rootind, aexp_snep, list_sourcetype, list_sourcenum)).T, outloc, format='no_header')


print("Done!")
