import numpy as np
from astropy.io import ascii

outlistdir = '/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/'
outloc = outlistdir + '/eagle_combined_sneplist.txt'

sniplist = outlistdir + 'eagle_snip_400_z20p00_z00p00.txt'
snaplist = outlistdir + 'eagle_outputs.txt'

snapdata = ascii.read(snaplist, guess=False, format='no_header')
snipdata = ascii.read(sniplist, guess=False, format='no_header')

aexp_snap = snapdata['col1']
n_snap = len(aexp_snap)

aexp_snip = snipdata['col1']
n_snip = len(aexp_snip)

aexp_snep = aexp_snip

for isnap, iaexp in enumerate(aexp_snap):
    abs_delta = np.abs(aexp_snip-iaexp)

    if (np.min(abs_delta) <= 1e-4):
        print("Snapshot {:d} close to snipshot {:d} -- skipping it" .format(isnap, np.argmin(abs_delta))) 
    else:
        aexp_snep = np.append(aexp_snep, iaexp) 

n_snep = len(aexp_snep)
print("Combined sneplist has {:d} elements." .format(n_snep))

aexp_snep.sort()

ascii.write(np.vstack((aexp_snep)), outloc, format='no_header')


print("Done!")
