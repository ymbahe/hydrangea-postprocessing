
import numpy as np
from pdb import set_trace
from astropy.io import fits

inloc_clusters = '/ptmp/mpa/ybahe/HYDRANGEA/RESULTS/cluster_growth_table_withrad.fit'
inloc_galaxies = '/ptmp/mpa/ybahe/HYDRANGEA/RESULTS/galaxy_pos_table_full.fit'
inloc_cat = '/ptmp/mpa/ybahe/HYDRANGEA/RESULTS/CombinedBasicGalaxyCatalogue_17OCT16_S29_starsplit_noradcut.fit'

outloc = '/ptmp/mpa/ybahe/HYDRANGEA/RESULTS/galaxy_infall_times.fit'

hdulist = fits.open(inloc_clusters)
rad_clusters = hdulist[1].data

hdulist = fits.open(inloc_galaxies)
galpos = hdulist[0].data
galshi = hdulist[1].data
satflag = hdulist[2].data

hdulist = fits.open(inloc_cat)
cat_in = hdulist[1].data

ngal = satflag.shape[0]

halo_list_all = cat_in['cluster_index'][:]
halo_int = np.zeros(len(halo_list_all),dtype=int) 

shi_z0 = cat_in['SHI'][:]



for ihu, hu in enumerate(halo_list_all):
    curr_hu = int(hu[1:])
    halo_int[ihu] = curr_hu

halo_int_unique = np.unique(halo_int)
haloes_rev = np.zeros(41, dtype = int)-1
haloes_rev[halo_int_unique] = np.arange(24,dtype=int)


# Determine when galaxy first became a satellite

sattime = np.zeros(ngal)-1
timevec = np.linspace(13.5, 0, num=28, endpoint=True)

for igal in range(ngal):
    satlist = satflag[igal,:]
    ind_sat = np.nonzero(satlist > 0)[0]
    
    if len(ind_sat) == 0:
        continue

    ind_sat_first = ind_sat[0]
    sattime[igal] = timevec[ind_sat_first]


# Determine time when galaxy first crossed r200

inftime = np.zeros(ngal)-1

for igal in range(ngal):
    poslist = galpos[igal,:,:]
    radlist = rad_clusters[haloes_rev[halo_int[igal]],:]

    ind_good = np.nonzero((poslist[:,0] >= 0) & (radlist >= 0))[0]
    

    if len(ind_good) == 0:
        continue
    
    ind_bcg = np.nonzero((shi_z0 == 0) & (halo_int == halo_int[igal]))[0]
    if len(ind_bcg) != 1:
        set_trace()

    ind_bcg = ind_bcg[0]

    pos_bcg = galpos[ind_bcg,:,:]

    pos_rel = poslist-pos_bcg
    rad = np.linalg.norm(pos_rel,axis=1)

    rad_rel = rad/radlist

    rad_rel = rad_rel[ind_good]

    ind_in = np.nonzero(rad_rel < 1.0)[0]

    if len(ind_in) == 0:
        continue
    
    first_in = ind_in[0]

    if first_in == 0:
        inftime[igal] = timevec[0]
        continue
 
    last_out = first_in-1
    
    inftime[igal] = timevec[ind_good[first_in]] + (timevec[ind_good[last_out]]-timevec[ind_good[first_in]])*(1.0-rad_rel[first_in])/(rad_rel[last_out]-rad_rel[first_in])

    if igal == 67787:
        set_trace()


hdu = fits.PrimaryHDU(sattime)
hdulist = fits.HDUList([hdu])

hdu2 = fits.PrimaryHDU(inftime)
hdulist.append(hdu2)

hdulist.writeto(outloc, clobber = True)

set_trace()

print("Done!")

