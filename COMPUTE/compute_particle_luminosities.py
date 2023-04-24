import sim_tools as st
import yb_utils as yb
import numpy as np
from pdb import set_trace
from scipy.interpolate import interp1d, griddata
from astropy.cosmology import Planck13

catloc = '/virgo/scratch/ybahe/HYDRANGEA/STAR_CATALOGUES/CE-16/CE-16_snap_029_core.hdf5'

ssploc = '/freya/ptmp/mpa/ybahe/HYDRANGEA/ANALYSIS/PARSEC_1.2S_SSPs_large.hdf5'

ssp_age = yb.read_hdf5(ssploc, 'age')
ssp_zmet = yb.read_hdf5(ssploc, 'Zmet')

sft = yb.read_hdf5(catloc, 'PartType4/StellarFormationTime')
zmet = yb.read_hdf5(catloc, 'PartType4/SmoothedMetallicity')

# Set up a fine interpolant to easily compute stellar ages:
zFine = np.arange(0, 20, 0.01)
ageFine = Planck13.age(zFine).value
csi_age = interp1d(zFine, ageFine, kind = 'cubic', fill_value = 'extrapolate')
birth_age = csi_age(1/sft-1)    # This is AGE OF THE UNIVERSE at formation

for band in ['u', 'g', 'r', 'i', 'z']:

    print("Processing band " + band + " ...")

    ssp_band = yb.read_hdf5(ssploc, band)
    star_za = np.meshgrid(ssp_zmet, ssp_age)

    set_trace()
    lum_band = griddata((ssp_zmet, ssp_age), ssp_band, star_za)

    #csi = interp2d(ssp_zmet, ssp_age, ssp_band, kind = 'cubic')
    #lum_band = csi(ssp_zmet, ssp_age)
    set_trace()

set_trace()

