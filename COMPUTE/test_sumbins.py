import numpy as np
import ctypes as c
from pdb import set_trace

print("Binning up masses... ", end = '', flush = True)
print("")

numPt = int(1e3)
pt_mass = np.zeros(numPt, dtype = np.float32) + 0.1

pt_code = np.zeros(numPt, dtype = np.int8)
pt_host_mass = np.zeros(numPt, dtype = np.int8)
pt_root_mass = np.zeros(numPt, dtype = np.int8)+1
pt_host_rad = np.zeros(numPt, dtype = np.int8)
pt_relrad = np.zeros(numPt, dtype = np.int8)

host_mass_nbins = 3
root_mass_nbins = 4
host_lograd_nbins = 10
host_relrad_nbins = 15

mass_binned = np.zeros((7,                  # 7 possible origin codes
                        host_mass_nbins,
                        root_mass_nbins,
                        host_lograd_nbins,
                        host_relrad_nbins), dtype = np.float64)

# *********** IMPORTANT ********************************
# This next line needs to be modified to point
# to the full path of where the library has been copied.
# *******************************************************

print(pt_mass.dtype)
print(pt_code.dtype)
print(pt_host_mass.dtype)
print(pt_root_mass.dtype)
print(pt_host_rad.dtype)
print(pt_relrad.dtype)


ObjectFile = "/u/ybahe/ANALYSIS/PACKAGES/lib/sumbins.so"

c_numPart = c.c_long(numPt)
c_nbins_code = c.c_byte(7)
c_nbins_massHost = c.c_byte(host_mass_nbins)
c_nbins_massRoot = c.c_byte(root_mass_nbins)
c_nbins_radHost = c.c_byte(host_lograd_nbins)
c_nbins_relradHost = c.c_byte(host_relrad_nbins) 

partMass_p = pt_mass.ctypes.data_as(c.c_void_p)

code_p = pt_code.ctypes.data_as(c.c_void_p)
hostMassBin_p = pt_host_mass.ctypes.data_as(c.c_void_p)
rootMassBin_p = pt_root_mass.ctypes.data_as(c.c_void_p)
hostRadBin_p = pt_host_rad.ctypes.data_as(c.c_void_p)
hostRelradBin_p = pt_relrad.ctypes.data_as(c.c_void_p)

result_p = mass_binned.ctypes.data_as(c.c_void_p)

nargs = 13
myargv = c.c_void_p * nargs
argv = myargv(c.addressof(c_numPart), 
              c.addressof(c_nbins_code),
              c.addressof(c_nbins_massHost),
              c.addressof(c_nbins_massRoot),
              c.addressof(c_nbins_radHost),
              c.addressof(c_nbins_relradHost),
              partMass_p, 
              code_p,
              hostMassBin_p, rootMassBin_p, hostRadBin_p,
              hostRelradBin_p, 
              result_p)
    
lib = c.cdll.LoadLibrary(ObjectFile)
succ = lib.sumbins(nargs, argv)

set_trace()
