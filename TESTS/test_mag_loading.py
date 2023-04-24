import numpy as np
import sim_tools as st
import hydrangea_tools as ht
import yb_utils as yb
from pdb import set_trace

rundir = '/virgo/simulations/Hydrangea/10r200/CE-0/HYDRO/'

snapdir = st.form_files(rundir, 29, 'snap')
magfile = ht.clone_dir(rundir) + '/highlev/StellarMagnitudes_029.hdf5'

posloc = rundir + '/highlev/GalaxyPositionsSnap.hdf5'
fgtloc = rundir + '/highlev/FullGalaxyTables.hdf5'
spiderloc = rundir + '/highlev/SpiderwebTables.hdf5'

galID = yb.read_hdf5(spiderloc, 'Subhalo/Snapshot_029/Galaxy')[5]
galPos = yb.read_hdf5(posloc, 'Centre')[galID, 29, :]

readReg = ht.ReadRegion(snapdir, 4, [*galPos*0.6777, 1.0])
ids = readReg.read_data("ParticleIDs")
shi = readReg.read_data("SubHaloIndex", singleFile = True, filename = magfile, verbose = True, PTName = '')


set_trace()
