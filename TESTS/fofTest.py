import halobuilder2
import numpy as np
from pdb import set_trace
import yb_utils as yb
import sim_tools as st
from scipy.spatial import cKDTree

np.random.seed(2205)

synthetic = False

rundir = '/virgo/simulations/Hydrangea/10r200/CE-29/HYDRO/'
snapdir = st.form_files(rundir, 29, 'snap')

if synthetic:

    nPart = 1000000

    pos = np.zeros((2*nPart+1, 3), dtype = np.float64)
    #pos[:1000, :] = 10*np.random.randn(1000, 3).astype(np.float64)
    pos[0:nPart, :] = 1.5*np.random.randn(nPart, 3).astype(np.float64)
    pos[nPart:2*nPart, :] = 1.5*np.random.randn(nPart, 3).astype(np.float64)

    pos[:nPart, 0] -= 5.0
    pos[nPart:2*nPart, 0] += 5.0
    pos[:, 2] = 0

    #pos[2000, 0] =

    #pos[0, :] = [0, 0, 0]
    #pos[1, :] = [1, 0, 0]

    linkLength = 0.01

else:

    mpc = 3.0856e22
    msun = 1.989e30
    gnewton_astro = 6.67e-11 / (mpc**3) * msun
    h0_astro = 67.77 * 1e3 / mpc

    pos = (st.eagleread(snapdir, 'PartType1/Coordinates', astro = True)[0]).astype(np.float64)    
    linkLengthCubed = 1e10 * st.m_dm(rundir, astro = True) * 8*np.pi*gnewton_astro/(0.307*3*h0_astro**2)
    linkLength = 0.2*linkLengthCubed**(1/3)
    #linkLength = 0.01

    print("Determined linkLength = {:.3f} cMpc" .format(linkLength))

grps = halobuilder2.halobuilder2(pos, linkLength, 1, 1, 32)

outloc = '/virgo/scratch/ybahe/DEVELOPMENT/TESTS/FOFtest.hdf5'

"""
pos_bar = np.zeros(0, 3)
pos_bar = np.concatenate((pos_bar, st.eagleread(snapdir, 'PartType0/Coordinates', astro = True)[0]))
pos_bar = np.concatenate((pos_bar, st.eagleread(snapdir, 'PartType4/Coordinates', astro = True)[0]))
pos_bar = np.concatenate((pos_bar, st.eagleread(snapdir, 'PartType5/Coordinates', astro = True)[0]))

tree_dm = cKDTree(pos)
tree_bar = cKDTree(pos_bar)

ngbs = tree_dm.query_ball_tree(tree_bar, linkLength*2)
"""

#yb.write_hdf5(pos, outloc, 'Pos', new = True)
#yb.write_hdf5(grps, outloc, 'Grp')

set_trace()
