import numpy as np
import yb_utils as yb
from pdb import set_trace

dataloc = '/virgo/scratch/ybahe/TESTS/G24-S15.hdf5'

pos = yb.read_hdf5(dataloc, 'Pos')
mass = yb.read_hdf5(dataloc, 'Mass')
vel = yb.read_hdf5(dataloc, 'Vel')

avVel = np.average(vel, weights=mass, axis = 0)

rad = np.linalg.norm(pos, axis = 1)
ind = np.nonzero(rad < 0.01)[0]

mit = np.zeros((3, 3))
for xx in range(3):
    for yy in range(3):
        mit[xx, yy] = np.sum(pos[ind, xx]*pos[ind, yy]*mass[ind])

# Now diagonalise MIT:
eigVal, eigVec = np.linalg.eig(mit)

# Find which eigenvector is the major, intermediate, and minor axis:
sorter = np.argsort(eigVal)
ax_a = sorter[0]  # Minor axis
ax_b = sorter[1]  # Intermediate axis
ax_c = sorter[2]  # Major axis

# Arrange eigenvectors appropriately for output:
axisVecs = np.zeros((3, 3))
axisVecs[:, 0] = eigVec[:, ax_a]
axisVecs[:, 1] = eigVec[:, ax_b]
axisVecs[:, 2] = eigVec[:, ax_c]
        
# Calculate axis ratios int/major, minor/major:
axisRatios = np.zeros(2)
axisRatios[0] = abs(eigVal[ax_b]/eigVal[ax_c])
axisRatios[1] = abs(eigVal[ax_a]/eigVal[ax_c])

set_trace()
