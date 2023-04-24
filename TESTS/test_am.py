"""
General-purpose program to make image of a galaxy.

 -- Started 21-Mar-2019
 -- Expanded 14-May-2019

"""

import sim_tools as st
import yb_utils as yb
import hydrangea_tools as ht
import image_routines as ir
from pdb import set_trace
from astropy.io import ascii
import numpy as np

import matplotlib
matplotlib.use('pdf')

import matplotlib.pyplot as plt

plt.style.use('dark_background')

matplotlib.rcParams['font.family'] = 'serif' 
matplotlib.rcParams['font.serif'][0] = 'palatino'

# Set up the simulation to image
rootdir = '/virgo/simulations/Hydrangea/10r200/'
simtype = 'HYDRO'
isim = 0

# Select the subhalo or galaxy to center on:
ish = 0                     # Only relevant if galID == None
galID = 24#77#257917#5186    # Will be derived from ish if None

follow_am = True

# Select snapshot, particle, and image type:
plot_snap = 17
ptype = 4
imtype = 'mass'     # 'gri' or 'mass', only for ptype=4
projectionPlane = 'xy'  # Image projection plane 
gas_tmax = None     # Only consider gas below a given T (can be None for all)

# Options to only plot particles in an individual galaxy:
# !! This is #not actually implemented yet !!
ref_galaxy = 24#None#24#1570#None#760    # Galaxy whose particles to consider exclusively 
ref_shi = None      # Subhalo whose particles to consider exclusively (unless
                    # ref_galaxy != None, or it is None)
ref_snap = plot_snap       # Snapshot in which particle membership is tested
sh_catalogue = 'Cantor' # Subhalo catalogue to use ('Subfind' or 'Cantor')

# Select image size and smoothing length
imsize = 0.05            # x/y size in Mpc (+/- vs. central galaxy)
zsize = 0.05             # z size in Mpc (+/- vs. mid-plane)
fixedSmoothingLength = 0  # Leave at 0 to compute adaptive smoothing lengths

# Options for indicating the locations of galaxies
mark_galaxies = False        # Highlight locations of galaxies?
mass_crit = 'MstarPeak'     # Selection criterion for which galaxies to show
                            # ('Mstar', 'MstarPeak', 'Msub', or 'MsubPeak')
mass_threshold = 9.0        # Minimum mass for galaxies to show
plot_dead_galaxies = True   # True: also show dead galaxies (in red).
                            # NB: only effective if mass_crit == '[...]Peak'
label_galaxies = True      # True: also print galaxy IDs in image
                            # (otherwise only a circle is drawn)
labelSize = 4.0             # Specify size of galaxy labels

# Options for storing the image:
save_maps = False     # Store images as HDF5 as well as picture
desNGB = 32           # Number of neighbours for smoothing calculation
numPix = 2400         # Image sidelength in pixels       

# Location for storing image file(s):
#plotloc = '/virgo/scratch/ybahe/HYDRANGEA/IMAGES/CE-29_BCG-x30p0-z5p0_'
#plotloc = '/virgo/scratch/ybahe/HYDRANGEA/IMAGES/CE-21_BCG_PT-{:d}_0p05_400p_' .format(ptype)
#plotloc = '/virgo/scratch/ybahe/HYDRANGEA/IMAGES/CE-21_BCG_PT-{:d}_1x0p2_IR_off0p25_' .format(ptype)
#plotloc = '/virgo/scratch/ybahe/HYDRANGEA/IMAGES/CE-28_BCG_PT-{:d}_1p5x1p5_IR_GRI_28-18-maas_' .format(ptype)
#plotloc = '/virgo/scratch/ybahe/HYDRANGEA/IMAGES/CE-28_BCG_PT-{:d}_1p5x1p5_IR_coolOnly_' .format(ptype)
#plotloc = '/virgo/scratch/ybahe/HYDRANGEA/IMAGES/CE-14_BCG_PT-{:d}_1p5_IR_GRI_' .format(ptype)
#plotloc = '/virgo/scratch/ybahe/HYDRANGEA/IMAGES/CE-29_G-2373_PT-{:d}_0p25_IR_mass_' .format(ptype)
plotloc = '/virgo/scratch/ybahe/HYDRANGEA/IMAGES/CE-0_G24AM_PT-{:d}_0p3_IR_mass_' .format(ptype)

# ---------------------------------------------- 

rundir = rootdir + 'CE-' + str(isim) + '/' + simtype + '/'
hldir = rundir + '/highlev/'

fgtloc = hldir + 'FullGalaxyTables.hdf5'
posloc = hldir + 'GalaxyPositionsSnap.hdf5'
spiderloc = hldir + 'SpiderwebTables.hdf5'

shi = yb.read_hdf5(fgtloc, 'SHI')

if sh_catalogue == 'Cantor':
    cantorloc = ht.clone_dir(hldir) + 'CantorCatalogue_7Jun19.hdf5'
    shiX = yb.read_hdf5(cantorloc, 'SubHaloIndexExtended') 

dpi = numPix/6

if galID is None:
    galID = yb.read_hdf5(spiderloc, 'Subhalo/Snapshot_' + str(plot_snap).zfill(3) + '/Galaxy')[ish]

pos_gal_all = yb.read_hdf5(posloc, 'Centre')[:, plot_snap, :]
galpos = pos_gal_all[galID, :]



# Set up reading region
readRad = max([imsize, zsize])*np.sqrt(3)

snaplistloc = rundir + "/sneplists/allsnaps.dat"
snaplist = ascii.read(snaplistloc)
aexpSnap = np.array(snaplist['aexp'])
aexp_factor = aexpSnap[plot_snap]

conv_astro_pos = aexpSnap[plot_snap]/0.6777
readRad_sim = readRad/conv_astro_pos
galpos_sim = galpos/conv_astro_pos
snapdir = st.form_files(rundir, plot_snap, 'snap')

if ref_galaxy is None:
    if ref_shi is not None:
        ref_galaxy = yb.read_hdf5(
            spiderloc, 'Subhalo/Snapshot_' + str(ref_snap).zfill(3) 
            + '/Galaxy')[ref_shi]

if ref_galaxy is not None:
    ref_shi = shiX[ref_galaxy, ref_snap]
    if sh_catalogue == 'Cantor':
        cantorloc = ht.clone_dir(hldir) + 'CantorCatalogue_7Jun19.hdf5'
        ref_ids_all = yb.read_hdf5(cantorloc, 'Snapshot_' 
                                   + str(ref_snap).zfill(3) 
                                   + '/IDs')
        ref_off = yb.read_hdf5(
            cantorloc, 'Snapshot_' + str(ref_snap).zfill(3) 
            + '/Subhalo/OffsetType')[ref_shi, ptype]
        ref_len = yb.read_hdf5(
            cantorloc, 'Snapshot_' + str(ref_snap).zfill(3) 
            + '/Subhalo/LengthType')[ref_shi, ptype]
        ref_ids = ref_ids_all[ref_off : ref_off+ref_len]

        am = yb.read_hdf5(
            cantorloc, 'Snapshot_' + str(ref_snap).zfill(3)
            + '/Subhalo/Centre/StellarAngularMomentum')[ref_shi, :]

        galvel = yb.read_hdf5(cantorloc, 'Snapshot_017/Subhalo/StellarCentreOfMass_Velocity')[ref_shi, :]

    else:
        print("Subfind not yet implemented...")
        set_trace()


if ptype == 4 and imtype == 'gri':
    
    subpartdir = st.form_files(rundir, plot_snap, 'subpart')
    magdir = "/virgo/scratch/ybahe/PARTICLE_MAGS/CE-" + str(isim) + "/HYDRO/data/partMags_EMILES_PDXX_DUST_CH_029_z000p000.0.hdf5"
    pos = st.eagleread(subpartdir, 'PartType4/Coordinates', astro = True)[0]
    mag_g = st.eagleread(magdir, "g-Magnitude", astro = False)
    mag_r = st.eagleread(magdir, "r-Magnitude", astro = False)
    mag_i = st.eagleread(magdir, "i-Magnitude", astro = False)
    
    mass = st.eagleread(subpartdir, 'PartType4/Mass', astro = True)[0]
    sft = st.eagleread(subpartdir, 'PartType4/StellarFormationTime', astro = False)

    lum_g = 10.0**(-0.4*mag_g)
    lum_r = 10.0**(-0.4*mag_r)
    lum_i = 10.0**(-0.4*mag_i)

else:
    readReg = ht.ReadRegion(snapdir, ptype, [*galpos_sim, readRad_sim])

        
    pos = readReg.read_data("Coordinates", astro = True)
    vel = readReg.read_data("Velocity", astro = True)

    if ptype != 1:
        mass = readReg.read_data("Mass", astro = True)
    else:
        mass = np.zeros(pos.shape[0])+st.m_dm(rundir, astro = True)

        
    if ptype == 0:
        temp = readReg.read_data("Temperature", astro = True)
        hsml = readReg.read_data("SmoothingLength", astro = True)
        sfr = readReg.read_data("StarFormationRate", astro = True)
        temp[np.nonzero(sfr > 0)] = 1e4

    if ref_galaxy is not None:
        ids = readReg.read_data("ParticleIDs", astro = False)

        gate = st.Gate(ids, ref_ids)
        ref_index = gate.in2()
        ind_in_ref = np.nonzero(ref_index >= 0)[0]

        pos = pos[ind_in_ref, :]
        vel = vel[ind_in_ref, :]
        mass = mass[ind_in_ref]
        
        if ptype == 0:
            temp = temp[ind_in_ref]
            hsml = hsml[ind_in_ref]
            sfr = sfr[ind_in_ref]

deltaPos = pos - galpos[None, :]

# Select only those particles that are actually within target sphere
rad = np.linalg.norm(deltaPos, axis = 1)
ind_sphere = np.nonzero((rad < imsize*np.sqrt(3)) & (np.abs(deltaPos[:, 2]) < zsize))[0]

pos = pos[ind_sphere, :]
mass = mass[ind_sphere]
vel = vel[ind_sphere, :]

if ptype == 4 and imtype == 'gri':
    lum_g = lum_g[ind_sphere]
    lum_r = lum_r[ind_sphere]
    lum_i = lum_i[ind_sphere]
    sft = sft[ind_sphere]

if ptype == 0:
    hsml = hsml[ind_sphere]

if ptype == 0 and gas_tmax is not None:

    ind_tsel = np.nonzero(temp[ind_sphere] < gas_tmax)[0]
    quant = temp[ind_sphere[ind_tsel]]
    mass = mass[ind_tsel]
    pos = pos[ind_tsel, :]
    
else:
    quant = mass

if ptype != 0:

    if fixedSmoothingLength > 0:
        hsml = np.zeros(len(mass), dtype = np.float32) + fixedSmoothingLength
    else:
        hsml = None
        
# Generate actual image

if ptype == 4 and imtype == 'gri':

    image_weight_all, image_quant, hsml_true = ir.make_sph_image_new_3d(pos, mass, mass, hsml, DesNgb=desNGB, imsize=numPix, zpix = 1, boxsize = imsize, CamPos = galpos, CamDir = [0,0,-1], ProjectionPlane = projectionPlane, CamAngle = [0,0,0], CamFOV = [0.0, 0.0], make_deepcopy = True, zrange = [-zsize, zsize], tau = 1e6, return_hsml = True)

    hsml = (hsml_true/1e-3)**((1.7-sft)) * 1e-3
    #ind_young = np.nonzero(sft > 0.9)[0]
    #hsml[ind_young] = (hsml_true[ind_young]*1e3)**(1.0) * 1e3
    hsml = np.clip(hsml, None, np.clip(0.1*(1-sft)**(0.5), 0.003, None))
    #hsml[ind_young] = np.clip(hsml[ind_young], None, 0.005)
    
    image_weight_all_g, image_quant = ir.make_sph_image_new_3d(pos, lum_g, lum_g, hsml, DesNgb=desNGB, imsize=numPix, zpix = 1, boxsize = imsize, CamPos = galpos, CamDir = [0,0,-1], ProjectionPlane = projectionPlane, CamAngle = [0,0,0], CamFOV = [0.0, 0.0], make_deepcopy = True, zrange = [-zsize, zsize], tau = 1e6, return_hsml = False)

    image_weight_all_r, image_quant = ir.make_sph_image_new_3d(pos, lum_r, lum_r, hsml, DesNgb=desNGB, imsize=numPix, zpix = 1, boxsize = imsize, CamPos = galpos, CamDir = [0,0,-1], ProjectionPlane = projectionPlane, CamAngle = [0,0,0], CamFOV = [0.0, 0.0], make_deepcopy = True, zrange = [-zsize, zsize], tau = 1e6, return_hsml = False)

    image_weight_all_i, image_quant = ir.make_sph_image_new_3d(pos, lum_i, lum_r, hsml, DesNgb=desNGB, imsize=numPix, zpix = 1, boxsize = imsize, CamPos = galpos, CamDir = [0,0,-1], ProjectionPlane = projectionPlane, CamAngle = [0,0,0], CamFOV = [0.0, 0.0], make_deepcopy = True, zrange = [-zsize, zsize], tau = 1e6, return_hsml = False)

    map_maas_g = -5/2*np.log10(image_weight_all_g[:, :, 1]+1e-5)+5*np.log10(180*3600/np.pi) + 25
    map_maas_r = -5/2*np.log10(image_weight_all_r[:, :, 1]+1e-5)+5*np.log10(180*3600/np.pi) + 25
    map_maas_i = -5/2*np.log10(image_weight_all_i[:, :, 1]+1e-5)+5*np.log10(180*3600/np.pi) + 25

else:

    if follow_am:
        image_weight_all, image_quant = ir.make_sph_image_new_3d(pos, mass, quant, hsml, DesNgb=desNGB, imsize=numPix, zpix = 1, boxsize = imsize, CamPos = galpos, CamDir = am, CamAngle = [0,0,0], CamFOV = [0.0, 0.0], make_deepcopy = True, zrange = [-zsize, zsize], tau = 1e6, return_hsml = False, treeAllocFac=10)
    else:
        image_weight_all, image_quant = ir.make_sph_image_new_3d(pos, mass, quant, hsml, DesNgb=desNGB, imsize=numPix, zpix = 1, boxsize = imsize, CamPos = galpos, CamDir = [0, 0, -1], ProjectionPlane = projectionPlane, CamAngle = [0,0,0], CamFOV = [0.0, 0.0], make_deepcopy = True, zrange = [-zsize, zsize], tau = 1e6, return_hsml = False)

if save_maps:
    maploc = plotloc + str(plot_snap).zfill(4) + '.hdf5'

    yb.write_hdf5(image_weight_all, maploc, 'Sigma', new = True)

    if ptype == 4 and imtype == 'gri':
        yb.write_hdf5(map_maas_g, maploc, 'Sigma_g')
        yb.write_hdf5(map_maas_r, maploc, 'Sigma_r')
        yb.write_hdf5(map_maas_i, maploc, 'Sigma_i')

    if ptype == 0:
        yb.write_hdf5(image_quant, maploc, 'Temperature')

    yb.write_hdf5(np.array((-imsize, imsize, -imsize, imsize)), maploc, 'Extent')

print("Obtained image, plotting...")

# Plot image...

fig = plt.figure(figsize = (6, 6))

if ptype == 4 and imtype == 'gri':

    vmin = -28.0 + np.array([-0.5, -0.25, 0.0])
    vmax = -18.0 + np.array([-0.5, -0.25, 0.0])
    
    clmap_rgb = np.zeros((numPix, numPix, 3))
    clmap_rgb[:, :, 2] = np.clip(((-map_maas_g)-vmin[0])/((vmax[0]-vmin[0])), 0, 1)
    clmap_rgb[:, :, 1] = np.clip(((-map_maas_r)-vmin[1])/((vmax[1]-vmin[1])), 0, 1)
    clmap_rgb[:, :, 0] = np.clip(((-map_maas_i)-vmin[2])/((vmax[2]-vmin[2])), 0, 1)
    
    #clmap_rgb[:, :, 0] = 0

    im = plt.imshow(clmap_rgb, extent = [-imsize, imsize, -imsize, imsize],
                    aspect = 'equal', interpolation = 'nearest', origin = 'lower',
                    alpha = 1.0)
    

else:
    sigma = np.log10(image_weight_all[:, :, 1]+1e-5)-2  # in Mpc / pc^2

    if ptype == 0:
        temp = np.log10(image_quant[:, :, 1])
        clmap_rgb = ir.make_double_image(sigma, temp)#, percSigma = [10.0, 99.99], rangeQuant= [6.0, 9.0])

        im = plt.imshow(clmap_rgb, extent = [-imsize, imsize, -imsize, imsize],
                    aspect = 'equal', interpolation = 'nearest', origin = 'lower',
                    alpha = 1.0)

    elif (pos.shape[0] > 32):
        if ptype == 1:
            #sigRange_use = np.percentile(sigma[np.nonzero(sigma >= 1e-4)], [0.01, 99.9])
            #vmin, vmax = 0.5, 3.2#sigRange_use[0], sigRange_use[1]#0, 4.5
            vmin, vmax = np.min(sigma), np.max(sigma)
            cmap = plt.cm.Greys_r

        else:
            vmin, vmax = -1, 3.0
            cmap = plt.cm.bone        
        
        im = plt.imshow(sigma, cmap = cmap, 
                        origin = 'lower', 
                        extent = [-imsize, imsize, -imsize, imsize], 
                        vmin = vmin, vmax = vmax, 
                        interpolation = 'none')

    else:
        plt.scatter(deltaPos[:, 0], deltaPos[:, 1], color = 'white')

if mark_galaxies:

    if mass_crit == 'MstarPeak':
        mpeak = yb.read_hdf5(fgtloc, 'Full/Mstar')
    elif mass_crit == 'MsubPeak':
        mpeak = yb.read_hdf5(fgtloc, 'Full/Msub')
    elif mass_crit == 'Mstar':
        mpeak = yb.read_hdf5(fgtloc, 'Mstar')[:, plot_snap]
    elif mass_crit == 'Msub':
        mpeak = yb.read_hdf5(fgtloc, 'Msub')[:, plot_snap]
    else:
        print("I do not understand mass_crit = '" + mass_crit + "'")
        set_trace()

    spiderloc = rundir + '/highlev/SpiderwebTables.hdf5'
    lastsnap = yb.read_hdf5(spiderloc, 'LastSnap')
    firstsnap = yb.read_hdf5(spiderloc, 'FirstSnap')
    satFlag = yb.read_hdf5(fgtloc, 'SatFlag')[:, plot_snap]

    ind_in_field = np.nonzero((np.max(np.abs(pos_gal_all-galpos[None, :]), axis = 1) < imsize) & (mpeak >= mass_threshold) & (aexpSnap[firstsnap] <= aexpSnap[plot_snap]))[0]
    
    sorter_lastsnap = np.argsort(lastsnap[ind_in_field])

    pos_in_field = (pos_gal_all[ind_in_field[sorter_lastsnap], :]-galpos[None, :])

    for iigal, igal in enumerate(ind_in_field[sorter_lastsnap]):
                
        if aexpSnap[lastsnap[igal]] < aexpSnap[plot_snap]:
            
            if plot_dead_galaxies:
                plotcol = 'red'
            else:
                continue
        else:
            plotcol = 'limegreen'
            
        if satFlag[igal]:
            marker = 'o'
        else:
            marker = 'D'

        plt.scatter(pos_in_field[iigal, 0], pos_in_field[iigal, 1], 5, edgecolor = plotcol, facecolor = 'none', alpha = 0.5, marker = marker)
            
        if label_galaxies:
            plt.text(pos_in_field[iigal, 0]+imsize/100, pos_in_field[iigal, 1]+imsize/100, str(igal), color = plotcol, va = 'bottom', ha = 'left', alpha = 0.5, fontsize = labelSize)



# Some embellishments on the image
plt.text(-0.045/0.05*imsize, 0.045/0.05*imsize, 'z = {:.3f}' 
         .format(1/aexp_factor - 1), va = 'center', 
         ha = 'left', color = 'white')

plt.text(0.045/0.05*imsize, 0.045/0.05*imsize, 'galID = {:d}' 
         .format(galID), va = 'center', 
         ha = 'right', color = 'white')
            
ax = plt.gca()
#ax.set_xlabel(r'$\Delta x$ [pMpc]')
#ax.set_ylabel(r'$\Delta y$ [pMpc]')
    
ax.set_xlim((-imsize, imsize))
ax.set_ylim((-imsize, imsize))

#plt.subplots_adjust(left = 0.13, right = 0.95, bottom = 0.12, top = 0.92)
plt.subplots_adjust(left = 0.1, right = 0.95, bottom = 0.1, top = 0.95)
plt.savefig(plotloc + str(plot_snap).zfill(4) + '.png', dpi = dpi)
plt.close()

            
    
print("Done!")



