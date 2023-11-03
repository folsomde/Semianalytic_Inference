'''The :mod:`satgen_preprocessing` module contains some helper functions for processing the output of SatGen.

The SatGen GitHub repository [1]_ has two scripts used to generate MW satellite systems: ``TreeGen`` to create
merger trees describing the MW's accretion history, and ``SatEvo`` to evolve satellites from the time of their 
infall to :math:`z=0`. The output arrays from ``SatEvo`` contain a great deal of information, though they can 
be large and inconvenient. The scripts provided here extract interesting parameters from the ``SatEvo`` output
for use in the study.

References
----------
.. [1] https://github.com/shergreen/SatGen

'''


import numpy as np
import re
from glob import glob

import astropy.units as u
import astropy.constants as const
import astropy.coordinates as crd

from scipy import optimize as opt
from scipy.signal import argrelmax, argrelmin
from SatGen.profiles import Dekel

import warnings
import os
from multiprocessing import Pool, cpu_count

host_dtype = [('virial_mass', np.float64), ('stellar_mass', np.float64), ('concentration', np.float64), 
              ('virial_radius', np.float64), ('virial_velocity', np.float64), ('DekelCAD', np.float64, (3,))]


subhalo_dtype = [
    ('virial_mass', np.float64),('virial_radius', np.float64), ('stellar_mass', np.float64), ('concentration', np.float64),
    ('v_max', np.float64), ('r_max', np.float64), ('mass_loss', np.float64), ('stellar_mass_loss', np.float64),
    ('position', np.float64, (3,)), ('velocity', np.float64, (3,)), 
    ('half_light_radius', np.float64), ('half_mass_radius', np.float64),  ('rho150', np.float64),
    ('peak_virial_mass', np.float64), ('peak_stellar_mass', np.float64), ('peak_v_max', np.float64), ('infall_time', np.float64),
    ('pericenter', np.float64), ('apocenter', np.float64), ('eccentricity', np.float64),('pericentric_passages', np.float64),
    ('DekelCAD', np.float64, (3,)), ('MCADz_infall', np.float64, (5,)), ('MCADz_first_infall', np.float64, (5,)), 
    ('parentID', np.int32), ('infall_order', np.int16), ('parent_mass', np.float64), ('infall_parent_mass', np.float64), ('parent_alive', bool), ('time_pre_infall', np.float64)
]


MASS_FLOOR = 6.5

def read_folder(directory):
    '''Load an entire folder of SatGen evolution outputs
    
    Parameters
    ----------
    directory : string
        the location of the SatGen SatEvo output
    
    Returns
    -------
    output : list
        a shape (N, 2) list, containing the hosts in the (i, 0) entry and the satellites in the (i, 1) entry
    '''
    allfiles = glob(f'{directory}/tree*.npz')
    allfiles.sort(key = lambda text: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)',text)])
    print(f'>>> {len(allfiles)} files found in {os.path.basename(os.path.normpath(directory))}')
    
    Ncores = int(os.getenv('OMP_NUM_THREADS', cpu_count()))
    print(f'>>> {Ncores} cores available for use')
    print(f'processing with minimum log10(Mvir) = {MASS_FLOOR}')
    with Pool(Ncores) as pool:  # use as many as requested
        sim = pool.map(read_file, allfiles, 1)
    
    return sim
 

def save_folder(directory, savename):
    '''Save an entire folder of SatGen SatEvo output to a specified file
    
    Parameters
    ----------
    directory : string
        the location of the SatGen SatEvo output
    savename : string
        output filename (will be saved in the parent directory of :attr:`directory` unless an absolute path is given)
    '''
    sim = read_folder(directory)
    
    sats = np.concatenate([np.array([(i,) + tuple(tup) for tup in arr], dtype = np.dtype([('hostID', np.int16),] + subhalo_dtype)) for i, (_, arr) in enumerate(sim)])
    hosts = np.array([h for (h, _) in sim])
    
    saveloc = savename if savename[0] == '/' else os.path.join(os.path.dirname(os.path.normpath(directory)), savename)
    print('>>> saving to', saveloc)
    np.savez_compressed(saveloc, sats = sats, hosts = hosts)
    


def read_file(file):
    '''Extract a number of interesting features from a ``SatGen`` SatEvo output file
    
    Parameters
    ----------
    file : str
        The name of the SatGen SatEvo output file
        
    Returns
    -------
    host : np.ndarray
        Information about the MW host of this satellite system
    sats_output : np.ndarray
        Information about the satellite system.
        
    Notes
    -----
    In particular, the satellite system output includes 
     - the virial mass (in :math:`M_\odot`), 
     - virial radius (in kpc), 
     - the stellar mass (in :math:`M_\odot`), 
     - concentration :math:`c_{-2} = r_\mathrm{vir}/r_{-2}`, 
     - :math:`v_\mathrm{max}` (in km/s), 
     - :math:`r_\mathrm{max}` (in kpc), 
     - halo mass loss :math:`M_\mathrm{vir}/M_\mathrm{peak}`, 
     - stellar mass loss (defined similarly), 
     - position and velocity in galactocentric cartesian coordinates (kpc and km/s, respectively), 
     - half-light radius (in kpc), 
     - half-mass radius (in kpc), 
     - instantaneous density at 150 pc :math:`\rho_{150}` (in :math:`M_\odot/\mathrm{kpc}^3`, 
     - peak virial mass (in :math:`M_\odot`), 
     - peak stellar mass (in :math:`M_\odot`), 
     - peak :math:`v_\mathrm{max}` (in km/s), 
     - infall time (in Gyr lookback), 
     - pericenter and apocenter (in kpc), 
     - eccentricity, 
     - number of pericentric passages, 
     - :math:`z = 0` Dekel profile parameters (concentration, innermost slope, and virial overdensity), 
     - Dekel profile parameters at the time of infall into the MW (virial mass, :math:`c`, :math:`\alpha`, :math:`\Delta`, and redshift of infall), 
     - Dekel profiles at the time of first infall (upon leaving the field), 
     - the index of the order-one parent satellite within the output array (with -1 in the case of direct accretion or a destroyed parent), 
     - the order of the satellite at the time of accretion onto the MW, 
     - the mass (in :math:`M_\odot`) of the original group host evaluated at :math:`z = 0` (equal to the MW mass in the case of direct accretion), 
     - the mass (in :math:`M_\odot`) of the original group host evaluated at the time of accretion onto the MW, 
     - whether or not the original group host survives to :math:`z = 0`, and 
     - the time spent evolving in a group system before MW accretion (in Gyr). 
    
    More information is in principle able to be extracted from the SatGen data, but this is beyond sufficient for the study here.
    '''

    with np.load(file) as f:
        virial_mass = f['mass']
        virial_radius = f['VirialRadius']
        stellar_mass = f['StellarMass']
        conc = f['concentration']
        coords = f['coordinates']
        vmax = (f['MaxCircularVelocity'] * u.kpc/u.Gyr).to(u.km/u.s).value
        half_light_radius = f['StellarSize']
        time = f['CosmicTime']
        order = f['order']
        parent = f['ParentID']

        cDekel = f['DekelConcentration']
        aDekel = f['DekelSlope']
        Delta = f['VirialOverdensity']

        time = f['CosmicTime']
        satgenz = f['redshift']

    #--- read in host properties
    host_mass = virial_mass[0,0]
    host_mstar = stellar_mass[0,0]
    host_conc = conc[0,0]
    host_radius = virial_radius[0,0]
    host_velocity = (np.sqrt(const.G * host_mass * u.Msun / (host_radius * u.kpc))).to(u.km/u.s).value
    
    mask = (virial_mass[:, 0] > 10**MASS_FLOOR) & (parent[:, 0] == 0)
    
    sats = [Dekel(M, c, alpha, delta) for M, c, alpha, delta in zip(virial_mass[mask, 0], cDekel[mask, 0], aDekel[mask, 0], Delta[mask, 0])]
    num_sats = np.count_nonzero(mask)
    
    half_mass_radius = np.array([opt.minimize_scalar(lambda r: np.abs(s.M(r) - s.Mh/2)).x for s in sats])
    # M_half = [sat.M(rhalf) for sat, rhalf in zip(sats, half_light_radius)]
    
    rho150 = [sat.rho(0.150) for sat in sats]
    
    peak_mvir = virial_mass[mask].max(axis=1)
    peak_mstar = stellar_mass[mask].max(axis=1)
    # note: satellites that have stellar mass loss greater than 50% are likely to be disrupted
    mass_loss = virial_mass[mask,0]/peak_mvir
    stellar_mass_loss = stellar_mass[mask,0]/peak_mstar
    peak_vmax = vmax[mask].max(axis=1)

    immediate_infall = ((virial_mass > 0) & (np.linalg.norm(coords, axis = 2) > 0)).argmin(axis = 1) - 1
    O1_parent = parent[mask, immediate_infall[mask]]
    acc_order = order[mask, immediate_infall[mask]]
    # O2 objects currently look at the correct parent, but O3+ only see the O2 infall, not the O1. 
    # this relabels O2idx = parent[O3idx] -> O1idx = parent[parent[O3idx]]
    # similarly, O3idx = parent[O4idx] -> O1idx = parent[parent[parent[O4idx]]]
    for o in range(acc_order.max(), 2, -1):
        to_fix = (acc_order >= o)
        O1_parent[to_fix] = parent[O1_parent[to_fix], immediate_infall[mask][to_fix]]

    infall_snap = np.where(acc_order == 1, immediate_infall[mask], immediate_infall[O1_parent])
    infall_time = time[0] - time[infall_snap] # in terms of lookback time (i.e. infall_time of 1 means the satellite fell in 1 Gyr ago)
    time_pre_infall = time[infall_snap] - time[immediate_infall[mask]]
    
    # account for relative parent coordinates
    fix_c, is_init = fix_coords(coords, order, parent, virial_mass > 0)
    distance = np.linalg.norm(fix_c[mask, :, :3], axis=2)
    distance_ma = np.ma.array(distance, mask = np.array([np.arange(len(time)) > i for i in infall_snap]))

    pericenter, apocenter, pericentric_passages = [], [], []
    for branch, infall in zip(distance_ma.data, infall_snap):
        peris = argrelmin(branch[:infall + 1], order = 3)[0]
        apos  = argrelmax(branch[:infall + 1], order = 3)[0]
        if len(peris):
            peri = peris[0]
            if branch[0] < branch[peri]:
                peri = 0
        else:
            pre_infall_loc = branch[:infall + 1]
            peri = 0 if len(pre_infall_loc) == 0 else pre_infall_loc.argmin()
            # if not len(pre_infall_loc):
            #     print(file, 'had a weird branch with positions',branch,'and infall snap',infall)

        pericenter.append(branch[peri])
        pericentric_passages.append(len(peris))

        if len(apos):
            apo = apos[0]
            if branch[0] > branch[apo]:
                apo = 0
        else:
            pre_infall_loc = branch[:infall + 1]
            apo = 0 if len(pre_infall_loc) == 0 else pre_infall_loc.argmax()
            # if not len(pre_infall_loc):
            #     print(file, 'had a weird branch with positions',branch,'and infall snap',infall)

        apocenter.append(branch[apo])
    pericenter = np.array(pericenter)
    apocenter = np.array(apocenter)
    pericentric_passages = np.array(pericentric_passages)

    
    infall_parent_mass = virial_mass[O1_parent, infall_snap]
    present_parent_mass = virial_mass[O1_parent, 0]
    parent_survived = mask[O1_parent] | (parent[mask, infall_snap] == 0)
    # if parent survived, find the number of it in this array
    pidx_in_output = np.searchsorted(np.where(mask)[0], O1_parent[parent_survived])
    pid = np.full(num_sats, -1)
    pid[parent_survived] = pidx_in_output
    
    # host_dtype = virial_mass, stellar_mass, concentration, virial_radius, virial_velocity, DekelCAD, 
    host = np.array((host_mass, host_mstar, host_conc, host_radius, host_velocity, [cDekel[0,0], aDekel[0, 0], Delta[0, 0]]), dtype=np.dtype(host_dtype))
    #               subhalo_dtype = virial_mass, virial_radius, stellar_mass, concentration, 
    sats_output = np.array(list(zip(virial_mass[mask, 0], virial_radius[mask, 0], stellar_mass[mask,0], conc[mask,0],
    #                               v_max, r_max, mass_loss, stellar_mass_loss, 
                                    vmax[mask,0], [sat.rmax for sat in sats], mass_loss, stellar_mass_loss,
    #                               position, velocity, 
                                    # the relative coordinates won't matter for the first-order satellites here
                                    fix_c[mask, 0, :3], (fix_c[mask,0, 3:]*u.kpc/u.Gyr).to(u.km/u.s).value,
    #                               half_light_radius, half_mass_radius, rho150, 
                                    half_light_radius[mask,0], half_mass_radius, rho150,
    #                               peak_virial_mass, peak_stellar_mass, peak_v_max, infall_time, 
                                    peak_mvir, peak_mstar, peak_vmax, infall_time,
    #                               pericenter, apocenter, eccentricity, pericentric_passages, 
                                    pericenter, apocenter, (apocenter-pericenter)/(apocenter+pericenter), pericentric_passages,
    #                               DekelCAD, 
                                    list(zip(cDekel[mask, 0], aDekel[mask, 0], Delta[mask, 0])),
    #                               MCADz_infall, 
                                    list(zip(virial_mass[mask, infall_snap], cDekel[mask, infall_snap], aDekel[mask, infall_snap], Delta[mask, infall_snap], satgenz[infall_snap])),
    #                               MCADz_first_infall, 
                                    list(zip(virial_mass[mask, immediate_infall[mask]], cDekel[mask, immediate_infall[mask]], aDekel[mask, immediate_infall[mask]], Delta[mask, immediate_infall[mask]], satgenz[immediate_infall[mask]])),
    #                               parentID, infall_order, parent_mass, infall_parent_mass, parent_alive, time_pre_infall, 
                                    pid, acc_order, present_parent_mass, infall_parent_mass, parent_survived, time_pre_infall
                                   )), dtype=subhalo_dtype)
    
    return host, sats_output

def fix_coords(coords, order, parent, initialized = None):
    '''transform satgen satellite evolution output into galactocentric cartesian coordinates
    
    Parameters
    ----------
    coords : np.ndarray(float)
        SatGen ``coordinates`` array
    order : np.ndarray(int)
        SatGen ``order`` array
    parent : np.ndarray(int)
        SatGen ``ParentID`` array
    initialized : np.ndarray(bool)
        Tracks whether the branch has had its orbit integrated at the given snapshot. In theory, equivalent to ``mass > -99``.
        
    Returns
    -------
    coords : np.ndarray(float)
        Equivalent to SatGen ``coordinates`` array, but in the MW reference frame with cartesian coordinates
    initialized : np.ndarray(bool)
        True if the branch and all of its parents have had their orbit integrated at the given snapshot.
    '''
    # fix uninitialized coords
    coords[~initialized] = np.tile([0.01, 0, 0, 0, 0, 0], (np.count_nonzero(~initialized),1))
    # transform to cartesian
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='invalid value encountered in divide')
        skyobj = crd.SkyCoord(frame='galactocentric', representation_type='cylindrical',
                           rho=coords[:,:,0] * u.kpc, phi=coords[:,:,1] * u.rad, z=coords[:,:,2]* u.kpc,
                           d_rho = coords[:,:,3] * u.kpc/u.Gyr, d_phi = np.where(coords[:,:,0], coords[:,:,4]/coords[:,:,0], coords[:,:,0]) * u.rad/u.Gyr, d_z = coords[:,:,5] * u.kpc/u.Gyr
                          )
        xyz = skyobj.cartesian.xyz.to(u.kpc).value
        vel = skyobj.cartesian.differentials['s'].d_xyz.to(u.kpc/u.Gyr).value
    # this is the same thing as SatGen `coordinates`, i.e. [branch, redshift, xv], but in cartesian coords
    coordinates = np.moveaxis(np.r_[xyz, vel], 0, 2) 
    
    # start at the top of the tree and propagate to children (first-order subhalos are already okay)
    for o in range(2, order.max() + 1):
        to_fix = (order == o)
        branch, redshift = np.where(to_fix)
        # add parent's location to where the child is
        coordinates[to_fix] = coordinates[to_fix] + coordinates[parent[to_fix], redshift]
        # also make sure that the parent is initialized
        if initialized is not None:
            initialized[to_fix] = initialized[to_fix] & initialized[parent[to_fix], redshift]
    
    #TODO: maybe transform back to cyl if desired. Cartesian seems generally nicer to use, though
    return coordinates, initialized
 
