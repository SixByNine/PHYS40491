import numpy as np
from astropy import units as u

def ew_line(spacing=400*u.m,nant=8):
    """
    Example of a two-element interferometer with a line of sight along the x-axis.
    
    Parameters
    ----------
    spacing : float
        Spacing between the two antennas (in meters).
    nant : int
        Number of antennas.

    Returns
    -------
    antenna_positions : ndarray, shape (2, 3)
        Positions of the two antennas in 3D space (x, y, z).
    """
    # Convert spacing to meters
    spacing_m = spacing.to(u.m).value
    
    # Antenna positions:
    antenna_positions = np.zeros((nant, 3))
    for i in range(nant):
        antenna_positions[i] = [i * spacing_m, 0, 0]
    return antenna_positions

def vla(config='A'):
    if config=='A':
        Ndist = [0,436,1433,2874,4709,6906,9443,12302,15470,18935]
        Wdist = [484,1589,3188,5230,7644,10420,13660,17270,21250]
        Edist = [484,1589,3188,5230,7644,10420,13660,17270,21250]
    elif config=='B':
        Ndist=[0,135,436,872,1433,2180,3121,4255,5582]
        Edist=[147,484,968,1589,2495,3606,4922,6443]
        Wdist=[147,484,968,1589,2495,3606,4922,6443]
    elif config=='C':
        Ndist=[1,54,134,266,436,644,872,1120,1140,1754]
        Edist=[49,147,294,484,727,1021,1378,1800]
        Wdist=[49,147,294,484,727,1021,1378,1800]
    elif config=='D':
        Ndist=[1,54,94,134,194,266,347,436,527]
        Edist=[24,73,147,242,358,484,631]
        Wdist=[24,73,147,242,358,484,631]
    else:
        raise ValueError("Invalid configuration. Choose from 'A', 'B', 'C', or 'D'.")
    # North arm is heading 355 deg
    # East arm is heading 115 deg
    # West arm is heading 236 deg
    N_xy = np.array([[np.sin(np.deg2rad(355)), np.cos(np.deg2rad(355))] for d in Ndist])
    E_xy = np.array([[np.sin(np.deg2rad(115)), np.cos(np.deg2rad(115))] for d in Edist])
    W_xy = np.array([[np.sin(np.deg2rad(236)), np.cos(np.deg2rad(236))] for d in Wdist])
    N_xy = N_xy * np.array(Ndist)[:, np.newaxis]
    E_xy = E_xy * np.array(Edist)[:, np.newaxis]
    W_xy = W_xy * np.array(Wdist)[:, np.newaxis]
    antenna_positions = np.vstack((N_xy, E_xy, W_xy))

    return antenna_positions


def gaussian_cloud(nant, dispersion=300*u.m,seed=0x3606A562):
    """
    Example of a random distribution of antennas with a Gaussian distribution of baselines.
    
    Parameters
    ----------
    nant : int
        Number of antennas.
    dispersion : float
        Standard deviation of the Gaussian distribution of baselines (in meters).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    antenna_positions : ndarray, shape (nant, 3)
        Positions of the antennas in 3D space (x, y, z).
    """
    # Store the seed to avoid affecting other results.
    store_seed=np.random.seed()
    np.random.seed(seed)
    # Generate random positions in 2D plane
    x = np.random.normal(0, dispersion.to(u.m).value, max(nant,1024))[:nant]
    y = np.random.normal(0, dispersion.to(u.m).value, max(nant,1024))[:nant]
    np.random.seed(store_seed)
    # Stack into a single array and add z=0 for 2D plane
    antenna_positions = np.vstack((x, y, np.zeros(nant))).T
    
    return antenna_positions


def spiral_array(nant, start_length, length_increment, angle_increment, start_angle=270*u.deg):
    """
    Example of a spiral array configuration.
    
    Parameters
    ----------
    nant : int
        Number of antennas.
    start_length : float
        Starting length of the spiral (in meters).
    length_increment : float
        Incremental length for each subsequent antenna (in meters).
    angle_increment : float
        Incremental angle for each subsequent antenna (in degrees).
    start_angle : float
        Starting angle for the first antenna (in degrees).

    Returns
    -------
    antenna_positions : ndarray, shape (nant, 3)
        Positions of the antennas in 3D space (x, y, z).
    """
    # Convert angles to radians
    start_angle_rad = start_angle.to(u.rad).value
    angle_increment_rad = angle_increment.to(u.rad).value
    
    # Generate positions in polar coordinates
    lengths = start_length.to(u.m).value + np.arange(nant) * length_increment.to(u.m).value
    angles = start_angle_rad + np.arange(nant) * angle_increment_rad
    
    # Convert to Cartesian coordinates
    x = lengths * np.cos(angles)
    y = lengths * np.sin(angles)
    
    # Stack into a single array and add z=0 for 2D plane
    antenna_positions = np.vstack((x, y, np.zeros(nant))).T
    
    return antenna_positions

def inner_outer_array(nin,nout,out_scale=3000*u.m, in_scale=800*u.m):
    
    antenna_inner = gaussian_cloud(nant=nin, dispersion=in_scale)
    antenna_outer = spiral_array(nant=nout, start_length=out_scale, length_increment=100*u.m, angle_increment=194.4*u.deg, start_angle=0*u.deg)
    
    antenna_positions = np.append(antenna_inner, antenna_outer, axis=0)
    
    return antenna_positions


def emerlin():
    """
    Example of the approximate e-MERLIN array configuration.

    """
    # e-MERLIN antenna positions in enu relative to the reference point 
    mk2 =[-641.64, 13.87, 0.01]
    defford =[10392.21, -126048.22, -1884.23]
    knockin =[-47309.44, -49096.41, -952.08]
    darnhall =[-16049.21, -8562.54, -632.34]
    tabley =[-9980.65, 6160.41, -628.45]
    cambridge =[159597.02, -116045.92, -3683.10]

    antenna_positions = np.array([mk2,defford, knockin, darnhall, tabley, cambridge])
    
    return antenna_positions