import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import astropy.units as u


def read_image_as_map(image_path, image_size=1.0*u.arcmin, peak_flux=1.0*u.Jy,invert=False, logscale=False,top_factor=1.0):
    """
    Read an image file and return it as a 2D numpy array (sky brightness map).
    
    Parameters
    ----------
    image_path : str
        Path to the image file.
    
    Returns
    -------
    sky_map : ndarray, shape (npix, npix)
        2D array representing the sky brightness map.
    """
    
    # Open the image file
    img = Image.open(image_path)
    
    # Convert to grayscale and then to a numpy array
    img_gray = img.convert('L')

    # crop to a square image
    width, height = img_gray.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    img_gray = img_gray.crop((left, top, right, bottom))

    npix = min_dim

    pixel_scale = (image_size / min_dim).to(u.rad).value  # radians per pixel
    

    if invert:
        img_gray = ImageOps.invert(img_gray)
    sky_map = np.array(img_gray).astype(float)

    if logscale:
        # Image was in log scale, convert back to linear scale
        sky_map = 10**(sky_map / 255.0)
        sky_map -= np.amin(sky_map)  # Shift to make minimum zero

    max_value = np.amax(sky_map)
    max_mask = sky_map > 0.99*max_value

    # Make sky map in Jy/steradian
    sky_map = (sky_map / np.amax(sky_map)) * peak_flux.to(u.Jy).value / (pixel_scale**2)  # Jy/steradian

    sky_map[max_mask] *= top_factor
    
    return sky_map, pixel_scale


def point_source_map(image_size, npix, source_positions=[(0, 0)], source_flux=1.0*u.Jy):
    """
    Create a 2D map with a point source.
    
    Parameters
    ----------
    image_size : float
        Angular size of the image
    npix : int
        Number of pixels along each image dimension.
    source_position : tuple of float
        Position of the point source in pixel coordinates (x, y).
    source_flux : float
        Flux of the point source (in Jy).
    
    Returns
    -------
    sky_map : ndarray, shape (npix, npix)
        2D array representing the sky brightness map with a point source.
    """

    pixel_scale = (image_size / npix).to(u.rad).value  # radians per pixel

    sky_map = np.zeros((npix, npix))
    
    # Convert source position from pixel coordinates to array indices
    for source_position in source_positions:
        x_idx = int(npix/2 + source_position[0])
        y_idx = int(npix/2 + source_position[1])
    
        # Ensure indices are within bounds
        if 0 <= x_idx < npix and 0 <= y_idx < npix:
            sky_map[y_idx, x_idx] = source_flux.to(u.Jy).value / (pixel_scale**2)  # Jy/steradian
        
    return sky_map, pixel_scale

def gaussian_beam(pixel_scale, npix,fwhm=30*u.arcsec):
    """
    Create a 2D Gaussian beam.
    
    Parameters
    ----------
    fwhm : float
        Full width at half maximum of the Gaussian beam (in arcseconds).
    
    Returns
    -------
    beam : ndarray, shape (npix, npix)
        2D array representing the Gaussian beam.
    """
    x = np.linspace(-npix/2, npix/2, npix)
    y = np.linspace(-npix/2, npix/2, npix)
    X, Y = np.meshgrid(x, y)
    # if pixel_scale is quantity, convert to value in radians
    if isinstance(pixel_scale, u.Quantity):
        pixel_scale = pixel_scale.to(u.rad).value
    
    sigma = (fwhm.to(u.rad).value / pixel_scale) / (2 * np.sqrt(2 * np.log(2)))  # convert FWHM to sigma
    print(sigma)
    beam = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    return beam




def make_uv_coverage(
    antenna_positions,
    pixel_scale,
    image_width,
    declination,
    latitude,
    hour_angles,
    wavelength,
    plot=False
):
    """
    Calculate the Fourier u-v grid and Earth-rotation u-v tracks.

    Parameters
    ----------
    antenna_positions : array_like, shape (N_ant, 2) or (N_ant, 3)
        Antenna positions in metres in a local ENU coordinate system:
        [East, North] or [East, North, Up].

    pixel_scale : float
        Image pixel scale in radians/pixel.

    image_width : int
        Number of pixels along each image dimension.

    declination : float
        Source declination in degrees.

    latitude : float
        Latitude of the array in degrees.

    hour_angles : array_like
        Hour angles at which to sample the observation, hours after the meridian.
        Positive H corresponds to hour angle west of the meridian.

    wavelength : float
        Observing wavelength in metres.

    plot : bool, optional
        If True, plot the antenna positions and u-v tracks.

    Returns
    -------
    u_grid, v_grid : 2D ndarray
        Fourier-plane coordinates in wavelengths.

    uv_tracks : ndarray, shape (N_baselines, N_times, 2)
        u,v coordinates for each baseline and time, in wavelengths.

    baselines : ndarray, shape (N_baselines, 3)
        Physical baseline vectors [E, N, U] in metres.

    """

    antenna_positions = np.asarray(antenna_positions, dtype=float)

    if antenna_positions.ndim != 2:
        raise ValueError("antenna_positions must be a 2D array")

    if antenna_positions.shape[1] == 2:
        # Add zero vertical coordinate
        antenna_positions = np.column_stack(
            [antenna_positions, np.zeros(len(antenna_positions))]
        )

    if antenna_positions.shape[1] != 3:
        raise ValueError(
            "antenna_positions must have shape (N,2) or (N,3)"
        )
    
    tspan=hour_angles[-1] - hour_angles[0]

    hour_angles = (np.asarray(hour_angles, dtype=float)*u.hourangle).to(u.radian).value

    # ------------------------------------------------------------
    # Fourier-plane grid
    # ------------------------------------------------------------

    # np.fft.fftfreq gives cycles per radian when the sample spacing
    # is in radians. These are therefore spatial frequencies in
    # wavelengths (u,v).
    freq = np.fft.fftfreq(
        image_width,
        d=pixel_scale
    )

    u_grid, v_grid = np.meshgrid(
        np.fft.fftshift(freq),
        np.fft.fftshift(freq)
    )

    # ------------------------------------------------------------
    # Form all unique antenna pairs
    # ------------------------------------------------------------

    baselines = []

    n_ant = len(antenna_positions)

    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            # Baseline from antenna i -> antenna j
            baseline = antenna_positions[j] - antenna_positions[i]
            baselines.append(baseline)

    baselines = np.asarray(baselines)

    # ------------------------------------------------------------
    # Project baselines into u,v,w
    # ------------------------------------------------------------

    E = baselines[:, 0][:, None]
    N = baselines[:, 1][:, None]
    U = baselines[:, 2][:, None]

    H = hour_angles[None, :]

    sinH = np.sin(H)
    cosH = np.cos(H)

    sin_d = np.sin(np.deg2rad(declination))
    cos_d = np.cos(np.deg2rad(declination))

    sin_phi = np.sin(np.deg2rad(latitude))
    cos_phi = np.cos(np.deg2rad(latitude))

    # Baseline components in wavelengths
    _u = (
        E * cosH
        - N * sin_phi * sinH
        + U * cos_phi * sinH
    ) / wavelength

    _v = (
        E * sin_d * sinH
        + N * (sin_phi * sin_d * cosH + cos_phi * cos_d)
        + U * (-cos_phi * sin_d * cosH + sin_phi * cos_d)
    ) / wavelength

    _w = (
        -E * cos_d * sinH
        + N * (sin_d * cos_phi - cos_d * sin_phi * cosH)
        + U * (sin_d * sin_phi + cos_d * cos_phi * cosH)
    ) / wavelength

    uv_tracks = np.stack([_u, _v], axis=-1)

    if plot:
        plt.figure(figsize=(14, 7))
        plt.subplot(1,2,1)
        plt.title("Antenna positions")
        plt.plot(antenna_positions[:, 0], antenna_positions[:, 1], "o")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis("equal")
        plt.subplot(1,2,2)
        plt.title(f"u-v tracks ({tspan:.1f} hr track)")
        mid_time=int(len(uv_tracks[0])/2)
        for track in uv_tracks:
            plt.plot(track[:, 0], track[:, 1], "-",lw=3,alpha=0.8)

            # Conjugate track
            plt.plot(-track[:, 0], -track[:, 1], "-",lw=3,alpha=0.8)
            plt.plot(track[mid_time, 0], track[mid_time, 1], "o", color="k",zorder=4)
            plt.plot(-track[mid_time, 0], -track[mid_time, 1], "o", color="k",zorder=4)

        plt.xlabel("$u$ (wavelengths)")
        plt.ylabel("$v$ (wavelengths)")
        plt.axis("equal")
        plt.show()


    return u_grid, v_grid, uv_tracks, baselines, _w



def make_uv_data(image, u_tracks, v_tracks, pixel_scale, weighting="Uniform",
                 taper_fwhm=None):
    """
    Fourier transform an image and sample it at specified u-v tracks.

    The supplied u-v tracks represent the measured visibilities. Because
    the sky brightness is real-valued, the conjugate visibility

        V(-u, -v) = V*(u, v)

    is also included automatically in the sampled Fourier plane.

    Parameters
    ----------
    image : ndarray, shape (N, N)
        Sky brightness distribution in Jy/sr.
        The image is assumed to be centred on the middle pixel.

    u_tracks, v_tracks : ndarray
        U-v coordinates of the measured points, in wavelengths.
        These should have the same shape, e.g. (n_baselines, n_times).

    pixel_scale : float
        Image pixel scale in radians/pixel.

    weighting : {"Natural", "Uniform"}, default "Uniform"
        Visibility weighting scheme (case-insensitive).

        "Natural": every measured sample has equal weight (equal noise
        variance), so a uv cell is weighted by the number of samples in it.
        This gives the best point-source sensitivity but a broader beam with
        higher sidelobes, because densely sampled short baselines dominate.

        "Uniform": each occupied uv cell has the same total weight (each
        sample is weighted by 1/number of samples in its cell). This gives a
        narrower beam and lower sidelobes at the cost of sensitivity.

    taper_fwhm : float or None, default None
        If given, multiply the weights by a circular Gaussian taper in the
        u-v plane with this FWHM in wavelengths (weight 0.5 at
        |uv| = taper_fwhm / 2). This down-weights long baselines, giving a
        broader, smoother beam and better sensitivity to extended emission.
        The taper can be combined with either weighting scheme.

    Returns
    -------
    _u : ndarray
        1-D u coordinate grid, in wavelengths.

    _v : ndarray
        1-D v coordinate grid, in wavelengths.

    V_full : ndarray
        Full Fourier transform of the image.

    V_masked : ndarray
        Fourier plane containing the weighted measured visibilities and their
        conjugate points. Unsampled pixels are zero. Weights are normalised
        to sum to 1, so the inverse FFT of V_masked is the dirty image.

    mask : ndarray of bool
        Sampling mask. True for both measured and conjugate points.

    dirty_beam : ndarray
        Dirty beam (point spread function) for the chosen weighting and
        taper, with shape (2*N, 2*N). Its pixel scale and peak normalization
        match the dirty image, but it covers twice the angular extent.
    """

    N = image.shape[0]

    if image.shape[1] != N:
        raise ValueError("image must be square")

    if u_tracks.shape != v_tracks.shape:
        raise ValueError("u_tracks and v_tracks must have the same shape")

    # ------------------------------------------------------------------
    # Fourier transform of the image
    # ------------------------------------------------------------------

    # Pixel solid angle. This converts Jy/sr -> Jy when integrating
    # the brightness distribution.
    dOmega = pixel_scale**2

    V_full = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(image)
        )
    ) * dOmega

    # ------------------------------------------------------------------
    # Fourier-plane coordinates
    # ------------------------------------------------------------------

    _u = np.fft.fftshift(
        np.fft.fftfreq(N, d=pixel_scale)
    )
    _v = np.fft.fftshift(
        np.fft.fftfreq(N, d=pixel_scale)
    )

    du = _u[1] - _u[0]
    dv = _v[1] - _v[0]

    # ------------------------------------------------------------------
    # Convert physical u-v coordinates to Fourier-grid indices
    # ------------------------------------------------------------------

    def uv_to_indices(u_points, v_points):

        iu = np.rint((u_points - _u[0]) / du).astype(int)
        iv = np.rint((v_points - _v[0]) / dv).astype(int)

        valid = (
            (iu >= 0) & (iu < N) &
            (iv >= 0) & (iv < N)
        )

        return iu, iv, valid

    # Measured points
    iu, iv, valid = uv_to_indices(u_tracks, v_tracks)

    # ------------------------------------------------------------------
    # Construct the sampling mask
    # ------------------------------------------------------------------

    mask = np.zeros((N, N), dtype=bool)

    # Add the actual measured u-v points
    mask[iv[valid], iu[valid]] = True

    # Number of samples in each uv cell (used for weighting)
    counts = np.zeros((N, N))
    np.add.at(counts, (iv[valid], iu[valid]), 1)

    # Add the conjugate points:
    #
    #       V(-u,-v) = V*(u,v)
    #
    # These are implied by the real-valued sky brightness.
    iu_conj, iv_conj, valid_conj = uv_to_indices(
        -u_tracks, -v_tracks
    )

    mask[iv_conj[valid_conj], iu_conj[valid_conj]] = True
    np.add.at(counts, (iv_conj[valid_conj], iu_conj[valid_conj]), 1)

    # ------------------------------------------------------------------
    # Visibility weights
    # ------------------------------------------------------------------

    scheme = str(weighting).lower()
    if scheme == "natural":
        weights = counts.copy()
    elif scheme == "uniform":
        weights = mask.astype(float)
    else:
        raise ValueError("weighting must be 'Natural' or 'Uniform'")

    if taper_fwhm is not None:
        if taper_fwhm <= 0:
            raise ValueError("taper_fwhm must be positive")
        uu, vv = np.meshgrid(_u, _v)
        sigma = taper_fwhm / (2 * np.sqrt(2 * np.log(2)))
        weights *= np.exp(-(uu**2 + vv**2) / (2 * sigma**2))

    weights[~mask] = 0.0

    total = weights.sum()
    if total == 0:
        raise ValueError("no u-v samples fall inside the Fourier grid")
    weights /= total

    # ------------------------------------------------------------------
    # Apply the mask to the full visibility plane
    # ------------------------------------------------------------------

    V_masked = np.zeros_like(V_full)

    V_masked = V_full* weights

    # ------------------------------------------------------------------
    # Compute the dirty beam
    # ------------------------------------------------------------------

    # Insert zeros between uv cells to halve the uv-grid spacing. The inverse
    # transform then samples the beam at the image pixel scale over twice the
    # field of view. Preserve the original FFT normalization for dirty-image
    # normalization below.
    beam_size = 2 * N
    beam_indices = 2 * (np.arange(N) - N // 2) + beam_size // 2
    beam_weights = np.zeros((beam_size, beam_size), dtype=weights.dtype)
    beam_weights[np.ix_(beam_indices, beam_indices)] = weights

    dirty_beam = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(beam_weights)
        )
    ).real * 4

    return _u, _v, V_full, V_masked, mask, dirty_beam

