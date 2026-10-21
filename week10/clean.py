import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import numpy as np


def gaussian_2d(coords, amplitude, x0, y0, sigma_major,
                sigma_minor, theta, offset):
    """Elliptical 2-D Gaussian, with theta in radians."""
    x, y = coords
    dx = x - x0
    dy = y - y0

    xp = dx * np.cos(theta) + dy * np.sin(theta)
    yp = -dx * np.sin(theta) + dy * np.cos(theta)

    return (
        offset
        + amplitude * np.exp(
            -0.5 * (
                (xp / sigma_major)**2
                + (yp / sigma_minor)**2
            )
        )
    )


def fit_clean_beam(dirty_beam, fit_radius=10):
    """
    Fit an elliptical Gaussian to the central lobe of a dirty beam.

    Parameters
    ----------
    dirty_beam : 2-D array
        Dirty beam / PSF, with its peak near the image centre.
    fit_radius : float
        Radius in pixels of the region used for the fit.
        Choose this to cover the central lobe, excluding sidelobes.

    Returns
    -------
    params : dict
        Gaussian parameters, including FWHM in pixels and radians.
        The position angle is measured from the positive x-axis.
    clean_beam : 2-D array
        Fitted Gaussian evaluated over the full image.
    """
    beam = np.asarray(dirty_beam, dtype=float)
    ny, nx = beam.shape

    # Locate the beam peak
    y0, x0 = np.unravel_index(np.nanargmax(beam), beam.shape)
    peak = beam[y0, x0]

    # Select the central fitting region
    y, x = np.indices(beam.shape)
    region = (
        (x - x0)**2 + (y - y0)**2 <= fit_radius**2
    ) & np.isfinite(beam)

    coords = (x[region], y[region])
    data = beam[region]

    # Initial guesses
    initial = [
        peak, x0, y0,
        fit_radius / 3, fit_radius / 3,
        0.0, 0.0
    ]

    # Constrain widths to be positive
    lower = [0, x0 - 1, y0 - 1, 0.1, 0.1, -np.pi, -np.inf]
    upper = [np.inf, x0 + 1, y0 + 1,
             2 * fit_radius, 2 * fit_radius, np.pi, np.inf]

    fitted, _ = curve_fit(
        gaussian_2d, coords, data,
        p0=initial, bounds=(lower, upper),
        maxfev=20000
    )

    amp, xc, yc, s1, s2, theta, offset = fitted

    # Sort widths so that sigma_major >= sigma_minor
    if s2 > s1:
        s1, s2 = s2, s1
        theta += np.pi / 2

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    fwhm_major = fwhm_factor * s1
    fwhm_minor = fwhm_factor * s2

    params = {
        "amplitude": amp,
        "x0": xc,
        "y0": yc,
        "sigma_major": s1,
        "sigma_minor": s2,
        "theta_rad": theta,
        "offset": offset,
        "fwhm_major_pixels": fwhm_major,
        "fwhm_minor_pixels": fwhm_minor,
        "position_angle_deg": np.degrees(theta),
    }

    clean_beam = gaussian_2d(
        (x, y), *fitted
    )

    return params, clean_beam

def apply_clean_step(current_residual_map, psf, current_model=None, current_clean_map=None, clean_beam_params=None, gain=0.1, plot=False, plot_vmin=None, plot_vmax=None):
    """
    Apply a single clean step to the current model and residual map.

    Parameters
    ----------
    current_residual_map : np.ndarray
        The current residual map.
    psf : np.ndarray
        The point spread function (PSF).
    clean_beam : np.ndarray
        The clean beam (Gaussian beam).
    current_model : np.ndarray
        The current model of the image, or None if no model exists yet.
    gain : float, optional
        The gain factor for the clean step. Default is 0.1.

    Returns
    -------
    updated_model : np.ndarray
        The updated model after applying the clean step.
    updated_residual_map : np.ndarray
        The updated residual map after applying the clean step.
    """
    # Find the peak in the residual map
    peak_value = np.max(current_residual_map)
    peak_position = np.unravel_index(np.argmax(current_residual_map), current_residual_map.shape)

    if clean_beam_params is None:
        clean_beam_params, _ = fit_clean_beam(psf)

    # Update the model at the peak position
    updated_model = current_model.copy() if current_model is not None else np.zeros_like(current_residual_map)
    updated_model[peak_position] += gain * peak_value
    clean_map = current_clean_map.copy() if current_clean_map is not None else np.zeros_like(current_residual_map)
    clean_map += gain * peak_value * gaussian_2d(
        (np.indices(current_residual_map.shape)), 
        clean_beam_params["amplitude"], 
        peak_position[0], 
        peak_position[1], 
        clean_beam_params["sigma_major"], 
        clean_beam_params["sigma_minor"], 
        clean_beam_params["theta_rad"], 
        0
    )


    # Subtract the scaled PSF from the residual map, shifted to the peak position
    # The PSF is centred at (n//2, m//2); place that centre on the peak, cropping
    # (not wrapping) any part that falls outside the map.
    psf_scaled = np.zeros_like(current_residual_map)
    offsets = [p - c for p, c in zip(peak_position, (psf.shape[0] // 2, psf.shape[1] // 2))]
    dst, src = [], []
    for off, n_map, n_psf in zip(offsets, current_residual_map.shape, psf.shape):
        lo, hi = max(off, 0), min(off + n_psf, n_map)
        dst.append(slice(lo, hi))
        src.append(slice(lo - off, hi - off))
    psf_scaled[tuple(dst)] = gain * peak_value * psf[tuple(src)]
    updated_residual_map = current_residual_map - psf_scaled

    if plot:
        
        plt.subplot(1, 3, 1)
        plt.imshow(current_residual_map, cmap='gray', vmin=plot_vmin, vmax=plot_vmax)
        # Add marker showing where the peak was found
        plt.plot(peak_position[1], peak_position[0], 'rx', markersize=3, label='Peak')
        plt.title('Input Residual Map')
        plt.subplot(1, 3, 2)
        plt.imshow(psf_scaled, cmap='gray', vmin=plot_vmin, vmax=plot_vmax)
        plt.title(' -  Scaled PSF')
        plt.subplot(1, 3, 3)
        plt.imshow(clean_map, cmap='gray', vmin=plot_vmin, vmax=plot_vmax)
        plt.title(' + Partial Clean Map')
        plt.show()

    return updated_residual_map, updated_model, clean_map