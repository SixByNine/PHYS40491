# PHYS40491 Interferometry demos

## What is this repo?
These are a series of demos produced for the Interferometry section of the PHYS40491 lecture course.
These python notebooks and scripts were produced by Dr Keith, but make extensive use of AI generated code.
The code is intended for teaching, but note that many approximations are used, and weird results may arise if extreme values are used in the simulations!

## Using the notebooks
Most standard tools that support ipython/Jupyter notebooks should work. They were developed in VScode, so that works well.

You will need to set up a python environment (e.g. venv) with the following packages
 * `numpy`
 * `scipy`
 * `matplotlib`
 * `astropy`

The animation of the spinning Earth requires the `globe` package.
Reading images for the Week 10 material requires the `pillow` package.

## Week 9
Week 9 focuses on the two element interferometer.

 * `two_element_interferometer.ipynb` Plots geometric delay, correlations and Visibility amplitude for point sources.
 * `two_element_interferometer_extended_source.ipynb` Computes visibility phase/amplitude for a choice of sources.
 * `visibility_amp_vs_baseline.ipynb` Computes a 1-d plot of coherence vs baseline length for Gaussian sources of different sizes
 * `baseline_aperture_synthesis.py` Generates the animated rotating earth with specific baselines drawn.

## Week 10
Week 10 focuses on u-v coverage, aperture synthesis and imaging.
 * `vanCittertZernike.ipynb` Simulates the dirty map for a given image and interferometer arrangement, including the u-v masking.
