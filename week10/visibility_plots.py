import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def amplitude_phase_plot(V_full, V_masked, _u, _v, img_name=None):
    # Show the u-v plane coverage and the masked visibility plane in phase and intensity
    plt.figure(figsize=(12, 12))

    plt.subplot(2,2,1)
    plt.title("log(Visibility amplitude)")
    plt.imshow(np.log(np.abs(V_masked)), cmap='gray',extent=[_u[0], _u[-1], _v[0], _v[-1]])
    plt.xlabel("$u$ (wavelengths)")
    plt.ylabel("$v$ (wavelengths)")
    colorbar = plt.colorbar()
    colorbar.set_label(r"$\ln(|V|/\mathrm{Jy})$")

    plt.subplot(2,2,2)
    plt.title("Visibility phase")
    plt.imshow(np.angle(V_masked), cmap='hsv',extent=[_u[0], _u[-1], _v[0], _v[-1]])
    plt.xlabel("$u$ (wavelengths)")
    plt.ylabel("$v$ (wavelengths)")
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.title("log(Visibility amplitude) - Full")
    plt.imshow(np.log(np.abs(V_full)), cmap='gray',extent=[_u[0], _u[-1], _v[0], _v[-1]])
    plt.xlabel("$u$ (wavelengths)")
    plt.ylabel("$v$ (wavelengths)")
    colorbar = plt.colorbar()
    colorbar.set_label(r"$\ln(|V|/\mathrm{Jy})$")

    plt.subplot(2,2,4)
    plt.title("Visibility phase - Full")
    plt.imshow(np.angle(V_full), cmap='hsv',extent=[_u[0], _u[-1], _v[0], _v[-1]])
    plt.xlabel("$u$ (wavelengths)")
    plt.ylabel("$v$ (wavelengths)")
    plt.colorbar()
    if img_name is not None:
        plt.savefig(img_name, dpi=300)
    plt.show()


def dirty_beam_plot(dirty_image, psf, uv_tracks, image_halfwidth, original_image,img_name=None):
    # Plot u-v tracks and the corresponding dirty beam (point spread function)
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.title("u-v tracks")
    mid_time = len(uv_tracks[0]) // 2
    for track in uv_tracks:
        plt.plot(track[:, 0], track[:, 1], "-", lw=1.5, alpha=0.65)
        plt.plot(-track[:, 0], -track[:, 1], "-", lw=1.5, alpha=0.65)
        plt.plot(track[mid_time, 0], track[mid_time, 1], "o", color="k", ms=3)
        plt.plot(-track[mid_time, 0], -track[mid_time, 1], "o", color="k", ms=3)
    plt.xlabel("$u$ (wavelengths)")
    plt.ylabel("$v$ (wavelengths)")
    plt.axis("equal")
    plt.subplot(2, 2, 2)
    plt.title("Dirty beam (point spread function)")
    beam_halfwidth = 2 * image_halfwidth
    plt.imshow(psf, cmap='gray', extent=[-beam_halfwidth, beam_halfwidth, -beam_halfwidth, beam_halfwidth],vmax=0.1,interpolation='nearest')
    plt.xlim(-image_halfwidth, image_halfwidth)
    plt.ylim(-image_halfwidth, image_halfwidth)
    plt.xlabel(r"$\theta_x$ [arcsec]")
    plt.ylabel(r"$\theta_y$ [arcsec]")
    plt.colorbar(label=r"$|B(\theta_x,\theta_y)|$")

    # below we plot the original and dirty images for comparison
    plt.subplot(2, 2, 3)
    plt.title("Original image")
    plt.imshow(original_image, cmap='gray', extent=[-image_halfwidth, image_halfwidth, -image_halfwidth, image_halfwidth])
    plt.xlabel(r"$\theta_x$ [arcsec]")
    plt.ylabel(r"$\theta_y$ [arcsec]")
    plt.colorbar(label=r"$B(\theta_x,\theta_y) \times P(\theta_x,\theta_y)$ [Jy/sr]")
    plt.subplot(2, 2, 4)
    plt.title("Dirty image")
    plt.imshow(dirty_image, cmap='gray', extent=[-image_halfwidth, image_halfwidth, -image_halfwidth, image_halfwidth])
    plt.xlabel(r"$\theta_x$ [arcsec]")
    plt.ylabel(r"$\theta_y$ [arcsec]")
    plt.colorbar(label=r"Dirty image [Jy/beam]")
    if img_name is not None:
        plt.savefig(img_name, dpi=300)
    plt.show()

def uv_and_dirty_image(V_full, V_masked, beam_peak, uv_tracks, antenna, _u, _v, image_halfwidth,img_name=None):
    # 2 rows x 3 columns:
    # Antenna | Full UV | Full-plane inverse FT
    # Tracks  | Sampled UV | Dirty image

    reconstructed_image = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(V_full))
    ).real
    dirty_image = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(V_masked))
    ).real / beam_peak  # Normalize dirty image by the peak of the dirty beam

    # Fourfold font scaling for lecture-slide readability, scoped to this figure.
    with plt.rc_context({"font.size": 20}):
        fig, axes = plt.subplots(
            2, 3, figsize=(24, 13.5), constrained_layout=True
        )
        for ax in axes.flat:
            ax.locator_params(axis="both", nbins=3)

        # Antenna layout
        ax = axes[0, 0]
        ax.set_title("Antenna")
        ax.plot(antenna[:, 0], antenna[:, 1], "o")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.axis("equal")

        # Full UV-plane visibility amplitude
        ax = axes[0, 1]
        image = ax.imshow(
            np.abs(V_full),
            cmap="viridis",
            origin="lower",
            extent=[_u[0], _u[-1], _v[0], _v[-1]],norm=LogNorm()
        )
        ax.set_title("Full UV")
        ax.set_xlabel("u [λ]")
        ax.set_ylabel("v [λ]")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)
        fig.colorbar(image, ax=ax)

        # Inverse Fourier transform of the full UV plane
        ax = axes[0, 2]
        image = ax.imshow(
            reconstructed_image,
            cmap="gray",
            extent=[-image_halfwidth, image_halfwidth, -image_halfwidth, image_halfwidth],
        )
        ax.set_title("Full-plane inverse FT")
        ax.set_xlabel(r"$\theta_x$ [arcsec]")
        ax.set_ylabel(r"$\theta_y$ [arcsec]")
        fig.colorbar(image, ax=ax)

        # UV tracks, including conjugate tracks for a real-valued sky image
        ax = axes[1, 0]
        ax.set_title("u-v Tracks")
        mid_time = len(uv_tracks[0]) // 2
        for track in uv_tracks:
            ax.plot(track[:, 0], track[:, 1], "-", lw=1.5, alpha=0.65)
            ax.plot(-track[:, 0], -track[:, 1], "-", lw=1.5, alpha=0.65)
            ax.plot(track[mid_time, 0], track[mid_time, 1], "o", color="k", ms=3,zorder=5)
            ax.plot(-track[mid_time, 0], -track[mid_time, 1], "o", color="k", ms=3,zorder=5)
        ax.set_xlabel("u [λ]")
        ax.set_ylabel("v [λ]")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)
        ax.axis("equal")

        # Sampled (masked) UV-plane visibility amplitude
        ax = axes[1, 1]
    
        image = ax.imshow(
            np.abs(V_masked),
            cmap="viridis",
            origin="lower",
            extent=[_u[0], _u[-1], _v[0], _v[-1]],
            norm=LogNorm()
        )
        ax.set_title("Sampled UV")
        ax.set_xlabel("u [λ]")
        ax.set_ylabel("v [λ]")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)
        fig.colorbar(image, ax=ax)

        # Dirty image from the sampled UV plane
        ax = axes[1, 2]
        image = ax.imshow(
            dirty_image,
            cmap="gray",
            extent=[-image_halfwidth, image_halfwidth, -image_halfwidth, image_halfwidth],
        )
        ax.set_title("Dirty image")
        ax.set_xlabel(r"$\theta_x$ [arcsec]")
        ax.set_ylabel(r"$\theta_y$ [arcsec]")
        fig.colorbar(image, ax=ax)
        if img_name is not None:
            plt.savefig(img_name, dpi=300)
        plt.show()