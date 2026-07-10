"""Atmospheric turbulence: von Karman / Kolmogorov phase screens and frozen flow.

The ground half of the wavefront program. A random phase screen is drawn by
Fourier filtering of white noise (the McGlamery method): white complex noise is
coloured by the square root of the von Karman phase power spectral density

    Phi_phi(f) = 0.023 r0^(-5/3) (f^2 + (1/L0)^2)^(-11/6)

and inverse-transformed. The normalization is exact -- the screen variance
equals the Riemann sum of the PSD over the frequency grid -- so the phase
structure function follows the Kolmogorov r^(5/3) law with the r0^(-5/3)
strength. Frozen flow slides one large screen across the aperture, the standard
Taylor-hypothesis model of a boiling wavefront. Screens are delivered as OPD
maps in nanometres, the same unit contract a ``physicaloptix.PhaseScreen`` and
the mode bases carry, so turbulence rides the same propagation and control path
as the space case.
"""

import jax
import jax.numpy as jnp


def _phase_screen(key, npix, dx_m, r0_m, l0_m):
    """A zero-mean Kolmogorov/von Karman phase screen (radians) on an npix grid."""
    f = jnp.fft.fftfreq(npix, d=dx_m)  # spatial frequency, cycles per metre
    fx, fy = jnp.meshgrid(f, f)
    f_squared = fx**2 + fy**2
    outer = 0.0 if l0_m is None else (1.0 / l0_m) ** 2
    psd = 0.023 * r0_m ** (-5.0 / 3.0) * (f_squared + outer) ** (-11.0 / 6.0)
    psd = psd.at[0, 0].set(0.0)  # remove piston (the DC/infinite-scale mode)
    cell = 1.0 / (npix * dx_m)  # frequency-grid spacing
    key_re, key_im = jax.random.split(key)
    noise = jax.random.normal(key_re, (npix, npix)) + 1j * jax.random.normal(
        key_im, (npix, npix)
    )
    # var of the screen below = sum(psd) * cell^2, the PSD integral (exact).
    spectrum = noise * jnp.sqrt(psd) * cell
    phase = jnp.real(jnp.fft.ifft2(spectrum)) * npix**2
    return phase - jnp.mean(phase)


def von_karman_screen(key, npix, dx_m, r0_m, wavelength_nm, l0_m=None):
    """Draw a von Karman atmospheric OPD screen.

    Args:
        key: A JAX PRNG key.
        npix: Screen side length in pixels.
        dx_m: Physical pixel pitch (metres).
        r0_m: Fried parameter (metres) at ``wavelength_nm``.
        wavelength_nm: Wavelength the ``r0`` is measured at; sets the OPD scale.
        l0_m: Outer scale (metres); ``None`` gives the pure Kolmogorov spectrum.

    Returns:
        An ``(npix, npix)`` OPD screen in nanometres (zero-mean, real).
    """
    phase = _phase_screen(key, npix, dx_m, r0_m, l0_m)
    return phase * (wavelength_nm / (2.0 * jnp.pi))


def frozen_flow_sequence(
    key, npix, dx_m, r0_m, wavelength_nm, *, n_frames, shift_px, l0_m=None
):
    """A frozen-flow OPD sequence: one large screen slid across the aperture.

    Under the Taylor hypothesis the turbulence is a frozen pattern blown past by
    the wind, so each frame is the previous one translated by ``shift_px``.

    Args:
        key: A JAX PRNG key.
        npix: Aperture window side length in pixels.
        dx_m: Physical pixel pitch (metres).
        r0_m: Fried parameter (metres) at ``wavelength_nm``.
        wavelength_nm: Wavelength the ``r0`` is measured at.
        n_frames: Number of frames.
        shift_px: Wind translation per frame, in pixels.
        l0_m: Outer scale (metres); ``None`` is pure Kolmogorov.

    Returns:
        An ``(n_frames, npix, npix)`` OPD sequence in nanometres.
    """
    span = npix + (n_frames - 1) * shift_px
    screen = _phase_screen(key, span, dx_m, r0_m, l0_m) * (
        wavelength_nm / (2.0 * jnp.pi)
    )
    return jnp.stack(
        [screen[:npix, k * shift_px : k * shift_px + npix] for k in range(n_frames)]
    )
