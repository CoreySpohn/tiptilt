"""Tests for the Zernike-wavefront-sensor low-order reconstruction."""

import jax.numpy as jnp
import numpy as np
from physicaloptix import Field, Grid, PlaneKind, ZernikeWavefrontSensor, zernike_basis
from physicaloptix.elements.basis import ModeBasis

from wavefronts.sensing import zwfs_calibrate, zwfs_reconstruct

WL = 500.0
NPUP = 48


def _low_order_basis(grid):
    """Noll 2-6 (tip, tilt, defocus, two astigmatisms) -- drop unobservable piston."""
    full = zernike_basis(grid, 6, rms_nm=1.0)
    return ModeBasis(B=full.B[1:6], coeffs=jnp.zeros(5))


def _aperture_field(grid):
    coords = np.asarray(grid.coords)
    xg, yg = np.meshgrid(coords, coords)
    disk = (xg**2 + yg**2 <= 0.25).astype(complex)
    return Field(data=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL)


class TestZwfsReconstruction:
    def test_recovers_injected_low_order_coefficients(self):
        grid = Grid.pupil(NPUP)
        sensor = ZernikeWavefrontSensor.build(npup=NPUP, q=4)
        aperture = _aperture_field(grid)
        modes = _low_order_basis(grid)
        reference, interaction = zwfs_calibrate(
            sensor, aperture, modes, wavelength_nm=WL
        )

        c_true = jnp.array([4.0, -3.0, 5.0, -2.0, 3.5])  # nm on each mode
        opd = jnp.tensordot(c_true, modes.B, axes=1)
        aberrated = Field(
            data=aperture.data * jnp.exp(1j * 2 * jnp.pi * opd / WL),
            grid=grid,
            plane=PlaneKind.PUPIL,
        )
        image = jnp.abs(sensor(aberrated).data) ** 2
        c_hat = zwfs_reconstruct(image, reference, interaction, regularization=1e-12)
        # Linear reconstruction of a nonlinear sensor: recovers to ~10% for
        # several-nanometre modes (good enough to close a low-order loop).
        np.testing.assert_allclose(np.asarray(c_hat), np.asarray(c_true), rtol=0.1)

    def test_flat_wavefront_reconstructs_near_zero(self):
        grid = Grid.pupil(NPUP)
        sensor = ZernikeWavefrontSensor.build(npup=NPUP, q=4)
        aperture = _aperture_field(grid)
        modes = _low_order_basis(grid)
        reference, interaction = zwfs_calibrate(
            sensor, aperture, modes, wavelength_nm=WL
        )
        c_hat = zwfs_reconstruct(
            reference, reference, interaction, regularization=1e-12
        )
        assert float(jnp.max(jnp.abs(c_hat))) < 1e-6
