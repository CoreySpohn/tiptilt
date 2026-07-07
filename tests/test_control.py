"""Tests for the single-DM electric-field-conjugation dark-hole loop."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from physicaloptix import (
    Field,
    Fraunhofer,
    Grid,
    ModeBasis,
    OpticalPath,
    PhaseScreen,
    PlaneKind,
    Stage,
)

from wavefronts.control import close_dark_hole

WL = 500.0
_KS = [(3, 1), (3, 0), (3, 2), (2, 1), (4, 1), (3, -1)]


def _fourier_dm(npix, amp_nm=5.0):
    x = np.asarray(Grid.pupil(npix).coords)
    x_grid, y_grid = np.meshgrid(x, x)
    modes = []
    for kx, ky in _KS:
        arg = 2 * np.pi * (kx * x_grid + ky * y_grid)
        modes.append(amp_nm * np.cos(arg))
        modes.append(amp_nm * np.sin(arg))
    stack = jnp.asarray(np.stack(modes))
    return ModeBasis(B=stack, coeffs=jnp.zeros(stack.shape[0]))


def _setup(npix=16):
    pupil = Grid.pupil(npix)
    focal = Grid.focal(32, 0.5)
    x = np.asarray(pupil.coords)
    x_grid, y_grid = np.meshgrid(x, x)
    aperture = (x_grid**2 + y_grid**2 <= 0.25).astype(float)
    aberration = 3.0 * np.cos(2 * np.pi * (3 * x_grid + 1 * y_grid))  # nm, freq (3,1)
    e_in = aperture * np.exp(1j * 2 * np.pi * aberration / WL)
    field = Field(data=jnp.asarray(e_in), grid=pupil, plane=PlaneKind.PUPIL)
    path = OpticalPath(
        stages=(
            Stage("dm", PhaseScreen(_fourier_dm(npix), pupil, wavelength_nm=WL)),
            Stage("science", Fraunhofer(grid_in=pupil, grid_out=focal)),
        )
    )
    fx = np.asarray(focal.coords)
    fx_grid, fy_grid = np.meshgrid(fx, fx)
    # A small one-sided dark zone around the aberration's speckle at (3, 1) lambda/D.
    mask = (np.abs(fx_grid - 3.0) < 0.4) & (np.abs(fy_grid - 1.0) < 0.4)
    return path, field, jnp.asarray(mask)


class TestCloseDarkHole:
    def test_reduces_dark_zone_intensity(self):
        """Best case: the aberration lies in the DM span and the null is
        underdetermined (fewer masked pixels than modes), so the loop reaches a
        deep null. A real dark hole floors at the out-of-DM-span residual."""
        path, field, mask = _setup()
        _, history = close_dark_hole(
            path, field, 0, mask, n_steps=20, gain=0.5, regularization=1e-6
        )
        assert history[-1] < 1e-3 * history[0]

    def test_rejects_empty_dark_zone(self):
        path, field, _ = _setup()
        empty = jnp.zeros((32, 32), dtype=bool)
        with pytest.raises(ValueError, match="dark_zone"):
            close_dark_hole(
                path, field, 0, empty, n_steps=3, gain=0.5, regularization=1e-6
            )

    def test_rejects_non_deformable_mirror_stage(self):
        path, field, mask = _setup()
        with pytest.raises(TypeError, match="PhaseScreen"):
            # stage 1 is the Fraunhofer, not a deformable mirror.
            close_dark_hole(
                path, field, 1, mask, n_steps=3, gain=0.5, regularization=1e-6
            )

    def test_rejects_nonpositive_regularization(self):
        path, field, mask = _setup()
        with pytest.raises(ValueError, match="regularization"):
            close_dark_hole(
                path, field, 0, mask, n_steps=3, gain=0.5, regularization=0.0
            )

    def test_is_differentiable_in_gain(self):
        path, field, mask = _setup()

        def final_contrast(gain):
            _, history = close_dark_hole(
                path, field, 0, mask, n_steps=8, gain=gain, regularization=1e-6
            )
            return history[-1]

        grad = jax.grad(final_contrast)(0.5)
        assert jnp.isfinite(grad)
