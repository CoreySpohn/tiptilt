"""A real dark hole dug in an actual vortex coronagraph.

This is the integration that makes the dark hole real: a deformable mirror in
the entrance pupil, a multi-scale vortex coronagraph and a Lyot stop, and a
two-dimensional dark zone dug against a broadband-in-frequency aberration -- by
the oracle loop (perfect knowledge) and by the honest estimated loop (the field
recovered from probe images).
"""

import jax.numpy as jnp
import numpy as np
from physicaloptix import (
    Field,
    Fraunhofer,
    Grid,
    MultiScaleVortex,
    OpticalPath,
    PhaseScreen,
    PlaneKind,
    Stage,
    fourier_dm_basis,
)
from physicaloptix.elements import SampledOptic

from wavefronts import close_dark_hole, probe_set

WL = 500.0
NPUP = 48


def _disk(grid, radius):
    x = np.asarray(grid.coords)
    xx, yy = np.meshgrid(x, x)
    return (xx**2 + yy**2) <= radius**2


def _vortex_path():
    pupil = Grid.pupil(NPUP)
    focal = Grid.focal(96, 0.25)  # FOV 12 lambda/D
    coords = np.asarray(pupil.coords)
    xg, yg = np.meshgrid(coords, coords)
    aperture = _disk(pupil, 0.5).astype(complex)
    lyot = _disk(pupil, 0.45).astype(float)
    dm = fourier_dm_basis(pupil, n_actuators=12, k_min=3.0, k_max=6.0)
    path = OpticalPath(
        stages=(
            Stage("dm", PhaseScreen(dm, pupil, wavelength_nm=WL)),
            Stage(
                "vortex",
                MultiScaleVortex.build(
                    charge=2, npup=NPUP, q=64, scaling_factor=4, window_size=16
                ),
            ),
            Stage(
                "lyot",
                SampledOptic(
                    transmission=jnp.asarray(lyot), grid=pupil, plane=PlaneKind.PUPIL
                ),
            ),
            Stage("science", Fraunhofer(grid_in=pupil, grid_out=focal)),
        )
    )
    # Broadband-in-frequency phase aberration -> speckles across the dark zone.
    opd = (
        5.0 * np.cos(2 * np.pi * 4 * xg)
        + 4.0 * np.cos(2 * np.pi * 5 * yg + 0.6)
        + 3.0 * np.cos(2 * np.pi * 4.5 * (xg + yg) / np.sqrt(2) + 1.1)
    )
    input_field = Field(
        data=jnp.asarray(aperture * np.exp(1j * 2 * np.pi * opd / WL)),
        grid=pupil,
        plane=PlaneKind.PUPIL,
    )
    model_field = Field(data=jnp.asarray(aperture), grid=pupil, plane=PlaneKind.PUPIL)
    fx = np.asarray(focal.coords)
    fxg, fyg = np.meshgrid(fx, fx)
    r = np.hypot(fxg, fyg)
    # One-sided (single-DM-nullable) D-shaped dark zone, a real 2D region.
    mask = jnp.asarray((r >= 3.0) & (r <= 6.0) & (fxg > 1.0))
    return path, dm, input_field, model_field, mask


class TestCoronagraphDarkHole:
    def test_oracle_digs_a_real_dark_hole_in_the_vortex(self):
        path, _dm, input_field, _model_field, mask = _vortex_path()
        _, history = close_dark_hole(
            path, input_field, 0, mask, n_steps=15, gain=0.5, regularization=1e-8
        )
        # Perfect knowledge carves the region deep: orders of magnitude.
        assert float(history[-1]) < 1e-3 * float(history[0])

    def test_estimated_loop_digs_a_real_dark_hole_in_the_vortex(self):
        path, dm, input_field, model_field, mask = _vortex_path()
        probes = probe_set(dm, amplitude_nm=2.0, n_probes=4)
        _, history = close_dark_hole(
            path, input_field, 0, mask, n_steps=18, gain=0.4, regularization=1e-6,
            estimator="pairwise", model_field=model_field, probes=probes,
        )
        # The honest loop digs a real hole; depth floors on the model mismatch.
        assert float(history[-1]) < 0.1 * float(history[0])
