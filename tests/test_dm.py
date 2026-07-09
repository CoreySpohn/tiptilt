"""Tests for the programmable actuator-grid deformable mirror."""

import jax.numpy as jnp
import numpy as np
import pytest
from physicaloptix import (
    Field,
    Fraunhofer,
    Grid,
    OpticalPath,
    PhaseScreen,
    PlaneKind,
    Stage,
)

from wavefronts.control import close_dark_hole
from wavefronts.dm import DeformableMirror, dm_influence_basis

WL = 500.0


def _nearest_pixel_value(mode, grid, center):
    coords = np.asarray(grid.coords)
    ix = int(np.argmin(np.abs(coords - center[0])))
    iy = int(np.argmin(np.abs(coords - center[1])))
    return float(np.asarray(mode)[iy, ix])


class TestInfluenceBasis:
    def test_shapes_peak_and_unit_contract(self):
        grid = Grid.pupil(32)
        basis, centers = dm_influence_basis(grid, n_actuators=8, coupling=0.15)
        n_kept = basis.B.shape[0]
        assert centers.shape == (n_kept, 2)
        assert basis.B.shape == (n_kept, 32, 32)
        assert basis.coeffs.shape == (n_kept,)
        # Peak-normalized: a unit coefficient pokes ~1 nm at the actuator.
        peak = float(jnp.max(basis.B[0]))
        assert 0.9 < peak <= 1.0 + 1e-12

    def test_coupling_parameterization_is_exact(self):
        """The influence function is coupling**((r/pitch)**2): check the
        sampled mode against the analytic form at the pixel nearest a
        neighboring actuator (the pixel sits ~half a pixel off the exact
        center, so compare at the PROBED position, not at r == pitch)."""
        grid = Grid.pupil(64)
        coupling = 0.12
        basis, centers = dm_influence_basis(grid, n_actuators=8, coupling=coupling)
        centers = np.asarray(centers)
        pitch = 1.0 / 8
        coords = np.asarray(grid.coords)
        k = 0
        neighbor = centers[k] + np.asarray([pitch, 0.0])
        ix = int(np.argmin(np.abs(coords - neighbor[0])))
        iy = int(np.argmin(np.abs(coords - neighbor[1])))
        probe = np.asarray([coords[ix], coords[iy]])
        r2 = float(np.sum((probe - centers[k]) ** 2))
        expected = coupling ** (r2 / pitch**2)
        value = float(np.asarray(basis.B[k])[iy, ix])
        assert value == pytest.approx(expected, rel=1e-9)
        # And the analytic value AT the neighbor is the coupling by design.
        assert coupling ** (pitch**2 / pitch**2) == pytest.approx(coupling)

    def test_actuators_beyond_the_margin_are_dropped(self):
        """With a half-pitch margin the 8x8 corners (r = 0.619) are dropped
        (keep radius 0.5625); a full-pitch margin keeps them (0.625)."""
        grid = Grid.pupil(32)
        n = 8
        basis, centers = dm_influence_basis(grid, n_actuators=n, margin_actuators=0.5)
        assert basis.B.shape[0] < n * n  # the square corners are gone
        radii = np.hypot(*np.asarray(centers).T)
        assert float(radii.max()) <= 0.5 + 0.5 / n + 1e-9
        full, _ = dm_influence_basis(grid, n_actuators=n, margin_actuators=1.0)
        assert full.B.shape[0] == n * n  # the generous margin keeps all

    def test_flat_poke_is_flat_in_the_interior(self):
        """Equal pokes on all actuators give a uniform interior surface."""
        grid = Grid.pupil(64)
        basis, _ = dm_influence_basis(grid, n_actuators=10, coupling=0.15)
        surface = np.asarray(
            jnp.tensordot(5.0 * jnp.ones(basis.B.shape[0]), basis.B, axes=1)
        )
        x = np.asarray(grid.coords)
        xg, yg = np.meshgrid(x, x)
        interior = np.hypot(xg, yg) < 0.3
        values = surface[interior]
        assert np.std(values) / np.mean(values) < 0.05


class TestDeformableMirror:
    def test_builds_a_phase_screen_element(self):
        grid = Grid.pupil(32)
        dm = DeformableMirror.build(grid, n_actuators=8, wavelength_nm=WL)
        assert isinstance(dm.screen, PhaseScreen)
        assert dm.n_active == dm.screen.basis.n_modes
        assert dm.centers.shape == (dm.n_active, 2)

    def test_clip_enforces_the_per_actuator_stroke_limit(self):
        grid = Grid.pupil(32)
        dm = DeformableMirror.build(
            grid, n_actuators=8, wavelength_nm=WL, stroke_limit_nm=3.0
        )
        wild = jnp.linspace(-10.0, 10.0, dm.n_active)
        clipped = dm.clip(wild)
        assert float(jnp.max(jnp.abs(clipped))) <= 3.0
        # No limit -> identity.
        free = DeformableMirror.build(grid, n_actuators=8, wavelength_nm=WL)
        np.testing.assert_array_equal(np.asarray(free.clip(wild)), np.asarray(wild))

    def test_surface_matches_the_screen_opd(self):
        grid = Grid.pupil(32)
        dm = DeformableMirror.build(grid, n_actuators=8, wavelength_nm=WL)
        command = 0.3 * jnp.arange(dm.n_active, dtype=float)
        np.testing.assert_allclose(
            np.asarray(dm.surface(command)),
            np.asarray(jnp.tensordot(command, dm.screen.basis.B, axes=1)),
            atol=1e-15,
        )

    @pytest.mark.slow
    def test_actuator_dm_digs_a_dark_hole_with_the_existing_driver(self):
        """The device drops into close_dark_hole unchanged: its screen is a
        PhaseScreen, its coefficients are per-actuator strokes."""
        npix = 32
        grid = Grid.pupil(npix)
        focal = Grid.focal(48, 0.4)
        dm = DeformableMirror.build(grid, n_actuators=12, wavelength_nm=WL)
        x = np.asarray(grid.coords)
        xg, yg = np.meshgrid(x, x)
        aperture = (xg**2 + yg**2 <= 0.25).astype(float)
        opd = 3.0 * np.cos(2 * np.pi * (3 * xg + yg))
        field = Field(
            data=jnp.asarray(aperture * np.exp(1j * 2 * np.pi * opd / WL)),
            grid=grid,
            plane=PlaneKind.PUPIL,
        )
        path = OpticalPath(
            stages=(
                Stage("dm", dm.screen),
                Stage("science", Fraunhofer(grid_in=grid, grid_out=focal)),
            )
        )
        fx = np.asarray(focal.coords)
        fxg, fyg = np.meshgrid(fx, fx)
        mask = jnp.asarray((np.abs(fxg - 3.0) < 0.8) & (np.abs(fyg - 1.0) < 0.8))
        command, history = close_dark_hole(
            path, field, 0, mask, n_steps=12, gain=0.6, regularization=1e-6
        )
        assert command.shape == (dm.n_active,)  # actuator-space command
        # An actuator DM has FITTING ERROR (a Gaussian-influence lattice
        # cannot exactly reproduce a Fourier cosine), so the floor sits at
        # the representation residual rather than machine depth.
        assert float(history[-1]) < 0.1 * float(history[0])
