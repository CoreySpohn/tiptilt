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

from tiptilt.control import close_dark_hole
from tiptilt.dm import DeformableMirror, dm_influence_basis

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


def _pupil_setup(npix=24, n_actuators=8, **hardware):
    grid = Grid.pupil(npix)
    device = DeformableMirror.build(
        grid, n_actuators=n_actuators, wavelength_nm=WL, **hardware
    )
    x = np.asarray(grid.coords)
    xg, yg = np.meshgrid(x, x)
    aperture = (xg**2 + yg**2 <= 0.25).astype(complex)
    field = Field(data=jnp.asarray(aperture), grid=grid, plane=PlaneKind.PUPIL)
    return grid, device, field


class TestHardwareDM:
    def test_ideal_hardware_matches_phase_screen(self):
        from tiptilt.dm import HardwareDM

        grid, device, field = _pupil_setup()
        ideal = device.screen  # plain PhaseScreen when no knobs are set
        assert type(ideal) is PhaseScreen
        hardware = HardwareDM(ideal.basis, grid, wavelength_nm=WL)  # knobs all default
        command = jnp.linspace(-3.0, 3.0, device.n_active)
        set_ideal = eqx_set(ideal, command)
        set_hw = eqx_set(hardware, command)
        assert jnp.allclose(set_hw(field).data, set_ideal(field).data)

    def test_quantization_staircase(self):
        _grid, device, field = _pupil_setup(dac_step_nm=2.0)
        screen = device.screen
        n = device.n_active
        sub_lsb = eqx_set(screen, jnp.full(n, 0.9))  # below step/2 -> rounds to 0
        flat = eqx_set(screen, jnp.zeros(n))
        assert jnp.allclose(sub_lsb(field).data, flat(field).data)
        one_step = eqx_set(screen, jnp.full(n, 1.4))  # rounds to 2.0
        exact = eqx_set(screen, jnp.full(n, 2.0))
        # quantized 1.4 nm command == exactly-2 nm command, != flat
        ideal_two = PhaseScreen(
            eqx_coeffs(screen.basis, jnp.full(n, 2.0)), _grid, wavelength_nm=WL
        )
        assert jnp.allclose(one_step(field).data, ideal_two(field).data)
        assert not jnp.allclose(one_step(field).data, flat(field).data)
        del exact

    def test_jacobian_stays_ideal_through_quantizer(self):
        from tiptilt.control import DarkZoneModel

        grid, device, field = _pupil_setup(dac_step_nm=2.0)
        _g2, ideal_dev, _f2 = _pupil_setup()
        focal = Grid.focal(32, 0.5)
        fx = np.asarray(focal.coords)
        fxg, fyg = np.meshgrid(fx, fx)
        mask = jnp.asarray((np.abs(fxg - 3.0) < 0.8) & (np.abs(fyg) < 0.8))

        def model_for(screen):
            path = OpticalPath(
                stages=(
                    Stage("dm", screen),
                    Stage("science", Fraunhofer(grid_in=grid, grid_out=focal)),
                )
            )
            return DarkZoneModel.build(path, 0, mask, jacobian_field=field)

        g_hw = model_for(device.screen).g_dz
        g_ideal = model_for(ideal_dev.screen).g_dz
        assert jnp.allclose(g_hw, g_ideal, atol=1e-12)

    def test_dead_actuator_ignores_command_and_zeroes_jacobian(self):
        from tiptilt.control import DarkZoneModel

        n_probe = DeformableMirror.build(
            Grid.pupil(24), n_actuators=8, wavelength_nm=WL
        ).n_active
        gains = jnp.ones(n_probe).at[3].set(0.0)
        grid, device, field = _pupil_setup(actuator_gains=gains)
        poke_dead = eqx_set(device.screen, jnp.zeros(n_probe).at[3].set(50.0))
        flat = eqx_set(device.screen, jnp.zeros(n_probe))
        assert jnp.allclose(poke_dead(field).data, flat(field).data)

        focal = Grid.focal(32, 0.5)
        fx = np.asarray(focal.coords)
        fxg, fyg = np.meshgrid(fx, fx)
        mask = jnp.asarray((np.abs(fxg - 3.0) < 0.8) & (np.abs(fyg) < 0.8))
        path = OpticalPath(
            stages=(
                Stage("dm", device.screen),
                Stage("science", Fraunhofer(grid_in=grid, grid_out=focal)),
            )
        )
        model = DarkZoneModel.build(path, 0, mask, jacobian_field=field)
        col = model.g_dz[:, 3]
        others = jnp.delete(model.g_dz, 3, axis=1)
        assert float(jnp.max(jnp.abs(col))) < 1e-14 * float(jnp.max(jnp.abs(others)))

    def test_stuck_actuator_offset_applies_without_command(self):
        n = DeformableMirror.build(
            Grid.pupil(24), n_actuators=8, wavelength_nm=WL
        ).n_active
        offsets = jnp.zeros(n).at[5].set(30.0)
        grid, device, field = _pupil_setup(
            actuator_gains=jnp.ones(n).at[5].set(0.0), actuator_offsets_nm=offsets
        )
        flat_cmd = eqx_set(device.screen, jnp.zeros(n))
        ideal = PhaseScreen(
            eqx_coeffs(device.screen.basis, offsets), grid, wavelength_nm=WL
        )
        assert jnp.allclose(flat_cmd(field).data, ideal(field).data)
        # commanding the stuck actuator changes nothing
        poke_stuck = eqx_set(device.screen, jnp.zeros(n).at[5].set(80.0))
        assert jnp.allclose(poke_stuck(field).data, flat_cmd(field).data)

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="dac_step_nm"):
            _pupil_setup(dac_step_nm=-1.0)
        with pytest.raises(ValueError, match="actuator_gains"):
            _pupil_setup(actuator_gains=jnp.ones(3))


class TestHardwareScenarios:
    def test_quantization_floor_ordering(self):
        from tiptilt.testbed import hardware_dm, run

        continuous = run(hardware_dm(), "oracle-efc").metrics.final_contrast
        fine = run(hardware_dm(dac_step_nm=0.5), "oracle-efc").metrics.final_contrast
        coarse = run(hardware_dm(dac_step_nm=3.0), "oracle-efc").metrics.final_contrast
        assert continuous < fine < coarse

    def test_stuck_actuator_still_digs(self):
        from tiptilt.testbed import hardware_dm, run

        result = run(hardware_dm(stuck_nm=40.0), "oracle-efc")
        open_loop = run(hardware_dm(stuck_nm=40.0), "open-loop")
        assert result.metrics.final_contrast < 0.3 * result.metrics.initial_contrast
        assert result.metrics.final_contrast < 0.3 * open_loop.metrics.final_contrast


def eqx_set(screen, command):
    import equinox as eqx

    return eqx.tree_at(lambda s: s.basis.coeffs, screen, jnp.asarray(command))


def eqx_coeffs(basis, coeffs):
    from physicaloptix import ModeBasis

    return ModeBasis(B=basis.B, coeffs=jnp.asarray(coeffs))
