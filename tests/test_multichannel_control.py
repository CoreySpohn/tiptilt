"""Tests for cross-channel (sense-one, correct-another) wavefront control."""

import jax.numpy as jnp
import numpy as np
import pytest
from physicaloptix import (
    BeamSplitter,
    Branch,
    Field,
    Fraunhofer,
    Grid,
    ModeBasis,
    OpticalPath,
    OpticalSystem,
    PhaseScreen,
    PlaneKind,
    Stage,
    linearize_shared,
)

from wavefronts.multichannel import (
    FeedForwardController,
    MultiChannelModel,
    run_multichannel,
    shared_dm_command,
)

WL, NPIX = 500.0, 16


def _basis(npix, freqs, amp_nm=4.0):
    x = np.asarray(Grid.pupil(npix).coords)
    xg, yg = np.meshgrid(x, x)
    modes = []
    for kx, ky in freqs:
        arg = 2 * np.pi * (kx * xg + ky * yg)
        modes.append(amp_nm * np.cos(arg))
        modes.append(amp_nm * np.sin(arg))
    stack = jnp.asarray(np.stack(modes))
    return ModeBasis(B=stack, coeffs=jnp.zeros(stack.shape[0]))


def _system(npix=NPIX, sci_ncpa_nm=0.0, wfs_ncpa_modes=None):
    """Trunk: drift screen + shared corrector; two focal arms with private NCPA."""
    grid = Grid.pupil(npix)
    focal = Grid.focal(32, 0.5)
    x = np.asarray(grid.coords)
    xg, yg = np.meshgrid(x, x)
    aperture = (xg**2 + yg**2 <= 0.25).astype(complex)
    shared = _basis(npix, [(3, 1), (3, 0)])  # 4 shared modes
    corrector = _basis(npix, [(3, 1), (3, 0)])
    private = _basis(npix, [(2, 1)], amp_nm=4.0)  # 2 private modes

    def screen(basis, coeffs=None):
        b = basis if coeffs is None else ModeBasis(B=basis.B, coeffs=coeffs)
        return PhaseScreen(b, grid, wavelength_nm=WL)

    trunk = OpticalPath(
        stages=(
            Stage("drift", screen(shared)),
            Stage("shared_dm", screen(corrector)),
        )
    )
    split = BeamSplitter.energy(0.5, grid=grid, plane=PlaneKind.PUPIL)
    sci_ncpa = jnp.asarray([sci_ncpa_nm / 4.0, 0.0])  # coeffs on private basis
    sci = OpticalPath(
        stages=(
            Stage("ncpa", screen(private, sci_ncpa)),
            Stage("science", Fraunhofer(grid_in=grid, grid_out=focal)),
        )
    )
    wfs_coeffs = jnp.zeros(2) if wfs_ncpa_modes is None else jnp.asarray(wfs_ncpa_modes)
    wfs = OpticalPath(
        stages=(
            Stage("wfs_ncpa", screen(private, wfs_coeffs)),
            Stage("sensorcam", Fraunhofer(grid_in=grid, grid_out=focal)),
        )
    )
    system = OpticalSystem(
        trunk=trunk,
        split=split,
        branches=(Branch("sci", "transmit", sci), Branch("wfs", "reflect", wfs)),
    )
    field = Field(data=jnp.asarray(aperture), grid=grid, plane=PlaneKind.PUPIL)
    fx = np.asarray(focal.coords)
    fxg, fyg = np.meshgrid(fx, fx)
    mask = jnp.asarray((np.abs(fxg - 3.0) < 0.6) & (np.abs(np.abs(fyg) - 0.5) < 0.7))
    masks = {"sci": mask, "wfs": mask}
    return system, field, masks


def _ramp(n_steps, n_modes, scale_nm=0.3):
    directions = np.asarray([1.0, -0.6, 0.8, -0.4])[:n_modes]
    return jnp.asarray(
        np.arange(1, n_steps + 1)[:, None] * scale_nm * directions[None, :]
    )


class TestSharedDmCommand:
    def test_recovers_the_drift_coefficients(self):
        """A clean sensing arm recovers the shared drift exactly (linear)."""
        system, field, masks = _system()
        mcl = linearize_shared(
            system, field, wavelength_nm=WL, shared_stage="shared_dm"
        )
        model = MultiChannelModel.build(mcl, masks)
        eps_true = jnp.asarray([0.4, -0.2, 0.3, 0.1])
        # Synthesize the sensed field linearly from the wfs shared block.
        e_hat = model.g_block("wfs") @ eps_true
        command, eps_hat = shared_dm_command(model, "wfs", e_hat, regularization=1e-12)
        np.testing.assert_allclose(np.asarray(eps_hat), np.asarray(eps_true), rtol=1e-8)
        np.testing.assert_allclose(
            np.asarray(command), -np.asarray(eps_true), rtol=1e-8
        )


class TestRunMultichannel:
    @pytest.mark.slow
    def test_feedforward_rejects_shared_drift_in_both_arms(self):
        """Sense on the WFS arm, correct the shared DM: BOTH arms hold, and
        the science arm floors at its private-NCPA differential."""
        system, field, masks = _system(sci_ncpa_nm=2.0)
        n_steps = 8
        drift = _ramp(n_steps, 4)
        result = run_multichannel(
            system,
            field,
            shared_dm_stage="shared_dm",
            drift_stage="drift",
            sense=("wfs",),
            science="sci",
            masks=masks,
            wavelength_nm=WL,
            drift_table=drift,
            n_steps=n_steps,
            gain=1.0,
            regularization=1e-10,
        )
        open_loop = run_multichannel(
            system,
            field,
            shared_dm_stage="shared_dm",
            drift_stage="drift",
            sense=("wfs",),
            science="sci",
            masks=masks,
            wavelength_nm=WL,
            drift_table=drift,
            n_steps=n_steps,
            gain=0.0,
            regularization=1e-10,
        )
        held = result["excess"]["sci"]
        lost = open_loop["excess"]["sci"]
        assert float(lost[-1]) > 10.0 * float(lost[0])
        assert float(held[-1]) < 0.1 * float(lost[-1])
        # The WFS arm is corrected by the SAME shared command (common mode).
        assert float(result["excess"]["wfs"][-1]) < 0.1 * float(
            open_loop["excess"]["wfs"][-1]
        )
        # The loop cannot dig below the static (NCPA + wings) floor: the
        # absolute contrast stays at the reference level, never below.
        assert float(result["history"]["sci"][-1]) >= 0.8 * float(
            result["history"]["sci"][0]
        )

    @pytest.mark.slow
    def test_wfs_private_drift_aliases_into_science(self):
        """Drift in the SENSING arm's private modes is misread as shared and
        INJECTED into the science arm -- the classic aliasing failure."""
        system, field, masks = _system()
        n_steps = 8
        no_shared_drift = jnp.zeros((n_steps, 4))
        wfs_private_drift = _ramp(n_steps, 2, scale_nm=0.5)
        kwargs = dict(
            shared_dm_stage="shared_dm",
            drift_stage="drift",
            sense=("wfs",),
            science="sci",
            masks=masks,
            wavelength_nm=WL,
            drift_table=no_shared_drift,
            n_steps=n_steps,
            regularization=1e-10,
            local_drift={"wfs": ("wfs_ncpa", wfs_private_drift)},
        )
        aliased = run_multichannel(system, field, gain=1.0, **kwargs)
        untouched = run_multichannel(system, field, gain=0.0, **kwargs)
        # With no shared drift the science arm should have stayed clean; the
        # feed-forward loop actively injects the sensor arm's private error.
        assert float(aliased["excess"]["sci"][-1]) > 5.0 * float(
            untouched["excess"]["sci"][-1] + 1e-16
        )

    @pytest.mark.slow
    def test_dual_science_common_mode_with_two_sense_arms(self):
        """Both arms sense; the equal-weight joint solve corrects the shared
        drift for both."""
        system, field, masks = _system()
        n_steps = 6
        drift = _ramp(n_steps, 4)
        result = run_multichannel(
            system,
            field,
            shared_dm_stage="shared_dm",
            drift_stage="drift",
            sense=("sci", "wfs"),
            science="sci",
            masks=masks,
            wavelength_nm=WL,
            drift_table=drift,
            n_steps=n_steps,
            gain=1.0,
            regularization=1e-10,
        )
        open_loop = run_multichannel(
            system,
            field,
            shared_dm_stage="shared_dm",
            drift_stage="drift",
            sense=("sci", "wfs"),
            science="sci",
            masks=masks,
            wavelength_nm=WL,
            drift_table=drift,
            n_steps=n_steps,
            gain=0.0,
            regularization=1e-10,
        )
        for name in ("sci", "wfs"):
            assert float(result["excess"][name][-1]) < 0.1 * float(
                open_loop["excess"][name][-1]
            )


class TestFeedForwardController:
    def test_command_delta_is_minus_gain_times_estimate(self):
        system, field, masks = _system()
        mcl = linearize_shared(
            system, field, wavelength_nm=WL, shared_stage="shared_dm"
        )
        model = MultiChannelModel.build(mcl, masks)
        controller = FeedForwardController.build(
            model, sense=("wfs",), gain=0.5, regularization=1e-12
        )
        eps_true = jnp.asarray([0.2, 0.1, -0.3, 0.05])
        e_hat = model.g_block("wfs") @ eps_true
        _, delta = controller.command_delta(e_hat)
        np.testing.assert_allclose(
            np.asarray(delta), -0.5 * np.asarray(eps_true), rtol=1e-6
        )
