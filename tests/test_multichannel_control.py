"""Tests for cross-channel (sense-one, correct-another) wavefront control."""

import jax.numpy as jnp
import numpy as np
import pytest
from optixstuff.grids import Frame, PixelGrid
from physicaloptix import (
    BeamSplitter,
    Branch,
    Field,
    Fraunhofer,
    ModeBasis,
    OpticalPath,
    OpticalSystem,
    PhaseScreen,
    PlaneKind,
    Stage,
    linearize_shared,
)

from tiptilt.multichannel import (
    FeedForwardController,
    MultiChannelModel,
    run_multichannel,
    shared_dm_command,
)

WL, NPIX = 500.0, 16


def _basis(npix, freqs, amp_nm=4.0):
    xg, yg = map(np.asarray, PixelGrid.pupil(npix).coordinates())
    modes = []
    for kx, ky in freqs:
        arg = 2 * np.pi * (kx * xg + ky * yg)
        modes.append(amp_nm * np.cos(arg))
        modes.append(amp_nm * np.sin(arg))
    stack = jnp.asarray(np.stack(modes))
    return ModeBasis(B=stack, coeffs=jnp.zeros(stack.shape[0]))


def _system(npix=NPIX, sci_ncpa_nm=0.0, wfs_ncpa_modes=None):
    """Trunk: drift screen + shared corrector; two focal arms with private NCPA."""
    grid = PixelGrid.pupil(npix)
    focal = PixelGrid.focal(32, 0.5, Frame(500.0, 1.0, "circumscribed"))
    xg, yg = map(np.asarray, grid.coordinates())
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
    fxg, fyg = map(np.asarray, focal.coordinates())
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


def _arm_model(*blocks):
    return MultiChannelModel(
        names=tuple(f"arm{i}" for i in range(len(blocks))),
        g_blocks=tuple(jnp.asarray(block) for block in blocks),
        e_refs=tuple(jnp.zeros(len(block), dtype=complex) for block in blocks),
        masks=tuple(jnp.ones((1, len(block)), dtype=bool) for block in blocks),
    )


class TestFeedForwardController:
    @pytest.mark.parametrize("weights", [[1, 2], [1, 1], [5, 0.1]])
    def test_consistent_arms_cancel_at_any_weight(self, weights):
        """Exact-algebra: both unit responses require -1 at every weight.

        Scaling only the response changes the solution for unequal weights.
        """
        model = _arm_model([[1 + 0j]], [[1 + 0j]])
        controller = FeedForwardController.build(
            model, sense=model.names, gain=1, regularization=0, weights=weights
        )
        _, delta = controller.command_delta(jnp.ones(2, dtype=complex))
        np.testing.assert_allclose(delta, [-1], atol=1e-12, rtol=0)

    @pytest.mark.parametrize(("weights", "expected"), [([1, 2], -3), ([0, 2], -4)])
    def test_inconsistent_arms_use_weighted_mean(self, weights, expected):
        """Exact-algebra: the unit-response fit is the weighted mean of 1 and 4.

        Scaling both response and estimate squares the weights.
        """
        model = _arm_model([[1 + 0j]], [[1 + 0j]])
        controller = FeedForwardController.build(
            model, sense=model.names, gain=1, regularization=0, weights=weights
        )
        _, delta = controller.command_delta(jnp.asarray([1 + 0j, 4 + 0j]))
        np.testing.assert_allclose(delta, [expected], atol=1e-12, rtol=0)

    def test_regularized_weighted_normal_equations(self):
        """NumPy float64 normal equations give a floating-point reference.

        Scaling the regularizer by the arm weight sum changes the solution.
        """
        rng = np.random.default_rng(82)
        sizes = (2, 3, 4)
        blocks = [rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2)) for n in sizes]
        weights = rng.uniform(0.2, 2, len(sizes))
        estimate = rng.normal(size=sum(sizes)) + 1j * rng.normal(size=sum(sizes))
        g = np.vstack(blocks)
        h = np.vstack((g.real, g.imag))
        w = np.diag(np.tile(np.repeat(weights, sizes), 2))
        z = np.concatenate((estimate.real, estimate.imag))
        gram = h.T @ w @ h
        reg = np.mean(np.diag(gram))
        gain = 0.7
        expected = -gain * np.linalg.solve(gram + reg * np.eye(2), h.T @ w @ z)

        model = _arm_model(*blocks)
        controller = FeedForwardController.build(
            model, sense=model.names, gain=gain, regularization=reg, weights=weights
        )
        _, delta = controller.command_delta(jnp.asarray(estimate))
        np.testing.assert_allclose(delta, expected, rtol=1e-10, atol=0)

    @pytest.mark.parametrize(
        ("weights", "message"),
        [
            ([-1, 2], "nonnegative"),
            ([np.nan, 1], "finite"),
            ([1, np.inf], "finite"),
            ([0, 0], "positive"),
            ([1, 2, 3], "weights must have shape"),
            ([[1], [2]], "weights must have shape"),
        ],
    )
    def test_rejects_invalid_arm_weights(self, weights, message):
        """Mutation: the statistical-weight domain requires each validation error.

        Omitting the corresponding check admits an invalid vector to the solve.
        """
        model = _arm_model([[1 + 0j]], [[1 + 0j]])
        with pytest.raises(ValueError, match=message):
            FeedForwardController.build(
                model, sense=model.names, gain=1, regularization=1, weights=weights
            )

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
