"""Tests for the estimator/controller seams (DarkZoneModel and the laws)."""

import jax.numpy as jnp
import numpy as np
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

from wavefronts.control import (
    DarkZoneModel,
    EFCController,
    PredictiveController,
    StrokeMinController,
)
from wavefronts.sensing import KalmanEstimator, OracleEstimator

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
    aberration = 3.0 * np.cos(2 * np.pi * (3 * x_grid + 1 * y_grid))
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
    mask = (np.abs(fx_grid - 3.0) < 0.4) & (np.abs(fy_grid - 1.0) < 0.4)
    return path, field, jnp.asarray(mask)


def _run(model, field, sensor, controller, n_steps):
    """A minimal seam-driven loop, mirroring the close_dark_hole driver."""
    command = jnp.zeros(model.n_total)
    history = []
    for _ in range(n_steps):
        history.append(model.contrast(model.focal_of(command, field)))
        sensor, e_hat = sensor.estimate(model, command)
        controller, delta = controller.command_delta(e_hat)
        command = command + delta
    return command, jnp.stack(history), sensor, controller


class TestEFCController:
    def test_command_delta_matches_the_manual_formula(self):
        path, field, mask = _setup()
        model = DarkZoneModel.build(path, 0, mask, jacobian_field=field)
        controller = EFCController.build(model, gain=0.5, regularization=1e-6)
        estimate = model.dark_zone_unweighted(
            model.focal_of(jnp.zeros(model.n_total), field)
        )
        _, delta = controller.command_delta(estimate)
        response = jnp.concatenate([jnp.real(model.g_dz), jnp.imag(model.g_dz)], axis=0)
        gram = response.T @ response + 1e-6 * jnp.eye(model.n_total)
        expected = -0.5 * (
            jnp.linalg.solve(gram, response.T)
            @ jnp.concatenate([jnp.real(estimate), jnp.imag(estimate)])
        )
        np.testing.assert_allclose(np.asarray(delta), np.asarray(expected), atol=1e-15)

    def test_oracle_seam_loop_digs(self):
        path, field, mask = _setup()
        model = DarkZoneModel.build(path, 0, mask, jacobian_field=field)
        controller = EFCController.build(model, gain=0.5, regularization=1e-6)
        sensor = OracleEstimator(input_field=field)
        _, history, _, _ = _run(model, field, sensor, controller, 20)
        assert float(history[-1]) < 1e-3 * float(history[0])


class TestStrokeMinController:
    def test_meets_target_with_less_stroke_than_efc(self):
        path, field, mask = _setup()
        model = DarkZoneModel.build(path, 0, mask, jacobian_field=field)
        sensor = OracleEstimator(input_field=field)
        initial = float(model.contrast(model.focal_of(jnp.zeros(model.n_total), field)))
        target = 0.05 * initial
        efc = EFCController.build(model, gain=1.0, regularization=1e-9)
        stroke = StrokeMinController.build(model, target_contrast=target)
        cmd_efc, hist_efc, _, _ = _run(model, field, sensor, efc, 8)
        cmd_stroke, hist_stroke, _, _ = _run(model, field, sensor, stroke, 8)
        assert float(hist_stroke[-1]) < 1.5 * target  # reaches the target
        # The dual objective: less command for a bounded contrast.
        assert float(jnp.linalg.norm(cmd_stroke)) < float(jnp.linalg.norm(cmd_efc))
        assert float(hist_efc[-1]) < float(hist_stroke[-1])  # EFC digs deeper


class TestPredictiveController:
    def test_alpha_zero_matches_efc(self):
        path, field, mask = _setup()
        model = DarkZoneModel.build(path, 0, mask, jacobian_field=field)
        sensor = OracleEstimator(input_field=field)
        efc = EFCController.build(model, gain=0.5, regularization=1e-6)
        pred = PredictiveController.build(
            model, gain=0.5, regularization=1e-6, alpha=0.0
        )
        _, hist_efc, _, _ = _run(model, field, sensor, efc, 6)
        _, hist_pred, _, _ = _run(model, field, sensor, pred, 6)
        np.testing.assert_allclose(
            np.asarray(hist_pred), np.asarray(hist_efc), rtol=1e-12
        )

    def test_state_advances(self):
        path, field, mask = _setup()
        model = DarkZoneModel.build(path, 0, mask, jacobian_field=field)
        pred = PredictiveController.build(
            model, gain=0.5, regularization=1e-6, alpha=1.0
        )
        estimate = model.dark_zone_unweighted(
            model.focal_of(jnp.zeros(model.n_total), field)
        )
        assert not bool(pred.primed)
        pred2, _ = pred.command_delta(estimate)
        assert bool(pred2.primed)
        np.testing.assert_allclose(
            np.asarray(pred2.prev_estimate), np.asarray(estimate.reshape(-1))
        )


class TestKalmanEstimatorState:
    def test_state_advances_per_call(self):
        path, field, mask = _setup()
        model = DarkZoneModel.build(path, 0, mask, jacobian_field=field)
        probes = [
            0.5 * jnp.ones(model.n_total).at[0].set(1.0),
            0.5 * jnp.ones(model.n_total).at[1].set(1.0),
        ]
        sensor = KalmanEstimator.build(
            model,
            input_field=field,
            model_field=field,
            probes=probes,
            probe_dm=0,
        )
        command = jnp.zeros(model.n_total)
        sensor2, _ = sensor.estimate(model, command)
        assert int(sensor2.step_index) == 1
        np.testing.assert_allclose(
            np.asarray(sensor2.last_command), np.asarray(command)
        )
        sensor3, _ = sensor2.estimate(model, command + 0.01)
        assert int(sensor3.step_index) == 2


class TestDarkZoneModelOperatingPoint:
    def test_jacobian_rotates_with_the_base_point(self):
        """The Jacobian at a nonzero command differs from the cold one (the
        PhaseScreen is a true exponential), which is what maintenance needs."""
        path, field, mask = _setup()
        cold = DarkZoneModel.build(path, 0, mask, jacobian_field=field)
        command0 = 0.3 * jnp.ones(cold.n_total)
        warm = DarkZoneModel.build(
            path, 0, mask, jacobian_field=field, operating_point=command0
        )
        assert float(jnp.linalg.norm(warm.g_dz - cold.g_dz)) > 1e-6
        np.testing.assert_allclose(
            np.asarray(warm.operating_point), np.asarray(command0)
        )
