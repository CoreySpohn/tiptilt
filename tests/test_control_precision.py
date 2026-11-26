"""Precision requirements of the control, design and sensing interfaces."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.leaves import canonical
from hwoutils.precision import PrecisionError
from physicaloptix import ZernikeWavefrontSensor
from physicaloptix.multichannel import ChannelLinearization, MultiChannelLinearization

import tiptilt as tt
from tiptilt.sensing import probe_measurement


@pytest.fixture(
    params=[
        pytest.param(
            "enable_x64", marks=pytest.mark.float32_only(reason="float64 unavailable")
        ),
        pytest.param(
            "canonical",
            marks=pytest.mark.float64_only(reason="float32 owner under x64"),
        ),
    ]
)
def refuses(request):
    def check(operation, *args, through=None, **kwargs):
        name = through or operation.__qualname__
        with pytest.raises(
            PrecisionError,
            match=rf"{re.escape(name)} requires float64.*{request.param}",
        ):
            operation(*args, **kwargs)

    return check


@pytest.fixture(scope="module")
def scenario():
    setup = tt.wfs_plus_science(n_steps=1)
    setup.params["input_field"] = canonical(
        setup.params["input_field"], precision="float32"
    )
    return setup


@pytest.fixture(scope="module")
def model(scenario):
    path = scenario.params["system"].trunk
    dark_shape = scenario.params["input_field"].data.shape
    mask = jnp.zeros(dark_shape, dtype=bool).at[0, 0].set(True)
    unit = jnp.ones(1, dtype=jnp.float32)
    response = jnp.ones((1, 4), dtype=jnp.complex64)
    return tt.DarkZoneModel(
        path,
        (1,),
        (),
        mask,
        unit,
        unit,
        unit,
        unit,
        response,
        jnp.zeros_like(response[0].real),
    )


def test_control_design_requires_float64(refuses, model, scenario):
    """Fails if a design can linearize or factor a float32 plant.

    Args:
        refuses: Checks the precision error and the remedy for this lane.
        model: Single-pixel dark-zone model with a complex64 Jacobian.
        scenario: Forked optical system with a float32 entrance field.
    """
    field = scenario.params["input_field"]
    refuses(tt.DarkZoneModel.build, model.path, 1, model.mask, jacobian_field=field)
    refuses(model.weighted_dark_zone, field.data)
    refuses(model.weighted_contrast, model.g_dz[0])
    refuses(model.contrast, field.data, through="DarkZoneModel.weighted_dark_zone")
    regularized = dict(gain=0.5, regularization=0.1)
    refuses(tt.EFCController.build, model, **regularized)
    refuses(tt.PredictiveController.build, model, **regularized)
    refuses(tt.StrokeMinController.build, model, target_contrast=0.1)
    refuses(
        tt.MatrixFreeEFCController.build, model, jacobian_field=field, **regularized
    )


def test_controller_steps_require_float64(refuses, model, scenario):
    """Fails if direct application or a traced step bypasses design-time refusal.

    Args:
        refuses: Checks the precision error and the remedy for this lane.
        model: Single-pixel dark-zone model with a complex64 Jacobian.
        scenario: Forked optical system with a float32 entrance field.
    """
    field = scenario.params["input_field"]
    estimate = jnp.ones(1, dtype=complex)
    response = jnp.concatenate((model.g_dz.real, model.g_dz.imag))
    efc = tt.EFCController(response.T, model.stack_weights, 0.5)
    pred = tt.PredictiveController(efc, estimate, 0.5, jnp.asarray(False))
    stroke = tt.StrokeMinController(
        response, response.T @ response, model, jnp.ones(1), 0.1, 0.5
    )
    free = tt.MatrixFreeEFCController(
        model, field, model.operating_point, 0.5, 0.1, 4, 1e-6
    )
    feed = tt.FeedForwardController(response.T, 0.5)
    for controller in (efc, pred, stroke, free, feed):
        refuses(controller.command_delta, estimate)
    refuses(free.relinearize, model.operating_point)
    refuses(stroke.contrast, estimate)
    # Supply the controller as a JIT argument so its matrix is a tracer.
    refuses(
        jax.jit(lambda law, e: law.command_delta(e)[1]),
        efc,
        estimate,
        through="EFCController.command_delta",
    )


def test_multichannel_design_requires_float64(refuses, model, scenario):
    """Fails if a float32 channel response is used in a shared-arm solve.

    Args:
        refuses: Checks the precision error and the remedy for this lane.
        model: Single-pixel dark-zone model with a complex64 Jacobian.
        scenario: Forked optical system with a float32 entrance field.
    """
    plane = jnp.ones(model.mask.shape, dtype=jnp.complex64)
    field = scenario.params["input_field"]
    channel = ChannelLinearization("wfs", plane, plane[None], field.grid)
    mcl = MultiChannelLinearization((channel,), 500.0, "opd", 1.0)
    refuses(tt.MultiChannelModel.build, mcl, {"wfs": model.mask})
    field_vector = plane[model.mask]
    multi = tt.MultiChannelModel(
        ("wfs",), (model.g_dz,), (field_vector,), (model.mask,)
    )
    refuses(
        tt.FeedForwardController.build,
        multi,
        sense=("wfs",),
        gain=1.0,
        regularization=0.1,
    )
    refuses(tt.shared_dm_command, multi, "wfs", field_vector, regularization=0.1)


def test_sensing_requires_float64(refuses, model, scenario):
    """Fails if a probe design or field reconstruction accepts float32 data.

    Args:
        refuses: Checks the precision error and the remedy for this lane.
        model: Single-pixel dark-zone model with a complex64 Jacobian.
        scenario: Forked optical system with a float32 entrance field.
    """
    field = scenario.params["input_field"]
    command = model.operating_point
    probes = (command, command)
    basis = model.path.stages[1].op.basis
    refuses(tt.probe_set, canonical(basis, precision="float32"), amplitude_nm=1.0)
    refuses(probe_measurement, model.path, field, field, 1, command, model.mask)
    refuses(tt.estimate_field_pairwise, model.path, field, 1, probes, model.mask)
    probe_fields = jnp.ones((2, 1), dtype=jnp.complex64)
    refuses(tt.pairwise_estimate, probe_fields, probe_fields.real)
    sensor = ZernikeWavefrontSensor.build(npup=field.data.shape[0])
    refuses(tt.zwfs_calibrate, sensor, field, basis, wavelength_nm=500.0)
    refuses(
        tt.zwfs_reconstruct, probe_fields.real, probe_fields.real, probe_fields.real
    )
    refuses(tt.OracleEstimator(field).estimate, model, command)
    refuses(tt.PairwiseEstimator(field, field, probes, 1).estimate, model, command)


@pytest.mark.float64_only(reason="host inputs to wavefront reconstruction")
def test_host_storage_width_preserves_the_float64_solve():
    """Fails if float32 storage rounds away the small eigenvalue of the Gram matrix.

    The dyadic inputs are exactly representable in both storage widths, so
    converting before forming the normal equations gives bitwise equal results.
    """
    response = np.array([[1.0, 1.0], [1.0, 1.0 + 2.0**-12]])
    signal = response @ np.array([1.0, -1.0])
    probe = ((response[:, 0] + 1j * response[:, 1]) / 4.0)[:, None]
    reference = tt.pairwise_estimate(probe, signal[:, None])
    stored = tt.pairwise_estimate(
        probe.astype(np.complex64), signal.astype(np.float32)[:, None]
    )
    np.testing.assert_array_equal(stored, reference)
    image = signal.astype(np.float32)
    stored = tt.zwfs_reconstruct(
        image, np.zeros_like(image), response.astype(np.float32)
    )
    reference = tt.zwfs_reconstruct(signal, np.zeros_like(signal), response)
    np.testing.assert_array_equal(stored, reference)
    assert np.all(np.isfinite(reference))


def test_kalman_requires_float64(refuses, model, scenario):
    """Fails if the recursive estimator creates or updates a float32 covariance.

    Args:
        refuses: Checks the precision error and the remedy for this lane.
        model: Single-pixel dark-zone model with a complex64 Jacobian.
        scenario: Forked optical system with a float32 entrance field.
    """
    field = scenario.params["input_field"]
    field_vector = jnp.ones(1, dtype=jnp.complex64)
    variance = jnp.ones((), dtype=jnp.float32)
    refuses(
        tt.KalmanFieldEstimator.init,
        1,
        initial_variance=variance,
        process_noise=0.1,
        measurement_noise=0.1,
    )
    prior = tt.KalmanFieldEstimator(
        jnp.zeros((1, 2)), jnp.eye(2)[None] * variance, 0.1, 0.1
    )
    refuses(prior.update, field_vector, field_vector.real)
    command = model.operating_point
    probes = (command,)
    refuses(
        tt.KalmanEstimator.build,
        model,
        input_field=field,
        model_field=field,
        probes=probes,
        probe_dm=1,
    )
    estimator = tt.KalmanEstimator(prior, probes, 0, command, field, field, 1)
    refuses(estimator.estimate, model, command)


def test_loop_drivers_require_float64(refuses, model, scenario):
    """Fails if any public loop reaches its simulation with a float32 field.

    Args:
        refuses: Checks the precision error and the remedy for this lane.
        model: Single-pixel dark-zone model with a complex64 Jacobian.
        scenario: Forked optical system with a float32 entrance field.
    """
    p = scenario.params
    field = p["input_field"]
    loop = dict(n_steps=1, gain=0.5, regularization=0.1)
    args = (model.path, field, 1, model.mask)
    refuses(tt.close_dark_hole, *args, **loop)
    maintenance = dict(drift=lambda _: model.operating_point, drift_stage=0, dt_s=1.0)
    refuses(tt.maintain_dark_hole, *args, **loop, **maintenance)
    refuses(
        tt.maintained_residual_field,
        *args,
        **loop,
        **maintenance,
        through="maintain_dark_hole",
    )
    refuses(tt.run_multichannel, **p, wavelength_nm=500.0)
    refuses(tt.lowfs_calibrate, p["system"], field, arm="wfs", stage="shared_dm")
    refuses(
        tt.run_pointing_loop,
        p["system"],
        field,
        **loop,
        corrector_stage="shared_dm",
        drift_stage="drift",
        sense="wfs",
        science="sci",
        mask=p["masks"]["sci"],
        drift_table=p["drift_table"],
    )
    refuses(tt.run, scenario, "feedforward")
    refuses(tt.sweep, {"science": scenario}, ["feedforward"], through="run")
