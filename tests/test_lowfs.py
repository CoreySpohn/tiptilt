"""Tests for the pickoff low-order sensor and the pointing loop."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from tiptilt.lowfs import lowfs_calibrate, run_pointing_loop
from tiptilt.maintenance import make_detector
from tiptilt.sensing import zwfs_reconstruct
from tiptilt.testbed import pointing_jitter, run


@pytest.fixture(scope="module")
def scenario():
    return pointing_jitter(npix=16, n_steps=6)


@pytest.fixture(scope="module")
def calibration(scenario):
    p = scenario.params
    reference, response = lowfs_calibrate(
        p["system"], p["input_field"], arm=p["sense"], stage=p["corrector_stage"]
    )
    return p, reference, response


def _fsm_index(system):
    return [stage.name for stage in system.trunk.stages].index("fsm")


def test_calibration_predicts_poked_image(calibration):
    p, reference, response = calibration
    poke = jnp.asarray([2.0, -1.0, 1.5])
    idx = _fsm_index(p["system"])
    poked = eqx.tree_at(
        lambda s: s.trunk.stages[idx].op.basis.coeffs, p["system"], poke
    )
    outputs, _ = poked.propagate(p["input_field"])
    image = jnp.abs(outputs[p["sense"]].data) ** 2
    predicted = reference + (response @ poke).reshape(reference.shape)
    rel = jnp.linalg.norm(image - predicted) / jnp.linalg.norm(image - reference)
    assert rel < 0.05


def test_reconstruct_recovers_drift_including_focus(calibration):
    p, reference, response = calibration
    injected = jnp.asarray([3.0, -2.0, 2.0])  # tip, tilt, focus in nm
    idx = [stage.name for stage in p["system"].trunk.stages].index("jitter")
    drifted = eqx.tree_at(
        lambda s: s.trunk.stages[idx].op.basis.coeffs, p["system"], injected
    )
    outputs, _ = drifted.propagate(p["input_field"])
    image = jnp.abs(outputs[p["sense"]].data) ** 2
    recovered = zwfs_reconstruct(image, reference, response)
    assert jnp.allclose(recovered, injected, rtol=0.1, atol=0.15)


def test_focus_needs_the_defocus_bias():
    biased = pointing_jitter(npix=16, n_steps=4).params
    unbiased = pointing_jitter(npix=16, n_steps=4, sensor_defocus_nm=0.0).params
    _, r_biased = lowfs_calibrate(
        biased["system"], biased["input_field"], arm="lowfs", stage="fsm"
    )
    _, r_unbiased = lowfs_calibrate(
        unbiased["system"], unbiased["input_field"], arm="lowfs", stage="fsm"
    )
    focus_biased = jnp.linalg.norm(r_biased[:, 2])
    focus_unbiased = jnp.linalg.norm(r_unbiased[:, 2])
    # An in-focus image has no first-order response to an even mode.
    assert focus_unbiased < 1e-6 * focus_biased


def test_closed_loop_rejects_jitter(scenario):
    p = scenario.params

    def residual_rms(gain):
        result = run_pointing_loop(
            p["system"],
            p["input_field"],
            corrector_stage=p["corrector_stage"],
            drift_stage=p["drift_stage"],
            sense=p["sense"],
            science=p["science"],
            mask=p["mask"],
            drift_table=p["drift_table"],
            n_steps=p["n_steps"],
            gain=gain,
            regularization=p["regularization"],
        )
        return jnp.sqrt(jnp.mean(result["residual_nm"][-2:] ** 2)), result

    open_rms, open_result = residual_rms(0.0)
    closed_rms, closed_result = residual_rms(p["gain"])
    assert closed_rms < 0.2 * open_rms
    assert closed_result["excess"][-1] < 0.3 * open_result["excess"][-1]
    assert closed_result["residual_nm"].shape == (p["n_steps"], 3)


def test_pointing_scenario_through_testbed(scenario):
    closed = run(scenario, "oracle-efc")  # gain None -> scenario gain
    open_loop = run(scenario, "open-loop")
    assert closed.metrics.final_contrast < 0.3 * open_loop.metrics.final_contrast
    assert closed.excess["residual_nm"].shape == (scenario.params["n_steps"], 3)


def test_detector_noise_loop_still_suppresses(scenario):
    p = scenario.params
    detector = make_detector(
        flux=1e7, exposure_time=1.0, read_noise_e=1.0, quantum_efficiency=0.9
    )
    result = run_pointing_loop(
        p["system"],
        p["input_field"],
        corrector_stage=p["corrector_stage"],
        drift_stage=p["drift_stage"],
        sense=p["sense"],
        science=p["science"],
        mask=p["mask"],
        drift_table=p["drift_table"],
        n_steps=p["n_steps"],
        gain=p["gain"],
        regularization=p["regularization"],
        detector=detector,
        key=jr.key(7),
    )
    open_rms = jnp.sqrt(jnp.mean(p["drift_table"][-2:] ** 2))
    noisy_rms = jnp.sqrt(jnp.mean(result["residual_nm"][-2:] ** 2))
    assert noisy_rms < 0.5 * open_rms


def test_validation_errors(scenario):
    p = scenario.params
    with pytest.raises(KeyError, match="arm"):
        lowfs_calibrate(p["system"], p["input_field"], arm="nope", stage="fsm")
    with pytest.raises(KeyError, match="stage"):
        lowfs_calibrate(p["system"], p["input_field"], arm="lowfs", stage="nope")
    with pytest.raises(ValueError, match="key"):
        run_pointing_loop(
            p["system"],
            p["input_field"],
            corrector_stage=p["corrector_stage"],
            drift_stage=p["drift_stage"],
            sense=p["sense"],
            science=p["science"],
            mask=p["mask"],
            drift_table=p["drift_table"],
            n_steps=2,
            gain=0.5,
            detector=lambda image, key: image,
        )
