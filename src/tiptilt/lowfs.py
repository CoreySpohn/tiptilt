"""Low-order sensing on a forked pickoff arm, and the pointing loop.

A pointing / low-order loop needs no field access: a fraction of the beam is
split off ahead of the science instrument, imaged onto a camera, and the
image's LINEAR intensity response to the corrector's modes is calibrated once
(a response matrix about the operating point, exactly the
:func:`tiptilt.sensing.zwfs_calibrate` construction, here through a forked
``OpticalSystem``). Closing an integrator on the reconstructed coefficients
holds the beam against line-of-sight jitter and low-order drift while the
science arm observes -- the loop the library is named for.

Even modes (focus and up) have no first-order intensity signature in an
in-focus image, so the sensor arm carries a static DEFOCUS BIAS -- the
standard defocused low-order sensor. The flight-style variant that senses
the light rejected by the coronagraph focal-plane mask needs a re-imaging
relay element and remains a follow-on; it exercises the same seams built
here (the fork, the response matrix, the integrator).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from physicaloptix import PhaseScreen

from tiptilt.sensing import zwfs_reconstruct


def _trunk_stage_index(system, name, what):
    names = [stage.name for stage in system.trunk.stages]
    if name not in names:
        raise KeyError(f"{what} stage {name!r} not in trunk stages {names}")
    index = names.index(name)
    if not isinstance(system.trunk.stages[index].op, PhaseScreen):
        raise TypeError(f"{what} stage {name!r} is not a PhaseScreen")
    return index


def _branch_check(system, name):
    names = [branch.name for branch in system.branches]
    if name not in names:
        raise KeyError(f"arm {name!r} not in branches {names}")


def lowfs_calibrate(system, input_field, *, arm, stage):
    """Reference image and response matrix of an arm's intensity to a stage.

    Linearizes the named sensing arm's camera intensity about the current
    coefficients of the named trunk ``PhaseScreen`` (via ``jax.jacfwd``) --
    the response-matrix calibration of a low-order wavefront sensor.
    Calibrating against the corrector stage makes the reconstructed
    coefficients directly commandable; any aberration sharing the mode
    shapes (a jitter carrier upstream) is sensed in the same units.

    Args:
        system: The forked ``OpticalSystem`` in its calibration state.
        input_field: The entrance field.
        arm: Branch name of the sensing camera.
        stage: Trunk stage name of the ``PhaseScreen`` to calibrate against.

    Returns:
        ``(reference, response)``: the operating-point image, shape
        ``(ny, nx)``, and the ``(n_pixels, n_modes)`` intensity response.

    Raises:
        KeyError: If ``arm`` or ``stage`` does not name a branch / trunk stage.
        TypeError: If the named stage is not a ``PhaseScreen``.
    """
    _branch_check(system, arm)
    index = _trunk_stage_index(system, stage, "calibration")
    base = system.trunk.stages[index].op.basis.coeffs

    def image_of(coeffs):
        updated = eqx.tree_at(
            lambda s: s.trunk.stages[index].op.basis.coeffs, system, coeffs
        )
        outputs, _ = updated.propagate(input_field)
        return jnp.abs(outputs[arm].data) ** 2

    reference = image_of(base)
    jacobian = jax.jacfwd(image_of)(base)  # (ny, nx, n_modes)
    return reference, jacobian.reshape(-1, base.shape[0])


def run_pointing_loop(
    system,
    input_field,
    *,
    corrector_stage,
    drift_stage,
    sense,
    science,
    mask,
    drift_table,
    n_steps,
    gain,
    regularization=0.0,
    detector=None,
    key=None,
):
    """Close an integrator on the low-order sensor while science observes.

    Each frame: advance the jitter carrier on the TRUE system, propagate the
    container once, read the sensing arm's intensity (optionally through a
    ``make_detector`` model), reconstruct low-order coefficients against the
    response matrix calibrated on the clean system, and integrate the
    corrector. The reconstruction sees INTENSITY ONLY; the true modal
    residual (drift plus command) is returned as a diagnostic, never fed
    back. The drift carrier and the corrector must share a mode basis so
    that residual is meaningful.

    Args:
        system: The forked ``OpticalSystem`` in its clean calibration state.
        input_field: The true entrance field.
        corrector_stage: Trunk stage name of the corrector ``PhaseScreen``.
        drift_stage: Trunk stage name of the jitter carrier ``PhaseScreen``.
        sense: Branch name of the sensing camera.
        science: Branch name of the science camera.
        mask: Boolean focal mask scoring the science arm.
        drift_table: ``(n_steps, n_modes)`` jitter coefficients per frame.
        n_steps: Number of frames.
        gain: Integrator gain (0 = open loop).
        regularization: Tikhonov term of the reconstruction.
        detector: Optional ``callable(image, key) -> image`` noise model
            applied to the sensor image (see
            :func:`tiptilt.maintenance.make_detector`).
        key: PRNG key, required when ``detector`` is given.

    Returns:
        Dict with ``command`` (final corrector command), ``history``
        (absolute science mask mean intensity per frame), ``excess``
        (reference-subtracted science mask intensity -- the jitter-tracking
        axis; the absolute history is floor-dominated), and ``residual_nm``
        (``(n_steps, n_modes)`` true post-correction modal residual).

    Raises:
        KeyError: If a stage or branch name does not resolve.
        TypeError: If a named stage is not a ``PhaseScreen``.
        ValueError: If drift and corrector mode counts differ, or a
            ``detector`` is given without a ``key``.
    """
    _branch_check(system, sense)
    _branch_check(system, science)
    drift_index = _trunk_stage_index(system, drift_stage, "drift")
    corrector_index = _trunk_stage_index(system, corrector_stage, "corrector")
    drift_table = jnp.asarray(drift_table)
    n_modes = system.trunk.stages[corrector_index].op.basis.coeffs.shape[0]
    if drift_table.shape[1] != n_modes:
        raise ValueError(
            f"drift table has {drift_table.shape[1]} modes, corrector has "
            f"{n_modes}; the pointing loop assumes a shared mode basis"
        )
    if detector is not None and key is None:
        raise ValueError("a detector model needs a PRNG key")

    reference, response = lowfs_calibrate(
        system, input_field, arm=sense, stage=corrector_stage
    )
    outputs, _ = system.propagate(input_field)
    e_ref = outputs[science].data[mask]

    def with_state(command, frame):
        return eqx.tree_at(
            lambda s: (
                s.trunk.stages[drift_index].op.basis.coeffs,
                s.trunk.stages[corrector_index].op.basis.coeffs,
            ),
            system,
            (drift_table[frame], command),
        )

    command = jnp.zeros(n_modes)
    history = []
    excess = []
    residuals = []
    for frame in range(n_steps):
        outputs, _ = with_state(command, frame).propagate(input_field)
        image = jnp.abs(outputs[sense].data) ** 2
        if detector is not None:
            key, subkey = jr.split(key)
            image = detector(image, subkey)
        estimate = zwfs_reconstruct(
            image, reference, response, regularization=regularization
        )
        command = command - gain * estimate
        outputs, _ = with_state(command, frame).propagate(input_field)
        dark_zone = outputs[science].data[mask]
        history.append(jnp.mean(jnp.abs(dark_zone) ** 2))
        excess.append(jnp.mean(jnp.abs(dark_zone - e_ref) ** 2))
        residuals.append(drift_table[frame] + command)
    return {
        "command": command,
        "history": jnp.stack(history),
        "excess": jnp.stack(excess),
        "residual_nm": jnp.stack(residuals),
    }


__all__ = ["lowfs_calibrate", "run_pointing_loop"]
