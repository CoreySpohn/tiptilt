"""Deformable-mirror wavefront control: a differentiable EFC dark-hole loop."""

import equinox as eqx
import jax
import jax.numpy as jnp
from physicaloptix import PhaseScreen

from wavefronts.sensing import (
    KalmanFieldEstimator,
    estimate_field_pairwise,
    probe_measurement,
)


def _dm_coeffs(path, dm_index):
    return path.stages[dm_index].op.basis.coeffs


def close_dark_hole(
    path,
    input_field,
    dm_indices,
    dark_zone_mask,
    *,
    n_steps,
    gain,
    regularization,
    estimator="oracle",
    model_field=None,
    probes=None,
    probe_dm=None,
    detector=None,
    key=None,
):
    """Dig a dark hole with deformable mirrors by electric-field conjugation.

    Linearizes the focal field with respect to the stacked DM command ONCE (the
    control Jacobian is constant to first order in the small-signal dark-hole
    regime), builds a regularized real control matrix, then runs a differentiable
    ``lax.scan`` that RE-PROPAGATES for the measurement and updates the command
    each step by swapping every DM's coefficients with ``eqx.tree_at`` -- never a
    reconstruction, which would re-run the propagator's construction-time gates.
    Because every propagation is a pure function, the loop differentiates through
    the feedback (e.g. the final contrast with respect to the loop gain).

    With one pupil DM the loop reaches only the PHASE quadrature, so it corrects a
    one-sided dark zone and floors on any amplitude speckle. Adding a second,
    out-of-pupil DM (a ``PhaseScreen`` at an ``INTERMEDIATE`` plane, reached
    through a ``Fresnel`` relay) supplies the amplitude quadrature via the Talbot
    conversion, which is what a symmetric (two-sided) or broadband dark hole needs.
    The commands of every listed DM are concatenated into one vector, jointly
    linearized, and solved together.

    A chromatic ``input_field`` digs a BROADBAND hole: the dark-zone response is
    stacked across wavelengths (each weighted by ``sqrt`` of the spectrum weight,
    so the solve is a weighted least-squares) and the reported contrast is the
    weight-averaged dark-zone intensity. Broadband correction is the second reason
    for two DMs, since the amplitude chromaticity a single DM leaves is exactly
    what the out-of-pupil DM cancels. The control Jacobian is a dense
    ``jax.jacfwd`` over the full command, so it materializes the
    ``(n_dark_zone, sum_modes)`` array -- fine for modest mode counts, but chunk it
    or reuse a streamed linearization at scale. ``dark_zone_mask`` must be a
    concrete (static) array for the loop to jit.

    The ``estimator`` selects how the dark-zone field is measured each step. The
    default ``"oracle"`` reads the true field by re-propagation (perfect
    knowledge, the achievable-contrast reference). ``"pairwise"`` instead
    estimates the field from probe images (see
    :func:`wavefronts.sensing.estimate_field_pairwise`), the hardware-realistic
    loop: it builds the control Jacobian on the ``model_field`` and floors on the
    model mismatch rather than digging arbitrarily deep. The estimated loop is
    monochromatic for now.

    Args:
        path: An ``OpticalPath`` ending at the focal plane whose ``dm_indices``
            stages are ``PhaseScreen`` deformable mirrors.
        input_field: The entrance field carrying the aberration to correct.
        dm_indices: A stage index, or a tuple of stage indices, each a
            ``PhaseScreen`` DM. A bare int is treated as a single-DM loop.
        dark_zone_mask: Boolean ``(y, x)`` focal-plane region to null (must
            select at least one pixel).
        n_steps: Number of control iterations.
        gain: Loop gain (the fraction of the computed correction applied).
        regularization: Positive Tikhonov regularization for the control-matrix
            inverse.
        estimator: ``"oracle"`` (read the true field) or ``"pairwise"`` (estimate
            it by probing).
        model_field: Design entrance field (no aberration) for the ``"pairwise"``
            control Jacobian and probe model; defaults to ``input_field``.
        probes: Probe command vectors for the probe deformable mirror (required
            for ``"pairwise"``; see :func:`wavefronts.sensing.probe_set`).
        probe_dm: Stage index of the probe deformable mirror; defaults to the
            first of ``dm_indices``.
        detector: Optional ``callable(image, key) -> image`` applying measurement
            noise to each probe image in the ``"pairwise"`` loop.
        key: PRNG key for the detector, split per step.

    Returns:
        ``(command, dark_zone_history)``: the final stacked DM command (the DMs'
        coefficients concatenated in ``dm_indices`` order) and the mean dark-zone
        intensity at each iteration.

    Raises:
        TypeError: If any ``dm_indices`` stage is not a ``PhaseScreen``.
        ValueError: If ``regularization`` is not positive, the dark zone is
            empty, ``estimator`` is unknown, or ``"pairwise"`` is missing
            ``probes``.
        NotImplementedError: If ``"pairwise"`` is used with a chromatic field.
    """
    indices = (dm_indices,) if isinstance(dm_indices, int) else tuple(dm_indices)
    for i in indices:
        stage_op = path.stages[i].op
        if not isinstance(stage_op, PhaseScreen):
            raise TypeError(
                f"stage {i} is not a PhaseScreen deformable mirror; got "
                f"{type(stage_op).__name__}"
            )
    if regularization <= 0.0:
        raise ValueError(f"regularization must be positive, got {regularization}")
    mask = jnp.asarray(dark_zone_mask)
    if not bool(jnp.any(mask)):
        raise ValueError("dark_zone_mask selects no pixels")
    if estimator not in ("oracle", "pairwise", "kalman"):
        raise ValueError(
            f"estimator must be 'oracle', 'pairwise', or 'kalman', got {estimator!r}"
        )
    estimated = estimator in ("pairwise", "kalman")
    if estimated:
        if probes is None:
            raise ValueError(f"estimator={estimator!r} requires probes")
        if input_field.spectrum is not None:
            raise NotImplementedError(
                "broadband estimated control is not yet supported"
            )
        if probe_dm is None:
            probe_dm = indices[0]
    # The control Jacobian is known from the model; the honest estimated loop
    # builds it on the unaberrated model field, not the (unknown) true field.
    jacobian_field = (
        model_field if (estimated and model_field is not None) else input_field
    )

    mode_counts = [path.stages[i].op.basis.n_modes for i in indices]
    n_total = sum(mode_counts)
    # Static split points that carve the stacked command back into per-DM chunks.
    split_points = []
    running = 0
    for count in mode_counts[:-1]:
        running += count
        split_points.append(running)

    spectrum = input_field.spectrum
    weights = jnp.ones(1) if spectrum is None else spectrum.weights
    sqrt_weights = jnp.sqrt(weights)

    def set_commands(command):
        chunks = jnp.split(command, split_points)
        return eqx.tree_at(
            lambda p: [_dm_coeffs(p, i) for i in indices], path, list(chunks)
        )

    def focal_of(command, field):
        out, _ = set_commands(command).propagate(field)
        return out.data

    def focal_field(command):
        return focal_of(command, input_field)

    def dark_zone(data):
        """Weighted, flattened complex dark-zone vector (mono or per-wavelength)."""
        if data.ndim == 2:
            return data[mask]
        return (sqrt_weights[:, jnp.newaxis] * data[:, mask]).reshape(-1)

    def contrast(data):
        """Weight-averaged mean dark-zone intensity (the broadband contrast)."""
        if data.ndim == 2:
            return jnp.mean(jnp.abs(data[mask]) ** 2)
        intensity = jnp.abs(data[:, mask]) ** 2  # (nlam, n_dz)
        return jnp.mean(jnp.tensordot(weights, intensity, axes=1))

    # Hoist: the joint dark-zone control Jacobian d(E_dz)/d(stacked command), once.
    g_dz = jax.jacfwd(lambda c: dark_zone(focal_of(c, jacobian_field)))(
        jnp.zeros(n_total)
    )
    # Real electric-field conjugation: a real command cancels a complex field, so
    # stack the real and imaginary response rows.
    response = jnp.concatenate([jnp.real(g_dz), jnp.imag(g_dz)], axis=0)
    gram = response.T @ response + regularization * jnp.eye(n_total)
    control_matrix = jnp.linalg.solve(gram, response.T)

    if estimator == "pairwise":
        # Probe-and-estimate loop: each step estimates the dark-zone field from
        # measured images instead of reading the true field. Unrolled in Python
        # (still a pure composition, so it differentiates), since probing is a
        # multi-propagation, keyed-noise step that does not fit a lax.scan.
        keys = (
            list(jax.random.split(key, n_steps))
            if key is not None
            else [None] * n_steps
        )
        model = input_field if model_field is None else model_field
        command = jnp.zeros(n_total)
        history = []
        for i in range(n_steps):
            history.append(contrast(focal_field(command)))
            e_hat = estimate_field_pairwise(
                set_commands(command),
                input_field,
                probe_dm,
                probes,
                mask,
                model_field=model,
                detector=detector,
                key=keys[i],
                regularization=regularization,
            )
            residual = jnp.concatenate([jnp.real(e_hat), jnp.imag(e_hat)])
            command = command - gain * (control_matrix @ residual)
        return command, jnp.stack(history)

    if estimator == "kalman":
        # One probe pair per step; the filter accumulates rank over time,
        # predicting with the known field change from the applied DM delta
        # (g_dz @ delta) and correcting from a single measurement.
        keys = (
            list(jax.random.split(key, n_steps))
            if key is not None
            else [None] * n_steps
        )
        model = input_field if model_field is None else model_field
        # Heuristic hyperparameters for a near-noiseless model: trust the
        # measurements (small R) but keep a little process noise so the filter
        # stays responsive to the residual model error as the hole digs.
        kalman = KalmanFieldEstimator.init(
            int(jnp.sum(mask)),
            initial_variance=1.0,
            process_noise=1e-12,
            measurement_noise=1e-16,
        )
        command = jnp.zeros(n_total)
        last_command = jnp.zeros(n_total)
        history = []
        for i in range(n_steps):
            history.append(contrast(focal_field(command)))
            field_change = g_dz @ (command - last_command)
            probe = probes[i % len(probes)]
            probe_field, diff = probe_measurement(
                set_commands(command), input_field, model, probe_dm, probe, mask,
                detector=detector, key=keys[i],
            )
            kalman = kalman.update(probe_field, diff, field_change=field_change)
            residual = jnp.concatenate([jnp.real(kalman.field), jnp.imag(kalman.field)])
            last_command = command
            command = command - gain * (control_matrix @ residual)
        return command, jnp.stack(history)

    def step(command, _):
        data = focal_field(command)  # exact re-propagation each step
        field_dz = dark_zone(data)
        residual = jnp.concatenate([jnp.real(field_dz), jnp.imag(field_dz)])
        command_next = command - gain * (control_matrix @ residual)
        return command_next, contrast(data)

    return jax.lax.scan(step, jnp.zeros(n_total), None, length=n_steps)
