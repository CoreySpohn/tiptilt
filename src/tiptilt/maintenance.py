"""Dark-hole maintenance: hold a dug hole against wavefront drift.

Digging and maintaining are different regimes of the same seams. Digging
starts cold with a large static error; maintenance starts from a DUG command
(``command0``) with a small time-varying drift injected upstream, senses it
interleaved with science, and corrects it every frame. The driver here
threads the same ``AbstractEstimator`` / ``AbstractController`` seams as
``close_dark_hole``, with two maintenance-specific pieces:

- the drift is applied to a named wavefront-error ``PhaseScreen`` in the TRUE
  system each step (the controller's model never sees it -- honesty is
  preserved by construction), sourced from a ``TabulatedSpeckleField.eps``
  accessor or any ``callable(time_s) -> coefficients``;
- the control Jacobian is linearized AT ``command0`` (the mirror is a true
  exponential, so the Jacobian rotates with the base point), the operating
  point a dig-from-cold loop does not have.

``make_detector`` wraps ``physicaloptix.read_detector`` into the
mean-preserving normalized-intensity form the estimators consume, so a
brighter star adds signal-to-noise, never loop gain.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from physicaloptix import PhaseScreen, read_detector

from tiptilt.control import DarkZoneModel, EFCController
from tiptilt.sensing import KalmanEstimator, OracleEstimator, PairwiseEstimator
from tiptilt.speckle import J2000_JD, TabulatedSpeckleField


def make_detector(
    *,
    flux,
    exposure_time=1.0,
    read_noise_e=0.0,
    quantum_efficiency=1.0,
    reference_peak=1.0,
    method="poisson",
):
    """A mean-preserving detector wrapper for normalized-intensity images.

    Scales an image (in units of ``reference_peak``) to photoelectrons,
    applies :func:`physicaloptix.read_detector`, and scales the counts back,
    so the expectation of the output equals the input image: more flux means
    LESS noise, never more loop gain. Photon budgets stay a separate metric
    axis.

    Args:
        flux: Photons per unit intensity per exposure (the star brightness).
        exposure_time: Exposure seconds per frame.
        read_noise_e: Gaussian read noise in electrons RMS.
        quantum_efficiency: Detected fraction.
        reference_peak: The intensity that corresponds to unit normalized
            image (e.g. the telescope PSF peak).
        method: ``"poisson"`` (exact) or ``"gaussian"`` (differentiable).

    Returns:
        ``callable(image, key) -> image`` in the same normalized units.
    """
    scale = flux * exposure_time * quantum_efficiency

    def detector(image, key):
        counts = read_detector(
            image / reference_peak,
            key,
            flux=flux,
            exposure_time=exposure_time,
            read_noise_e=read_noise_e,
            quantum_efficiency=quantum_efficiency,
            method=method,
        )
        return reference_peak * counts / scale

    return detector


def maintain_dark_hole(
    path,
    input_field,
    dm_indices,
    dark_zone_mask,
    *,
    drift,
    drift_stage,
    n_steps,
    dt_s,
    gain,
    regularization,
    command0=None,
    linearize_at_command0=True,
    controller=None,
    estimator="oracle",
    model_field=None,
    probes=None,
    probe_dm=None,
    detector=None,
    key=None,
    process_noise=1e-10,
    measurement_noise=1e-16,
    return_trajectory=False,
):
    """Hold a dark hole against injected wavefront drift.

    Each step: advance the drift trajectory on the TRUE system's
    wavefront-error screen, measure (via the chosen estimator, optionally
    through a noisy detector), and correct with the controller built ONCE at
    the operating point. The model side (Jacobian, probe model) never sees
    the drift, so the loop is honest by construction.

    Args:
        path: The ``OpticalPath`` containing both the drift screen and the
            deformable mirrors.
        input_field: The true entrance field.
        dm_indices: DM stage index or tuple of indices.
        dark_zone_mask: Boolean focal-plane dark-zone mask.
        drift: A ``TabulatedSpeckleField`` (its ``eps`` accessor is used) or
            any ``callable(time_s) -> coefficients`` for the drift screen.
        drift_stage: Stage index of the wavefront-error ``PhaseScreen`` that
            carries the drift.
        n_steps: Number of maintenance frames.
        dt_s: Seconds per frame (the drift clock).
        gain: Loop gain (0 disables correction -- the open-loop reference).
        regularization: Tikhonov term of the default EFC controller.
        command0: The pre-dug DM command to hold (defaults to zeros).
        linearize_at_command0: Build the Jacobian at ``command0`` (the honest
            maintenance operating point); ``False`` reuses the cold Jacobian.
        controller: Optional ``AbstractController`` overriding the default
            EFC (e.g. a ``PredictiveController``).
        estimator: ``"oracle"``, ``"pairwise"``, or ``"kalman"``.
        model_field: Design entrance field for the honest estimators.
        probes: Probe commands (required for the estimated loops).
        probe_dm: Probe mirror stage index; defaults to the first DM.
        detector: Optional noise model ``callable(image, key) -> image``.
        key: PRNG key for the detector, split per frame.
        process_noise: Kalman process-noise variance (drift-tuned random
            walk; larger than the dig-from-cold default).
        measurement_noise: Kalman measurement-noise variance.
        return_trajectory: Also return the per-frame command trajectory
            (the command HELD during each frame, aligned with the history).

    Returns:
        ``(command, contrast_history)``: the final stacked command and the
        TRUE (drifted) dark-zone contrast at each frame. With
        ``return_trajectory=True``, ``(command, contrast_history,
        trajectory)`` where ``trajectory`` has shape ``(n_steps, n_total)``
        and ``trajectory[0]`` is the operating point ``command0``.

    Raises:
        TypeError: If ``drift_stage`` is not a ``PhaseScreen``.
        ValueError: For the same conditions as ``close_dark_hole``.
    """
    if not isinstance(path.stages[drift_stage].op, PhaseScreen):
        raise TypeError(
            f"drift_stage {drift_stage} is not a PhaseScreen; got "
            f"{type(path.stages[drift_stage].op).__name__}"
        )
    if estimator not in ("oracle", "pairwise", "kalman"):
        raise ValueError(
            f"estimator must be 'oracle', 'pairwise', or 'kalman', got {estimator!r}"
        )
    estimated = estimator in ("pairwise", "kalman")
    if estimated and probes is None:
        raise ValueError(f"estimator={estimator!r} requires probes")

    indices = (dm_indices,) if isinstance(dm_indices, int) else tuple(dm_indices)
    if estimated and probe_dm is None:
        probe_dm = indices[0]
    jacobian_field = (
        model_field if (estimated and model_field is not None) else input_field
    )
    eps_of = drift.eps if hasattr(drift, "eps") else drift

    dz_model = DarkZoneModel.build(
        path,
        indices,
        dark_zone_mask,
        jacobian_field=jacobian_field,
        operating_point=(command0 if linearize_at_command0 else None),
    )
    if command0 is None:
        command0 = jnp.zeros(dz_model.n_total)
    if controller is None:
        controller = EFCController.build(
            dz_model, gain=gain, regularization=regularization
        )

    model = input_field if model_field is None else model_field
    if estimator == "pairwise":
        sensor = PairwiseEstimator(
            input_field=input_field,
            model_field=model,
            probes=tuple(probes),
            probe_dm=probe_dm,
            detector=detector,
            regularization=regularization,
        )
    elif estimator == "kalman":
        sensor = KalmanEstimator.build(
            dz_model,
            input_field=input_field,
            model_field=model,
            probes=tuple(probes),
            probe_dm=probe_dm,
            detector=detector,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )
    else:
        sensor = OracleEstimator(input_field=input_field)

    keys = list(jax.random.split(key, n_steps)) if key is not None else [None] * n_steps

    def drifted(model_view, time_s):
        """The model view with the TRUE system's drift screen advanced."""
        return eqx.tree_at(
            lambda m: m.path.stages[drift_stage].op.basis.coeffs,
            model_view,
            jnp.asarray(eps_of(time_s)),
        )

    command = command0
    history = []
    trajectory = []
    for i in range(n_steps):
        dz_true = drifted(dz_model, i * dt_s)
        trajectory.append(command)
        history.append(dz_true.contrast(dz_true.focal_of(command, input_field)))
        sensor, e_hat = sensor.estimate(dz_true, command, key=keys[i])
        controller, delta = controller.command_delta(e_hat)
        command = command + delta
    if return_trajectory:
        return command, jnp.stack(history), jnp.stack(trajectory)
    return command, jnp.stack(history)


def maintained_residual_field(
    path,
    input_field,
    dm_indices,
    dark_zone_mask,
    *,
    drift,
    drift_stage,
    n_steps,
    dt_s,
    gain,
    regularization,
    normalization=1.0,
    pixel_scale_lod=0.25,
    epoch_jd=J2000_JD,
    coherent=True,
    **maintain_kwargs,
):
    """The maintenance loop's residual, delivered through the speckle seam.

    Runs :func:`maintain_dark_hole` and packages what the CONTROLLER leaves
    behind as a ``TabulatedSpeckleField``: the nominal field is the dug
    operating point, the sensitivity stacks the drift screen's modes with
    the deformable mirrors' modes, and the tabulated trajectory is the true
    per-frame state ``[drift(t_k), command(t_k) - command0]``. Because the
    result implements the ``optixstuff.AbstractSpeckleField`` contract, the
    imaging and yield layers consume the controller-conditioned residual --
    the honest speckle floor -- with no change on their side. The coherent
    pinning term is on by default: residual speckles near a dug hole
    interfere with the static floor, which is the statistics that matter
    there.

    Args:
        path: The ``OpticalPath`` containing drift screen and mirrors.
        input_field: The true entrance field.
        dm_indices: DM stage index or tuple of indices.
        dark_zone_mask: Boolean focal-plane dark-zone mask.
        drift: A ``TabulatedSpeckleField`` (its ``eps``) or
            ``callable(time_s) -> coefficients`` for the drift screen.
        drift_stage: Stage index of the drift-carrying ``PhaseScreen``.
        n_steps: Number of maintenance frames.
        dt_s: Seconds per frame.
        gain: Loop gain (0 = the open-loop reference field).
        regularization: Tikhonov term of the default EFC controller.
        normalization: Intensity mapping to unit contrast (the telescope
            PSF peak the imaging layer references).
        pixel_scale_lod: Native focal pixel scale in lambda/D per pixel.
        epoch_jd: Julian Date mapping to ``time_s = 0``.
        coherent: Include the pinning cross term (default True).
        **maintain_kwargs: Forwarded to :func:`maintain_dark_hole`
            (``command0``, ``estimator``, ``probes``, ``detector``, ...).

    Returns:
        ``(field, contrast_history)``: the ``TabulatedSpeckleField`` and the
        maintenance loop's true dark-zone contrast per frame.
    """
    _command, history, trajectory = maintain_dark_hole(
        path,
        input_field,
        dm_indices,
        dark_zone_mask,
        drift=drift,
        drift_stage=drift_stage,
        n_steps=n_steps,
        dt_s=dt_s,
        gain=gain,
        regularization=regularization,
        return_trajectory=True,
        **maintain_kwargs,
    )
    indices = (dm_indices,) if isinstance(dm_indices, int) else tuple(dm_indices)
    command0 = trajectory[0]
    dz_model = DarkZoneModel.build(
        path,
        indices,
        dark_zone_mask,
        jacobian_field=input_field,
        operating_point=command0,
    )
    eps_of = drift.eps if hasattr(drift, "eps") else drift
    n_drift = path.stages[drift_stage].op.basis.coeffs.shape[0]
    zero_drift = jnp.zeros(n_drift)

    def full_focal(drift_coeffs, command):
        dz_true = eqx.tree_at(
            lambda m: m.path.stages[drift_stage].op.basis.coeffs,
            dz_model,
            drift_coeffs,
        )
        return dz_true.focal_of(command, input_field)

    e_nom = full_focal(zero_drift, command0)
    g_drift = jax.jacfwd(full_focal, argnums=0)(zero_drift, command0)
    g_command = jax.jacfwd(full_focal, argnums=1)(zero_drift, command0)
    sensitivity = jnp.concatenate(
        [jnp.moveaxis(g_drift, -1, 0), jnp.moveaxis(g_command, -1, 0)], axis=0
    )
    times_s = dt_s * jnp.arange(n_steps)
    drift_table = jnp.stack([jnp.asarray(eps_of(float(t))) for t in times_s])
    eps_table = jnp.concatenate([drift_table, trajectory - command0], axis=1)
    field = TabulatedSpeckleField(
        e_nom,
        sensitivity,
        times_s,
        eps_table,
        normalization,
        pixel_scale_lod=pixel_scale_lod,
        epoch_jd=epoch_jd,
        coherent=coherent,
    )
    return field, history
