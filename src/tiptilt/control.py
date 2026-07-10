"""Deformable-mirror wavefront control: models, controllers, and the EFC loop.

The control problem factors onto two symmetric seams sharing one currency:

- ``DarkZoneModel``: the controller-facing linearization of a path -- the
  stacked dark-zone Jacobian ``g_dz = d(E_dz)/d(command)`` (broadband via
  sqrt-weighted per-wavelength stacking), built ONCE at an operating point,
  plus the command plumbing (``set_commands``/``focal_of``/``contrast``).
- ``AbstractController``: ``command_delta(estimate) -> (new_self, delta)``.
  Stateless laws (EFC / stroke minimization) return themselves unchanged;
  stateful ones (a predictive AR feed-forward) advance their state.
- ``AbstractEstimator`` (in ``tiptilt.sensing``): ``estimate(model,
  command, key) -> (new_self, e_hat)``, the measurement half.

``close_dark_hole`` is a thin driver over these seams: the oracle loop is a
``lax.scan`` with ONE propagation per step (read the true field, correct);
the estimated loops are Python-unrolled (probing is a multi-propagation,
keyed-noise step). Every propagation is pure, so the loops differentiate
through the feedback.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from physicaloptix import PhaseScreen

from tiptilt.sensing import KalmanEstimator, PairwiseEstimator


def _dm_coeffs(path, dm_index):
    return path.stages[dm_index].op.basis.coeffs


class DarkZoneModel(eqx.Module):
    """The controller-facing dark-zone linearization of one optical path.

    Carries the stacked control Jacobian and the command plumbing every
    estimator and controller shares. The dark-zone ordering is
    wavelength-major, pixel-minor; ``stack_weights`` (sqrt of the spectrum
    weights, ones for mono) map an UNWEIGHTED field estimate onto the
    Jacobian's weighted stacking.

    Attributes:
        path: The optical path (commands applied by ``set_commands``).
        indices: DM stage indices, in command-stacking order.
        split_points: Cumulative mode counts carving the stacked command.
        mask: Boolean dark-zone mask.
        weights: Spectrum weights (ones for a monochromatic field).
        sqrt_weights: Their square roots.
        stack_weights: Per-(wavelength, pixel) sqrt weights, stacked.
        g_dz: The dark-zone Jacobian, shape ``(n_stack, n_total)``.
        operating_point: The command the Jacobian was built at.
    """

    path: eqx.Module
    indices: tuple = eqx.field(static=True)
    split_points: tuple = eqx.field(static=True)
    mask: Array
    weights: Array
    sqrt_weights: Array
    stack_weights: Array
    g_dz: Array
    operating_point: Array

    @property
    def n_total(self):
        """Total stacked command length."""
        return self.g_dz.shape[1]

    def set_commands(self, command):
        """The path with the stacked ``command`` split onto its mirrors."""
        chunks = jnp.split(command, list(self.split_points))
        return eqx.tree_at(
            lambda p: [_dm_coeffs(p, i) for i in self.indices],
            self.path,
            list(chunks),
        )

    def focal_of(self, command, field):
        """The focal data for a command applied to a given entrance field."""
        out, _ = self.set_commands(command).propagate(field)
        return out.data

    def dark_zone_unweighted(self, data):
        """Flattened complex dark-zone vector, unweighted (mono or stacked)."""
        if data.ndim == 2:
            return data[self.mask]
        return data[:, self.mask].reshape(-1)

    def contrast(self, data):
        """Weight-averaged mean dark-zone intensity (broadband contrast)."""
        if data.ndim == 2:
            return jnp.mean(jnp.abs(data[self.mask]) ** 2)
        intensity = jnp.abs(data[:, self.mask]) ** 2  # (nlam, n_dz)
        return jnp.mean(jnp.tensordot(self.weights, intensity, axes=1))

    @classmethod
    def build(
        cls, path, dm_indices, dark_zone_mask, *, jacobian_field, operating_point=None
    ):
        """Linearize a path's dark zone about an operating point.

        Args:
            path: An ``OpticalPath`` whose ``dm_indices`` stages are
                ``PhaseScreen`` deformable mirrors.
            dm_indices: A stage index or tuple of stage indices.
            dark_zone_mask: Boolean focal-plane mask (static).
            jacobian_field: The entrance field the Jacobian is built on (the
                DESIGN model for an honest loop; the true field for an
                oracle).
            operating_point: Stacked command to linearize at; defaults to
                zeros (the dig-from-cold base point). A maintenance loop
                passes the pre-dug command.

        Returns:
            A ``DarkZoneModel``.
        """
        indices = (dm_indices,) if isinstance(dm_indices, int) else tuple(dm_indices)
        for i in indices:
            stage_op = path.stages[i].op
            if not isinstance(stage_op, PhaseScreen):
                raise TypeError(
                    f"stage {i} is not a PhaseScreen deformable mirror; got "
                    f"{type(stage_op).__name__}"
                )
        mask = jnp.asarray(dark_zone_mask)
        if not bool(jnp.any(mask)):
            raise ValueError("dark_zone_mask selects no pixels")

        mode_counts = [path.stages[i].op.basis.n_modes for i in indices]
        n_total = sum(mode_counts)
        split_points = []
        running = 0
        for count in mode_counts[:-1]:
            running += count
            split_points.append(running)

        spectrum = jacobian_field.spectrum
        weights = jnp.ones(1) if spectrum is None else spectrum.weights
        sqrt_weights = jnp.sqrt(weights)
        n_dark = int(jnp.sum(mask))
        stack_weights = jnp.repeat(sqrt_weights, n_dark)
        if operating_point is None:
            operating_point = jnp.zeros(n_total)

        # A weighted view for the Jacobian only; the model's public
        # dark-zone vector stays unweighted (the estimators' convention).
        def weighted_dark_zone(data):
            if data.ndim == 2:
                return data[mask]
            return (sqrt_weights[:, jnp.newaxis] * data[:, mask]).reshape(-1)

        model = cls(
            path=path,
            indices=indices,
            split_points=tuple(split_points),
            mask=mask,
            weights=weights,
            sqrt_weights=sqrt_weights,
            stack_weights=stack_weights,
            g_dz=jnp.zeros((0, n_total)),  # placeholder, replaced below
            operating_point=operating_point,
        )
        g_dz = jax.jacfwd(
            lambda c: weighted_dark_zone(model.focal_of(c, jacobian_field))
        )(operating_point)
        return eqx.tree_at(lambda m: m.g_dz, model, g_dz)


class AbstractController(eqx.Module):
    """The control seam: a field estimate in, a command delta out.

    ``command_delta`` returns ``(new_self, delta)`` so stateful laws (a
    predictive feed-forward carrying an AR state) advance while stateless
    ones (EFC, stroke minimization) return themselves unchanged.
    """

    def command_delta(self, estimate):
        """The command update for an unweighted dark-zone field estimate."""
        raise NotImplementedError


class EFCController(AbstractController):
    """Electric-field conjugation: one regularized real least squares.

    The classic dark-hole law: stack the Jacobian's real and imaginary rows
    (a real command cancels a complex field), Tikhonov-regularize, and apply
    a fixed gain. Energy minimization with a fixed multiplier is the same
    matrix; stroke minimization differs only in how the multiplier is chosen.

    Attributes:
        control_matrix: ``(n_total, 2 n_stack)`` solve of the regularized
            normal equations.
        stack_weights: The model's per-(wavelength, pixel) sqrt weights.
        gain: Loop gain (a differentiable leaf).
    """

    control_matrix: Array
    stack_weights: Array
    gain: Array

    @classmethod
    def build(cls, model, *, gain, regularization):
        """The EFC law for a dark-zone model.

        Args:
            model: The ``DarkZoneModel`` (its ``g_dz`` is the plant).
            gain: Loop gain.
            regularization: Positive Tikhonov term.

        Returns:
            An ``EFCController``.
        """
        if regularization <= 0.0:
            raise ValueError(f"regularization must be positive, got {regularization}")
        response = jnp.concatenate([jnp.real(model.g_dz), jnp.imag(model.g_dz)], axis=0)
        gram = response.T @ response + regularization * jnp.eye(model.n_total)
        return cls(
            control_matrix=jnp.linalg.solve(gram, response.T),
            stack_weights=model.stack_weights,
            gain=jnp.asarray(gain),
        )

    def command_delta(self, estimate):
        """Weight the estimate, stack Re/Im, and apply the control matrix."""
        weighted = self.stack_weights * estimate.reshape(-1)
        residual = jnp.concatenate([jnp.real(weighted), jnp.imag(weighted)])
        return self, -self.gain * (self.control_matrix @ residual)


class StrokeMinController(AbstractController):
    """Stroke minimization: the least command that reaches a target contrast.

    The dual framing of the same convex program as EFC: instead of a fixed
    Tikhonov weight, pick per step the LARGEST multiplier (least stroke)
    whose predicted linear residual still meets ``target_contrast``, falling
    back to the deepest available correction when the target is out of
    reach. Stateless.

    Attributes:
        response: Stacked real Jacobian rows ``[Re g_dz; Im g_dz]``.
        rtr: Its Gram matrix.
        stack_weights: The model's stacked sqrt weights.
        n_dark: Dark-zone pixel count (contrast normalization).
        mu_grid: Candidate multipliers, ascending.
        target_contrast: The contrast the step tries to reach.
        gain: Step gain on the chosen delta.
    """

    response: Array
    rtr: Array
    stack_weights: Array
    n_dark: int = eqx.field(static=True)
    mu_grid: Array
    target_contrast: Array
    gain: Array

    @classmethod
    def build(cls, model, *, target_contrast, mu_grid=None, gain=1.0):
        """The stroke-minimizing law for a dark-zone model.

        Args:
            model: The ``DarkZoneModel``.
            target_contrast: Dark-zone mean intensity to reach per step.
            mu_grid: Candidate Lagrange multipliers (ascending); defaults to
                a wide log grid.
            gain: Step gain on the chosen delta.

        Returns:
            A ``StrokeMinController``.
        """
        if mu_grid is None:
            mu_grid = jnp.logspace(-12, 0, 13)
        response = jnp.concatenate([jnp.real(model.g_dz), jnp.imag(model.g_dz)], axis=0)
        n_dark = int(jnp.sum(model.mask))
        return cls(
            response=response,
            rtr=response.T @ response,
            stack_weights=model.stack_weights,
            n_dark=n_dark,
            mu_grid=jnp.asarray(mu_grid),
            target_contrast=jnp.asarray(target_contrast),
            gain=jnp.asarray(gain),
        )

    def command_delta(self, estimate):
        """Pick the least-stroke multiplier that meets the target contrast."""
        weighted = self.stack_weights * estimate.reshape(-1)
        residual = jnp.concatenate([jnp.real(weighted), jnp.imag(weighted)])
        rhs = self.response.T @ residual
        eye = jnp.eye(self.rtr.shape[0])

        def candidate(mu):
            delta = -jnp.linalg.solve(self.rtr + mu * eye, rhs)
            predicted = residual + self.response @ delta
            contrast = jnp.sum(jnp.abs(predicted) ** 2) / self.n_dark
            return delta, contrast

        deltas, contrasts = jax.vmap(candidate)(self.mu_grid)
        feasible = contrasts <= self.target_contrast
        # Largest feasible mu = least stroke; else the deepest correction.
        least_stroke = jnp.argmax(
            jnp.where(feasible, jnp.arange(self.mu_grid.shape[0]), -1)
        )
        deepest = jnp.argmin(contrasts)
        chosen = jnp.where(jnp.any(feasible), least_stroke, deepest)
        return self, self.gain * deltas[chosen]


class PredictiveController(AbstractController):
    """A linear predictive feed-forward wrapped around EFC. Stateful.

    Extrapolates the field estimate one step ahead
    (``e_pred = e + alpha (e - e_prev)``) before applying the EFC law, so a
    steadily drifting field is corrected at its predicted, not lagged,
    value. ``alpha = 0`` reduces exactly to EFC.

    Attributes:
        efc: The inner EFC law.
        prev_estimate: Last step's estimate (the carried state).
        alpha: Extrapolation weight.
        primed: Whether ``prev_estimate`` is real data yet.
    """

    efc: EFCController
    prev_estimate: Array
    alpha: Array
    primed: Array

    @classmethod
    def build(cls, model, *, gain, regularization, alpha=1.0):
        """A predictive law sharing EFC's matrix.

        Args:
            model: The ``DarkZoneModel``.
            gain: Loop gain of the inner EFC.
            regularization: Tikhonov term of the inner EFC.
            alpha: Extrapolation weight (0 = plain EFC).

        Returns:
            A ``PredictiveController``.
        """
        n_stack = model.stack_weights.shape[0]
        return cls(
            efc=EFCController.build(model, gain=gain, regularization=regularization),
            prev_estimate=jnp.zeros(n_stack, dtype=complex),
            alpha=jnp.asarray(alpha),
            primed=jnp.asarray(False),
        )

    def command_delta(self, estimate):
        """Extrapolate the estimate, apply EFC, and advance the state."""
        flat = estimate.reshape(-1)
        predicted = jnp.where(
            self.primed, flat + self.alpha * (flat - self.prev_estimate), flat
        )
        _, delta = self.efc.command_delta(predicted)
        new_self = eqx.tree_at(
            lambda c: (c.prev_estimate, c.primed),
            self,
            (flat, jnp.asarray(True)),
        )
        return new_self, delta


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
    :func:`tiptilt.sensing.estimate_field_pairwise`), the hardware-realistic
    loop: it builds the control Jacobian on the ``model_field`` and floors on the
    model mismatch rather than digging arbitrarily deep. A chromatic
    ``input_field`` drives a BROADBAND estimated loop by sub-band probing: the
    field is estimated per wavelength (``probe_measurement`` reads a per-sub-band
    image) and the DMs are driven against the same stacked, ``sqrt``-weighted
    per-wavelength response the oracle broadband loop uses.

    This is a thin driver over the ``DarkZoneModel`` / ``AbstractEstimator`` /
    ``AbstractController`` seams; swap the law or the sensor by driving those
    seams directly (the maintenance driver does).

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
            for ``"pairwise"``; see :func:`tiptilt.sensing.probe_set`).
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
            empty, ``estimator`` is unknown, or an estimated loop is missing
            ``probes``.
    """
    indices = (dm_indices,) if isinstance(dm_indices, int) else tuple(dm_indices)
    if estimator not in ("oracle", "pairwise", "kalman"):
        raise ValueError(
            f"estimator must be 'oracle', 'pairwise', or 'kalman', got {estimator!r}"
        )
    estimated = estimator in ("pairwise", "kalman")
    if estimated:
        if probes is None:
            raise ValueError(f"estimator={estimator!r} requires probes")
        if probe_dm is None:
            probe_dm = indices[0]
    # The control Jacobian is known from the model; the honest estimated loop
    # builds it on the unaberrated model field, not the (unknown) true field.
    jacobian_field = (
        model_field if (estimated and model_field is not None) else input_field
    )
    dz_model = DarkZoneModel.build(
        path, indices, dark_zone_mask, jacobian_field=jacobian_field
    )
    controller = EFCController.build(dz_model, gain=gain, regularization=regularization)

    if estimated:
        # Probe-and-estimate loops: Python-unrolled (still a pure composition,
        # so they differentiate); probing is a multi-propagation, keyed step.
        keys = (
            list(jax.random.split(key, n_steps))
            if key is not None
            else [None] * n_steps
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
        else:
            sensor = KalmanEstimator.build(
                dz_model,
                input_field=input_field,
                model_field=model,
                probes=tuple(probes),
                probe_dm=probe_dm,
                detector=detector,
            )
        command = jnp.zeros(dz_model.n_total)
        history = []
        for i in range(n_steps):
            history.append(dz_model.contrast(dz_model.focal_of(command, input_field)))
            sensor, e_hat = sensor.estimate(dz_model, command, key=keys[i])
            controller, delta = controller.command_delta(e_hat)
            command = command + delta
        return command, jnp.stack(history)

    def step(command, _):
        data = dz_model.focal_of(command, input_field)  # ONE oracle read
        estimate = dz_model.dark_zone_unweighted(data)
        _, delta = controller.command_delta(estimate)
        return command + delta, dz_model.contrast(data)

    return jax.lax.scan(step, jnp.zeros(dz_model.n_total), None, length=n_steps)
