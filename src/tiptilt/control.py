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
    wavelength-major, pixel-minor; ``stack_weights`` map an UNWEIGHTED field
    estimate onto the Jacobian's weighted stacking.

    The model already weighted the WAVELENGTH axis (a broadband hole digs
    each sub-band in proportion to its spectral weight). ``pixel_weights``
    opens the same door on the SPATIAL axis, so "dig where it matters" is a
    controller input rather than a binary mask choice: the two compose as
    ``stack_weights[l, k] = sqrt(spectrum[l] * pixel[k])``, and weighted
    least squares is the same normal equations with the weights folded into
    the rows. Uniform weights reproduce the unweighted behavior exactly.
    Weights are used only through ratios, so their overall scale is
    irrelevant and the library deliberately does not normalize them (a
    caller may be carrying posterior probability mass and want it readable).

    Attributes:
        path: The optical path (commands applied by ``set_commands``).
        indices: DM stage indices, in command-stacking order.
        split_points: Cumulative mode counts carving the stacked command.
        mask: Boolean dark-zone mask.
        weights: Spectrum weights (ones for a monochromatic field).
        sqrt_weights: Their square roots.
        pixel_weights: Per-dark-zone-pixel weights (ones when uniform).
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
    pixel_weights: Array
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
        """Weight-averaged mean dark-zone intensity (broadband contrast).

        Averaged over the pixel axis with ``pixel_weights`` (uniform weights
        reduce to the plain mean) and over the wavelength axis with the
        spectrum weights.
        """
        pixel_w = self.pixel_weights
        denom = jnp.sum(pixel_w)
        if data.ndim == 2:
            return jnp.sum(pixel_w * jnp.abs(data[self.mask]) ** 2) / denom
        intensity = jnp.abs(data[:, self.mask]) ** 2  # (nlam, n_dz)
        return jnp.sum(pixel_w * jnp.tensordot(self.weights, intensity, axes=1)) / denom

    @classmethod
    def build(
        cls,
        path,
        dm_indices,
        dark_zone_mask,
        *,
        jacobian_field,
        operating_point=None,
        pixel_weights=None,
        materialize_jacobian=True,
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
            pixel_weights: Optional nonnegative per-dark-zone-pixel weights,
                shape ``(n_dz,)`` in the mask's flattened order, defaulting to
                uniform. Only ratios matter, so the scale is free and the
                library does not normalize them. Weights are STATIC per build
                (they change per target or epoch, not per iteration), so the
                intended pattern is rebuilding the model per visit. A pixel
                weighted zero leaves the solve entirely, which also removes it
                from what the Tikhonov regularization sees.
            materialize_jacobian: Build the dense ``g_dz`` by ``jacfwd``
                (default). ``False`` leaves ``g_dz`` as an EMPTY
                ``(0, n_total)`` placeholder for the matrix-free controller,
                which linearizes on demand instead; the dense controllers
                refuse such a model.

        Returns:
            A ``DarkZoneModel``.

        Raises:
            ValueError: If ``dark_zone_mask`` selects no pixels, or
                ``pixel_weights`` has the wrong shape or a negative entry.
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
        if pixel_weights is None:
            pixel_w = jnp.ones(n_dark)
        else:
            pixel_w = jnp.asarray(pixel_weights, dtype=float)
            if pixel_w.shape != (n_dark,):
                raise ValueError(
                    f"pixel_weights has shape {pixel_w.shape}; expected "
                    f"({n_dark},) to match the dark-zone pixel count"
                )
            if bool(jnp.any(pixel_w < 0.0)):
                raise ValueError("pixel_weights must be nonnegative")
        sqrt_pixel_w = jnp.sqrt(pixel_w)
        # Composed sqrt weights: stack_weights[l, k] = sqrt(spectrum_l * pixel_k),
        # so the spectral and spatial axes multiply rather than compete.
        stack_weights = (sqrt_weights[:, jnp.newaxis] * sqrt_pixel_w).reshape(-1)
        if operating_point is None:
            operating_point = jnp.zeros(n_total)

        # A weighted view for the Jacobian only; the model's public
        # dark-zone vector stays unweighted (the estimators' convention).
        # Scaling a row by sqrt(w) is what turns the least squares the
        # controllers solve into the WEIGHTED least squares.
        def weighted_dark_zone(data):
            if data.ndim == 2:
                return sqrt_pixel_w * data[mask]
            return (
                sqrt_weights[:, jnp.newaxis] * sqrt_pixel_w * data[:, mask]
            ).reshape(-1)

        model = cls(
            path=path,
            indices=indices,
            split_points=tuple(split_points),
            mask=mask,
            weights=weights,
            sqrt_weights=sqrt_weights,
            pixel_weights=pixel_w,
            stack_weights=stack_weights,
            g_dz=jnp.zeros((0, n_total)),  # placeholder, replaced below
            operating_point=operating_point,
        )
        if not materialize_jacobian:
            return model
        g_dz = jax.jacfwd(
            lambda c: weighted_dark_zone(model.focal_of(c, jacobian_field))
        )(operating_point)
        return eqx.tree_at(lambda m: m.g_dz, model, g_dz)

    @property
    def has_jacobian(self):
        """Whether the dense ``g_dz`` was materialized."""
        return self.g_dz.shape[0] > 0

    def weighted_dark_zone(self, data):
        """The sqrt-weighted stacked dark-zone vector (the Jacobian's rows)."""
        return self.stack_weights * self.dark_zone_unweighted(data)


def _require_jacobian(model, law):
    if not model.has_jacobian:
        raise ValueError(
            f"{law} needs a materialized Jacobian; build the DarkZoneModel with "
            "materialize_jacobian=True or use MatrixFreeEFCController"
        )


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
        _require_jacobian(model, "EFCController")
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
        _require_jacobian(model, "StrokeMinController")
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


class MatrixFreeEFCController(AbstractController):
    """Electric-field conjugation that never stores the Jacobian.

    The same regularized real least squares as :class:`EFCController`, but the
    normal equations ``(R^T R + reg I) delta = -R^T residual`` are solved by
    conjugate gradients with the two matrix-vector products supplied by
    autodiff: ``R v`` is the linearized propagation (``jax.linearize`` at the
    controller's ``command``) and ``R^T w`` its transpose. Nothing of size
    ``(n_stack, n_total)`` ever exists, which is what a flight-scale mirror
    (tens of thousands of actuators against tens of thousands of dark-zone
    pixels per wavelength) needs. Each solve costs ``2 * max_iterations``
    linearized propagations instead of one dense factorization, and the
    controller RE-LINEARIZES wherever it is pointed (``relinearize``), so it
    doubles as a relinearizing EFC for large excursions.

    The linearization point is state (``command``); ``command_delta`` is
    otherwise stateless, so the seam contract holds.

    Jitting a step that closes over the controller embeds the path's arrays
    as compile-time constants; at large pupils XLA's constant folder then
    evaluates the MFT kernel products with its slow single-threaded
    evaluator (tens of CPU-minutes at 2048 px). Set
    ``XLA_FLAGS=--xla_disable_hlo_passes=constant_folding`` before the
    backend initializes when driving this law at scale.

    Attributes:
        model: The ``DarkZoneModel`` (its path, mask, and weights; ``g_dz``
            may be empty).
        field: The entrance field the linearization is taken on.
        command: The stacked command the controller is linearized at.
        gain: Loop gain (a differentiable leaf).
        regularization: Positive Tikhonov term.
        max_iterations: Conjugate-gradient iteration cap per solve.
        tol: Conjugate-gradient relative tolerance.
    """

    model: DarkZoneModel
    field: eqx.Module
    command: Array
    gain: Array
    regularization: float = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)
    tol: float = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        model,
        *,
        jacobian_field,
        gain,
        regularization,
        max_iterations=100,
        tol=1e-6,
        operating_point=None,
    ):
        """The matrix-free EFC law for a dark-zone model.

        Args:
            model: The ``DarkZoneModel`` (dense ``g_dz`` not required).
            jacobian_field: The entrance field to linearize on.
            gain: Loop gain.
            regularization: Positive Tikhonov term.
            max_iterations: Conjugate-gradient iteration cap per solve.
            tol: Conjugate-gradient relative tolerance.
            operating_point: Initial linearization command; defaults to the
                model's.

        Returns:
            A ``MatrixFreeEFCController``.
        """
        if regularization <= 0.0:
            raise ValueError(f"regularization must be positive, got {regularization}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        command = model.operating_point if operating_point is None else operating_point
        return cls(
            model=model,
            field=jacobian_field,
            command=jnp.asarray(command),
            gain=jnp.asarray(gain),
            regularization=float(regularization),
            max_iterations=int(max_iterations),
            tol=float(tol),
        )

    def relinearize(self, command):
        """The same law linearized at ``command``."""
        return eqx.tree_at(lambda c: c.command, self, jnp.asarray(command))

    def _stacked(self, command):
        data = self.model.focal_of(command, self.field)
        weighted = self.model.weighted_dark_zone(data)
        return jnp.concatenate([jnp.real(weighted), jnp.imag(weighted)])

    def command_delta(self, estimate):
        """Solve the regularized normal equations by CG on jvp/vjp products."""
        _, jvp_fn = jax.linearize(self._stacked, self.command)
        vjp_fn = jax.linear_transpose(jvp_fn, self.command)

        def normal(v):
            return vjp_fn(jvp_fn(v))[0] + self.regularization * v

        weighted = self.model.stack_weights * estimate.reshape(-1)
        residual = jnp.concatenate([jnp.real(weighted), jnp.imag(weighted)])
        rhs = -vjp_fn(residual)[0]
        delta, _ = jax.scipy.sparse.linalg.cg(
            normal, rhs, maxiter=self.max_iterations, tol=self.tol
        )
        return self, self.gain * delta


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
    pixel_weights=None,
    jacobian="dense",
    cg_iterations=100,
    cg_tol=1e-6,
):
    """Dig a dark hole with deformable mirrors by electric-field conjugation.

    By default linearizes the focal field with respect to the stacked DM command
    ONCE (the control Jacobian is constant to first order in the small-signal
    dark-hole regime), builds a regularized real control matrix, then runs a
    differentiable ``lax.scan`` that RE-PROPAGATES for the measurement and updates
    the command each step by swapping every DM's coefficients with
    ``eqx.tree_at`` -- never a reconstruction, which would re-run the propagator's
    construction-time gates. Because every propagation is a pure function, the
    loop differentiates through the feedback (e.g. the final contrast with
    respect to the loop gain). ``jacobian="matrix-free"`` replaces the stored
    control matrix with :class:`MatrixFreeEFCController` (re-linearized every
    step, nothing of Jacobian size stored) for mirrors too large to
    materialize; that loop is Python-unrolled.

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
        pixel_weights: Optional per-dark-zone-pixel weights passed through to
            :meth:`DarkZoneModel.build`, so the loop digs where the weight is
            rather than uniformly across the mask. Uniform by default.
        jacobian: ``"dense"`` (the ``jacfwd`` control matrix, built once) or
            ``"matrix-free"`` (:class:`MatrixFreeEFCController`: CG on
            autodiff products, re-linearized at every step, nothing of
            Jacobian size stored). Matrix-free loops are Python-unrolled and
            do not support the Kalman estimator (it needs the dense model).
        cg_iterations: Matrix-free only: CG iteration cap per solve.
        cg_tol: Matrix-free only: CG relative tolerance.

    Returns:
        ``(command, dark_zone_history)``: the final stacked DM command (the DMs'
        coefficients concatenated in ``dm_indices`` order) and the mean dark-zone
        intensity at each iteration.

    Raises:
        TypeError: If any ``dm_indices`` stage is not a ``PhaseScreen``.
        ValueError: If ``regularization`` is not positive, the dark zone is
            empty, ``estimator`` or ``jacobian`` is unknown, an estimated loop
            is missing ``probes``, or the Kalman estimator is combined with
            ``jacobian="matrix-free"``.
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
    if jacobian not in ("dense", "matrix-free"):
        raise ValueError(f"jacobian must be 'dense' or 'matrix-free', got {jacobian!r}")
    matrix_free = jacobian == "matrix-free"
    if matrix_free and estimator == "kalman":
        raise ValueError("the kalman estimator needs a dense Jacobian")
    # The control Jacobian is known from the model; the honest estimated loop
    # builds it on the unaberrated model field, not the (unknown) true field.
    jacobian_field = (
        model_field if (estimated and model_field is not None) else input_field
    )
    dz_model = DarkZoneModel.build(
        path,
        indices,
        dark_zone_mask,
        jacobian_field=jacobian_field,
        pixel_weights=pixel_weights,
        materialize_jacobian=not matrix_free,
    )
    if matrix_free:
        controller = MatrixFreeEFCController.build(
            dz_model,
            jacobian_field=jacobian_field,
            gain=gain,
            regularization=regularization,
            max_iterations=cg_iterations,
            tol=cg_tol,
        )
    else:
        controller = EFCController.build(
            dz_model, gain=gain, regularization=regularization
        )

    if matrix_free and not estimated:
        # The controller (and its model's boolean mask) is a closure constant,
        # not a jit argument, so the mask indexing stays concrete.
        @eqx.filter_jit
        def mf_step(command):
            data = dz_model.focal_of(command, input_field)
            estimate = dz_model.dark_zone_unweighted(data)
            _, delta = controller.relinearize(command).command_delta(estimate)
            return command + delta, dz_model.contrast(data)

        command = jnp.zeros(dz_model.n_total)
        history = []
        for _ in range(n_steps):
            command, contrast = mf_step(command)
            history.append(contrast)
        return command, jnp.stack(history)

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
            if matrix_free:
                controller = controller.relinearize(command)
            controller, delta = controller.command_delta(e_hat)
            command = command + delta
        return command, jnp.stack(history)

    def step(command, _):
        data = dz_model.focal_of(command, input_field)  # ONE oracle read
        estimate = dz_model.dark_zone_unweighted(data)
        _, delta = controller.command_delta(estimate)
        return command + delta, dz_model.contrast(data)

    return jax.lax.scan(step, jnp.zeros(dz_model.n_total), None, length=n_steps)
