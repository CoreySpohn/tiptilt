"""Focal-plane wavefront sensing: pairwise-probe electric-field estimation.

A coronagraph detector measures intensity, not the complex focal field the
control loop needs. Pairwise probing recovers the field by applying equal
positive and negative deformable-mirror probes: the difference image cancels
the quadratic terms and the unknown incoherent bias, leaving the linear
cross-term ``dI_j = 4 Re{conj(E) p_j}`` (Give'on 2011, Groff 2016). Stacking a
few probes with independent quadratures and solving the per-pixel least squares
gives the complex field, replacing the ``jax.jacfwd``-on-truth shortcut that a
real system cannot take.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from physicaloptix import Field, PlaneKind


def probe_set(basis, *, amplitude_nm, n_probes=3, seed=0):
    """Deformable-mirror probe commands for pairwise field estimation.

    Each probe is a fixed pseudo-random combination of the deformable mirror's
    modes, scaled so its wavefront error is ``amplitude_nm`` RMS. A random
    combination of the band-limited control modes spreads the probe field over
    the whole controllable region, and several probes give the per-pixel
    quadrature diversity the batch solve needs (``n_probes >= 2``, three or more
    to condition the inverse). This is the simplest probe that covers an annular
    dark hole uniformly; localized sinc probes (Give'on 2011) are a later
    signal-to-noise refinement.

    Args:
        basis: The deformable mirror's ``ModeBasis`` (its ``B`` sets the OPD
            scale of a unit coefficient).
        amplitude_nm: Per-probe RMS wavefront amplitude in nanometres. Small
            enough to stay in the linear regime (a few nanometres) yet bright
            enough for signal to noise.
        n_probes: Number of probes (each applied as a positive and a negative
            pair).
        seed: Base PRNG seed, so a probe set is deterministic.

    Returns:
        A list of ``n_probes`` command vectors (each shape ``(n_modes,)``).
    """
    modes = basis.B
    n_modes = modes.shape[0]
    probes = []
    for key in jax.random.split(jax.random.PRNGKey(seed), n_probes):
        coeffs = jax.random.normal(key, (n_modes,))
        opd = jnp.tensordot(coeffs, modes, axes=1)
        rms = jnp.sqrt(jnp.mean(opd**2))
        probes.append(coeffs * (amplitude_nm / rms))
    return probes


def _with_command(path, dm_index, command):
    """A copy of ``path`` with deformable-mirror ``dm_index`` set to ``command``."""
    return eqx.tree_at(lambda p: p.stages[dm_index].op.basis.coeffs, path, command)


def _focal_field(path, field, dm_index, command):
    """The complex focal field for a given deformable-mirror command."""
    out, _ = _with_command(path, dm_index, command).propagate(field)
    return out.data


def _mask_dark_zone(data, mask):
    """Flatten a focal array over the dark zone, chromatic-stack aware.

    ``(ny, nx) -> (n_dark,)`` and ``(nlam, ny, nx) -> (nlam, n_dark)``.
    """
    return data[mask] if data.ndim == 2 else data[:, mask]


def probe_measurement(
    path,
    input_field,
    model_field,
    dm_index,
    probe,
    dark_zone_mask,
    *,
    detector=None,
    key=None,
):
    """One probe's model field and measured difference image over a dark zone.

    Applies the probe as a positive and negative deformable-mirror pair around
    the deformable mirror's current command, reads the two focal images off the
    true ``input_field`` (optionally through ``detector``), and computes the
    symmetric model probe field ``(f(+) - f(-))/2`` from ``model_field``. The
    symmetric form cancels the even (quadratic) deformable-mirror nonlinearity,
    matching the symmetric difference image.

    Returns:
        ``(probe_field, diff_image)``: complex model field and real ``I_+ - I_-``,
        both flattened over ``dark_zone_mask`` to ``(n_dark,)``, or ``(nlam,
        n_dark)`` for a chromatic field (one row per wavelength / sub-band).
    """
    command = path.stages[dm_index].op.basis.coeffs
    e_plus = _focal_field(path, input_field, dm_index, command + probe)
    e_minus = _focal_field(path, input_field, dm_index, command - probe)
    i_plus = jnp.abs(e_plus) ** 2
    i_minus = jnp.abs(e_minus) ** 2
    if detector is not None:
        key_plus, key_minus = jax.random.split(key)
        i_plus = detector(i_plus, key_plus)
        i_minus = detector(i_minus, key_minus)
    diff = _mask_dark_zone(i_plus - i_minus, dark_zone_mask)
    model_plus = _focal_field(path, model_field, dm_index, command + probe)
    model_minus = _focal_field(path, model_field, dm_index, command - probe)
    probe_field = 0.5 * _mask_dark_zone(model_plus - model_minus, dark_zone_mask)
    return probe_field, diff


def estimate_field_pairwise(
    path,
    input_field,
    dm_index,
    probes,
    dark_zone_mask,
    *,
    model_field=None,
    command=None,
    detector=None,
    key=None,
    regularization=0.0,
):
    """Pairwise-probe estimate of the dark-zone field through an optical path.

    Applies each probe as a positive and negative deformable-mirror pair, reads
    the two focal images off the TRUE ``input_field`` (optionally through a
    ``detector`` for noise), and computes each probe's added field from the
    ``model_field`` (the design pupil, no aberration -- keeping the estimate
    honest, since a real system has only the model, not the true field). The
    per-pixel batch least squares then returns the complex dark-zone field.

    Args:
        path: The ``OpticalPath`` ending at the focal plane; ``dm_index`` is the
            probe deformable mirror (a ``PhaseScreen``).
        input_field: The true entrance field (carries the unknown aberration).
        dm_index: Stage index of the probe deformable mirror.
        probes: Probe command vectors (see :func:`probe_set`).
        dark_zone_mask: Boolean focal-plane region to estimate.
        model_field: The design entrance field for the model probe response;
            defaults to ``input_field`` (an oracle model).
        command: Current deformable-mirror command the probes are applied
            around; defaults to zeros.
        detector: Optional ``callable(image, key) -> image`` applying noise (in
            normalized-intensity units) to each focal image.
        key: PRNG key for the detector (split per image); required if
            ``detector`` is given.
        regularization: Tikhonov term for the per-pixel solve.

    Returns:
        Complex field estimate over the masked pixels, shape ``(n_dark,)``, or
        ``(nlam, n_dark)`` for a chromatic ``input_field`` -- one independent
        sub-band estimate per wavelength (the per-pixel solve treats each
        wavelength/pixel as its own unknown).
    """
    if command is not None:
        path = _with_command(path, dm_index, command)  # probe around this command
    if model_field is None:
        model_field = input_field
    keys = (
        list(jax.random.split(key, len(probes)))
        if detector is not None
        else [None] * len(probes)
    )

    probe_fields, diff_images = [], []
    for probe, probe_key in zip(probes, keys, strict=True):
        probe_field, diff = probe_measurement(
            path,
            input_field,
            model_field,
            dm_index,
            probe,
            dark_zone_mask,
            detector=detector,
            key=probe_key,
        )
        probe_fields.append(probe_field)
        diff_images.append(diff)

    # Fold any wavelength axis into the per-pixel solve, then restore its shape:
    # (n_probes, n_dark) stays (n_dark,); (n_probes, nlam, n_dark) -> (nlam, n_dark).
    stacked_fields = jnp.stack(probe_fields)
    stacked_diffs = jnp.stack(diff_images)
    trailing = stacked_fields.shape[1:]
    estimate = pairwise_estimate(
        stacked_fields.reshape(stacked_fields.shape[0], -1),
        stacked_diffs.reshape(stacked_diffs.shape[0], -1),
        regularization=regularization,
    )
    return estimate.reshape(trailing)


def pairwise_estimate(probe_fields, diff_images, *, regularization=0.0):
    """Batch pairwise-probe estimate of the complex focal field per pixel.

    Solves ``dI_j = 4 Re{conj(E) p_j}`` for ``E`` at every pixel. With
    ``x = [Re E; Im E]`` and observation matrix ``H = 4 [Re p_j, Im p_j]`` the
    per-pixel normal equations are ``x = (H^T H + reg I)^{-1} H^T z``. At least
    two probes with linearly independent ``(Re p, Im p)`` directions are needed
    for ``H`` to have rank two; a single probe recovers only one quadrature.

    Args:
        probe_fields: Complex model probe fields, shape ``(n_probes, n_pixels)``
            (the field each probe adds at the estimated pixels).
        diff_images: Measured probe-difference intensities ``I_+ - I_-``, shape
            ``(n_probes, n_pixels)``.
        regularization: Non-negative Tikhonov term stabilizing the per-pixel
            inverse (helps where the probe directions are near-degenerate).

    Returns:
        Complex field estimate, shape ``(n_pixels,)``.
    """
    # H per pixel: (n_pixels, n_probes, 2), stacking the real and imag rows.
    h_rows = 4.0 * jnp.stack([jnp.real(probe_fields), jnp.imag(probe_fields)], axis=-1)
    h = jnp.moveaxis(h_rows, 1, 0)  # (n_pixels, n_probes, 2)
    z = jnp.moveaxis(diff_images, 1, 0)[..., jnp.newaxis]  # (n_pixels, n_probes, 1)

    gram = jnp.einsum("kna,knb->kab", h, h)  # (n_pixels, 2, 2)
    rhs = jnp.einsum("kna,knb->kab", h, z)  # (n_pixels, 2, 1)
    reg = regularization * jnp.eye(2)
    x = jnp.linalg.solve(gram + reg, rhs)[..., 0]  # (n_pixels, 2)
    return x[:, 0] + 1j * x[:, 1]


class KalmanFieldEstimator(eqx.Module):
    """Recursive pairwise-probe field estimator (one probe pair per iteration).

    The batch estimator needs a full probe set every iteration because it keeps
    no memory. A Kalman filter carries the field estimate and its covariance
    forward through the control history, so the rank the batch needed all at
    once is instead accumulated over time and a single probe pair per iteration
    suffices (Groff 2016). The per-pixel state is ``[Re E; Im E]`` with a 2x2
    covariance; ``update`` runs a predict step (an optional known field change
    from the last deformable-mirror command, plus process noise) and a
    measurement update from one probe pair.

    Attributes:
        state: Per-pixel ``[Re E; Im E]``, shape ``(n_pixels, 2)``.
        covariance: Per-pixel error covariance, shape ``(n_pixels, 2, 2)``.
        process_noise: Scalar process-noise variance (drift between steps).
        measurement_noise: Scalar measurement-noise variance (detector).
    """

    state: jax.Array
    covariance: jax.Array
    process_noise: float = eqx.field(static=True)
    measurement_noise: float = eqx.field(static=True)

    @classmethod
    def init(cls, n_pixels, *, initial_variance, process_noise, measurement_noise):
        """A diffuse prior: zero field, ``initial_variance`` on each quadrature."""
        return cls(
            state=jnp.zeros((n_pixels, 2)),
            covariance=jnp.tile(initial_variance * jnp.eye(2), (n_pixels, 1, 1)),
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )

    @property
    def field(self):
        """The current complex field estimate, shape ``(n_pixels,)``."""
        return self.state[:, 0] + 1j * self.state[:, 1]

    def update(self, probe_field, diff_image, *, field_change=None):
        """One predict-and-update step from a single probe pair.

        Args:
            probe_field: The probe's model field, shape ``(n_pixels,)`` complex.
            diff_image: The measured difference ``I_+ - I_-``, shape
                ``(n_pixels,)``.
            field_change: Known field change since the last step (the applied
                deformable-mirror delta propagated to the focal plane); defaults
                to none.

        Returns:
            A new ``KalmanFieldEstimator`` with the updated state and covariance.
        """
        n_pixels = self.state.shape[0]
        if field_change is None:
            shift = jnp.zeros((n_pixels, 2))
        else:
            shift = jnp.stack([jnp.real(field_change), jnp.imag(field_change)], axis=-1)
        x_pred = self.state + shift
        p_pred = self.covariance + self.process_noise * jnp.eye(2)

        # Measurement: z = H x + noise, H = 4 [Re p, Im p] (one row per pixel).
        h = 4.0 * jnp.stack([jnp.real(probe_field), jnp.imag(probe_field)], axis=-1)
        h = h[:, jnp.newaxis, :]  # (n_pixels, 1, 2)
        z = diff_image[:, jnp.newaxis]  # (n_pixels, 1)

        ph_t = jnp.einsum("kij,klj->kil", p_pred, h)  # (n_pixels, 2, 1)
        s = jnp.einsum("kij,kjl->kil", h, ph_t) + self.measurement_noise  # (k,1,1)
        gain = ph_t / s  # (n_pixels, 2, 1)
        innovation = z - jnp.einsum("kij,kj->ki", h, x_pred)  # (n_pixels, 1)
        x_new = x_pred + gain[..., 0] * innovation  # (n_pixels, 2)
        kh = jnp.einsum("kil,klj->kij", gain, h)  # (n_pixels, 2, 2)
        p_new = jnp.einsum("kij,kjl->kil", jnp.eye(2) - kh, p_pred)
        return eqx.tree_at(lambda e: (e.state, e.covariance), self, (x_new, p_new))


def zwfs_calibrate(sensor, aperture_field, mode_basis, *, wavelength_nm):
    """Reference image and per-mode interaction matrix of a Zernike WFS.

    Linearizes the sensor image about the flat wavefront: pushes each mode of
    ``mode_basis`` through ``sensor`` (via ``jax.jacfwd``) and stacks the
    intensity response, so the low-order phase is recovered by inverting the
    interaction matrix. This is the standard low-order-WFS calibration.

    Args:
        sensor: A ``ZernikeWavefrontSensor``.
        aperture_field: The flat (unaberrated) pupil field.
        mode_basis: The low-order ``ModeBasis`` to sense (drop piston).
        wavelength_nm: Wavelength for the OPD-to-phase conversion.

    Returns:
        ``(reference_image, interaction)``: the flat-wavefront sensor image and
        the ``(n_pixels, n_modes)`` intensity response matrix.
    """
    modes = mode_basis.B
    n_modes = modes.shape[0]
    aperture = aperture_field.data
    grid = aperture_field.grid
    spectrum = aperture_field.spectrum

    def image_of(coeffs):
        opd = jnp.tensordot(coeffs, modes, axes=1)
        field = Field(
            data=aperture * jnp.exp(1j * 2.0 * jnp.pi * opd / wavelength_nm),
            grid=grid,
            plane=PlaneKind.PUPIL,
            spectrum=spectrum,
        )
        return jnp.abs(sensor(field).data) ** 2

    reference = image_of(jnp.zeros(n_modes))
    jac = jax.jacfwd(image_of)(jnp.zeros(n_modes))  # (npix, npix, n_modes)
    return reference, jac.reshape(-1, n_modes)


def zwfs_reconstruct(image, reference, interaction, *, regularization=0.0):
    """Least-squares low-order coefficients from a Zernike-WFS image.

    Inverts the calibrated interaction matrix: solves
    ``(Z^T Z + reg I) c = Z^T (image - reference)`` for the mode coefficients.

    Args:
        image: The measured sensor image.
        reference: The flat-wavefront reference image from :func:`zwfs_calibrate`.
        interaction: The ``(n_pixels, n_modes)`` interaction matrix.
        regularization: Non-negative Tikhonov term for the inverse.

    Returns:
        The reconstructed mode coefficients, shape ``(n_modes,)``.
    """
    diff = (image - reference).reshape(-1)
    n_modes = interaction.shape[1]
    gram = interaction.T @ interaction + regularization * jnp.eye(n_modes)
    return jnp.linalg.solve(gram, interaction.T @ diff)


class AbstractEstimator(eqx.Module):
    """The estimation seam: a dark-zone field estimate per control step.

    ``estimate(model, command, key) -> (new_self, e_hat)`` where ``model`` is
    the loop's ``DarkZoneModel``, ``command`` the current stacked DM command,
    and ``e_hat`` the UNWEIGHTED complex dark-zone estimate (``(n_dark,)``
    mono, ``(nlam, n_dark)`` broadband). Stateless estimators return
    themselves unchanged; recursive ones (Kalman) advance their carried
    state. The estimated drivers unroll in Python, so estimators may hold
    plain-Python bookkeeping.
    """

    def estimate(self, model, command, *, key=None):
        """One measurement of the dark-zone field at the given command."""
        raise NotImplementedError


class OracleEstimator(AbstractEstimator):
    """Perfect knowledge: read the true field by re-propagation.

    The achievable-contrast reference every honest estimator is compared
    against (a real system cannot take this shortcut).

    Attributes:
        input_field: The true entrance field.
    """

    input_field: eqx.Module

    def estimate(self, model, command, *, key=None):
        """Propagate the true field and read the dark zone."""
        data = model.focal_of(command, self.input_field)
        return self, model.dark_zone_unweighted(data)


class PairwiseEstimator(AbstractEstimator):
    """Batch pairwise probing: a full probe set every step. Stateless.

    Wraps :func:`estimate_field_pairwise` on the seam: each step applies the
    probe set as positive/negative pairs around the current command, reads
    the difference images off the TRUE field (optionally through a
    detector), and solves the per-pixel least squares against the DESIGN
    model's probe fields.

    Attributes:
        input_field: The true entrance field (measurements).
        model_field: The design entrance field (probe model).
        probes: Probe command vectors.
        probe_dm: Stage index of the probe mirror.
        detector: Optional noise model ``callable(image, key) -> image``.
        regularization: Tikhonov term of the per-pixel solve.
    """

    input_field: eqx.Module
    model_field: eqx.Module
    probes: tuple
    probe_dm: int = eqx.field(static=True)
    detector: object = eqx.field(static=True, default=None)
    regularization: float = eqx.field(static=True, default=0.0)

    def estimate(self, model, command, *, key=None):
        """A batch pairwise estimate at the current command."""
        e_hat = estimate_field_pairwise(
            model.set_commands(command),
            self.input_field,
            self.probe_dm,
            list(self.probes),
            model.mask,
            model_field=self.model_field,
            detector=self.detector,
            key=key,
            regularization=self.regularization,
        )
        return self, e_hat


class KalmanEstimator(AbstractEstimator):
    """Recursive estimation: one probe pair per step, rank over time.

    Wraps :class:`KalmanFieldEstimator` on the seam. Each step predicts the
    field change from the applied command delta through the model Jacobian
    (unweighted), measures ONE probe pair, and updates the per-pixel filter.
    The carried state is the filter plus the last command and the probe
    cycle position.

    Attributes:
        filter: The per-pixel Kalman filter.
        probes: Probe command vectors, cycled one per step.
        step_index: Position in the probe cycle.
        last_command: The command the previous estimate was made at.
        input_field: The true entrance field.
        model_field: The design entrance field.
        probe_dm: Stage index of the probe mirror.
        detector: Optional noise model.
    """

    filter: KalmanFieldEstimator
    probes: tuple
    step_index: Array
    last_command: Array
    input_field: eqx.Module
    model_field: eqx.Module
    probe_dm: int = eqx.field(static=True)
    detector: object = eqx.field(static=True, default=None)

    @classmethod
    def build(
        cls,
        model,
        *,
        input_field,
        model_field,
        probes,
        probe_dm,
        detector=None,
        initial_variance=1.0,
        process_noise=1e-12,
        measurement_noise=1e-16,
    ):
        """A fresh filter sized to the model's stacked dark zone.

        The default hyperparameters suit a near-noiseless model: trust the
        measurements (small R) but keep a little process noise so the filter
        stays responsive to residual model error as the hole digs.

        Args:
            model: The loop's ``DarkZoneModel``.
            input_field: The true entrance field.
            model_field: The design entrance field.
            probes: Probe command vectors (cycled).
            probe_dm: Stage index of the probe mirror.
            detector: Optional noise model.
            initial_variance: Diffuse prior variance per quadrature.
            process_noise: Per-step process-noise variance.
            measurement_noise: Detector-noise variance.

        Returns:
            A ``KalmanEstimator``.
        """
        return cls(
            filter=KalmanFieldEstimator.init(
                model.stack_weights.shape[0],
                initial_variance=initial_variance,
                process_noise=process_noise,
                measurement_noise=measurement_noise,
            ),
            probes=tuple(probes),
            step_index=jnp.asarray(0),
            last_command=jnp.zeros(model.n_total),
            input_field=input_field,
            model_field=model_field,
            probe_dm=probe_dm,
            detector=detector,
        )

    def estimate(self, model, command, *, key=None):
        """Predict through the command delta, update from one probe pair."""
        # The filter state is the UNWEIGHTED per-(wavelength, pixel) field, so
        # predict with the unweighted field change; the sqrt-weighting enters
        # only in the controller's residual.
        g_unweighted = model.g_dz / model.stack_weights[:, jnp.newaxis]
        field_change = g_unweighted @ (command - self.last_command)
        probe = self.probes[int(self.step_index) % len(self.probes)]
        probe_field, diff = probe_measurement(
            model.set_commands(command),
            self.input_field,
            self.model_field,
            self.probe_dm,
            probe,
            model.mask,
            detector=self.detector,
            key=key,
        )
        new_filter = self.filter.update(
            probe_field.reshape(-1), diff.reshape(-1), field_change=field_change
        )
        new_self = eqx.tree_at(
            lambda e: (e.filter, e.step_index, e.last_command),
            self,
            (new_filter, self.step_index + 1, command),
        )
        return new_self, new_filter.field
