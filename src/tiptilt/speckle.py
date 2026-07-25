"""The tiptilt speckle-field family: aberration realizations as residuals.

These implement optixstuff's ``AbstractSpeckleField`` on top of the
``(E_nom, G)`` linearization a propagation engine provides, differing only in
how they source the drifting mode coefficients ``eps(t)``.

``correlated_drift_field`` covers stationary drift with a target cross-mode
covariance (segment or Zernike statistics coupled by a screen), which the
per-mode-independent ``SpeckleProcess`` cannot: it synthesizes a correlated
spectral realization and reuses physicaloptix's analytic cosine-sum field. Its
covariance guarantee is ensemble-mean, so it fits broadband, whitish drift; a
single realization of a red PSD is unrepresentative (see the function docstring).

``TabulatedSpeckleField`` replays a precomputed coefficient trajectory. It is
the escape hatch for drift a stationary spectral synthesis cannot represent --
an autoregressive or random-walk process (variance growing in time), or any
sampled time series (a STOP thermal run) -- which a finite sum of cosines with
bounded, time-constant variance structurally cannot hold. The realization is
fixed at construction and interpolated by elapsed time, so it stays
deterministic and differentiable, as the contract requires.

The trajectory builders fill that table. :func:`ou_trajectory` is the exact
stationary process (per-mode Ornstein-Uhlenbeck, autocorrelation exactly
``exp(-lag/tau_k)`` at every step size), and it is the right choice over the
cosine-sum builders whenever long lags matter: a finite line sum is
almost-periodic, so its kernel revives rather than decaying.
:func:`random_walk_trajectory` and :func:`creep_trajectory` cover the
non-stationary drift the stationary family cannot express at all -- variance
growing in time, and one-sided creep with a skewed marginal --
and :func:`compose_trajectories` sums them into the single table the field
replays::

    times_s = jnp.arange(0.0, 8 * 3600.0, 60.0)
    eps = compose_trajectories(
        ou_trajectory(covariance_nm2, timescales_s, key=key, times_s=times_s),
        creep_trajectory(rates_nm_per_s, times_s=times_s),
    )
    field = TabulatedSpeckleField(e_nom, G, times_s, eps, normalization)
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from optixstuff.speckle import AbstractSpeckleField
from physicaloptix import AnalyticSpeckleField, lambda_scaled_channels
from physicaloptix.speckle import _check_chromatic_layout, _select_channel

J2000_JD = 2451545.0

# Below this effective frequency count a single frozen realization's modal
# covariance is an unreliable sample of the target (scatter ~ 1/sqrt(N_eff)).
_NEFF_WARN = 8.0


def _line_powers(frequencies_hz, psd, df_weighted):
    """Per-line temporal power from a PSD sampled on a frequency grid.

    ``psd`` is a power spectral DENSITY, so the power carried by line ``j`` is
    ``S(f_j) df_j`` for the grid's quadrature widths, not the bare ordinate
    ``S(f_j)``. The distinction is invisible on a uniform grid (the widths are
    a constant that normalizes away) and decisive on a log grid, where using
    the ordinate synthesizes ``S(f) / f`` instead of ``S(f)`` and the field
    decorrelates far too slowly.

    Args:
        frequencies_hz: Frequency grid, shape ``(f,)``, increasing.
        psd: Density sampled on that grid, shape ``(f,)``.
        df_weighted: Apply the trapezoid widths. ``False`` reproduces the
            pre-2026-07-25 behavior.

    Returns:
        The unnormalized per-line powers, a numpy array of shape ``(f,)``.

    Raises:
        ValueError: If the shapes disagree or the total power is not positive.
    """
    f = np.asarray(frequencies_hz, dtype=float)
    power = np.asarray(psd, dtype=float)
    if f.shape != power.shape:
        raise ValueError(
            f"frequencies_hz {f.shape} and psd {power.shape} must have the same shape"
        )
    if df_weighted and f.size > 1:
        half = 0.5 * np.diff(f)
        df = np.concatenate([[half[0]], half[1:] + half[:-1], [half[-1]]])
        power = power * df
    if float(power.sum()) <= 0.0:
        raise ValueError("psd must have positive total power")
    return power


def _psd_sqrt(matrix):
    """A square root of a symmetric positive-semidefinite matrix.

    An eigendecomposition (``A = V diag(sqrt(max(lambda, 0)))``) rather than a
    Cholesky factorization, so a rank-deficient covariance works: the modal
    statistics that matter here (a screen coupling a few segment modes, a
    covariance clipped to its leading eigenvectors) are routinely singular.

    Args:
        matrix: Symmetric positive-semidefinite ``(m, m)`` array.

    Returns:
        An ``(m, m)`` array ``A`` with ``A A^T`` equal to ``matrix``.
    """
    eigvals, eigvecs = jnp.linalg.eigh(matrix)
    return eigvecs * jnp.sqrt(jnp.maximum(eigvals, 0.0))


def _check_covariance(covariance_nm2, name="covariance_nm2"):
    """Validate a modal covariance and return it as a numpy array.

    Args:
        covariance_nm2: Candidate ``(m, m)`` covariance.
        name: Argument name to quote in the error messages.

    Returns:
        The validated covariance as a numpy array.

    Raises:
        ValueError: If it is not square, not symmetric, or has a materially
            negative eigenvalue.
    """
    covariance = np.asarray(covariance_nm2)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be square (m, m), got {covariance.shape}")
    if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-6):
        raise ValueError(f"{name} must be symmetric")
    eigvals = np.linalg.eigvalsh(covariance)
    tol = 1e-8 * max(float(np.abs(eigvals).max()), 1.0)
    if eigvals.min() < -tol:
        raise ValueError(
            f"{name} must be positive semidefinite; min eigenvalue {eigvals.min():.3e}"
        )
    return covariance


def _check_timescales(timescales_s, n_modes):
    """Broadcast per-mode decorrelation timescales to ``(m,)`` and validate.

    Args:
        timescales_s: Scalar (shared by every mode) or ``(m,)`` timescales.
        n_modes: The number of modes the covariance carries.

    Returns:
        The timescales as a numpy array of shape ``(m,)``.

    Raises:
        ValueError: If the shape disagrees with ``n_modes`` or any timescale is
            not strictly positive.
    """
    tau = np.atleast_1d(np.asarray(timescales_s, dtype=float))
    if tau.size == 1:
        tau = np.full(n_modes, float(tau[0]))
    if tau.shape != (n_modes,):
        raise ValueError(
            f"timescales_s has shape {tau.shape}; expected a scalar or "
            f"({n_modes},) to match the covariance"
        )
    if not np.all(tau > 0.0):
        raise ValueError("timescales_s must be positive")
    return tau


def _check_times(times_s):
    """Validate a sample-time grid and return it as a numpy array.

    Args:
        times_s: Candidate sample times, shape ``(t,)``.

    Returns:
        The times as a numpy array.

    Raises:
        ValueError: If they are not 1D, empty, or not strictly ascending.
    """
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError(f"times_s must be a non-empty 1D array, got {times.shape}")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("times_s must be strictly ascending")
    return times


def _draw_correlated_spectrum(covariance_nm2, key, weights):
    """Draw a correlated spectral realization: per-mode ``(amplitudes, phases)``.

    Colours a circularly-symmetric complex-normal spectrum by a square root of
    the target modal covariance so that the synthesized coefficient process
    ``eps_k(t) = sum_j a_kj cos(2 pi f_j t + phi_kj)`` has equal-time covariance
    ``covariance_nm2`` when the temporal ``weights`` sum to 2. The square root is
    an eigendecomposition (``A = V diag(sqrt(max(lambda, 0)))``), robust to a
    rank-deficient, positive-semidefinite covariance where a Cholesky would fail.

    Args:
        covariance_nm2: Target ``(m, m)`` equal-time modal covariance in nm^2.
        key: A JAX PRNG key; the drawn realization is frozen into the result.
        weights: Per-frequency temporal weights, shape ``(f,)``, summing to 2.

    Returns:
        ``(amplitudes, phases)``, each a real ``(m, f)`` array.
    """
    covariance = jnp.asarray(covariance_nm2)
    factor = _psd_sqrt(covariance)
    n_modes = covariance.shape[0]
    n_freq = weights.shape[0]
    key_real, key_imag = jax.random.split(key)
    z = (
        jax.random.normal(key_real, (n_modes, n_freq))
        + 1j * jax.random.normal(key_imag, (n_modes, n_freq))
    ) / jnp.sqrt(2.0)
    c = jnp.sqrt(weights)[None, :] * (factor @ z)
    return jnp.abs(c), jnp.angle(c)


def correlated_drift_field(
    e_nom,
    G,
    covariance_nm2,
    *,
    key,
    frequencies_hz,
    psd,
    normalization,
    pixel_scale_lod=0.25,
    epoch_jd=J2000_JD,
    coherent=False,
    df_weighted=True,
):
    """Build a stationary speckle field with a target cross-mode covariance.

    Reuses physicaloptix's analytic cosine-sum field, but synthesizes its
    ``(amplitudes, phases)`` so the mode-coefficient process is a stationary
    Gaussian process whose ENSEMBLE equal-time modal covariance is
    ``covariance_nm2`` (nm^2), with ensemble power distributed over
    ``frequencies_hz`` in proportion to ``psd``. This supplies the cross-mode
    correlation (segment or Zernike statistics coupled by a screen) that the
    per-mode-independent ``SpeckleProcess`` cannot.

    The returned field is ONE frozen realization of that process. Its own
    time-averaged modal covariance is an unbiased but random sample of
    ``covariance_nm2`` with relative scatter ``~ 1/sqrt(N_eff)``, where
    ``N_eff = (sum w)^2 / sum(w^2)`` is the participation ratio of the per-line
    powers ``w`` (``psd * df`` under ``df_weighted``) -- set by the PSD SHAPE
    and the grid, not by the number of frequencies alone. A finite
    cosine sum is also almost-periodic: its lag covariance revives on ``~1/df``.
    So this is the right tool for broadband, whitish stationary drift (large
    ``N_eff``); for a red PSD (``N_eff`` near 1), a genuine autoregressive /
    random-walk process, or fidelity over long baselines, generate an explicit
    trajectory and use :class:`TabulatedSpeckleField` instead. A low ``N_eff``
    emits a warning. The coherent cross term needs float64 inputs (x64 enabled).

    Args:
        e_nom: Complex nominal focal field, shape ``(y, x)``.
        G: Complex sensitivity ``d(E_focal)/d(mode)``, shape ``(m, y, x)``.
        covariance_nm2: Target ``(m, m)`` real symmetric positive-semidefinite
            ensemble modal covariance in nm^2.
        key: A JAX PRNG key freezing the drawn realization.
        frequencies_hz: Temporal frequency grid, shape ``(f,)``.
        psd: Temporal power spectral density at each frequency, shape ``(f,)``
            (nonnegative, positive total; only its shape matters, it is
            renormalized).
        normalization: Intensity that maps to unit contrast.
        pixel_scale_lod: Native pixel scale in lambda/D per pixel.
        epoch_jd: Julian Date mapping to ``time_s = 0``. Default J2000.
        coherent: Include the pinning cross term. Default ``False``.
        df_weighted: Treat ``psd`` as a density and give line ``j`` the power
            ``psd[j] df_j`` (trapezoid widths of ``frequencies_hz``), so the
            realized temporal kernel is the PSD's transform. Default ``True``;
            a uniform grid is unaffected either way. Pass ``False`` to
            reproduce ensembles drawn before 2026-07-25.

    Returns:
        A physicaloptix ``AnalyticSpeckleField``.

    Raises:
        ValueError: If ``covariance_nm2`` is not square, not symmetric, or has a
            materially negative eigenvalue, or if ``psd`` has non-positive total.
    """
    _check_covariance(covariance_nm2)

    power = _line_powers(frequencies_hz, psd, df_weighted)
    total = float(power.sum())
    n_eff = total**2 / float((power**2).sum())
    if n_eff < _NEFF_WARN:
        warnings.warn(
            f"correlated_drift_field: effective frequency count N_eff={n_eff:.1f} "
            "is low, so a single frozen realization's modal covariance scatters by "
            "~1/sqrt(N_eff) around covariance_nm2 and the finite cosine sum revives "
            "on ~1/df. Use more frequencies, or TabulatedSpeckleField for red / "
            "long-baseline drift.",
            stacklevel=2,
        )

    weights = jnp.asarray(2.0 * power / total)
    amplitudes, phases = _draw_correlated_spectrum(covariance_nm2, key, weights)
    return AnalyticSpeckleField(
        e_nom,
        G,
        amplitudes,
        jnp.asarray(frequencies_hz),
        phases,
        normalization,
        pixel_scale_lod=pixel_scale_lod,
        epoch_jd=epoch_jd,
        coherent=coherent,
    )


def grouped_drift_field(
    e_nom,
    G,
    groups,
    *,
    key,
    normalization,
    pixel_scale_lod=0.25,
    epoch_jd=J2000_JD,
    coherent=False,
    df_weighted=True,
):
    """Stationary field whose mode BLOCKS drift on different timescales.

    :func:`correlated_drift_field` gives every mode one shared temporal PSD, so
    the cross-spectral density is forced to factorize as ``C_a x psd(f)``: the
    modes may be correlated in space and may carry different variances, but
    they all decorrelate on the same schedule. Real observatories do not work
    that way -- segment piston/tip/tilt rattles fast while a thermal bulk mode
    creeps -- and no single ``(covariance, psd)`` pair can express that.

    This builder takes a SEQUENCE of blocks, each with its own covariance, its
    own frequency grid, and its own PSD, and concatenates them along the mode
    axis (the pattern :func:`correlated_channel_fields` already uses for local
    non-common-path blocks). Blocks are drawn independently, so this supplies
    within-block spatial correlation on a per-block timescale; it deliberately
    does not create cross-block correlation, which would need a full
    cross-spectral density rather than a list.

    Args:
        e_nom: Complex nominal focal field, shape ``(y, x)``.
        G: Complex sensitivity ``d(E_focal)/d(mode)``, shape ``(m, y, x)``,
            whose mode axis is the blocks concatenated in order.
        groups: Sequence of ``(covariance_nm2, frequencies_hz, psd)``. Every
            block needs the same NUMBER of frequencies (they stack into one
            ``(m, f)`` array) but the grids and PSD shapes may differ.
        key: A JAX PRNG key freezing the drawn realization.
        normalization: Intensity that maps to unit contrast.
        pixel_scale_lod: Native pixel scale in lambda/D per pixel.
        epoch_jd: Julian Date mapping to ``time_s = 0``. Default J2000.
        coherent: Include the pinning cross term. Default ``False``.
        df_weighted: Treat each PSD as a density (see
            :func:`correlated_drift_field`). Default ``True``.

    Returns:
        A physicaloptix ``AnalyticSpeckleField`` over all blocks' modes. Its
        ``frequencies_hz`` is ``(f,)`` when every block shares one grid and
        ``(m, f)`` otherwise.

    Raises:
        ValueError: If ``groups`` is empty, the blocks' frequency counts
            disagree, or the block sizes do not sum to ``G``'s mode axis.
    """
    if len(groups) == 0:
        raise ValueError("groups must contain at least one (covariance, freqs, psd)")
    n_freq = np.asarray(groups[0][1]).shape[-1]

    amplitude_blocks, phase_blocks, freq_blocks = [], [], []
    for i, (covariance, freqs, psd) in enumerate(groups):
        freqs = np.asarray(freqs, dtype=float)
        if freqs.shape[-1] != n_freq:
            raise ValueError(
                f"group {i} has {freqs.shape[-1]} frequencies but group 0 has "
                f"{n_freq}; blocks stack into one (m, f) array so the counts "
                "must agree (resample the grids, or build separate fields)"
            )
        power = _line_powers(freqs, psd, df_weighted)
        weights = jnp.asarray(2.0 * power / float(power.sum()))
        amplitudes, phases = _draw_correlated_spectrum(
            covariance, jax.random.fold_in(key, i), weights
        )
        amplitude_blocks.append(amplitudes)
        phase_blocks.append(phases)
        freq_blocks.append(jnp.broadcast_to(jnp.asarray(freqs), amplitudes.shape))

    amplitudes = jnp.concatenate(amplitude_blocks, axis=0)
    phases = jnp.concatenate(phase_blocks, axis=0)
    if amplitudes.shape[0] != G.shape[0]:
        raise ValueError(
            f"the blocks supply {amplitudes.shape[0]} modes but G has "
            f"{G.shape[0]}; G's mode axis must be the blocks concatenated "
            "in order"
        )
    frequencies = jnp.concatenate(freq_blocks, axis=0)
    # Collapse to the shared 1D grid when every block agrees, so a
    # single-timescale call is indistinguishable from correlated_drift_field.
    if bool(jnp.all(frequencies == frequencies[:1])):
        frequencies = frequencies[0]
    return AnalyticSpeckleField(
        e_nom,
        G,
        amplitudes,
        frequencies,
        phases,
        normalization,
        pixel_scale_lod=pixel_scale_lod,
        epoch_jd=epoch_jd,
        coherent=coherent,
    )


def correlated_channel_fields(
    mcl,
    shared_covariance_nm2,
    *,
    key,
    frequencies_hz,
    psd,
    normalizations,
    local=None,
    epoch_jd=J2000_JD,
    coherent=True,
    df_weighted=True,
):
    """Correlated per-channel speckle fields from one shared-mode draw.

    Draws ONE correlated spectral realization of the SHARED modes (the
    segment/trunk drift every channel sees) and imprints it through each
    channel's own ``g_shared`` block, so the returned fields realize the
    exact cross-channel covariance ``G_a Sigma (G_b)^H`` -- the physics that
    makes differential imaging work. Optional per-channel LOCAL blocks
    (non-common-path drift) are drawn INDEPENDENTLY per channel and
    appended; they are the part no cross-channel difference removes.

    Normalization is PER CHANNEL (each channel's own reference peak,
    including its split fraction), so a real split ratio cancels in
    per-channel contrast. ``coherent=True`` by default -- the common-mode
    signal is a FIELD effect (the pinning cross term carries it), a
    deliberate divergence from the single-field default.

    Args:
        mcl: A ``physicaloptix.MultiChannelLinearization`` (the per-channel
            ``e_nom`` / ``g_shared`` blocks of one shared basis).
        shared_covariance_nm2: Target ``(m, m)`` shared-mode covariance.
        key: PRNG key freezing the shared draw (and seeding the local ones).
        frequencies_hz: Temporal frequency grid, shape ``(f,)``.
        psd: Temporal PSD shape over those frequencies.
        normalizations: Dict mapping EVERY channel name to the intensity
            that maps to unit contrast in that channel.
        local: Optional dict mapping a channel name to ``(g_local,
            covariance_local_nm2)`` -- that channel's independent
            non-common-path block.
        epoch_jd: Julian Date mapping to ``time_s = 0``.
        coherent: Include the pinning cross term. Default ``True``.
        df_weighted: Treat ``psd`` as a density (see
            :func:`correlated_drift_field`). Default ``True``.

    Returns:
        A dict mapping each channel name to an ``AnalyticSpeckleField``.

    Raises:
        ValueError: If a channel is missing a normalization, or the shared
            covariance fails ``correlated_drift_field``'s validity checks.
    """
    missing = [name for name in mcl.names if name not in normalizations]
    if missing:
        raise ValueError(
            f"normalizations missing for channel(s) {missing}; every channel "
            "needs its own reference peak (a split ratio changes it)"
        )
    power = _line_powers(frequencies_hz, psd, df_weighted)
    weights = jnp.asarray(2.0 * power / float(power.sum()))

    key_shared, key_locals = jax.random.split(key)
    shared_amp, shared_phase = _draw_correlated_spectrum(
        shared_covariance_nm2, key_shared, weights
    )

    fields = {}
    for i, name in enumerate(mcl.names):
        channel = mcl[name]
        g = channel.g_shared
        amplitudes, phases = shared_amp, shared_phase
        if local is not None and name in local:
            g_local, covariance_local = local[name]
            local_amp, local_phase = _draw_correlated_spectrum(
                covariance_local, jax.random.fold_in(key_locals, i), weights
            )
            g = jnp.concatenate([g, jnp.asarray(g_local)], axis=0)
            amplitudes = jnp.concatenate([shared_amp, local_amp], axis=0)
            phases = jnp.concatenate([shared_phase, local_phase], axis=0)
        fields[name] = AnalyticSpeckleField(
            channel.e_nom,
            g,
            amplitudes,
            jnp.asarray(frequencies_hz),
            phases,
            normalizations[name],
            pixel_scale_lod=channel.pixel_scale_lod,
            epoch_jd=epoch_jd,
            coherent=coherent,
        )
    return fields


class TabulatedSpeckleField(AbstractSpeckleField):
    """Speckle field replaying a precomputed mode-coefficient trajectory.

    ``realize`` interpolates the tabulated ``eps`` at the requested elapsed
    time (holding the endpoints outside the sampled range) and returns the
    contrast delta ``(I(t) - |E_nom|^2) / normalization``, never the floor
    itself. With ``coherent=False`` (default) it returns the strictly positive
    incoherent halo ``|G eps|^2 / normalization``; with ``coherent=True`` it
    adds the pinning cross term via ``2 Re(E_nom* . G eps) + |G eps|^2``, the
    numerically stable form of ``|E_nom + G eps|^2 - |E_nom|^2`` (it avoids
    subtracting two floor-magnitude numbers), and needs the complex ``E_nom``.

    Monochromatic by default: ``realize`` then ignores ``wavelength_nm``.
    With ``wavelengths_nm`` set, ``e_nom`` / ``G`` (and optionally
    ``normalization``) carry a leading channel axis and ``realize`` selects
    the channel nearest the requested wavelength; the tabulated trajectory
    stays shared across channels (a wavefront error in nanometres is
    achromatic). Build the stacks per sub-band for an exact model, or via
    :meth:`broadened` for the lambda-scaling approximation. The cross term
    needs float64 inputs.
    """

    e_nom: Array  # complex (y, x) or (w, y, x): nominal focal field
    G: Array  # complex (m, y, x) or (w, m, y, x): d(E_focal)/d(mode)
    times_s: Array  # float (t,): ascending sample times in seconds
    eps_table: Array  # float (t, m): the coefficient trajectory
    normalization: Array
    pixel_scale_lod: float
    epoch_jd: float
    wavelengths_nm: Array | None
    coherent: bool = eqx.field(static=True)

    def __init__(
        self,
        e_nom,
        G,
        times_s,
        eps_table,
        normalization,
        *,
        pixel_scale_lod=0.25,
        epoch_jd=J2000_JD,
        coherent=False,
        wavelengths_nm=None,
    ):
        """Build a tabulated speckle field from a coefficient trajectory.

        Args:
            e_nom: Complex nominal focal field, shape ``(y, x)`` -- or
                ``(w, y, x)`` with ``wavelengths_nm`` set.
            G: Complex sensitivity ``d(E_focal)/d(mode)``, shape
                ``(m, y, x)`` -- or ``(w, m, y, x)`` with
                ``wavelengths_nm`` set.
            times_s: Ascending sample times in seconds, shape ``(t,)``.
            eps_table: Mode coefficients at each sample time, shape ``(t, m)``.
            normalization: Intensity that maps to unit contrast (the telescope
                PSF peak the focal field is referenced to); a scalar, or one
                value per channel for a chromatic field.
            pixel_scale_lod: Native pixel scale in lambda/D per pixel
                (shared by every channel: the maps live in lambda/D units).
            epoch_jd: Julian Date mapping to ``time_s = 0``. Default J2000.
            coherent: Include the pinning cross term. Default ``False``.
            wavelengths_nm: Channel wavelengths, shape ``(w,)``, enabling
                the chromatic layout above. ``None`` (default) for a
                monochromatic field.
        """
        self.e_nom = e_nom
        self.G = G
        self.times_s = times_s
        self.eps_table = eps_table
        self.normalization = jnp.asarray(normalization, dtype=float)
        self.pixel_scale_lod = pixel_scale_lod
        self.epoch_jd = epoch_jd
        self.wavelengths_nm = (
            None if wavelengths_nm is None else jnp.asarray(wavelengths_nm, dtype=float)
        )
        self.coherent = coherent

    def __check_init__(self):
        """Validate the trajectory shapes and the (chromatic) layout."""
        if self.eps_table.ndim != 2:
            raise ValueError(
                f"eps_table must be 2D (t, m), got shape {self.eps_table.shape}"
            )
        n_times, n_modes = self.eps_table.shape
        if self.times_s.shape != (n_times,):
            raise ValueError(
                f"times_s has shape {self.times_s.shape}; expected ({n_times},) "
                "to match eps_table's time axis"
            )
        _check_chromatic_layout(
            self.e_nom, self.G, self.normalization, self.wavelengths_nm
        )
        mode_axis = 0 if self.wavelengths_nm is None else 1
        if self.G.shape[mode_axis] != n_modes:
            raise ValueError(
                f"G has {self.G.shape[mode_axis]} modes but eps_table has {n_modes}"
            )

    def _eps(self, time_s):
        """Coefficients at ``time_s`` by per-mode linear interpolation, ``(m,)``."""
        t = jnp.asarray(time_s)
        return jax.vmap(lambda col: jnp.interp(t, self.times_s, col), in_axes=1)(
            self.eps_table
        )

    def eps(self, time_s):
        """The mode coefficients at an elapsed time, shape ``(m,)``.

        The public accessor a maintenance driver uses to inject this
        trajectory into a wavefront-error screen (the endpoints are held
        outside the sampled range, matching ``realize``).

        Args:
            time_s: Elapsed time in seconds.

        Returns:
            The interpolated coefficients in the basis's length unit.
        """
        return self._eps(time_s)

    def realize(self, *, wavelength_nm, time_s=0.0):
        """Speckle contrast delta at ``time_s`` (see class docstring)."""
        e_nom, g, normalization = _select_channel(
            self.e_nom, self.G, self.normalization, self.wavelengths_nm, wavelength_nm
        )
        g_eps = jnp.tensordot(self._eps(time_s), g, axes=1)
        if self.coherent:
            delta = 2.0 * jnp.real(jnp.conj(e_nom) * g_eps) + jnp.abs(g_eps) ** 2
        else:
            delta = jnp.abs(g_eps) ** 2
        return delta / normalization

    def broadened(self, *, reference_wavelength_nm, wavelengths_nm):
        """A chromatic copy under the lambda-scaling approximation.

        See ``physicaloptix.lambda_scaled_channels`` for the physics and
        its limits (``G`` scales as ``lambda0/lambda``; ``e_nom`` is held
        fixed; the lambda/D morphology is achromatic).

        Args:
            reference_wavelength_nm: The wavelength this field's ``G`` was
                generated at.
            wavelengths_nm: Channel wavelengths for the broadened field.

        Returns:
            A chromatic ``TabulatedSpeckleField`` sharing this field's
            trajectory.
        """
        if self.wavelengths_nm is not None:
            raise ValueError("field is already chromatic")
        e_stack, g_stack = lambda_scaled_channels(
            self.e_nom, self.G, reference_wavelength_nm, wavelengths_nm
        )
        return TabulatedSpeckleField(
            e_stack,
            g_stack,
            self.times_s,
            self.eps_table,
            self.normalization,
            pixel_scale_lod=self.pixel_scale_lod,
            epoch_jd=self.epoch_jd,
            coherent=self.coherent,
            wavelengths_nm=wavelengths_nm,
        )


def ou_covariance(covariance_nm2, timescales_s):
    """The equal-time modal covariance an OU trajectory actually realizes.

    Per-mode timescales and cross-mode correlation cannot both be imposed
    freely: two processes with different spectra cannot be perfectly
    correlated. Driving the modes with a common white noise of covariance
    ``C`` gives the stationary covariance ``C * O`` (elementwise), where

        O_kl = 2 sqrt(tau_k tau_l) / (tau_k + tau_l)

    is the overlap of the two Lorentzian spectra. The DIAGONAL is untouched
    (``O_kk = 1``), so every mode has exactly the requested variance and
    exactly the autocorrelation ``exp(-lag/tau_k)``; only the cross terms are
    damped, by how far apart the two timescales are (a decade of separation
    keeps 57 percent of the correlation, two decades 20 percent). ``O`` is a
    Gram matrix, so the product stays positive semidefinite (Schur) and the
    process is drawable.

    Use this wherever the realized covariance matters -- a PASTIS budget, a
    moment oracle, a mode-allocation inversion -- rather than assuming the
    trajectory delivers ``covariance_nm2`` itself.

    Args:
        covariance_nm2: Driving ``(m, m)`` modal covariance in nm^2.
        timescales_s: Per-mode decorrelation timescale, scalar or ``(m,)``.

    Returns:
        The realized ``(m, m)`` equal-time covariance in nm^2.

    Raises:
        ValueError: If the covariance or the timescales fail validation.
    """
    covariance = _check_covariance(covariance_nm2)
    tau = _check_timescales(timescales_s, covariance.shape[0])
    overlap = 2.0 * np.sqrt(np.outer(tau, tau)) / (tau[:, None] + tau[None, :])
    return jnp.asarray(covariance_nm2) * jnp.asarray(overlap)


def ou_trajectory(covariance_nm2, timescales_s, *, key, times_s):
    """Exact stationary drift trajectory with per-mode decorrelation times.

    Each mode is an Ornstein-Uhlenbeck process,
    ``d eps_k = -eps_k dt / tau_k + noise``, driven by a white noise with
    cross-mode covariance ``covariance_nm2``, sampled by its EXACT discrete
    transition (the AR(1) update ``eps <- a eps + L z`` with
    ``a_k = exp(-dt/tau_k)`` and ``L L^T = Sigma * (1 - a a^T)``). There is no
    time-discretization error at any step size, uniform or not, and the first
    sample is drawn from the stationary distribution, so there is no burn-in.

    This is the trajectory to reach for whenever long lags matter. The
    cosine-sum builders (:func:`correlated_drift_field`,
    :func:`grouped_drift_field`) synthesize a finite line spectrum, which is
    almost-periodic: past a few decorrelation times its kernel stops decaying
    and wanders, so decorrelation-limited quantities (post-processing floors
    at long lags, multi-epoch scheduling gains) inherit an artifact. The OU
    kernel is ``exp(-lag/tau_k)`` at every lag, by construction. Being exactly
    Gaussian, it also needs no renormalization and carries no excess kurtosis.

    The realized equal-time covariance is :func:`ou_covariance`, NOT
    ``covariance_nm2`` itself, whenever the timescales differ across modes;
    the diagonal is exact either way.

    Args:
        covariance_nm2: Driving ``(m, m)`` real symmetric positive-semidefinite
            modal covariance in nm^2 (equal to the realized covariance when
            every mode shares one timescale).
        timescales_s: Per-mode decorrelation timescale in seconds, a scalar
            (shared) or shape ``(m,)``. This is the ``1/e`` time of the modal
            autocorrelation, ``tau = 1 / (2 pi f_knee)`` against a Lorentzian
            knee frequency.
        key: A JAX PRNG key freezing the realization.
        times_s: Strictly ascending sample times in seconds, shape ``(t,)``.
            A non-uniform grid is exact too, at the cost of one matrix square
            root per step (``(t, m, m)`` of work, versus one factorization
            reused by a uniform grid).

    Returns:
        The mode-coefficient trajectory in nm, shape ``(t, m)``, ready to pass
        to :class:`TabulatedSpeckleField` alongside ``times_s``.

    Raises:
        ValueError: If the covariance, the timescales, or the time grid fail
            validation.
    """
    covariance = _check_covariance(covariance_nm2)
    n_modes = covariance.shape[0]
    tau = _check_timescales(timescales_s, n_modes)
    times = _check_times(times_s)

    sigma = ou_covariance(covariance_nm2, tau)
    inverse_tau = jnp.asarray(1.0 / tau)
    noise = jax.random.normal(key, (times.size, n_modes))
    start = _psd_sqrt(sigma) @ noise[0]
    if times.size == 1:
        return start[None, :]

    steps_s = jnp.asarray(np.diff(times))
    # 1 - a_k a_l via expm1 so a step much shorter than the timescales keeps
    # its precision (the naive difference cancels to nothing there).
    pair_rate = inverse_tau[:, None] + inverse_tau[None, :]

    def advance(eps, xs):
        decay, lower, z = xs
        moved = decay * eps + lower @ z
        return moved, moved

    decays = jnp.exp(-steps_s[:, None] * inverse_tau)  # (t - 1, m)
    if np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-9, atol=0.0):
        lower = _psd_sqrt(sigma * -jnp.expm1(-steps_s[0] * pair_rate))

        def advance_uniform(eps, z):
            return advance(eps, (decays[0], lower, z))

        _, moved = jax.lax.scan(advance_uniform, start, noise[1:])
    else:
        lowers = jax.vmap(lambda dt: _psd_sqrt(sigma * -jnp.expm1(-dt * pair_rate)))(
            steps_s
        )
        _, moved = jax.lax.scan(advance, start, (decays, lowers, noise[1:]))
    return jnp.concatenate([start[None, :], moved], axis=0)


def random_walk_trajectory(diffusion_nm2_per_s, *, key, times_s, start_nm=None):
    """Non-stationary random-walk drift: variance growing linearly in time.

    Correlated Brownian increments, so the coefficient covariance at elapsed
    time ``t`` is ``diffusion_nm2_per_s * t`` and grows without bound. This is
    the regime no stationary synthesis can represent -- a finite cosine sum has
    bounded, time-constant variance by construction -- and it is what an
    uncontrolled thermal or mechanical mode looks like between corrections.

    The wavefront-control literature usually quotes drift as an rms per
    control iteration (``sigma`` per ``sqrt(iteration)``); that maps here as
    ``diffusion = sigma^2 / dt_iteration``.

    Args:
        diffusion_nm2_per_s: ``(m, m)`` real symmetric positive-semidefinite
            covariance of the increment PER SECOND, in nm^2/s.
        key: A JAX PRNG key freezing the realization.
        times_s: Strictly ascending sample times in seconds, shape ``(t,)``.
        start_nm: Coefficients at the first sample time, shape ``(m,)``.
            Default ``None``, meaning start at the origin (the walk then
            carries the drift SINCE the reference state, which is what a
            residual after a dark-hole dig is).

    Returns:
        The mode-coefficient trajectory in nm, shape ``(t, m)``.

    Raises:
        ValueError: If the diffusion or the time grid fail validation.
    """
    diffusion = _check_covariance(diffusion_nm2_per_s, name="diffusion_nm2_per_s")
    n_modes = diffusion.shape[0]
    times = _check_times(times_s)
    origin = jnp.zeros(n_modes) if start_nm is None else jnp.asarray(start_nm)
    if times.size == 1:
        return jnp.broadcast_to(origin, (1, n_modes))

    factor = _psd_sqrt(jnp.asarray(diffusion_nm2_per_s))
    steps_s = jnp.asarray(np.diff(times))
    noise = jax.random.normal(key, (times.size - 1, n_modes))
    increments = jnp.sqrt(steps_s)[:, None] * (noise @ factor.T)
    walk = jnp.concatenate(
        [jnp.zeros((1, n_modes)), jnp.cumsum(increments, axis=0)], axis=0
    )
    return walk + origin


def creep_trajectory(rates_nm_per_s, *, times_s, key=None, rate_shape=None):
    """One-sided linear creep, optionally with a drawn rate.

    A deterministic ramp ``eps_k(t) = rate_k (t - t_0)`` -- the slow,
    monotone component of measured observatory drift, which sits alongside a
    faster stationary component rather than replacing it (compose the two with
    :func:`compose_trajectories`).

    With ``rate_shape`` the per-mode rate is drawn from a gamma distribution
    of that shape scaled to the requested mean, which keeps the sign of
    ``rates_nm_per_s`` in EVERY realization: the creep is one-sided, and the
    resulting coefficient marginal is skewed by ``2 / sqrt(rate_shape)``
    rather than Gaussian. That is the knob for the modal-skewness row of the
    speckle-statistics ledger, where the heterodyne variance term responds
    linearly to modal skewness and a symmetric process cannot probe it. Large
    ``rate_shape`` recovers the deterministic ramp. Rates are drawn
    independently per mode.

    Args:
        rates_nm_per_s: Per-mode creep rate in nm/s, shape ``(m,)``. Its sign
            sets the direction of the creep.
        times_s: Strictly ascending sample times in seconds, shape ``(t,)``.
            Elapsed time is measured from the FIRST sample, so the trajectory
            starts at zero.
        key: A JAX PRNG key, required when ``rate_shape`` is set.
        rate_shape: Gamma shape parameter for the drawn rate. Default
            ``None``, a deterministic rate.

    Returns:
        The mode-coefficient trajectory in nm, shape ``(t, m)``.

    Raises:
        ValueError: If the time grid fails validation, if ``rate_shape`` is
            set without a ``key``, or if it is not strictly positive.
    """
    times = _check_times(times_s)
    rates = jnp.asarray(rates_nm_per_s, dtype=float)
    if rates.ndim != 1:
        raise ValueError(f"rates_nm_per_s must be 1D (m,), got {rates.shape}")
    if rate_shape is not None:
        if key is None:
            raise ValueError("rate_shape needs a key: the drawn rate is random")
        if not float(rate_shape) > 0.0:
            raise ValueError(f"rate_shape must be positive, got {rate_shape}")
        gamma = jax.random.gamma(key, float(rate_shape), shape=rates.shape)
        rates = rates * gamma / float(rate_shape)
    elapsed_s = jnp.asarray(times - times[0])
    return elapsed_s[:, None] * rates


def compose_trajectories(*eps_tables):
    """Sum trajectories sampled on a shared time grid into one table.

    Drift is a sum of regimes -- a fast stationary component, a slow creep, a
    random walk between corrections -- and a speckle field replays ONE
    coefficient table. Adding the tables is exact: the mode coefficients enter
    the field linearly, so the composed table realizes the composed process
    with no field-level composition machinery.

    Args:
        *eps_tables: Trajectories of identical shape ``(t, m)``, all sampled
            on the same ``times_s``.

    Returns:
        Their sum, shape ``(t, m)``.

    Raises:
        ValueError: If no table is given or the shapes disagree.
    """
    if not eps_tables:
        raise ValueError("compose_trajectories needs at least one trajectory")
    shapes = {jnp.asarray(table).shape for table in eps_tables}
    if len(shapes) != 1:
        raise ValueError(
            f"every trajectory must have the same shape (t, m), got {sorted(shapes)}; "
            "build them on one shared times_s"
        )
    return sum(jnp.asarray(table) for table in eps_tables)
