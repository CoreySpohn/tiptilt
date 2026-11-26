"""The tiptilt speckle-field family: aberration realizations as residuals.

These implement optixstuff's ``LinearSpeckleField`` on top of the
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
    response = FocalResponse(
        e_nom, G, grid, e_in, grid.frame.reference_wavelength_nm, wavelength_law="opd"
    )
    field = TabulatedSpeckleField(response, times_s, eps)
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import as_float_leaf, is_traced
from hwoutils.numerics import COARSEST, eps
from hwoutils.precision import like, require_float64
from jaxtyping import Array
from optixstuff import FocalResponse, LinearSpeckleField
from optixstuff.spectral import trapezoid_weights
from physicaloptix import AnalyticSpeckleField

from ._table_clock import _TableClock

# Below this effective frequency count a single frozen realization's modal
# covariance is an unreliable sample of the target (scatter ~ 1/sqrt(N_eff)).
_NEFF_WARN = 8.0


def _line_powers(frequencies_hz, psd):
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
    # A singleton is a discrete line power, not a finite-span PSD integral.
    if f.size > 1:
        # The static quadrature determines both the draw weights and N_eff.
        with (
            jax.ensure_compile_time_eval(),
            jax.enable_x64(),  # precision-ok: concrete host frequency quadrature
        ):
            df = np.asarray(trapezoid_weights(f))
        power = power * df
    if float(power.sum()) <= 0.0:
        raise ValueError("psd must have positive total power")
    return power


def _stored_frequencies(frequencies_hz):
    """Build a checked frequency leaf that resolves each delivered line.

    Distinct lines that collapse in float32 require float64 storage.

    Args:
        frequencies_hz: Concrete frequency grid in Hz.

    Returns:
        The frequency leaf at the synthesis precision.
    """
    with jax.ensure_compile_time_eval():
        stored = as_float_leaf(frequencies_hz)
        host = np.asarray(frequencies_hz)
        grid = np.asarray(stored)
        if np.any(
            (host[..., 1:] != host[..., :-1]) & (grid[..., 1:] == grid[..., :-1])
        ):
            require_float64("distinct frequency lines in spectral synthesis", stored)
        return stored


def _scale_covariance(matrix):
    """Rescale a covariance by an even power of two for its eigensolver.

    Args:
        matrix: Finite real square matrix.

    Returns:
        The scaled matrix and the exponent for restoring its square root.
    """
    info = jnp.finfo(matrix.dtype)
    bits = jnp.abs(matrix).view(np.dtype(f"u{matrix.dtype.itemsize}"))
    subnormal = bits < (1 << info.nmant)
    _, exponent = jnp.frexp(jnp.max(jnp.abs(matrix)))
    # A subnormal matrix has no normal exponent on a flush-to-zero backend.
    exponent = jnp.where(jnp.all(subnormal) & jnp.any(bits != 0), info.minexp, exponent)
    half_exponent = exponent // 2
    unit = jnp.ldexp(matrix, -2 * half_exponent)
    # Recover subnormal entries from their integer counts of the smallest unit.
    units = jnp.copysign(bits.astype(matrix.dtype), matrix)
    restored = jnp.ldexp(units, info.minexp - info.nmant - 2 * half_exponent)
    correction = jnp.where(subnormal, restored - unit, 0)
    return unit + jax.lax.stop_gradient(correction), half_exponent


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
    # Scale before the eigensolver symmetrizes the input or forms its norms.
    unit, exponent = _scale_covariance(matrix)
    eigvals, eigvecs = jnp.linalg.eigh(unit)
    return jnp.ldexp(eigvecs * jnp.sqrt(jnp.maximum(eigvals, 0.0)), exponent)


def _covariance_admissible(unit):
    """Check a normalized covariance on the shared decision grid.

    Args:
        unit: Power-of-two-scaled float32 matrix.

    Returns:
        Boolean scalar for symmetry and positive-semidefinite admission.
    """
    tolerance = eps(COARSEST) * jnp.max(jnp.abs(unit))
    delta = jnp.abs(unit - unit.T)
    # At a rounded tie, try the equivalent difference with the tolerance
    # first. Within a factor of two, Sterbenz's lemma makes it exact.
    # Outside that range the original difference cannot round a violation
    # down to the boundary.
    larger = jnp.maximum(jnp.abs(unit), jnp.abs(unit.T))
    opposite_sign = jnp.signbit(unit) != jnp.signbit(unit.T)
    smaller = jnp.minimum(jnp.abs(unit), jnp.abs(unit.T))
    signed_smaller = jnp.where(opposite_sign, -smaller, smaller)
    excess_at_tie = (
        (larger >= tolerance / 2)
        & (larger <= tolerance * 2)
        & (larger - tolerance > signed_smaller)
    )
    symmetric = jnp.all((delta < tolerance) | ((delta == tolerance) & ~excess_at_tie))
    eigvals = jnp.linalg.eigvalsh((unit + unit.T) / 2)
    # Each row has m rounded entries: ||delta C||_2 <= m * eps * ||C||_2.
    bound = unit.shape[0] * eps(COARSEST) * jnp.max(jnp.abs(eigvals))
    return symmetric & (eigvals[0] >= -bound)


def _check_covariance(covariance_nm2, name="covariance_nm2"):
    """Check symmetry and positive semidefiniteness relative to the matrix scale.

    Host inputs are normalized before narrowing to the same decision grid
    used by eager and traced arrays.

    Args:
        covariance_nm2: Candidate real, finite ``(m, m)`` covariance.
        name: Argument name to quote in the error message.

    Returns:
        The covariance at its working precision.

    Raises:
        ValueError: If the matrix is empty, nonsquare, nonfinite, asymmetric,
            or has a materially negative eigenvalue.
    """
    dynamic = any(isinstance(x, jax.Array) for x in jax.tree.leaves(covariance_nm2))
    xp = jnp if dynamic else np
    values = xp.asarray(covariance_nm2)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not values.size:
        raise ValueError(
            f"{name} must be nonempty and square (m, m), got {values.shape}"
        )
    if xp.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    message = f"{name} must be finite, symmetric and positive semidefinite"
    if dynamic:
        values = jax.lax.stop_gradient(as_float_leaf(values))
        finite = jnp.isfinite(values)
        unit, _ = _scale_covariance(jnp.where(finite, values, 0))
        valid = jnp.all(finite) & _covariance_admissible(unit.astype(COARSEST))
        if is_traced(valid):
            return eqx.error_if(as_float_leaf(covariance_nm2), ~valid, message)
        if not valid:
            raise ValueError(message)
        return as_float_leaf(covariance_nm2)

    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(message)
    _, exponent = np.frexp(np.max(np.abs(values)))
    unit = np.ldexp(values, -2 * (exponent // 2)).astype(COARSEST)
    with jax.ensure_compile_time_eval():
        if not _covariance_admissible(jnp.asarray(unit)):
            raise ValueError(message)
        return as_float_leaf(covariance_nm2)


def _check_covariance_modes(covariance, n_modes, name):
    """Match a covariance block to the modes its coefficients drive."""
    shape = jnp.shape(covariance)
    if shape != (n_modes, n_modes):
        raise ValueError(
            f"{name} has shape {shape} but the response has {n_modes} modes"
        )


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

    Colors a circularly-symmetric complex-normal spectrum by a square root of
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
    response: FocalResponse,
    covariance_nm2,
    *,
    key,
    frequencies_hz,
    psd,
    epoch_d=0.0,
):
    """Build a stationary speckle field with a target cross-mode covariance.

    Reuses physicaloptix's analytic cosine-sum field, but synthesizes its
    ``(amplitudes, phases)`` so the mode-coefficient process is a stationary
    Gaussian process whose ENSEMBLE equal-time modal covariance is
    ``covariance_nm2``, with ensemble power distributed over
    ``frequencies_hz`` in proportion to ``psd``. This supplies the cross-mode
    correlation (segment or Zernike statistics coupled by a screen) that the
    per-mode-independent ``SpeckleProcess`` cannot.

    The returned field is ONE frozen realization of that process. Its own
    time-averaged modal covariance is an unbiased but random sample of
    ``covariance_nm2`` with relative scatter ``~ 1/sqrt(N_eff)``, where
    ``N_eff = (sum w)^2 / sum(w^2)`` is the participation ratio of the per-line
    powers ``w = psd * df`` -- set by the PSD SHAPE
    and the grid, not by the number of frequencies alone. A finite
    cosine sum is also almost-periodic: its lag covariance revives on ``~1/df``.
    So this is the right tool for broadband, whitish stationary drift (large
    ``N_eff``); for a red PSD (``N_eff`` near 1), a genuine autoregressive /
    random-walk process, or fidelity over long baselines, generate an explicit
    trajectory and use :class:`TabulatedSpeckleField` instead. A low ``N_eff``
    emits a warning. The coherent cross term needs float64 inputs (x64 enabled).

    Args:
        response: Caller-supplied monochromatic response defining the wavelength
            law and mode order.
        covariance_nm2: Target ``(m, m)`` real symmetric positive-semidefinite
            ensemble modal covariance in squared coefficient units (nm^2 for OPD).
        key: A JAX PRNG key freezing the drawn realization.
        frequencies_hz: Temporal frequency grid, shape ``(f,)``.
        psd: Temporal power spectral density at each frequency, shape ``(f,)``
            (nonnegative, positive total; only its shape matters, it is
            renormalized).
        epoch_d: Days from the scene epoch mapping to ``time_s = 0``. Default 0.0.

    Returns:
        A physicaloptix ``AnalyticSpeckleField``.

    Raises:
        ValueError: If the covariance does not match the modes, is asymmetric,
            or has a materially negative eigenvalue, or if ``psd`` has
            non-positive total.
    """
    covariance = _check_covariance(covariance_nm2)
    _check_covariance_modes(covariance, response.G.shape[-3], "covariance_nm2")

    power = _line_powers(frequencies_hz, psd)
    frequencies = _stored_frequencies(frequencies_hz)
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
    amplitudes, phases = _draw_correlated_spectrum(covariance, key, weights)
    return AnalyticSpeckleField(
        response,
        amplitudes=amplitudes,
        frequencies_hz=frequencies,
        phases=phases,
        epoch_d=epoch_d,
    )


def grouped_drift_field(
    response: FocalResponse,
    groups,
    *,
    key,
    epoch_d=0.0,
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
        response: Caller-supplied monochromatic response defining the wavelength
            law and mode order.
        groups: Sequence of ``(covariance_nm2, frequencies_hz, psd)``. Every
            block needs the same NUMBER of frequencies (they stack into one
            ``(m, f)`` array) but the grids and PSD shapes may differ.
        key: A JAX PRNG key freezing the drawn realization.
        epoch_d: Days from the scene epoch mapping to ``time_s = 0``. Default 0.0.

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
        power = _line_powers(freqs, psd)
        weights = jnp.asarray(2.0 * power / float(power.sum()))
        covariance = _check_covariance(covariance, f"group {i} covariance")
        amplitudes, phases = _draw_correlated_spectrum(
            covariance, jax.random.fold_in(key, i), weights
        )
        amplitude_blocks.append(amplitudes)
        phase_blocks.append(phases)
        freq_blocks.append(np.broadcast_to(freqs, amplitudes.shape))

    amplitudes = jnp.concatenate(amplitude_blocks, axis=0)
    phases = jnp.concatenate(phase_blocks, axis=0)
    if amplitudes.shape[0] != response.G.shape[0]:
        raise ValueError(
            f"the blocks supply {amplitudes.shape[0]} modes but G has "
            f"{response.G.shape[0]}; G's mode axis must be the blocks concatenated "
            "in order"
        )
    frequencies = np.concatenate(freq_blocks, axis=0)
    # Collapse to the shared 1D grid when every block agrees, so a
    # single-timescale call is indistinguishable from correlated_drift_field.
    if np.all(frequencies == frequencies[:1]):
        frequencies = frequencies[0]
    return AnalyticSpeckleField(
        response,
        amplitudes=amplitudes,
        frequencies_hz=_stored_frequencies(frequencies),
        phases=phases,
        epoch_d=epoch_d,
    )


def correlated_channel_fields(
    mcl,
    shared_covariance_nm2,
    *,
    key,
    frequencies_hz,
    psd,
    local=None,
    epoch_d=0.0,
):
    """Correlated per-channel speckle fields from one shared-mode draw.

    Draws ONE correlated spectral realization of the SHARED modes (the
    segment/trunk drift every channel sees) and imprints it through each
    channel's own ``g_shared`` block, so the returned fields realize the
    exact cross-channel covariance ``G_a Sigma (G_b)^H`` -- the physics that
    makes differential imaging work. Optional per-channel LOCAL blocks
    (non-common-path drift) are drawn INDEPENDENTLY per channel and
    appended; they are the part no cross-channel difference removes.

    Photometry derives from the SHARED primitives: every channel references
    the one entrance-field energy ``mcl.input_energy`` with its own
    ``grid.pixel_area``, so a channel's flux-fraction map carries its split
    fraction inside the map -- the same convention that puts coronagraph mask
    losses inside a YIP's maps. The shared law always includes the pinning
    cross term; supply zero nominal amplitudes for a pure halo.

    Args:
        mcl: A ``physicaloptix.MultiChannelLinearization`` (the per-channel
            ``e_nom`` / ``g_shared`` blocks of one shared basis; it records
            the entrance field's ``input_energy``).
        shared_covariance_nm2: Target ``(m, m)`` shared-mode covariance.
        key: PRNG key freezing the shared draw (and seeding the local ones).
        frequencies_hz: Temporal frequency grid, shape ``(f,)``.
        psd: Temporal PSD shape over those frequencies.
        local: Optional dict mapping a channel name to ``(g_local,
            covariance_local_nm2)`` -- that channel's independent
            non-common-path block. Its modes must follow the same wavelength law
            as the shared modes, given by ``mcl.kind``.
        epoch_d: Days from the scene epoch mapping to ``time_s = 0``. Default 0.0.

    Returns:
        A dict mapping each channel name to an ``AnalyticSpeckleField``.

    Raises:
        ValueError: If a covariance does not match its modes or the shared
            basis kind has no monochromatic law.
    """
    if mcl.kind not in ("opd", "amplitude"):
        raise ValueError(
            f"channel fields need an 'opd' or 'amplitude' basis kind, got {mcl.kind!r}"
        )
    for channel in mcl.channels:
        _check_covariance_modes(
            shared_covariance_nm2,
            channel.g_shared.shape[-3],
            f"{channel.name} shared covariance",
        )
    power = _line_powers(frequencies_hz, psd)
    weights = jnp.asarray(2.0 * power / float(power.sum()))

    frequencies = _stored_frequencies(frequencies_hz)
    key_shared, key_locals = jax.random.split(key)
    shared = _check_covariance(shared_covariance_nm2, "shared covariance")
    shared_amp, shared_phase = _draw_correlated_spectrum(shared, key_shared, weights)

    fields = {}
    for i, name in enumerate(mcl.names):
        channel = mcl[name]
        g = channel.g_shared
        amplitudes, phases = shared_amp, shared_phase
        if local is not None and name in local:
            g_local, covariance_local = local[name]
            g_local = jnp.asarray(g_local)
            _check_covariance_modes(
                covariance_local, g_local.shape[-3], f"{name} local covariance"
            )
            covariance_local = _check_covariance(
                covariance_local, f"{name} local covariance"
            )
            local_amp, local_phase = _draw_correlated_spectrum(
                covariance_local, jax.random.fold_in(key_locals, i), weights
            )
            g = jnp.concatenate([g, g_local], axis=0)
            amplitudes = jnp.concatenate([shared_amp, local_amp], axis=0)
            phases = jnp.concatenate([shared_phase, local_phase], axis=0)
        fields[name] = AnalyticSpeckleField(
            FocalResponse(
                channel.e_nom,
                g,
                channel.grid,
                mcl.input_energy,
                mcl.wavelength_nm,
                wavelength_law=mcl.kind,
            ),
            amplitudes=amplitudes,
            frequencies_hz=frequencies,
            phases=phases,
            epoch_d=epoch_d,
        )
    return fields


class TabulatedSpeckleField(LinearSpeckleField):
    """Replay modal trajectories through a monochromatic focal response.

    The caller's focal response defines the wavelength law.
    The inherited law includes interference with e_nom; a zero nominal
    response produces a pure halo. ``coefficients(*, elapsed_s)``
    interpolates ``eps_table`` at ``elapsed_s`` on ``times_s`` (seconds since
    the clock origin), holding the endpoints outside the table.
    """

    _clock: _TableClock
    eps_table: Array

    def __init__(self, response, times_s, eps_table, *, epoch_d=0.0):
        """Build a replay field on a clock local to the first table knot.

        Args:
            response: Monochromatic focal amplitudes and normalization.
            times_s: Ascending knot times in seconds since the clock origin.
            eps_table: Per-mode coefficients at each knot.
            epoch_d: Scene days corresponding to zero elapsed seconds.
        """
        self.response = response
        self.epoch_d = epoch_d
        self.eps_table = as_float_leaf(eps_table)
        self._clock = _TableClock(times_s, self.eps_table)

    def __check_init__(self):
        """Require one coefficient row per knot and one column per mode.

        Raises:
            ValueError: If the coefficient axes do not match the knots and modes.
        """
        if self.eps_table.ndim != 2:
            raise ValueError("eps_table must be 2D (t, m)")
        n_times, n_modes = self.eps_table.shape
        if self._clock.offsets_s.shape != (n_times,):
            raise ValueError("times_s must match eps_table's nonempty time axis")
        if self.response.G.shape[0] != n_modes:
            raise ValueError(
                f"G has {self.response.G.shape[0]} modes but eps_table has {n_modes}"
            )

    @property
    def times_s(self) -> Array:
        """Return knot times in seconds since the clock origin.

        Returns:
            Absolute knot coordinates at the stored working precision.
        """
        return self._clock.times_s

    def coefficients(self, *, elapsed_s):
        """Interpolate the table and hold the endpoints outside its span.

        Args:
            elapsed_s: Seconds since the clock origin.

        Returns:
            An array of shape ``time_shape + (n_modes,)``.
        """
        t = self._clock.offset(elapsed_s)
        return jax.vmap(
            lambda column: jnp.interp(t, self._clock.offsets_s, column),
            in_axes=1,
            out_axes=-1,
        )(self.eps_table)


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
    return _stationary_ou_covariance(covariance, tau)


def _stationary_ou_covariance(covariance, tau):
    """Apply the Lorentzian spectral overlap to the driving covariance.

    Args:
        covariance: Checked driving covariance in squared coefficient units.
        tau: Validated per-mode decorrelation timescales in seconds.

    Returns:
        Stationary covariance at the working precision of the driven modes.
    """
    roots = np.sqrt(tau)
    root_ratio = np.minimum(roots[:, None], roots) / np.maximum(roots[:, None], roots)
    overlap = like(2.0 * root_ratio / (1.0 + root_ratio**2), covariance)
    # Preserve subnormal entries at unit overlap even on a flush-to-zero backend.
    return jnp.where(overlap == 1.0, covariance, covariance * overlap)


def _ou_innovation(sigma, inverse_tau):
    """Build the innovation-covariance map for one stationary OU process.

    Returns a function of the step length, closing over the stationary
    covariance and the per-mode rates so a scan can reuse it. The factor is
    written with ``expm1`` because ``1 - a_k a_l`` cancels to nothing for a
    step much shorter than the timescales, which is exactly the regime a fine
    time grid sits in.

    Args:
        sigma: Stationary ``(m, m)`` covariance (already overlap-damped).
        inverse_tau: Per-mode rates ``1 / tau``, shape ``(m,)``.

    Returns:
        A callable mapping a step length in seconds to the ``(m, m)``
        innovation covariance.
    """
    pair_rate = inverse_tau[:, None] + inverse_tau[None, :]

    def innovation(step_s):
        return sigma * -jnp.expm1(-step_s * pair_rate)

    return innovation


def ou_innovation_covariance(covariance_nm2, timescales_s, step_s):
    """Covariance of the OU innovation over one step: what a step cannot predict.

    Splitting the process into the part a step carries forward and the part it
    cannot, ``eps(t + dt) = D_a eps(t) + innovation``, this is the covariance of
    that second piece, ``Sigma - D_a Sigma D_a`` for ``a_k = exp(-dt/tau_k)``.

    It answers two different questions with one object. As PHYSICS it is the
    unpredictable content of a step, so its eigenspectrum -- against the
    stationary covariance's -- is what says whether predictive control has less
    structure to work with than the raw residual does. As a CONTRACT it is the
    validity condition on the process: a stationary AR(1) exists only if this
    matrix is positive semidefinite at every step, which is not automatic for a
    freely chosen covariance and per-mode timescales (see :func:`ou_covariance`)
    but is guaranteed for the damped covariance this returns it for.

    Args:
        covariance_nm2: Driving ``(m, m)`` modal covariance in nm^2.
        timescales_s: Per-mode decorrelation timescale, scalar or ``(m,)``.
        step_s: Step length in seconds.

    Returns:
        The ``(m, m)`` innovation covariance in nm^2. It goes to zero as the
        step does (a short step predicts almost everything) and to the full
        stationary covariance as the step grows past every timescale (a long
        step predicts nothing).

    Raises:
        ValueError: If the covariance or the timescales fail validation.
    """
    covariance = _check_covariance(covariance_nm2)
    tau = _check_timescales(timescales_s, covariance.shape[0])
    sigma = _stationary_ou_covariance(covariance, tau)
    return _ou_innovation(sigma, like(1.0 / tau, sigma))(like(step_s, sigma))


def ou_lag_covariance(covariance_nm2, timescales_s, lag_s):
    """The two-time modal covariance ``Cov[eps(t), eps(t + lag)]``.

    Column ``l`` of the equal-time covariance decayed by that mode's own
    factor, ``Sigma_kl exp(-lag / tau_l)`` -- equivalently
    ``ou_covariance(...) @ diag(rho)``. It is NOT symmetric when the modes
    carry different timescales and the covariance is not diagonal, which is
    the whole content of the per-mode generalization: how correlated mode
    ``k`` now is with mode ``l`` later depends on which of the two is doing
    the waiting.

    This is the object the two-time variance forms consume, so a closed-form
    prediction and this generator can be evaluated against the same matrix
    instead of against two separately-derived ones. Do not symmetrize it: a
    symmetric surrogate describes a different process.

    Args:
        covariance_nm2: Driving ``(m, m)`` modal covariance in nm^2.
        timescales_s: Per-mode decorrelation timescale, scalar or ``(m,)``.
        lag_s: Lag in seconds. Negative lags give the transpose.

    Returns:
        The ``(m, m)`` lag covariance in nm^2, in the convention
        ``result[k, l] = Cov[eps_k(t), eps_l(t + lag_s)]``.

    Raises:
        ValueError: If the covariance or the timescales fail validation.
    """
    covariance = _check_covariance(covariance_nm2)
    tau = _check_timescales(timescales_s, covariance.shape[0])
    sigma = _stationary_ou_covariance(covariance, tau)
    lag = like(lag_s, sigma)
    decay = jnp.exp(-jnp.abs(lag) / like(tau, sigma))
    return jnp.where(lag >= 0.0, sigma * decay[None, :], sigma * decay[:, None])


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
    the diagonal is exact either way, and :func:`ou_lag_covariance` gives the
    two-time law. That damping is what makes the process well posed rather
    than a limitation of this implementation: a covariance and a set of
    per-mode timescales are not a free pair, because a stationary AR(1)
    requires the innovation covariance ``Sigma - D_a Sigma D_a`` to stay
    positive semidefinite. Imposing an arbitrary covariance exactly while
    giving the modes different timescales violates that for most inputs (it
    is not a process at all), whereas deriving the damping from the
    timescales satisfies it by construction, for every covariance and every
    step size.

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

    sigma = _stationary_ou_covariance(covariance, tau)
    inverse_tau = like(1.0 / tau, sigma)
    noise = jax.random.normal(key, (times.size, n_modes), dtype=sigma.dtype)
    start = _psd_sqrt(sigma) @ noise[0]
    if times.size == 1:
        return start[None, :]

    # Reuse a factor only for identical host intervals. Near-uniform grids
    # retain their actual gaps without a precision-dependent uniformity test.
    gaps, indices = np.unique(np.diff(times), return_inverse=True)
    steps_s = like(gaps, sigma)
    indices = jnp.asarray(indices, dtype=jnp.int32)
    decays = jnp.exp(-steps_s[:, None] * inverse_tau)
    innovation = _ou_innovation(sigma, inverse_tau)
    lowers = jax.vmap(lambda dt: _psd_sqrt(innovation(dt)))(steps_s)

    def advance(eps, xs):
        index, z = xs
        moved = decays[index] * eps + lowers[index] @ z
        return moved, moved

    _, moved = jax.lax.scan(advance, start, (indices, noise[1:]))
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
    origin = (
        jnp.zeros(n_modes, dtype=diffusion.dtype)
        if start_nm is None
        else like(start_nm, diffusion)
    )
    if times.size == 1:
        # Keep the checked diffusion in the traced result even without a draw.
        origin = eqx.error_if(
            origin, ~jnp.all(jnp.isfinite(diffusion)), "diffusion must be finite"
        )
        return jnp.broadcast_to(origin, (1, n_modes))

    factor = _psd_sqrt(diffusion)
    steps_s = like(np.diff(times), diffusion)
    noise = jax.random.normal(key, (times.size - 1, n_modes), dtype=diffusion.dtype)
    increments = jnp.sqrt(steps_s)[:, None] * (noise @ factor.T)
    walk = jnp.concatenate(
        [
            jnp.zeros((1, n_modes), dtype=diffusion.dtype),
            jnp.cumsum(increments, axis=0),
        ],
        axis=0,
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
    rates = as_float_leaf(rates_nm_per_s)
    if rates.ndim != 1:
        raise ValueError(f"rates_nm_per_s must be 1D (m,), got {rates.shape}")
    if rate_shape is not None:
        if key is None:
            raise ValueError("rate_shape needs a key: the drawn rate is random")
        if not float(rate_shape) > 0.0:
            raise ValueError(f"rate_shape must be positive, got {rate_shape}")
        gamma = jax.random.gamma(
            key, float(rate_shape), shape=rates.shape, dtype=rates.dtype
        )
        rates = rates * gamma / float(rate_shape)
    elapsed_s = like(times - times[0], rates)
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
