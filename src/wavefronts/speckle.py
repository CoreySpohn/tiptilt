"""The wavefronts speckle-field family: aberration realizations as residuals.

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
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from optixstuff.speckle import AbstractSpeckleField
from physicaloptix import AnalyticSpeckleField

J2000_JD = 2451545.0

# Below this effective frequency count a single frozen realization's modal
# covariance is an unreliable sample of the target (scatter ~ 1/sqrt(N_eff)).
_NEFF_WARN = 8.0


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
    eigvals, eigvecs = jnp.linalg.eigh(covariance)
    factor = eigvecs * jnp.sqrt(jnp.maximum(eigvals, 0.0))
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
    ``N_eff = (sum psd)^2 / sum(psd^2)`` is the participation ratio of the
    weighting -- set by the PSD SHAPE, not the number of frequencies. A finite
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

    Returns:
        A physicaloptix ``AnalyticSpeckleField``.

    Raises:
        ValueError: If ``covariance_nm2`` is not square, not symmetric, or has a
            materially negative eigenvalue, or if ``psd`` has non-positive total.
    """
    covariance = np.asarray(covariance_nm2)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            f"covariance_nm2 must be square (m, m), got {covariance.shape}"
        )
    if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-6):
        raise ValueError("covariance_nm2 must be symmetric")
    eigvals = np.linalg.eigvalsh(covariance)
    tol = 1e-8 * max(float(np.abs(eigvals).max()), 1.0)
    if eigvals.min() < -tol:
        raise ValueError(
            "covariance_nm2 must be positive semidefinite; min eigenvalue "
            f"{eigvals.min():.3e}"
        )

    psd = np.asarray(psd, dtype=float)
    total = float(psd.sum())
    if total <= 0.0:
        raise ValueError("psd must have positive total power")
    n_eff = total**2 / float((psd**2).sum())
    if n_eff < _NEFF_WARN:
        warnings.warn(
            f"correlated_drift_field: effective frequency count N_eff={n_eff:.1f} "
            "is low, so a single frozen realization's modal covariance scatters by "
            "~1/sqrt(N_eff) around covariance_nm2 and the finite cosine sum revives "
            "on ~1/df. Use more frequencies, or TabulatedSpeckleField for red / "
            "long-baseline drift.",
            stacklevel=2,
        )

    weights = jnp.asarray(2.0 * psd / total)
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

    Monochromatic in v1: ``realize`` ignores ``wavelength_nm`` (kept for
    interface conformance). The cross term needs float64 inputs.
    """

    e_nom: Array  # complex (y, x): nominal focal field
    G: Array  # complex (m, y, x): d(E_focal)/d(mode)
    times_s: Array  # float (t,): ascending sample times in seconds
    eps_table: Array  # float (t, m): the coefficient trajectory
    normalization: float
    pixel_scale_lod: float
    epoch_jd: float
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
    ):
        """Build a tabulated speckle field from a coefficient trajectory.

        Args:
            e_nom: Complex nominal focal field, shape ``(y, x)``.
            G: Complex sensitivity ``d(E_focal)/d(mode)``, shape ``(m, y, x)``.
            times_s: Ascending sample times in seconds, shape ``(t,)``.
            eps_table: Mode coefficients at each sample time, shape ``(t, m)``.
            normalization: Intensity that maps to unit contrast (the telescope
                PSF peak the focal field is referenced to).
            pixel_scale_lod: Native pixel scale in lambda/D per pixel. Must
                equal the coronagraph's plate scale for the speckle path.
            epoch_jd: Julian Date mapping to ``time_s = 0``. Default J2000.
            coherent: Include the pinning cross term. Default ``False``.
        """
        self.e_nom = e_nom
        self.G = G
        self.times_s = times_s
        self.eps_table = eps_table
        self.normalization = normalization
        self.pixel_scale_lod = pixel_scale_lod
        self.epoch_jd = epoch_jd
        self.coherent = coherent

    def __check_init__(self):
        """Validate the trajectory shapes against the mode count."""
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
        if self.G.shape[0] != n_modes:
            raise ValueError(
                f"G has {self.G.shape[0]} modes but eps_table has {n_modes}"
            )

    def _eps(self, time_s):
        """Coefficients at ``time_s`` by per-mode linear interpolation, ``(m,)``."""
        t = jnp.asarray(time_s)
        return jax.vmap(lambda col: jnp.interp(t, self.times_s, col), in_axes=1)(
            self.eps_table
        )

    def realize(self, *, wavelength_nm, time_s=0.0):
        """Speckle contrast delta at ``time_s`` (see class docstring)."""
        g_eps = jnp.tensordot(self._eps(time_s), self.G, axes=1)
        if self.coherent:
            delta = 2.0 * jnp.real(jnp.conj(self.e_nom) * g_eps) + jnp.abs(g_eps) ** 2
        else:
            delta = jnp.abs(g_eps) ** 2
        return delta / self.normalization
