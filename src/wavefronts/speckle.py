"""The wavefronts speckle-field family: aberration realizations as residuals.

These implement optixstuff's ``AbstractSpeckleField`` on top of the
``(E_nom, G)`` linearization a propagation engine provides, differing only in
how they source the drifting mode coefficients ``eps(t)``.

``TabulatedSpeckleField`` replays a precomputed coefficient trajectory. It is
the escape hatch for drift a stationary spectral synthesis cannot represent --
an autoregressive or random-walk process (variance growing in time), or any
sampled time series (a STOP thermal run) -- which a finite sum of cosines with
bounded, time-constant variance structurally cannot hold. The realization is
fixed at construction and interpolated by elapsed time, so it stays
deterministic and differentiable, as the contract requires.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from optixstuff.speckle import AbstractSpeckleField

J2000_JD = 2451545.0


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
