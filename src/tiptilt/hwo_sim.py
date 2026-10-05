"""A reader and a speckle field for the hwo_sim linear E-field format.

hwo_sim writes the two ingredients of the linear speckle model
``I(t) = |E_nom + G eps(t)|^2`` as FITS files in one directory::

    E0.fits             E_nom, sqrt(contrast)
                        (num_wl, num_dz_pix, re/im)
    sensitivities.fits  G, sqrt(contrast) per nm RMS wavefront error
                        (num_wl, num_dz_pix, re/im, num_seg, num_zern)
    dark_zone.fits      (ny, nx) mask; its True pixels, in C order, are the
                        pixel axis of the two files above
    detector_axis.fits  (nx,) focal-plane sample positions in lambda/D

:func:`load_hwo_sim` reads them and :class:`HwoSimSpeckleField` evaluates the
model as an ``optixstuff.AbstractSpeckleField``. The field differs from
``TabulatedSpeckleField`` where the format does:

- The fields live on the dark-zone pixel list and are scattered onto the full
  grid on output, so the field is zero outside the dark zone, where the format
  carries nothing.
- hwo_sim normalizes by the peak of the unocculted image, so ``|E|^2`` is
  peak-referenced contrast. One scalar, ``contrast_to_flux_fraction``, takes it
  to the flux fraction per pixel the contract asks for.
- The pinning cross term ``2 Re(E_nom* G eps)`` is always kept, which is why
  the complex ``E_nom`` is stored rather than an intensity map.

The format carries no drift ``eps(t)``. The field takes it in one of two forms:
a sum of sinusoids per mode (``amplitude``, ``frequency_hz``, ``phase``), or a
tabulated series (``drift_time_s``, ``drift_coefficients``) interpolated
linearly and held at its endpoints. :func:`segment_zernike_series` shapes a
flat per-(segment, Zernike) coefficient series, such as a fit of a
structural-thermal wavefront history to the format's own basis, into the
tabulated form. Which form is in use is a provenance fact about every rendered
frame, so :attr:`HwoSimSpeckleField.drift_is_tabulated` reports it.
"""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from jax.typing import ArrayLike
from jaxtyping import Array
from optixstuff.speckle import AbstractSpeckleField


def load_hwo_sim(
    data_dir: Path | str,
    *,
    wavelengths_nm: ArrayLike,
    num_zern: int | None = None,
    wl_indices: tuple[int, ...] | None = None,
    dtype: type = np.complex64,
) -> dict:
    """Load an hwo_sim linear E-field directory, sliced to bound memory.

    The sensitivity matrix dominates the footprint: it holds every (segment,
    Zernike) column at every wavelength and dark-zone pixel in float64.
    Slicing the Zernike axis is the cheapest large saving, since the low
    radial orders carry nearly all of a realistic drift. The slice is taken
    from the memory-mapped file, so the full matrix is never read.

    Args:
        data_dir: Directory holding E0.fits, sensitivities.fits,
            dark_zone.fits and detector_axis.fits.
        wavelengths_nm: The wavelength of each E0 plane, in nanometers. The
            files do not record it; it is the filter definition of the
            hwo_sim run that wrote them.
        num_zern: Keep only the first ``num_zern`` Zernike TERMS per segment,
            Noll-ordered. Terms are not radial orders: 11 means Noll 1-11,
            radial orders 0-3 complete plus spherical. None keeps all.
        wl_indices: Wavelength indices to keep. None keeps all.
        dtype: Complex dtype for the loaded fields.

    Returns:
        Dict with keys ``E0`` (num_wl, num_pix), ``G`` (num_wl, num_pix,
        num_seg, num_zern), ``dark_zone`` (ny, nx) bool, ``detector_axis``
        (nx,) in lambda/D, ``wavelengths_nm``, and ``pixel_scale_lod``.

    Raises:
        ValueError: If ``wavelengths_nm`` does not have one entry per E0 plane.
    """
    data_dir = Path(data_dir)
    dark_zone = fits.getdata(data_dir / "dark_zone.fits").astype(bool)
    axis = np.asarray(
        fits.getdata(data_dir / "detector_axis.fits"), dtype=float
    ).ravel()
    pixel_scale_lod = float(np.mean(np.diff(axis)))

    e0_raw = fits.getdata(data_dir / "E0.fits")
    e0 = (e0_raw[..., 0] + 1j * e0_raw[..., 1]).astype(dtype)

    wl_nm = np.asarray(wavelengths_nm, dtype=float)
    if e0.shape[0] != wl_nm.size:
        raise ValueError(
            f"E0 has {e0.shape[0]} wavelengths but wavelengths_nm has "
            f"{wl_nm.size}; pass the filter definition of the run that wrote "
            "these files."
        )

    z_slice = slice(None) if num_zern is None else slice(0, num_zern)
    with fits.open(data_dir / "sensitivities.fits", memmap=True) as hdul:
        raw = hdul[0].data  # (wl, pix, re/im, seg, zern)
        if wl_indices is None:
            block = np.asarray(raw[:, :, :, :, z_slice])
        else:
            block = np.stack([np.asarray(raw[i, :, :, :, z_slice]) for i in wl_indices])
    g = (block[:, :, 0] + 1j * block[:, :, 1]).astype(dtype)
    del block

    if wl_indices is not None:
        keep = np.asarray(wl_indices, dtype=int)
        e0 = e0[keep]
        wl_nm = wl_nm[keep]

    return {
        "E0": e0,
        "G": g,
        "dark_zone": dark_zone,
        "detector_axis": axis,
        "wavelengths_nm": wl_nm,
        "pixel_scale_lod": pixel_scale_lod,
    }


def segment_zernike_series(
    time_s: ArrayLike,
    coefficients: ArrayLike,
    *,
    num_seg: int,
    num_zern: int | None = None,
    recenter: bool = False,
) -> tuple[Array, Array]:
    """Shape a flat per-(segment, Zernike) series for the tabulated drift.

    The slicing must match the ``num_zern`` passed to :func:`load_hwo_sim`
    exactly, or the coefficients contract against the wrong Jacobian columns
    and the result is silently wrong rather than an error. Both orderings
    are the format's own: the flat mode axis is ``(segment, Zernike)`` in C
    order and the Zernike axis is Noll-ordered, so a leading slice of each is
    the subset the loader's ``[..., :num_zern]`` keeps.

    Args:
        time_s: Sample times, (T,) seconds.
        coefficients: The series, (T, num_seg * Z) in nm RMS wavefront error,
            with Z Zernike terms per segment.
        num_seg: Number of segments.
        num_zern: Keep the first this many Noll terms per segment. None keeps
            all Z.
        recenter: Subtract the series mean, so ``eps = 0`` is the window
            average rather than the series' own zero. Leave it False when the
            series is already referenced to the wavefront ``E_nom`` was
            computed at and the question is the departure from it; set it when
            the question is the spread about a maintained mean.

    Returns:
        Tuple ``(time_s, coefficients)``, shaped ``(T,)`` and
        ``(T, num_seg, num_zern)``, sorted by time.

    Raises:
        ValueError: If the lengths disagree, the mode count does not divide
            into ``num_seg`` segments, or ``num_zern`` exceeds Z.
    """
    time_s = np.asarray(time_s, dtype=float).ravel()
    coeffs = np.asarray(coefficients, dtype=float)
    if coeffs.shape[0] != time_s.size:
        raise ValueError(
            f"coefficients has {coeffs.shape[0]} frames but time_s has {time_s.size}"
        )
    if coeffs.shape[1] % num_seg:
        raise ValueError(
            f"{coeffs.shape[1]} modes is not divisible by {num_seg} segments"
        )
    coeffs = coeffs.reshape(time_s.size, num_seg, -1)
    if num_zern is not None:
        if num_zern > coeffs.shape[2]:
            raise ValueError(
                f"asked for {num_zern} Zernikes but the series has "
                f"{coeffs.shape[2]} per segment"
            )
        coeffs = coeffs[:, :, :num_zern]
    if recenter:
        coeffs = coeffs - coeffs.mean(axis=0, keepdims=True)
    order = np.argsort(time_s)
    return jnp.asarray(time_s[order]), jnp.asarray(coeffs[order])


class HwoSimSpeckleField(AbstractSpeckleField):
    """Speckle field driven by an hwo_sim linear E-field model.

    Evaluates ``I(t) = |E_nom + G eps(t)|^2`` on the dark-zone pixel list and
    returns the excess over the static floor, per the speckle contract, as
    flux fraction per pixel: ``|E_nom + G eps|^2 - |E_nom|^2`` times
    ``contrast_to_flux_fraction``, zero outside the dark zone.

    Wavelength handling is nearest-neighbor over the stored planes.
    Interpolating a complex field across a broad band would wrap phase
    between samples, so it is not attempted; evaluate at or near the stored
    wavelengths.

    Exactly one drift form must be supplied (see the module docstring). Set
    ``epoch_jd`` near the observations being simulated. JAX defaults to
    float32, in which the sinusoid phase ``2 pi f t`` loses absolute precision
    as the elapsed time grows, so a clock origin left thousands of days away
    quietly corrupts the drift. The tabulated form has the same exposure
    through its interpolation abscissa. :meth:`max_safe_elapsed_s` reports
    the limit for whichever form is loaded.
    """

    pixel_scale_lod: float
    epoch_jd: float
    wavelengths_nm: Array
    e_nom: Array  # complex (num_wl, num_pix): nominal field on the pixel list
    gain: Array  # complex (num_wl, num_pix, num_mode): d(E)/d(mode), flat modes
    dark_zone_index: Array  # int32 (num_pix,): flat grid index of each pixel
    grid_shape: tuple[int, int] = eqx.field(static=True)
    amplitude: Array | None
    frequency_hz: Array | None
    phase: Array | None
    drift_time_s: Array | None
    drift_coefficients: Array | None
    contrast_to_flux_fraction: float

    def __init__(
        self,
        e_nom: ArrayLike,
        G: ArrayLike,
        dark_zone: ArrayLike,
        *,
        wavelengths_nm: ArrayLike,
        pixel_scale_lod: float,
        epoch_jd: float,
        contrast_to_flux_fraction: float,
        amplitude: Array | None = None,
        frequency_hz: Array | None = None,
        phase: Array | None = None,
        drift_time_s: Array | None = None,
        drift_coefficients: Array | None = None,
    ):
        """Assemble from the loaded arrays and a drift specification.

        The first five arguments are the like-named entries of
        :func:`load_hwo_sim`'s result (``E0``, ``G``, ``dark_zone``,
        ``wavelengths_nm``, ``pixel_scale_lod``).

        Args:
            e_nom: Nominal field, (num_wl, num_pix) complex, sqrt(contrast).
            G: Sensitivities, (num_wl, num_pix, ...) complex, sqrt(contrast)
                per nm; the trailing mode axes are flattened in C order.
            dark_zone: (ny, nx) mask whose True pixels, in C order, are the
                pixel axis of ``e_nom`` and ``G``.
            wavelengths_nm: Wavelength of each plane, (num_wl,).
            pixel_scale_lod: Plate scale in lambda/D per pixel.
            epoch_jd: Julian Date mapping to ``time_s = 0``.
            contrast_to_flux_fraction: Scale from peak-referenced contrast to
                flux fraction per pixel.
            amplitude: Sinusoid drift amplitudes, (num_seg, num_zern,
                n_components), nm RMS.
            frequency_hz: Sinusoid drift frequencies, same shape.
            phase: Sinusoid drift phases, same shape.
            drift_time_s: Abscissa of a tabulated drift, (T,) seconds since
                ``epoch_jd``, strictly increasing.
            drift_coefficients: Tabulated drift, (T, num_seg, num_zern), nm
                RMS wavefront error.
        """
        dark_zone = np.asarray(dark_zone)
        num_wl, num_pix = np.shape(e_nom)

        self.pixel_scale_lod = float(pixel_scale_lod)
        self.epoch_jd = float(epoch_jd)
        self.wavelengths_nm = jnp.asarray(wavelengths_nm, dtype=float)
        self.e_nom = jnp.asarray(e_nom)
        # Flat modes, so the drift contracts as a single matrix-vector product
        # per wavelength.
        self.gain = jnp.asarray(G).reshape(num_wl, num_pix, -1)
        self.dark_zone_index = jnp.asarray(
            np.flatnonzero(dark_zone.ravel()), dtype=jnp.int32
        )
        self.grid_shape = tuple(int(n) for n in dark_zone.shape)
        self.amplitude = None if amplitude is None else jnp.asarray(amplitude)
        self.frequency_hz = None if frequency_hz is None else jnp.asarray(frequency_hz)
        self.phase = None if phase is None else jnp.asarray(phase)
        self.drift_time_s = (
            None if drift_time_s is None else jnp.asarray(drift_time_s, dtype=float)
        )
        self.drift_coefficients = (
            None if drift_coefficients is None else jnp.asarray(drift_coefficients)
        )
        self.contrast_to_flux_fraction = float(contrast_to_flux_fraction)

    def __check_init__(self):
        """Require exactly one drift form, with consistent shapes."""
        if self.dark_zone_index.shape[0] != self.e_nom.shape[1]:
            raise ValueError(
                f"dark_zone has {self.dark_zone_index.shape[0]} pixels but e_nom "
                f"has {self.e_nom.shape[1]}"
            )
        if self.wavelengths_nm.shape != (self.e_nom.shape[0],):
            raise ValueError(
                f"wavelengths_nm has shape {self.wavelengths_nm.shape} but e_nom "
                f"has {self.e_nom.shape[0]} planes"
            )
        sinusoid = self.amplitude is not None
        tabulated = self.drift_coefficients is not None
        if sinusoid == tabulated:
            raise ValueError(
                "supply exactly one drift form: amplitude / frequency_hz / "
                "phase, or drift_time_s / drift_coefficients."
            )
        if sinusoid:
            if self.amplitude.ndim != 3:
                raise ValueError(
                    "amplitude must be (num_seg, num_zern, n_components), got "
                    f"shape {self.amplitude.shape}"
                )
            if self.frequency_hz is None or self.phase is None:
                raise ValueError(
                    "a sinusoid drift needs all of amplitude, frequency_hz, phase"
                )
            modes = self.amplitude.shape[0] * self.amplitude.shape[1]
        else:
            if self.drift_time_s is None:
                raise ValueError(
                    "a tabulated drift needs drift_time_s alongside drift_coefficients"
                )
            t_tab, c_tab = self.drift_time_s, self.drift_coefficients
            if t_tab.ndim != 1 or c_tab.ndim != 3:
                raise ValueError(
                    "drift_time_s must be (T,) and drift_coefficients "
                    f"(T, num_seg, num_zern); got {t_tab.shape} and {c_tab.shape}"
                )
            if t_tab.shape[0] != c_tab.shape[0]:
                raise ValueError(
                    f"drift_time_s has {t_tab.shape[0]} samples but "
                    f"drift_coefficients has {c_tab.shape[0]}"
                )
            modes = c_tab.shape[1] * c_tab.shape[2]
        if modes != self.gain.shape[-1]:
            raise ValueError(
                f"the drift spans {modes} modes but G has {self.gain.shape[-1]}; "
                "slice it with the same num_zern passed to load_hwo_sim."
            )

    @property
    def num_mode(self) -> int:
        """Number of (segment, Zernike) modes retained."""
        return int(self.gain.shape[-1])

    @property
    def drift_is_tabulated(self) -> bool:
        """Whether the drift is a tabulated series rather than sinusoids."""
        return self.drift_coefficients is not None

    def coefficients(self, time_s: ArrayLike) -> Array:
        """Per-segment Zernike coefficients at a time, in nm RMS wavefront.

        Returns shape ``(num_seg, num_zern)``. For a sinusoid drift the
        per-mode components are summed here, so callers never see the
        component axis. For a tabulated drift the series is interpolated
        linearly in time and held at its end values outside its span, which
        keeps a frame deterministic, differentiable and jittable exactly as
        the sinusoid path does.

        The branch is on ``None``-ness, which is PyTree structure rather than
        a traced value, so it resolves at trace time under ``jit``.
        """
        t = jnp.asarray(time_s, dtype=float)
        if self.drift_coefficients is None:
            wave = self.amplitude * jnp.cos(
                2.0 * jnp.pi * self.frequency_hz * t + self.phase
            )
            return jnp.sum(wave, axis=-1)
        table = self.drift_coefficients
        flat = table.reshape(table.shape[0], -1)
        # jnp.interp clamps outside the abscissa: a series covers a finite
        # window and the wavefront outside it is unknown, so holding the
        # endpoint is honest where extrapolating a thermal ramp would not be.
        interp = jax.vmap(lambda column: jnp.interp(t, self.drift_time_s, column))
        return interp(flat.T).reshape(table.shape[1], table.shape[2])

    def _wavelength_index(self, wavelength_nm: ArrayLike) -> Array:
        return jnp.argmin(jnp.abs(self.wavelengths_nm - jnp.asarray(wavelength_nm)))

    def _scatter(self, values: Array) -> Array:
        flat = jnp.zeros(self.grid_shape[0] * self.grid_shape[1], dtype=values.dtype)
        return flat.at[self.dark_zone_index].set(values).reshape(self.grid_shape)

    def static_contrast(self, wavelength_nm: ArrayLike) -> Array:
        """Static coronagraphic floor ``|E_nom|^2`` as peak-referenced contrast."""
        index = self._wavelength_index(wavelength_nm)
        return self._scatter(jnp.abs(jnp.take(self.e_nom, index, axis=0)) ** 2)

    def static_flux_fraction(self, wavelength_nm: ArrayLike) -> Array:
        """Static coronagraphic floor as flux fraction per pixel.

        ``realize`` excludes it, per the speckle contract: an image simulator
        applies its own static floor, and the two differ wherever the
        coronagraph models differ.
        """
        return self.static_contrast(wavelength_nm) * self.contrast_to_flux_fraction

    def with_drift(
        self,
        *,
        epoch_jd: float | None = None,
        amplitude: Array | None = None,
        frequency_hz: Array | None = None,
        phase: Array | None = None,
    ) -> "HwoSimSpeckleField":
        """Copy with a new sinusoid drift realization, reusing the field arrays.

        Rebuilding through ``__init__`` would re-upload the gain matrix, which
        is hundreds of megabytes at full size. This swaps only the drift
        leaves, so an independent wavefront realization per visit is cheap.
        That is the right model for revisits separated by more than the drift
        correlation time.

        Args:
            epoch_jd: New clock origin. None keeps the current one.
            amplitude: New drift amplitudes. None keeps the current ones.
            frequency_hz: New drift frequencies. None keeps the current ones.
            phase: New drift phases. None keeps the current ones.

        Returns:
            A new :class:`HwoSimSpeckleField` sharing this one's field arrays.

        Raises:
            ValueError: If this field carries a tabulated drift, which has no
                per-visit resampling interpretation. Use
                :meth:`with_tabulated_drift`.
        """
        if self.drift_is_tabulated:
            raise ValueError(
                "this field carries a tabulated drift; use with_tabulated_drift "
                "to point it at a different window or series."
            )
        updates = {
            "epoch_jd": float(self.epoch_jd if epoch_jd is None else epoch_jd),
            "amplitude": self.amplitude
            if amplitude is None
            else jnp.asarray(amplitude),
            "frequency_hz": (
                self.frequency_hz if frequency_hz is None else jnp.asarray(frequency_hz)
            ),
            "phase": self.phase if phase is None else jnp.asarray(phase),
        }
        return eqx.tree_at(
            lambda s: (s.epoch_jd, s.amplitude, s.frequency_hz, s.phase),
            self,
            tuple(
                updates[k] for k in ("epoch_jd", "amplitude", "frequency_hz", "phase")
            ),
        )

    def with_tabulated_drift(
        self,
        *,
        epoch_jd: float | None = None,
        drift_time_s: Array | None = None,
        drift_coefficients: Array | None = None,
    ) -> "HwoSimSpeckleField":
        """Copy pointing at a different tabulated drift, reusing the arrays.

        The tabulated counterpart of :meth:`with_drift`. The use is not a
        fresh random realization but a different slice of history: another
        window of the same series, another control scenario, or another run.
        Re-anchoring ``epoch_jd`` to the window being rendered is usually part
        of that, since the abscissa is seconds from the clock origin.

        Args:
            epoch_jd: New clock origin. None keeps the current one.
            drift_time_s: New abscissa, (T,). None keeps the current one.
            drift_coefficients: New series, (T, num_seg, num_zern). None keeps
                the current one.

        Returns:
            A new :class:`HwoSimSpeckleField` sharing this one's field arrays.

        Raises:
            ValueError: If this field carries a sinusoid drift, or if the two
                tabulated arrays disagree on length.
        """
        if not self.drift_is_tabulated:
            raise ValueError(
                "this field carries a sinusoid drift; use with_drift, or "
                "construct a new field with drift_coefficients."
            )
        t_new = (
            self.drift_time_s
            if drift_time_s is None
            else jnp.asarray(drift_time_s, dtype=float)
        )
        c_new = (
            self.drift_coefficients
            if drift_coefficients is None
            else jnp.asarray(drift_coefficients)
        )
        if t_new.shape[0] != c_new.shape[0]:
            raise ValueError(
                f"drift_time_s has {t_new.shape[0]} samples but "
                f"drift_coefficients has {c_new.shape[0]}"
            )
        return eqx.tree_at(
            lambda s: (s.epoch_jd, s.drift_time_s, s.drift_coefficients),
            self,
            (float(self.epoch_jd if epoch_jd is None else epoch_jd), t_new, c_new),
        )

    def max_safe_elapsed_s(
        self, phase_tol_rad: float = 0.05, *, spacing_tol_frac: float = 0.05
    ) -> float:
        """Elapsed time beyond which float32 corrupts the drift.

        The sinusoid argument is ``2 pi f t``. In float32 its absolute error
        grows with magnitude, at a relative resolution of ``2 ** -24``, so the
        phase error for the fastest mode is ``2 pi f t * 2 ** -24``. Solving
        for the tolerance gives the returned bound. Anchor ``epoch_jd`` inside
        it, or enable ``jax_enable_x64``.

        A tabulated drift has the same exposure through a different quantity.
        There is no phase, but the interpolation abscissa is read at absolute
        time, whose float32 quantization is also ``t * 2 ** -24``. Once that
        reaches an appreciable fraction of the finest sample spacing the
        interpolation starts landing on the wrong side of a sample, so the
        bound becomes that spacing over the same relative resolution.

        Args:
            phase_tol_rad: Acceptable drift phase error, in radians. Sinusoid
                drift only.
            spacing_tol_frac: Acceptable time error as a fraction of the finest
                sample spacing. Tabulated drift only.

        Returns:
            Bound on ``time_s`` in seconds.
        """
        eps = float(np.finfo(np.float32).eps)
        if self.drift_is_tabulated:
            spacing = float(jnp.min(jnp.diff(self.drift_time_s)))
            return spacing_tol_frac * spacing / eps
        f_max = float(jnp.max(self.frequency_hz))
        return phase_tol_rad / (2.0 * np.pi * eps * f_max)

    def decompose_delta(
        self, *, wavelength_nm: ArrayLike, time_s: ArrayLike = 0.0
    ) -> tuple[Array, Array]:
        """The two terms of the speckle delta, separately, as flux fractions.

        Returns ``(cross, quadratic)`` for ``2 Re(E_nom* G eps)`` and
        ``|G eps|^2``. Their ratio says which regime the field is in: cross
        term dominant is the pinned, near-linear regime where the delta is
        roughly sign-symmetric, while quadratic dominant means an incoherent
        halo skewed positive. Useful for checking that a chosen drift
        amplitude sits where the linear E-field model is meant to be used.
        """
        index = self._wavelength_index(wavelength_nm)
        e_nom = jnp.take(self.e_nom, index, axis=0)
        gain = jnp.take(self.gain, index, axis=0)
        delta_e = gain @ self.coefficients(time_s).ravel().astype(gain.dtype)
        cross = 2.0 * jnp.real(jnp.conj(e_nom) * delta_e)
        quadratic = jnp.abs(delta_e) ** 2
        return (
            self._scatter(cross) * self.contrast_to_flux_fraction,
            self._scatter(quadratic) * self.contrast_to_flux_fraction,
        )

    def realize(
        self,
        *,
        wavelength_nm: ArrayLike,
        time_s: ArrayLike = 0.0,
    ) -> Array:
        """Wavefront-error-induced excess over the static floor.

        Returns ``|E_nom + G eps(t)|^2 - |E_nom|^2`` as flux fraction per
        pixel, zero outside the dark zone.
        """
        index = self._wavelength_index(wavelength_nm)
        e_nom = jnp.take(self.e_nom, index, axis=0)
        gain = jnp.take(self.gain, index, axis=0)
        delta_e = gain @ self.coefficients(time_s).ravel().astype(gain.dtype)
        delta = 2.0 * jnp.real(jnp.conj(e_nom) * delta_e) + jnp.abs(delta_e) ** 2
        return self._scatter(delta.real) * self.contrast_to_flux_fraction
