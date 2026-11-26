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
  grid on output. There is no data outside the dark zone; the array holds 0.0
  there and ``data_mask`` is False.
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

import copy
import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from hwoutils.leaves import as_bulk_leaf, as_float_leaf, is_traced
from hwoutils.numerics import COARSEST, is_uniform
from hwoutils.precision import like
from jax.typing import ArrayLike
from jaxtyping import Array
from optixstuff.grids import Frame, Image, Measure, PixelGrid
from optixstuff.speckle import AbstractSpeckleField

from ._table_clock import _TableClock


def _checked_field_table(value, name):
    """Convert a delivered field, checking host narrowing before staging it.

    Args:
        value: Nominal or sensitivity field samples.
        name: Argument name for validation diagnostics.

    Returns:
        The checked bulk leaf.

    Raises:
        ValueError: If a component is nonfinite or fails checked narrowing.
    """
    message = f"{name} field components must be finite"
    if any(is_traced(x) for x in jax.tree.leaves(value)):
        storage = as_bulk_leaf(value)
        return eqx.error_if(storage, ~jnp.all(jnp.isfinite(storage)), message)
    with jax.ensure_compile_time_eval():
        storage = as_bulk_leaf(value)
        if not np.all(np.isfinite(np.asarray(storage))):
            raise ValueError(message)
        return storage


def _complex_plane(real, imag, dtype):
    """Pack selected FITS components into checked host complex storage.

    Args:
        real: Real field samples.
        imag: Imaginary field samples.
        dtype: Requested complex storage dtype.

    Returns:
        Complex field at the requested precision.

    Raises:
        ValueError: If the dtype is real or a component does not fit.
    """
    dtype = np.dtype(dtype)
    if dtype.kind != "c":
        raise ValueError("dtype must be complex")
    field = np.empty(real.shape, dtype=dtype)
    for before, after in ((real, field.real), (imag, field.imag)):
        with np.errstate(over="ignore", under="ignore"):
            after[...] = before
        if not np.all(np.isfinite(after)):
            raise ValueError(f"field components must fit finite {dtype} storage")
        if before.dtype.itemsize > after.dtype.itemsize and np.any(
            (before != 0) & (after == 0)
        ):
            raise ValueError(f"field components underflow to zero in {dtype} storage")
    return field


def load_hwo_sim(
    data_dir: Path | str,
    *,
    wavelengths_nm: ArrayLike,
    frame: Frame,
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
        frame: Diameter convention and reference wavelength from run provenance.
        num_zern: Keep only the first ``num_zern`` Zernike TERMS per segment,
            Noll-ordered. Terms are not radial orders: 11 means Noll 1-11,
            radial orders 0-3 complete plus spherical. None keeps all.
        wl_indices: Wavelength indices to keep. None keeps all.
        dtype: Complex dtype for the loaded fields.

    Returns:
        Dict with keys ``E0`` (num_wl, num_pix), ``G`` (num_wl, num_pix,
        num_seg, num_zern), ``dark_zone`` (ny, nx) bool, ``detector_axis``
        (nx,) in lambda/D, ``wavelengths_nm``, and ``grid`` with its Frame.

    Raises:
        ValueError: If the detector axis is nonuniform, the number of wavelengths
            disagrees with E0, or a selected field component does not fit the
            requested complex dtype.
    """
    data_dir = Path(data_dir)
    dark_zone = fits.getdata(data_dir / "dark_zone.fits").astype(bool)
    axis = np.asarray(
        fits.getdata(data_dir / "detector_axis.fits"), dtype=float
    ).ravel()
    if (
        axis.size != dark_zone.shape[1]
        or axis.size < 2
        or not np.all(np.isfinite(axis))
        or not np.all(axis[1:] > axis[:-1])
        or not is_uniform(axis - axis[0])
    ):
        raise ValueError("detector_axis must be ascending and uniformly sampled")
    step = float(np.mean(np.diff(axis)))
    # The format uses the same coordinate origin and spacing on both axes.
    grid = PixelGrid.regular(
        dark_zone.shape,
        step,
        "lod",
        center=(-axis[0] / step, -axis[0] / step),
        frame=frame,
    )

    e0_raw = fits.getdata(data_dir / "E0.fits")
    wl_nm = np.asarray(wavelengths_nm, dtype=float)
    if e0_raw.shape[0] != wl_nm.size:
        raise ValueError(
            f"E0 has {e0_raw.shape[0]} wavelengths but wavelengths_nm has "
            f"{wl_nm.size}; pass the filter definition of the run that wrote "
            "these files."
        )
    if wl_indices is not None:
        keep = np.asarray(wl_indices, dtype=int)
        e0_raw = e0_raw[keep]
        wl_nm = wl_nm[keep]
    e0 = _complex_plane(e0_raw[..., 0], e0_raw[..., 1], dtype)

    z_slice = slice(None) if num_zern is None else slice(0, num_zern)
    with fits.open(data_dir / "sensitivities.fits", memmap=True) as hdul:
        raw = hdul[0].data  # (wl, pix, re/im, seg, zern)
        if wl_indices is None:
            block = np.asarray(raw[:, :, :, :, z_slice])
        else:
            block = np.stack([np.asarray(raw[i, :, :, :, z_slice]) for i in wl_indices])
    g = _complex_plane(block[:, :, 0], block[:, :, 1], dtype)
    del block

    return {
        "E0": e0,
        "G": g,
        "dark_zone": dark_zone,
        "detector_axis": axis,
        "wavelengths_nm": wl_nm,
        "grid": grid,
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
        ``(T, num_seg, num_zern)``, sorted by time. The times remain on
        the host in binary64 so the field can rebase them before storage.

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
    return time_s[order], jnp.asarray(coeffs[order])


def _optional_float_leaf(value):
    return None if value is None else as_float_leaf(value)


class HwoSimSpeckleField(AbstractSpeckleField):
    """Speckle field driven by an hwo_sim linear E-field model.

    Evaluates ``I(t) = |E_nom + G eps(t)|^2`` on the dark-zone pixel list and
    returns the excess over the static floor, per the speckle contract, as
    flux fraction per pixel: ``|E_nom + G eps|^2 - |E_nom|^2`` times
    ``contrast_to_flux_fraction``. There is no data outside the dark zone; the
    array holds 0.0 there and :meth:`data_mask` is False.

    Wavelength handling is nearest-neighbor over the delivered planes, with
    ties resolved in delivery order. Centers must be distinct on the
    float32 wavelength grid.
    Interpolating a complex field across a broad band would wrap phase
    between samples, so it is not attempted; evaluate at or near the stored
    wavelengths.

    Exactly one drift form must be supplied (see the module docstring). Set
    ``epoch_d`` near the observations being simulated. JAX defaults to
    float32, in which the sinusoid phase ``2 pi f t`` loses absolute precision
    as the elapsed time grows, so a clock origin left thousands of days away
    quietly corrupts the drift. The tabulated form has the same exposure
    through its interpolation offset from the first knot.
    :meth:`max_safe_elapsed_s` estimates the working-precision limit.

    ``epoch_d`` is placement metadata, a static host binary64 float.
    :meth:`coefficients`, :meth:`realize` and :meth:`decompose_delta` take
    ``elapsed_s``, seconds since the clock origin, and do not read
    ``epoch_d``. Convert scene time with ``elapsed_s(time_d)``, which refuses
    a float32 time at a nonzero epoch. In float32 the drift is exact only
    while ``elapsed_s`` stays under :meth:`max_safe_elapsed_s`.

    Attributes:
        grid: Native focal sampling and frame.
        epoch_d: Scene days at zero elapsed seconds.
        wavelengths_nm: Wavelength metadata in delivery order.
        e_nom: Nominal field at each wavelength and dark-zone pixel.
        gain: Sensitivities with flattened mode axes.
        dark_zone_index: Flat row-major indices of delivered pixels.
        grid_shape: Height and width of the full detector grid.
        amplitude: Component amplitudes for sinusoid drift.
        frequency_hz: Component frequencies for sinusoid drift.
        phase: Component phases for sinusoid drift.
        drift_coefficients: Per-knot coefficients for tabulated drift.
        contrast_to_flux_fraction: Scale from peak contrast to per-pixel flux.
    """

    grid: PixelGrid
    epoch_d: float = eqx.field(static=True, converter=float)
    wavelengths_nm: Array
    _plane_order: tuple[int, ...] = eqx.field(static=True)
    _wavelength_breaks: tuple[float, ...] = eqx.field(static=True)
    e_nom: Array = eqx.field(converter=as_bulk_leaf)  # (num_wl, num_pix)
    gain: Array = eqx.field(converter=as_bulk_leaf)  # (num_wl, num_pix, num_mode)
    dark_zone_index: Array  # int32 (num_pix,): flat grid index of each pixel
    grid_shape: tuple[int, int] = eqx.field(static=True)
    amplitude: Array | None
    frequency_hz: Array | None
    phase: Array | None
    _clock: _TableClock | None
    drift_coefficients: Array | None
    contrast_to_flux_fraction: Array = eqx.field(converter=as_float_leaf)

    def __init__(
        self,
        e_nom: ArrayLike,
        G: ArrayLike,
        dark_zone: ArrayLike,
        *,
        wavelengths_nm: ArrayLike,
        grid: PixelGrid,
        epoch_d: float = 0.0,
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
        ``wavelengths_nm``, ``grid``).

        Args:
            e_nom: Nominal field, (num_wl, num_pix) complex, sqrt(contrast).
            G: Sensitivities, (num_wl, num_pix, ...) complex, sqrt(contrast)
                per nm; the trailing mode axes are flattened in C order.
            dark_zone: (ny, nx) mask whose True pixels, in C order, are the
                pixel axis of ``e_nom`` and ``G``.
            wavelengths_nm: Concrete delivered wavelengths, (num_wl,).
            grid: Native sampling and Frame from replay provenance.
            epoch_d: Days from the scene epoch mapping to ``time_s = 0``. Default 0.0.
            contrast_to_flux_fraction: Scale from peak-referenced contrast to
                flux fraction per pixel.
            amplitude: Sinusoid drift amplitudes, (num_seg, num_zern,
                n_components), nm RMS.
            frequency_hz: Sinusoid drift frequencies, same shape.
            phase: Sinusoid drift phases, same shape.
            drift_time_s: Abscissa of a tabulated drift, (T,) seconds since
                ``epoch_d``, strictly increasing.
            drift_coefficients: Tabulated drift, (T, num_seg, num_zern), nm
                RMS wavefront error.

        Raises:
            ValueError: If the selected plane centers or fields are invalid.
        """
        dark_zone = np.asarray(dark_zone)
        num_wl, num_pix = np.shape(e_nom)

        self.grid = grid
        if grid.unit != "lod" or grid.shape != dark_zone.shape:
            raise ValueError("grid must be native lod and match dark_zone")
        self.epoch_d = epoch_d
        if is_traced(wavelengths_nm):
            raise ValueError("wavelengths_nm must be concrete plane centers")
        centers = np.asarray(wavelengths_nm, dtype=np.float64)
        if centers.shape != (num_wl,) or not num_wl:
            raise ValueError("wavelengths_nm must match the nonempty plane axis")
        if not np.all(np.isfinite(centers)) or np.any(centers <= 0.0):
            raise ValueError("wavelengths_nm must be finite and positive")
        order = np.argsort(centers)
        sorted_centers = centers[order]
        breaks = sorted_centers[:-1] + np.diff(sorted_centers) / 2
        with np.errstate(over="ignore", under="ignore"):
            grid_centers = sorted_centers.astype(COARSEST)
        if (
            not np.all(np.isfinite(grid_centers))
            or np.any(grid_centers <= 0)
            or np.any(grid_centers[1:] <= grid_centers[:-1])
            or np.any(breaks <= sorted_centers[:-1])
            or np.any(breaks >= sorted_centers[1:])
        ):
            raise ValueError(
                "wavelengths_nm must have distinct centers on the float32 grid"
            )
        self._plane_order = tuple(map(int, order))
        self._wavelength_breaks = tuple(map(float, breaks))
        with jax.ensure_compile_time_eval():
            self.wavelengths_nm = as_float_leaf(wavelengths_nm)
        self.e_nom = _checked_field_table(e_nom, "e_nom")
        # Flat modes, so the drift contracts as a single matrix-vector product
        # per wavelength.
        self.gain = _checked_field_table(G, "gain").reshape(num_wl, num_pix, -1)
        self.dark_zone_index = jnp.asarray(
            np.flatnonzero(dark_zone.ravel()), dtype=jnp.int32
        )
        self.grid_shape = tuple(int(n) for n in dark_zone.shape)
        self.amplitude = _optional_float_leaf(amplitude)
        self.frequency_hz = _optional_float_leaf(frequency_hz)
        self.phase = _optional_float_leaf(phase)
        self.drift_coefficients = _optional_float_leaf(drift_coefficients)
        clock_owner = (
            self.drift_coefficients if self.amplitude is None else self.amplitude
        )
        self._clock = (
            None
            if drift_time_s is None or clock_owner is None
            else _TableClock(drift_time_s, clock_owner)
        )
        self.contrast_to_flux_fraction = contrast_to_flux_fraction

    def __check_init__(self):
        """Require a finite clock origin and exactly one consistent drift form."""
        if not math.isfinite(self.epoch_d):
            raise ValueError("epoch_d must be finite")
        if self.dark_zone_index.shape[0] != self.e_nom.shape[1]:
            raise ValueError(
                f"dark_zone has {self.dark_zone_index.shape[0]} pixels but e_nom "
                f"has {self.e_nom.shape[1]}"
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
            if self._clock is None:
                raise ValueError(
                    "a tabulated drift needs drift_time_s alongside drift_coefficients"
                )
            t_tab, c_tab = self._clock.offsets_s, self.drift_coefficients
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
    def drift_time_s(self) -> Array | None:
        """Return the drift knot times for inspection.

        Returns:
            Knot times in seconds at the storage precision, or None.
        """
        return None if self._clock is None else self._clock.times_s

    @property
    def drift_is_tabulated(self) -> bool:
        """Whether the drift is a tabulated series rather than sinusoids."""
        return self.drift_coefficients is not None

    def coefficients(self, *, elapsed_s: ArrayLike) -> Array:
        """Per-segment Zernike coefficients at ``elapsed_s``, in nm RMS wavefront.

        ``elapsed_s`` is seconds since the clock origin.

        Returns shape ``(num_seg, num_zern)``. For a sinusoid drift the
        per-mode components are summed here, so callers never see the
        component axis. For a tabulated drift the series is interpolated
        linearly in time and held at its end values outside its span, which
        keeps a frame deterministic, differentiable and jittable exactly as
        the sinusoid path does.

        The branch is on ``None``-ness, which is PyTree structure rather than
        a traced value, so it resolves at trace time under ``jit``.

        Args:
            elapsed_s: Seconds since the field's clock origin.

        Returns:
            Coefficients for each segment and Zernike mode.
        """
        if self.drift_coefficients is None:
            t = like(elapsed_s, self.frequency_hz)
            wave = self.amplitude * jnp.cos(
                2.0 * jnp.pi * self.frequency_hz * t + self.phase
            )
            return jnp.sum(wave, axis=-1)
        t = self._clock.offset(elapsed_s)
        table = self.drift_coefficients
        flat = table.reshape(table.shape[0], -1)
        # jnp.interp clamps outside the abscissa: a series covers a finite
        # window and the wavefront outside it is unknown, so holding the
        # endpoint is honest where extrapolating a thermal ramp would not be.
        interp = jax.vmap(lambda column: jnp.interp(t, self._clock.offsets_s, column))
        return interp(flat.T).reshape(table.shape[1], table.shape[2])

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the plane switches used by the nearest-neighbor lookup.

        An exact midpoint selects the first plane in delivery order.

        Returns:
            Sorted midpoints in nanometers between delivered plane centers.
        """
        return np.asarray(self._wavelength_breaks, dtype=np.float64)

    def _wavelength_index(self, wavelength_nm: ArrayLike) -> Array:
        if np.shape(wavelength_nm) != ():
            raise ValueError("wavelength_nm must be scalar")
        breaks = self.spectral_breaks_nm()
        order = np.asarray(self._plane_order, dtype=np.int32)
        tie_right = order[1:] < order[:-1]
        if not is_traced(wavelength_nm):
            query = np.asarray(wavelength_nm, dtype=np.float64)
            right = (query > breaks) | ((query == breaks) & tie_right)
            return jnp.asarray(order[np.count_nonzero(right)])

        # Compare conditioned differences on the common grid. Splitting each
        # midpoint keeps float32 queries from inheriting a rounded boundary.
        high = breaks.astype(COARSEST)
        low = breaks - high
        query = jnp.asarray(wavelength_nm)
        if not jnp.issubdtype(query.dtype, jnp.floating):
            query = query.astype(COARSEST)
        delta = ((query - high) - jnp.asarray(low, dtype=query.dtype)).astype(COARSEST)
        right = (delta > 0) | ((delta == 0) & tie_right)
        return jnp.asarray(order)[jnp.count_nonzero(right)]

    def _plane(self, table, index):
        return like(jnp.take(table, index, axis=0), self.contrast_to_flux_fraction)

    def _scatter(self, values: Array) -> Array:
        flat = jnp.zeros(self.grid_shape[0] * self.grid_shape[1], dtype=values.dtype)
        return flat.at[self.dark_zone_index].set(values).reshape(self.grid_shape)

    def data_mask(self) -> Array:
        """The dark zone on ``grid``: True where the delivery holds data.

        The field is known on the dark-zone pixel list only, whatever the size
        of its array, so a consumer's support ends at the list.

        Returns:
            Boolean of shape ``grid.shape``, True on the dark-zone pixels.
        """
        return self._scatter(jnp.ones(self.dark_zone_index.shape, dtype=bool))

    def static_contrast(self, wavelength_nm: ArrayLike) -> Array:
        """Static coronagraphic floor ``|E_nom|^2`` as peak-referenced contrast."""
        index = self._wavelength_index(wavelength_nm)
        return self._scatter(jnp.abs(self._plane(self.e_nom, index)) ** 2)

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
        epoch_d: float | None = None,
        amplitude: Array | None = None,
        frequency_hz: Array | None = None,
        phase: Array | None = None,
    ) -> "HwoSimSpeckleField":
        """Copy with a new sinusoid drift realization, reusing the field arrays.

        Only the supplied values are converted; every other field, including
        an unused drift array and the dtype it was stored in, passes through
        unchanged and uncopied. The gain matrix, hundreds of megabytes at full
        size, is never uploaded again, so an independent wavefront realization
        per visit is cheap. That is the right model for revisits separated by
        more than the drift correlation time. The drift arrays may be traced,
        so the copy can be made inside ``eqx.filter_jit``.

        Args:
            epoch_d: New clock origin. None keeps the current one.
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
        return self._reanchor(
            epoch_d, amplitude=amplitude, frequency_hz=frequency_hz, phase=phase
        )

    def with_tabulated_drift(
        self,
        *,
        epoch_d: float | None = None,
        drift_time_s: Array | None = None,
        drift_coefficients: Array | None = None,
    ) -> "HwoSimSpeckleField":
        """Copy pointing at a different tabulated drift, reusing the arrays.

        The tabulated counterpart of :meth:`with_drift`. The use is not a
        fresh random realization but a different slice of history: another
        window of the same series, another control scenario, or another run.
        Re-anchoring ``epoch_d`` to the window being rendered is usually part
        of that, since the abscissa is seconds from the clock origin. As in
        :meth:`with_drift`, only the supplied values are converted.

        Args:
            epoch_d: New clock origin. None keeps the current one.
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
        return self._reanchor(
            epoch_d, drift_time_s=drift_time_s, drift_coefficients=drift_coefficients
        )

    def _reanchor(self, epoch_d, **drift) -> "HwoSimSpeckleField":
        """Copy with the supplied clock origin and drift arrays, all else as stored.

        The supplied drift arrays go in through ``eqx.tree_at`` after the same
        conversion the constructor applies. ``epoch_d`` is a static field,
        which ``eqx.tree_at`` cannot reach, so it is set on a fresh object the
        way Equinox's own unflattening sets fields. The result is always a new
        object, even with nothing supplied, and this field is never changed.
        The class checks then run on the result, as the constructor would run
        them.
        """
        supplied = {
            name: as_float_leaf(value)
            for name, value in drift.items()
            if value is not None and name != "drift_time_s"
        }
        if drift.get("drift_time_s") is not None:
            table = supplied.get("drift_coefficients", self.drift_coefficients)
            supplied["_clock"] = _TableClock(drift["drift_time_s"], table)
        if supplied:
            names = tuple(supplied)
            new = eqx.tree_at(
                lambda f: tuple(getattr(f, name) for name in names),
                self,
                tuple(supplied.values()),
            )
        else:
            new = copy.copy(self)
        if epoch_d is not None:
            object.__setattr__(new, "epoch_d", float(epoch_d))
        for cls in type(new).__mro__:
            check = cls.__dict__.get("__check_init__")
            if check is not None:
                check(new)
        return new

    def max_safe_elapsed_s(
        self, phase_tol_rad: float = 0.05, *, spacing_tol_frac: float = 0.05
    ) -> float:
        """Elapsed time beyond which float32 corrupts the drift.

        The sinusoid argument is ``2 pi f t``. In float32 its absolute error
        grows with magnitude, at a relative resolution of ``2 ** -24``, so the
        phase error for the fastest mode is ``2 pi f t * 2 ** -24``. Solving
        for the tolerance gives the returned bound. Anchor ``epoch_d`` inside
        it, or enable ``jax_enable_x64``.

        For a tabulated drift the bound applies to the offset from the first
        knot. An elapsed time already rounded by the caller also carries
        its original rounding error.

        Args:
            phase_tol_rad: Acceptable drift phase error, in radians. Sinusoid
                drift only.
            spacing_tol_frac: Acceptable time error as a fraction of the finest
                sample spacing. Tabulated drift only.

        Returns:
            Bound in seconds from the sinusoid epoch or the first table knot.
        """
        eps = float(np.finfo(np.float32).eps)
        if self.drift_is_tabulated:
            offsets = self._clock.offsets_s
            if offsets.size == 1:
                return math.inf
            spacing = float(jnp.min(jnp.diff(offsets)))
            return spacing_tol_frac * spacing / eps
        f_max = float(jnp.max(self.frequency_hz))
        return phase_tol_rad / (2.0 * np.pi * eps * f_max)

    def decompose_delta(
        self, *, wavelength_nm: ArrayLike, elapsed_s: ArrayLike = 0.0
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
        e_nom = self._plane(self.e_nom, index)
        gain = self._plane(self.gain, index)
        delta_e = gain @ self.coefficients(elapsed_s=elapsed_s).ravel().astype(
            gain.dtype
        )
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
        elapsed_s: ArrayLike = 0.0,
    ) -> Image:
        """Wavefront-error-induced excess over the static floor.

        Returns ``|E_nom + G eps(t)|^2 - |E_nom|^2`` as flux fraction per
        pixel. There is no data outside the dark zone; the array holds 0.0
        there and :meth:`data_mask` is False.
        """
        index = self._wavelength_index(wavelength_nm)
        e_nom = self._plane(self.e_nom, index)
        gain = self._plane(self.gain, index)
        delta_e = gain @ self.coefficients(elapsed_s=elapsed_s).ravel().astype(
            gain.dtype
        )
        delta = 2.0 * jnp.real(jnp.conj(e_nom) * delta_e) + jnp.abs(delta_e) ** 2
        return Image(
            self._scatter(delta.real) * self.contrast_to_flux_fraction,
            self.grid,
            (),
            Measure.integral(),
            "stellar_flux_fraction",
        )
