"""Tests for the hwo_sim reader and its speckle field, on synthetic files."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from hwoutils.numerics import roundoff, sum_roundoff
from optixstuff.grids import Frame, Measure, PixelGrid
from optixstuff.speckle import AbstractSpeckleField

from tiptilt.hwo_sim import HwoSimSpeckleField, load_hwo_sim, segment_zernike_series

NUM_WL, NUM_SEG, NUM_ZERN = 3, 2, 4
WAVELENGTHS_NM = np.array([500.0, 550.0, 600.0])
PIXEL_SCALE = 0.25
SCALE = 7.0e-3  # contrast_to_flux_fraction


@pytest.fixture
def grid():
    """A 5x6 grid whose dark zone is an irregular subset of pixels."""
    rng = np.random.default_rng(1)
    dark_zone = rng.random((5, 6)) > 0.4
    return dark_zone, int(dark_zone.sum())


@pytest.fixture
def delivery_dir(tmp_path, grid):
    """Write a synthetic hwo_sim directory and return it with its arrays."""
    dark_zone, num_pix = grid
    rng = np.random.default_rng(2)
    e0 = rng.standard_normal((NUM_WL, num_pix, 2)) * 1e-5
    sens = rng.standard_normal((NUM_WL, num_pix, 2, NUM_SEG, NUM_ZERN)) * 1e-4
    axis = (np.arange(dark_zone.shape[1]) - 2.5) * PIXEL_SCALE
    fits.writeto(tmp_path / "E0.fits", e0)
    fits.writeto(tmp_path / "sensitivities.fits", sens)
    fits.writeto(tmp_path / "dark_zone.fits", dark_zone.astype(np.uint8))
    fits.writeto(tmp_path / "detector_axis.fits", axis)
    return tmp_path, e0, sens


def _field(delivery, **drift):
    return HwoSimSpeckleField(
        delivery["E0"],
        delivery["G"],
        delivery["dark_zone"],
        wavelengths_nm=delivery["wavelengths_nm"],
        grid=delivery["grid"],
        epoch_d=0.0,
        contrast_to_flux_fraction=SCALE,
        **drift,
    )


def _sinusoid(num_zern=NUM_ZERN, n_components=3):
    rng = np.random.default_rng(3)
    shape = (NUM_SEG, num_zern, n_components)
    return {
        "amplitude": jnp.asarray(rng.random(shape)),
        "frequency_hz": jnp.asarray(rng.random(shape) * 1e-3),
        "phase": jnp.asarray(rng.random(shape) * 2 * np.pi),
    }


def _tabulated(num_zern=NUM_ZERN):
    rng = np.random.default_rng(4)
    times = jnp.asarray([0.0, 100.0, 300.0])
    return {
        "drift_time_s": times,
        "drift_coefficients": jnp.asarray(rng.standard_normal((3, NUM_SEG, num_zern))),
    }


def _assert_epoch_is_static(field):
    """``epoch_d`` is a Python float and no PyTree leaf at all, filtered or not."""
    assert type(field.epoch_d) is float
    paths = jax.tree_util.tree_leaves_with_path(field)
    assert not any("epoch_d" in jax.tree_util.keystr(path) for path, _ in paths)


def _assert_bitwise_equal_except(new, old, changed):
    for name in (
        "wavelengths_nm",
        "e_nom",
        "gain",
        "dark_zone_index",
        "amplitude",
        "frequency_hz",
        "phase",
        "drift_time_s",
        "drift_coefficients",
        "contrast_to_flux_fraction",
    ):
        if name in changed:
            continue
        a, b = getattr(new, name), getattr(old, name)
        if b is None:
            assert a is None, name
            continue
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), name
    assert new.grid == old.grid and new.grid_shape == old.grid_shape
    _assert_epoch_is_static(new)


def _two_planes(wavelengths_nm, *, e_nom=None, gain=None):
    grid = PixelGrid.focal(1, 0.25, Frame(500.0, 1.0, "circumscribed"))
    return HwoSimSpeckleField(
        np.array([[1.0j], [2.0j]]) if e_nom is None else e_nom,
        np.ones((2, 1, 1), dtype=np.complex128) if gain is None else gain,
        np.ones((1, 1), dtype=bool),
        wavelengths_nm=wavelengths_nm,
        grid=grid,
        contrast_to_flux_fraction=1.0,
        drift_time_s=[0.0],
        drift_coefficients=[[[0.0]]],
    )


@pytest.mark.float32_only(reason="checked narrowing to complex64 storage")
@pytest.mark.parametrize("leaf", ["e_nom", "gain"])
@pytest.mark.parametrize(
    ("component", "message"),
    [(2.0 * float(np.finfo(np.float32).max), "do not fit"), (2.0**-151, "underflow")],
)
def test_static_field_components_are_checked_in_compiled_builds(
    leaf, component, message
):
    """A nonzero imaginary component must survive checked storage.

    Args:
        leaf: Delivery array containing the component.
        component: Host value outside the complex64 component range.
        message: Expected range-check diagnostic.
    """
    values = np.full((2, 1), complex(1.0, component))
    if leaf == "gain":
        values = values[..., None]

    def build():
        return getattr(_two_planes([500.0, 550.0], **{leaf: values}), leaf)

    for evaluate in (build, jax.jit(build)):
        with pytest.raises(ValueError, match=message):
            evaluate()


@pytest.mark.parametrize("leaf", ["e_nom", "gain"])
def test_staged_field_components_must_be_finite(leaf):
    """Reject a staged infinity even if the array needs no conversion.

    Args:
        leaf: Delivery array containing the nonfinite component.
    """
    values = jnp.asarray([[complex(1.0, np.inf)], [1.0 + 0.0j]])
    if leaf == "gain":
        values = values[..., None]

    def build(data):
        return getattr(_two_planes([500.0, 550.0], **{leaf: data}), leaf)

    with pytest.raises((ValueError, RuntimeError), match="finite"):
        build(values)
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        np.asarray(jax.jit(build)(values))


@pytest.mark.parametrize("reverse", [False, True])
def test_declared_break_matches_actual_plane_switch(reverse, precision):
    """Read exact squared field values on each side, with delivery-order ties.

    Args:
        reverse: Put the higher-wavelength plane first when true.
        precision: Real floating dtype of the active test lane.
    """
    step = float(np.spacing(np.float32(525.0)))
    centers = [500.0, 550.0]
    centers[0] += step / 8
    field = _two_planes(centers[::-1] if reverse else centers)
    (boundary,) = field.spectral_breaks_nm()
    assert boundary == 525.0 + step / 16
    left, right = (4.0, 1.0) if reverse else (1.0, 4.0)

    def read(wavelength):
        return field.static_contrast(wavelength)

    compiled = jax.jit(read)
    for offset, expected in ((-step / 4, left), (0, 1.0), (step / 4, right)):
        wl = np.float64(boundary + offset)
        assert read(wl).item() == expected
        assert jax.jit(lambda wl=wl: read(wl))().item() == expected
        if precision == np.float64:
            assert compiled(jnp.asarray(wl)).item() == expected
    for wl, expected in ((525.0, left), (525.0 + step, right)):
        assert read(wl).item() == expected
        assert compiled(jnp.asarray(wl, dtype=jnp.float32)).item() == expected


def test_nearby_wavelength_centers_are_addressable_or_refused():
    """Both centers must own a plane on the supported decision grid."""
    first = np.float32(500.0)
    second = np.nextafter(first, np.float32(np.inf))
    field = _two_planes([first, second])
    for read in (field.static_contrast, jax.jit(lambda w: field.static_contrast(w))):
        assert read(jnp.asarray(first)).item() == 1.0
        assert read(jnp.asarray(second)).item() == 4.0
    unresolved = [float(first), np.nextafter(float(first), np.inf)]
    for build in (
        lambda: _two_planes(unresolved).e_nom,
        jax.jit(lambda: _two_planes(unresolved).e_nom),
    ):
        with pytest.raises(ValueError, match=r"distinct.*float32"):
            build()


class TestLoadHwoSim:
    def test_assembles_complex_fields(self, delivery_dir, grid):
        path, e0, sens = delivery_dir
        d = load_hwo_sim(
            path,
            wavelengths_nm=WAVELENGTHS_NM,
            dtype=np.complex128,
            frame=Frame(550.0, 6.5, "circumscribed"),
        )
        np.testing.assert_array_equal(d["E0"], e0[..., 0] + 1j * e0[..., 1])
        np.testing.assert_array_equal(d["G"], sens[:, :, 0] + 1j * sens[:, :, 1])
        np.testing.assert_array_equal(d["dark_zone"], grid[0])
        assert d["dark_zone"].dtype == bool
        assert d["grid"].step == pytest.approx(PIXEL_SCALE)
        assert d["grid"].frame == Frame(550.0, 6.5, "circumscribed")
        np.testing.assert_allclose(
            d["grid"].coordinates().x[0],
            d["detector_axis"],
            rtol=0,
            atol=np.finfo(float).eps,
        )
        np.testing.assert_array_equal(d["wavelengths_nm"], WAVELENGTHS_NM)

    def test_slices_leading_noll_terms(self, delivery_dir):
        path, _, sens = delivery_dir
        d = load_hwo_sim(
            path,
            wavelengths_nm=WAVELENGTHS_NM,
            num_zern=2,
            frame=Frame(550.0, 6.5, "circumscribed"),
        )
        assert d["G"].shape == (NUM_WL, sens.shape[1], NUM_SEG, 2)
        expected = sens[:, :, 0, :, :2] + 1j * sens[:, :, 1, :, :2]
        expected = expected.astype(np.complex64)
        np.testing.assert_array_equal(d["G"], expected)

    def test_keeps_selected_wavelengths(self, delivery_dir):
        """Fails if a discarded wavelength is narrowed before selection.

        Args:
            delivery_dir: Synthetic FITS directory and source planes.
        """
        path, e0, _ = delivery_dir
        full = load_hwo_sim(
            path,
            wavelengths_nm=WAVELENGTHS_NM,
            frame=Frame(550.0, 6.5, "circumscribed"),
        )
        e0[1, 0, 0] = 2.0 * float(np.finfo(np.float32).max)
        fits.writeto(path / "E0.fits", e0, overwrite=True)
        d = load_hwo_sim(
            path,
            wavelengths_nm=WAVELENGTHS_NM,
            wl_indices=(2, 0),
            frame=Frame(550.0, 6.5, "circumscribed"),
        )
        np.testing.assert_array_equal(d["E0"], full["E0"][[2, 0]])
        np.testing.assert_array_equal(d["G"], full["G"][[2, 0]])
        np.testing.assert_array_equal(d["wavelengths_nm"], WAVELENGTHS_NM[[2, 0]])

    @pytest.mark.parametrize("nominal", [True, False], ids=["E0", "gain"])
    def test_refuses_selected_field_overflow(self, delivery_dir, nominal, precision):
        """Fails if a finite FITS component turns infinite in complex64 storage.

        Args:
            delivery_dir: Synthetic FITS directory and source planes.
            nominal: Inject overflow in E0 or in the modal sensitivity.
            precision: Dtype available for constructing a field from complex128.
        """
        path, e0, sens = delivery_dir
        large = 2.0 * float(np.finfo(np.float32).max)
        if nominal:
            e0[1, 0, 0] = large
            fits.writeto(path / "E0.fits", e0, overwrite=True)
        else:
            sens[1, 0, 1, 0, 0] = large
            fits.writeto(path / "sensitivities.fits", sens, overwrite=True)
        kwargs = dict(
            wavelengths_nm=WAVELENGTHS_NM,
            wl_indices=(1,),
            frame=Frame(550.0, 6.5, "circumscribed"),
        )
        with pytest.raises(ValueError, match="complex64"):
            load_hwo_sim(path, **kwargs)
        wide = load_hwo_sim(path, dtype=np.complex128, **kwargs)
        if precision == np.float32:
            with pytest.raises(ValueError, match="complex64"):
                _field(wide, **_tabulated())

    def test_detector_spacing_uses_axis_offsets(self, delivery_dir):
        """Fails if a translated detector axis changes the uniformity decision.

        Args:
            delivery_dir: Synthetic FITS directory and source planes.
        """
        path = delivery_dir[0]
        axis = 2.0**20 + np.arange(6) * PIXEL_SCALE
        kwargs = dict(
            wavelengths_nm=WAVELENGTHS_NM, frame=Frame(550.0, 6.5, "circumscribed")
        )
        for shift, accepted in ((2.0**-20, True), (0.0625, False), (0.25, False)):
            perturbed = axis.copy()
            perturbed[3] += shift
            fits.writeto(path / "detector_axis.fits", perturbed, overwrite=True)
            if accepted:
                assert load_hwo_sim(path, **kwargs)["grid"].step == PIXEL_SCALE
            else:
                with pytest.raises(ValueError, match="detector_axis"):
                    load_hwo_sim(path, **kwargs)

    def test_rejects_a_wavelength_grid_of_the_wrong_length(self, delivery_dir):
        with pytest.raises(ValueError, match="wavelengths"):
            load_hwo_sim(
                delivery_dir[0],
                wavelengths_nm=WAVELENGTHS_NM[:2],
                frame=Frame(550.0, 6.5, "circumscribed"),
            )


class TestSegmentZernikeSeries:
    def test_reshapes_slices_and_sorts(self):
        """Sorting keeps exact host knots within one float32 spacing."""
        origin = 2.0**26
        times = origin + np.array([2.0, 0.0, 1.0])
        flat = np.arange(3 * NUM_SEG * 5, dtype=float).reshape(3, NUM_SEG * 5)
        t, c = segment_zernike_series(times, flat, num_seg=NUM_SEG, num_zern=3)
        np.testing.assert_array_equal(t - origin, [0.0, 1.0, 2.0])
        expected = flat.reshape(3, NUM_SEG, 5)[[1, 2, 0], :, :3]
        np.testing.assert_array_equal(c, expected)

    def test_recenter_removes_the_window_mean(self):
        """Bound the residual mean by centering, storage and reduction roundoff."""
        flat = np.random.default_rng(5).standard_normal((6, NUM_SEG * 3))
        _, c = segment_zernike_series(
            np.arange(6.0), flat, num_seg=NUM_SEG, recenter=True
        )
        scale = np.max(abs(flat), axis=0).reshape(c.shape[1:])
        # Mean, subtraction and storage round once, in addition to the sum.
        bound = roundoff(scale, n_ops=3, dtype=c.dtype) + sum_roundoff(
            scale, len(flat), dtype=c.dtype
        )
        assert np.all(np.abs(np.asarray(c).mean(axis=0)) <= bound)

    @pytest.mark.parametrize(
        ("times", "shape", "num_zern", "match"),
        [
            (np.arange(4.0), (3, 6), None, "frames"),
            (np.arange(3.0), (3, 7), None, "divisible"),
            (np.arange(3.0), (3, 6), 4, "Zernikes"),
        ],
    )
    def test_rejects_inconsistent_series(self, times, shape, num_zern, match):
        with pytest.raises(ValueError, match=match):
            segment_zernike_series(
                times, np.zeros(shape), num_seg=NUM_SEG, num_zern=num_zern
            )


class TestHwoSimSpeckleField:
    @pytest.fixture
    def delivery(self, delivery_dir):
        return load_hwo_sim(
            delivery_dir[0],
            wavelengths_nm=WAVELENGTHS_NM,
            dtype=np.complex128,
            frame=Frame(550.0, 6.5, "circumscribed"),
        )

    def test_is_a_speckle_field(self, delivery):
        assert isinstance(_field(delivery, **_sinusoid()), AbstractSpeckleField)

    def test_realize_is_the_linear_model_excess_on_the_dark_zone(self, delivery, grid):
        """Allow two dot-product roundings per mode and eight for intensity and scale.

        The absolute scale sums unsigned field contributions to cover cancellation.

        Args:
            delivery: Synthetic linearized field.
            grid: Pixel mask and count.
        """
        dark_zone, _ = grid
        field = _field(delivery, **_tabulated())
        np.testing.assert_array_equal(field.spectral_breaks_nm(), [525.0, 575.0])
        image = field.realize(wavelength_nm=552.0, elapsed_s=100.0)
        assert image.grid == delivery["grid"]
        assert image.measure == Measure.integral()
        assert image.quantity == "stellar_flux_fraction"
        out = np.asarray(image.data)
        e = delivery["E0"][1]
        g = delivery["G"][1].reshape(e.size, -1)
        coeffs = np.asarray(_tabulated()["drift_coefficients"][1]).ravel()
        de = g @ coeffs
        expected = np.zeros(dark_zone.shape)
        expected[dark_zone] = (np.abs(e + de) ** 2 - np.abs(e) ** 2) * SCALE
        envelope = np.abs(g) @ np.abs(coeffs)
        scale = (2 * np.abs(e) + envelope) * envelope * SCALE
        bound = roundoff(scale, n_ops=2 * coeffs.size + 8, dtype=out.dtype)
        assert np.all(np.abs(out[dark_zone] - expected[dark_zone]) <= bound)
        assert np.all(out[~dark_zone] == 0.0)
        # The data mask is the delivery's own dark zone, not the array: a field
        # that reported its array would let a consumer support the zero fill.
        mask = np.asarray(field.data_mask())
        assert mask.dtype == bool and mask.shape == image.grid.shape
        np.testing.assert_array_equal(mask, dark_zone)

    def test_decomposition_sums_to_realize(self, delivery):
        """Distributing the scale over a sum incurs two products and an addition.

        Args:
            delivery: Synthetic linearized field.
        """
        field = _field(delivery, **_sinusoid())
        cross, quad = field.decompose_delta(wavelength_nm=500.0, elapsed_s=40.0)
        total = field.realize(wavelength_nm=500.0, elapsed_s=40.0).data
        assert np.all(
            np.abs(cross + quad - total) <= roundoff(abs(cross) + quad, n_ops=3)
        )
        assert np.all(np.asarray(quad) >= 0.0)

    def test_static_floor_is_the_nominal_intensity(self, delivery, grid):
        """Allow storage, complex magnitude and square roundoff, then one scale.

        Args:
            delivery: Synthetic linearized field.
            grid: Pixel mask and count.
        """
        dark_zone, _ = grid
        field = _field(delivery, **_sinusoid())
        floor = np.asarray(field.static_contrast(600.0))
        expected = np.abs(delivery["E0"][2]) ** 2
        assert np.all(
            np.abs(floor[dark_zone] - expected)
            <= roundoff(expected, n_ops=5, dtype=floor.dtype)
        )
        flux = np.asarray(field.static_flux_fraction(600.0))
        assert np.all(
            np.abs(flux - floor.astype(np.float64) * SCALE) <= roundoff(flux, n_ops=2)
        )

    def test_sinusoid_coefficients_sum_components(self, delivery):
        """Propagate angle roundoff through cosine (absolute slope at most one).

        Three products/adds form the angle. Cosine and amplitude each round
        once, followed by the component reduction.

        Args:
            delivery: Synthetic linearized field.
        """
        drift = _sinusoid()
        field = _field(delivery, **drift)
        t = 123.0
        amplitude = np.asarray(drift["amplitude"], dtype=np.float64)
        angle = 2 * np.pi * np.asarray(
            drift["frequency_hz"], dtype=np.float64
        ) * t + np.asarray(drift["phase"])
        expected = np.sum(amplitude * np.cos(angle), axis=-1)
        got = np.asarray(field.coefficients(elapsed_s=t))
        scale = np.sum(amplitude * (3 * abs(angle) + 2), axis=-1)
        bound = roundoff(scale, dtype=got.dtype) + sum_roundoff(
            amplitude.sum(axis=-1), amplitude.shape[-1], dtype=got.dtype
        )
        assert np.all(abs(got - expected) <= bound)

    def test_tabulated_coefficients_interpolate_and_hold(self, delivery):
        """Check midpoint difference, division, product and sum, and exact clamps.

        Args:
            delivery: Synthetic linearized field.
        """
        drift = _tabulated()
        field = _field(delivery, **drift)
        table = np.asarray(drift["drift_coefficients"])
        midpoint = 0.5 * (table[1].astype(np.float64) + table[2])
        bound = roundoff(abs(table[1]) + abs(table[2]), n_ops=4)
        assert np.all(
            abs(np.asarray(field.coefficients(elapsed_s=200.0)) - midpoint) <= bound
        )
        np.testing.assert_array_equal(
            np.asarray(field.coefficients(elapsed_s=-50.0)), table[0]
        )
        np.testing.assert_array_equal(
            np.asarray(field.coefficients(elapsed_s=1e4)), table[2]
        )
        assert field.drift_is_tabulated
        assert field.num_mode == NUM_SEG * NUM_ZERN

    def test_realize_jits(self, delivery):
        """Compare fused and eager arithmetic on the unsigned intensity scale.

        Each path interpolates (four operations), contracts the modal field
        (two operations per mode) and forms and scales the intensity (eight).

        Args:
            delivery: Synthetic linearized field.
        """
        field = _field(delivery, **_tabulated())
        jitted = eqx.filter_jit(
            lambda f, t: f.realize(wavelength_nm=550.0, elapsed_s=t).data
        )
        eager = np.asarray(field.realize(wavelength_nm=550.0, elapsed_s=150.0).data)
        bound_coeff = np.max(
            abs(np.asarray(field.drift_coefficients)[1:]), axis=0
        ).ravel()
        envelope = np.abs(delivery["G"][1].reshape(-1, field.num_mode)) @ bound_coeff
        scale = (2 * abs(delivery["E0"][1]) + envelope) * envelope * SCALE
        bound = roundoff(scale, n_ops=2 * (12 + 2 * field.num_mode), dtype=eager.dtype)
        difference = abs(np.asarray(jitted(field, 150.0)) - eager)
        mask = delivery["dark_zone"]
        assert np.all(difference[mask] <= bound)
        np.testing.assert_array_equal(difference[~mask], 0.0)

    def test_with_drift_swaps_only_the_drift(self, delivery):
        """Fails if re-anchoring drops, converts or copies an untouched field.

        Covers an unused ``drift_time_s`` kept on a sinusoid field, a pure
        re-anchor (a new object; the source keeps its epoch), a call with
        nothing supplied (still a new object, keeping a nonzero epoch on an
        already anchored field), and an amplitude swap traced
        inside ``eqx.filter_jit`` (a host round trip on the dark-zone index
        raises there). Truth source:
        the original field; every array other than the swapped drift is
        bitwise equal and the gain is not copied. Exact comparisons, except the
        doubled read, which is linear in the amplitude.
        """
        field = _field(delivery, **_sinusoid(), drift_time_s=jnp.asarray([0.0, 10.0]))
        swapped = field.with_drift(amplitude=2.0 * field.amplitude, epoch_d=1.0)
        assert swapped.epoch_d == 1.0
        assert swapped.gain is field.gain
        _assert_bitwise_equal_except(swapped, field, ("amplitude",))
        np.testing.assert_allclose(
            np.asarray(swapped.coefficients(elapsed_s=10.0)),
            2.0 * np.asarray(field.coefficients(elapsed_s=10.0)),
        )
        anchored = field.with_drift(epoch_d=2.0)
        assert anchored.epoch_d == 2.0
        assert anchored is not field and field.epoch_d == 0.0
        _assert_bitwise_equal_except(anchored, field, ())
        unchanged = field.with_drift()
        assert unchanged is not field
        _assert_bitwise_equal_except(unchanged, field, ())
        again = anchored.with_drift()
        assert again is not anchored and again.epoch_d == 2.0
        _assert_bitwise_equal_except(again, anchored, ())
        traced = eqx.filter_jit(lambda f, a: f.with_drift(amplitude=a))(
            field, 3.0 * field.amplitude
        )
        _assert_bitwise_equal_except(traced, field, ("amplitude",))
        np.testing.assert_array_equal(
            np.asarray(traced.amplitude), 3.0 * np.asarray(field.amplitude)
        )
        with pytest.raises(ValueError, match="tabulated"):
            _field(delivery, **_tabulated()).with_drift(epoch_d=1.0)

    def test_with_tabulated_drift_swaps_the_series(self, delivery):
        """Fails if the tabulated copy drops a field or loses the new epoch.

        The unused sinusoid ``frequency_hz`` and ``phase`` supplied with the
        series must survive, ``amplitude`` must stay None, and a call with
        nothing supplied keeps a nonzero epoch on an already anchored field.
        Truth source:
        the original field and the negated series. Exact comparisons, except
        the interpolated read, which is linear in the table.
        """
        sinusoid = _sinusoid()
        field = _field(
            delivery,
            **_tabulated(),
            frequency_hz=sinusoid["frequency_hz"],
            phase=sinusoid["phase"],
        )
        table = field.drift_coefficients
        swapped = field.with_tabulated_drift(drift_coefficients=-table)
        np.testing.assert_allclose(
            np.asarray(swapped.coefficients(elapsed_s=100.0)),
            -np.asarray(field.coefficients(elapsed_s=100.0)),
        )
        anchored = field.with_tabulated_drift(epoch_d=2.5)
        assert anchored.epoch_d == 2.5
        assert anchored is not field and field.epoch_d == 0.0
        assert anchored.gain is field.gain
        _assert_bitwise_equal_except(anchored, field, ())
        unchanged = field.with_tabulated_drift()
        assert unchanged is not field
        _assert_bitwise_equal_except(unchanged, field, ())
        again = anchored.with_tabulated_drift()
        assert again is not anchored and again.epoch_d == 2.5
        _assert_bitwise_equal_except(again, anchored, ())
        with pytest.raises(ValueError, match="samples"):
            field.with_tabulated_drift(drift_coefficients=table[:2])
        with pytest.raises(ValueError, match="sinusoid"):
            _field(delivery, **_sinusoid()).with_tabulated_drift(epoch_d=1.0)

    def test_max_safe_elapsed_follows_float32_resolution(self, delivery):
        eps32 = float(np.finfo(np.float32).eps)
        sinusoid = _field(delivery, **_sinusoid())
        f_max = float(np.max(np.asarray(sinusoid.frequency_hz)))
        assert sinusoid.max_safe_elapsed_s() == pytest.approx(
            0.05 / (2 * np.pi * eps32 * f_max)
        )
        tabulated = _field(delivery, **_tabulated())
        assert tabulated.max_safe_elapsed_s() == pytest.approx(0.05 * 100.0 / eps32)

    def test_requires_exactly_one_drift_form(self, delivery):
        with pytest.raises(ValueError, match="exactly one"):
            _field(delivery)
        with pytest.raises(ValueError, match="exactly one"):
            _field(delivery, **_sinusoid(), **_tabulated())

    def test_rejects_a_drift_on_the_wrong_mode_count(self, delivery):
        with pytest.raises(ValueError, match="modes"):
            _field(delivery, **_tabulated(num_zern=NUM_ZERN - 1))
        with pytest.raises(ValueError, match="modes"):
            _field(delivery, **_sinusoid(num_zern=NUM_ZERN - 1))

    def test_rejects_a_bare_2d_amplitude(self, delivery):
        drift = _sinusoid()
        drift["amplitude"] = drift["amplitude"][..., 0]
        with pytest.raises(ValueError, match="n_components"):
            _field(delivery, **drift)
