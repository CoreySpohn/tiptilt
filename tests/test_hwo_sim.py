"""Tests for the hwo_sim reader and its speckle field, on synthetic files."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from optixstuff.grids import Frame, Measure
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
        path = delivery_dir[0]
        full = load_hwo_sim(
            path,
            wavelengths_nm=WAVELENGTHS_NM,
            frame=Frame(550.0, 6.5, "circumscribed"),
        )
        d = load_hwo_sim(
            path,
            wavelengths_nm=WAVELENGTHS_NM,
            wl_indices=(2, 0),
            frame=Frame(550.0, 6.5, "circumscribed"),
        )
        np.testing.assert_array_equal(d["E0"], full["E0"][[2, 0]])
        np.testing.assert_array_equal(d["G"], full["G"][[2, 0]])
        np.testing.assert_array_equal(d["wavelengths_nm"], WAVELENGTHS_NM[[2, 0]])

    def test_rejects_a_wavelength_grid_of_the_wrong_length(self, delivery_dir):
        with pytest.raises(ValueError, match="wavelengths"):
            load_hwo_sim(
                delivery_dir[0],
                wavelengths_nm=WAVELENGTHS_NM[:2],
                frame=Frame(550.0, 6.5, "circumscribed"),
            )


class TestSegmentZernikeSeries:
    def test_reshapes_slices_and_sorts(self):
        times = np.array([20.0, 0.0, 10.0])
        flat = np.arange(3 * NUM_SEG * 5, dtype=float).reshape(3, NUM_SEG * 5)
        t, c = segment_zernike_series(times, flat, num_seg=NUM_SEG, num_zern=3)
        np.testing.assert_array_equal(t, [0.0, 10.0, 20.0])
        expected = flat.reshape(3, NUM_SEG, 5)[[1, 2, 0], :, :3]
        np.testing.assert_array_equal(c, expected)

    def test_recenter_removes_the_window_mean(self):
        flat = np.random.default_rng(5).standard_normal((6, NUM_SEG * 3))
        _, c = segment_zernike_series(
            np.arange(6.0), flat, num_seg=NUM_SEG, recenter=True
        )
        np.testing.assert_allclose(np.asarray(c).mean(axis=0), 0.0, atol=1e-14)

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
        de = g @ np.asarray(_tabulated()["drift_coefficients"][1]).ravel()
        expected = np.zeros(dark_zone.shape)
        expected[dark_zone] = (np.abs(e + de) ** 2 - np.abs(e) ** 2) * SCALE
        np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-30)
        assert np.all(out[~dark_zone] == 0.0)
        # The data mask is the delivery's own dark zone, not the array: a field
        # that reported its array would let a consumer support the zero fill.
        mask = np.asarray(field.data_mask())
        assert mask.dtype == bool and mask.shape == image.grid.shape
        np.testing.assert_array_equal(mask, dark_zone)

    def test_decomposition_sums_to_realize(self, delivery):
        field = _field(delivery, **_sinusoid())
        cross, quad = field.decompose_delta(wavelength_nm=500.0, elapsed_s=40.0)
        total = field.realize(wavelength_nm=500.0, elapsed_s=40.0).data
        np.testing.assert_allclose(np.asarray(cross + quad), np.asarray(total))
        assert np.all(np.asarray(quad) >= 0.0)

    def test_static_floor_is_the_nominal_intensity(self, delivery, grid):
        dark_zone, _ = grid
        field = _field(delivery, **_sinusoid())
        floor = np.asarray(field.static_contrast(600.0))
        np.testing.assert_allclose(floor[dark_zone], np.abs(delivery["E0"][2]) ** 2)
        np.testing.assert_allclose(
            np.asarray(field.static_flux_fraction(600.0)), floor * SCALE
        )

    def test_sinusoid_coefficients_sum_components(self, delivery):
        drift = _sinusoid()
        field = _field(delivery, **drift)
        t = 123.0
        expected = np.sum(
            np.asarray(drift["amplitude"])
            * np.cos(
                2 * np.pi * np.asarray(drift["frequency_hz"]) * t
                + np.asarray(drift["phase"])
            ),
            axis=-1,
        )
        np.testing.assert_allclose(
            np.asarray(field.coefficients(elapsed_s=t)), expected
        )

    def test_tabulated_coefficients_interpolate_and_hold(self, delivery):
        drift = _tabulated()
        field = _field(delivery, **drift)
        table = np.asarray(drift["drift_coefficients"])
        np.testing.assert_allclose(
            np.asarray(field.coefficients(elapsed_s=200.0)), 0.5 * (table[1] + table[2])
        )
        np.testing.assert_allclose(
            np.asarray(field.coefficients(elapsed_s=-50.0)), table[0]
        )
        np.testing.assert_allclose(
            np.asarray(field.coefficients(elapsed_s=1e4)), table[2]
        )
        assert field.drift_is_tabulated
        assert field.num_mode == NUM_SEG * NUM_ZERN

    def test_realize_jits(self, delivery):
        field = _field(delivery, **_tabulated())
        jitted = eqx.filter_jit(
            lambda f, t: f.realize(wavelength_nm=550.0, elapsed_s=t).data
        )
        np.testing.assert_allclose(
            np.asarray(jitted(field, 150.0)),
            np.asarray(field.realize(wavelength_nm=550.0, elapsed_s=150.0).data),
        )

    def test_with_drift_swaps_only_the_drift(self, delivery):
        """Fails if re-anchoring drops, converts or copies an untouched field.

        Covers an unused ``drift_time_s`` kept on a sinusoid field, a pure
        re-anchor (a new object; the source keeps its epoch), a call with
        nothing supplied (still a new object), and an amplitude swap traced
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
        series must survive, and ``amplitude`` must stay None. Truth source:
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
