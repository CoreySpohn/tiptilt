"""Tests for the tiptilt speckle-field family."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.numerics import eps, roundoff
from optixstuff import FocalResponse
from optixstuff.grids import Frame, Measure, PixelGrid

from tiptilt.hwo_sim import HwoSimSpeckleField
from tiptilt.speckle import (
    TabulatedSpeckleField,
    _draw_correlated_spectrum,
    correlated_drift_field,
    grouped_drift_field,
)


def _e_in(norm, du=0.25):
    """input_energy that makes the derived normalization equal ``norm``."""
    return norm * du * du


@pytest.fixture
def ingredients():
    rng = np.random.default_rng(0)
    m, ny, nx = 2, 4, 4
    g = rng.standard_normal((m, ny, nx)) + 1j * rng.standard_normal((m, ny, nx))
    e = rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))
    return {
        "response": FocalResponse(
            e,
            g,
            PixelGrid.focal(4, 0.25, Frame(500.0, 1.0, "circumscribed")),
            _e_in(3.0),
            500.0,
            wavelength_law="opd",
        ),
        "times_s": jnp.asarray([0.0, 10.0, 20.0]),
        "eps_table": jnp.asarray([[0.0, 0.0], [1.0, -1.0], [2.0, 0.5]]),
    }


def _synthetic_lin(seed=2, m=2, ny=3, nx=3):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((m, ny, nx)) + 1j * rng.standard_normal((m, ny, nx))
    e = rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))
    return jnp.asarray(e), jnp.asarray(g)


def _focal_response(e_nom, g):
    grid = PixelGrid.regular(
        e_nom.shape, 0.25, "lod", frame=Frame(500.0, 1.0, "circumscribed")
    )
    return FocalResponse(e_nom, g, grid, _e_in(1.0), 500.0, wavelength_law="opd")


def _ensemble_cov(covariance, weights, n_samples):
    def eps0(key):
        amplitudes, phases = _draw_correlated_spectrum(covariance, key, weights)
        return jnp.sum(amplitudes * jnp.cos(phases), axis=1)

    keys = jax.random.split(jax.random.PRNGKey(7), n_samples)
    samples = np.asarray(jax.vmap(eps0)(keys))
    return np.cov(samples.T)


class TestCorrelatedDriftField:
    def test_covariance_matches_response_modes(self, ingredients):
        """The covariance dimension counts driving modes (mutation, exact sizes).

        Fails if a single coefficient can broadcast across the two response modes.
        """
        response = ingredients["response"]
        kwargs = dict(
            key=jax.random.PRNGKey(0),
            frequencies_hz=jnp.arange(1, 17),
            psd=jnp.ones(16),
        )
        with pytest.raises(ValueError, match=r"\(1, 1\).*2 modes"):
            correlated_drift_field(response, jnp.eye(1), **kwargs)
        field = correlated_drift_field(response, jnp.eye(2), **kwargs)
        assert field.coefficients(elapsed_s=0.0).shape == (2,)

    def test_returns_an_analytic_speckle_field(self):
        e_nom, g = _synthetic_lin()
        covariance = jnp.asarray([[4.0, 1.0], [1.0, 9.0]])
        freqs = jnp.linspace(1e-4, 1e-2, 16)
        psd = jnp.ones(16)
        field = correlated_drift_field(
            _focal_response(e_nom, g),
            covariance,
            key=jax.random.PRNGKey(0),
            frequencies_hz=freqs,
            psd=psd,
        )
        from physicaloptix import AnalyticSpeckleField

        assert isinstance(field, AnalyticSpeckleField)
        assert field.amplitudes.shape == (2, 16)
        assert field.frequencies_hz.shape == (16,)

    def test_recovers_the_target_covariance(self):
        covariance = jnp.asarray([[4.0, 1.5], [1.5, 9.0]])
        weights = 2.0 * jnp.ones(32) / 32.0
        cov = _ensemble_cov(covariance, weights, 8000)
        np.testing.assert_allclose(cov, np.asarray(covariance), atol=0.4)

    def test_is_deterministic_given_a_key(self):
        e_nom, g = _synthetic_lin()
        covariance = jnp.asarray([[4.0, 1.0], [1.0, 9.0]])
        freqs = jnp.linspace(1e-4, 1e-2, 16)
        psd = jnp.ones(16)
        args = (_focal_response(e_nom, g), covariance)
        kw = dict(
            key=jax.random.PRNGKey(3),
            frequencies_hz=freqs,
            psd=psd,
        )
        first = correlated_drift_field(*args, **kw)
        second = correlated_drift_field(*args, **kw)
        np.testing.assert_array_equal(
            np.asarray(first.amplitudes), np.asarray(second.amplitudes)
        )

    def test_rank_deficient_covariance_is_handled(self):
        """A singular (rank-1) covariance must not crash and is still
        recovered -- this is why the draw uses an eigendecomposition, not a
        Cholesky factorization."""
        v = jnp.asarray([1.0, 2.0])
        covariance = jnp.outer(v, v)  # rank-1 PSD, singular
        weights = 2.0 * jnp.ones(32) / 32.0
        cov = _ensemble_cov(covariance, weights, 8000)
        np.testing.assert_allclose(cov, np.asarray(covariance), atol=0.4)

    def _builder_kwargs(self, psd):
        e_nom, g = _synthetic_lin()
        return e_nom, g, jnp.asarray([[4.0, 1.0], [1.0, 9.0]]), psd

    def test_rejects_zero_total_psd(self):
        e_nom, g, cov, psd = self._builder_kwargs(jnp.zeros(8))
        with pytest.raises(ValueError, match="psd"):
            correlated_drift_field(
                _focal_response(e_nom, g),
                cov,
                key=jax.random.PRNGKey(0),
                frequencies_hz=jnp.linspace(1e-4, 1e-2, 8),
                psd=psd,
            )

    def test_rejects_indefinite_covariance(self):
        e_nom, g = _synthetic_lin()
        indefinite = jnp.asarray([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues -1, 3
        with pytest.raises(ValueError, match="semidefinite"):
            correlated_drift_field(
                _focal_response(e_nom, g),
                indefinite,
                key=jax.random.PRNGKey(0),
                frequencies_hz=jnp.linspace(1e-4, 1e-2, 8),
                psd=jnp.ones(8),
            )

    def test_rejects_asymmetric_covariance(self):
        e_nom, g = _synthetic_lin()
        asymmetric = jnp.asarray([[4.0, 1.0], [0.5, 9.0]])
        with pytest.raises(ValueError, match="symmetric"):
            correlated_drift_field(
                _focal_response(e_nom, g),
                asymmetric,
                key=jax.random.PRNGKey(0),
                frequencies_hz=jnp.linspace(1e-4, 1e-2, 8),
                psd=jnp.ones(8),
            )

    def test_warns_when_neff_is_low(self):
        """A steeply red PSD concentrates power in one frequency, so a single
        realization is unrepresentative: the builder warns."""
        e_nom, g, cov, red = self._builder_kwargs(
            jnp.asarray([1.0, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
        )
        with pytest.warns(UserWarning, match="N_eff"):
            correlated_drift_field(
                _focal_response(e_nom, g),
                cov,
                key=jax.random.PRNGKey(0),
                frequencies_hz=jnp.linspace(1e-4, 1e-2, 8),
                psd=red,
            )


class TestPsdIsADensity:
    """``psd`` is a DENSITY, so line j carries ``psd[j] df_j``.

    On the uniform grids the older tests use, the widths are a constant that
    normalizes away and nothing changes. On a LOG grid -- the natural choice
    for drift spanning decades, and what physicaloptix's SpeckleProcess
    builds -- using the bare ordinate synthesizes ``S(f) / f`` and the field
    decorrelates far too slowly.
    """

    KNEE = 1e-3
    TAU_C = 1.0 / (2.0 * np.pi * KNEE)

    def _log_grid(self, n=64, decades_below=1.7, decades_above=2.3):
        lo = np.log10(self.KNEE) - decades_below
        hi = np.log10(self.KNEE) + decades_above
        f = np.logspace(lo, hi, n)
        return jnp.asarray(f), jnp.asarray(1.0 / (1.0 + (f / self.KNEE) ** 2))

    def _measured_rho(self, freqs, psd, lags, *, n=4000):
        e_nom, g = _synthetic_lin(m=1)
        cov = jnp.asarray([[1.0]])

        def eps_at(key):
            field = correlated_drift_field(
                _focal_response(e_nom, g),
                cov,
                key=key,
                frequencies_hz=freqs,
                psd=psd,
            )
            return jnp.stack(
                [field.coefficients(elapsed_s=float(t))[0] for t in [0.0, *lags]]
            )

        keys = jax.random.split(jax.random.PRNGKey(11), n)
        eps = np.asarray(jax.vmap(eps_at)(keys))  # (n, 1 + n_lags)
        return np.mean(eps[:, :1] * eps[:, 1:], axis=0) / np.var(eps[:, 0])

    def test_uniform_grid_changes_only_at_the_band_edges(self):
        """On a uniform grid the interior widths are a constant that
        normalizes away, so a uniform-grid caller sees no reweighting except
        at the two extreme lines, which a trapezoid rule half-weights because
        the band stops there rather than continuing. Two rounded frequency
        nodes determine each half-width. Compare with the host analytic
        spacing using binary64 host epsilon at the largest frequency.
        """
        from tiptilt.speckle import _line_powers

        freqs = np.linspace(1e-4, 1e-2, 16)
        psd = np.linspace(1.0, 0.1, 16)
        with_df = _line_powers(freqs, psd)
        ratio = with_df / psd
        spacing = (freqs[-1] - freqs[0]) / (freqs.size - 1)
        expected = np.full_like(freqs, spacing)
        expected[[0, -1]] *= 0.5
        np.testing.assert_allclose(
            ratio, expected, rtol=0, atol=2 * eps(freqs.dtype) * freqs[-1]
        )

    def test_log_grid_lorentzian_decorrelates_on_schedule(self):
        """A Lorentzian PSD on a log grid must realize the OU kernel
        exp(-tau / tau_c) that its knee names."""
        freqs, psd = self._log_grid()
        lags = [0.25 * self.TAU_C, self.TAU_C]
        rho = self._measured_rho(freqs, psd, lags)
        target = np.exp(-np.array(lags) / self.TAU_C)
        np.testing.assert_allclose(rho, target, atol=0.04)

    def test_mismatched_shapes_are_rejected(self):
        e_nom, g = _synthetic_lin()
        with pytest.raises(ValueError, match="same shape"):
            correlated_drift_field(
                _focal_response(e_nom, g),
                jnp.asarray([[4.0, 1.0], [1.0, 9.0]]),
                key=jax.random.PRNGKey(0),
                frequencies_hz=jnp.linspace(1e-4, 1e-2, 8),
                psd=jnp.ones(7),
            )


class TestGroupedDriftField:
    """Mode blocks may drift on their OWN timescales (upgrades #1).

    A single (covariance, psd) pair forces the cross-spectral density to
    factorize as C_a x psd(f): every mode decorrelates on one schedule. These
    pin that a fast block and a slow block coexist in one field.
    """

    def _grid(self, knee, n=48):
        f = np.logspace(np.log10(knee) - 1.7, np.log10(knee) + 2.3, n)
        return jnp.asarray(f), jnp.asarray(1.0 / (1.0 + (f / knee) ** 2))

    def _blocks(self, knee_fast=1e-2, knee_slow=1e-4):
        return [
            (jnp.asarray([[1.0]]), *self._grid(knee_fast)),
            (jnp.asarray([[1.0]]), *self._grid(knee_slow)),
        ]

    def test_blocks_keep_their_own_timescales(self):
        """At a lag between the two decorrelation times, the fast block has
        lost its correlation and the slow block has not."""
        e_nom, g = _synthetic_lin(m=2)
        groups = self._blocks()
        lag = 200.0  # 1/(2 pi 1e-2) = 16 s << lag << 1/(2 pi 1e-4) = 1592 s

        def eps_at(key):
            field = grouped_drift_field(
                _focal_response(e_nom, g),
                groups,
                key=key,
            )
            return jnp.stack(
                [field.coefficients(elapsed_s=0.0), field.coefficients(elapsed_s=lag)]
            )

        keys = jax.random.split(jax.random.PRNGKey(5), 3000)
        eps = np.asarray(jax.vmap(eps_at)(keys))  # (n, 2, m)
        rho = np.mean(eps[:, 0] * eps[:, 1], axis=0) / np.var(eps[:, 0], axis=0)
        assert rho[0] < 0.4  # fast block: decorrelated
        assert rho[1] > 0.85  # slow block: still correlated

    def test_per_block_grids_produce_a_two_dimensional_frequency_axis(self):
        e_nom, g = _synthetic_lin(m=2)
        field = grouped_drift_field(
            _focal_response(e_nom, g),
            self._blocks(),
            key=jax.random.PRNGKey(0),
        )
        assert field.frequencies_hz.shape == (2, 48)
        assert field.amplitudes.shape == (2, 48)

    def test_one_shared_grid_collapses_to_the_single_timescale_case(self):
        """When every block names the same grid the result is indistinguishable
        in shape from correlated_drift_field's, so nothing downstream has to
        care whether a caller used blocks."""
        e_nom, g = _synthetic_lin(m=2)
        freqs, psd = self._grid(1e-3)
        groups = [
            (jnp.asarray([[1.0]]), freqs, psd),
            (jnp.asarray([[1.0]]), freqs, psd),
        ]
        field = grouped_drift_field(
            _focal_response(e_nom, g),
            groups,
            key=jax.random.PRNGKey(0),
        )
        assert field.frequencies_hz.shape == (48,)

    def test_within_block_covariance_is_realized(self):
        """A correlated 2x2 block keeps its cross-mode covariance."""
        e_nom, g = _synthetic_lin(m=2)
        cov = jnp.asarray([[4.0, 1.5], [1.5, 9.0]])
        freqs, psd = self._grid(1e-3)

        def eps0(key):
            field = grouped_drift_field(
                _focal_response(e_nom, g),
                [(cov, freqs, psd)],
                key=key,
            )
            return field.coefficients(elapsed_s=0.0)

        keys = jax.random.split(jax.random.PRNGKey(9), 8000)
        samples = np.asarray(jax.vmap(eps0)(keys))
        np.testing.assert_allclose(np.cov(samples.T), np.asarray(cov), atol=0.6)

    def test_mode_count_must_match_g(self):
        e_nom, g = _synthetic_lin(m=2)
        freqs, psd = self._grid(1e-3)
        with pytest.raises(ValueError, match="G has"):
            grouped_drift_field(
                _focal_response(e_nom, g),
                [(jnp.asarray([[1.0]]), freqs, psd)],
                key=jax.random.PRNGKey(0),
            )

    def test_frequency_counts_must_agree(self):
        e_nom, g = _synthetic_lin(m=2)
        with pytest.raises(ValueError, match="frequencies but group 0"):
            grouped_drift_field(
                _focal_response(e_nom, g),
                [
                    (jnp.asarray([[1.0]]), *self._grid(1e-2, n=48)),
                    (jnp.asarray([[1.0]]), *self._grid(1e-4, n=32)),
                ],
                key=jax.random.PRNGKey(0),
            )

    def test_empty_groups_are_rejected(self):
        e_nom, g = _synthetic_lin(m=2)
        with pytest.raises(ValueError, match="at least one"):
            grouped_drift_field(
                _focal_response(e_nom, g),
                [],
                key=jax.random.PRNGKey(0),
            )


class TestPhysicaloptixIntegration:
    def test_tabulated_field_from_a_linearized_path(self):
        """A real (E_nom, G) from physicaloptix.linearize over a Zernike basis
        drives a tabulated drift into a finite, floor-excluded contrast map."""
        from physicaloptix import (
            Field,
            Fraunhofer,
            OpticalPath,
            PlaneKind,
            Stage,
            linearize,
            zernike_basis,
        )

        pupil = PixelGrid.pupil(48)
        xx, yy = map(np.asarray, pupil.coordinates())
        aperture = (xx**2 + yy**2 <= 0.25).astype(float)
        path = OpticalPath(
            stages=(
                Stage(
                    "science",
                    Fraunhofer(
                        grid_in=pupil,
                        grid_out=PixelGrid.focal(
                            32, 0.5, Frame(500.0, 1.0, "circumscribed")
                        ),
                    ),
                ),
            )
        )
        field = Field(
            data=jnp.asarray(aperture, dtype=complex),
            grid=pupil,
            plane=PlaneKind.PUPIL,
        )
        lin = linearize(
            path, field, zernike_basis(pupil, 5, rms_nm=10.0), wavelength_nm=500.0
        )

        times = jnp.asarray([0.0, 100.0])
        eps_table = jnp.asarray(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.5, -0.3, 0.2, -0.1]]
        )
        speckle = TabulatedSpeckleField(
            FocalResponse(
                lin.e_nom,
                lin.G,
                lin.grid,
                lin.input_energy,
                500.0,
                wavelength_law="opd",
            ),
            times,
            eps_table,
        )
        out = speckle.realize(wavelength_nm=500.0, elapsed_s=50.0).data
        assert out.shape == (32, 32)
        assert jnp.all(jnp.isfinite(out))
        at_zero = speckle.realize(wavelength_nm=500.0, elapsed_s=0.0).data
        np.testing.assert_allclose(np.asarray(at_zero), 0.0, atol=1e-30)


def test_shared_law_matches_density_parent_formula(ingredients):
    """Check coherent density against a host binary64 contraction.

    Allow 16 roundings on the unsigned intensity for interpolation, two
    complex-by-real modal products, their sum, modulus square and scaling.

    Args:
        ingredients: Synthetic tabulated drift and optical response.
    """
    from optixstuff.speckle import LinearSpeckleField

    field = TabulatedSpeckleField(**ingredients)
    assert isinstance(field, LinearSpeckleField)
    assert type(field).realize is LinearSpeckleField.realize
    assert type(field).broadened is LinearSpeckleField.broadened
    response = ingredients["response"]
    eps = np.array([0.7, -0.7])
    gain = np.asarray(response.G, dtype=np.complex128)
    de = np.einsum("m,myx->yx", eps, gain)
    nominal = np.asarray(response.e_nom, dtype=np.complex128)
    parent = 2 * np.real(nominal.conj() * de) + abs(de) ** 2
    normalization = response.grid.pixel_area / float(response.input_energy)
    parent *= normalization
    image = field.realize(wavelength_nm=500.0, elapsed_s=7.0)
    envelope = np.einsum("m,myx->yx", abs(eps), abs(gain))
    scale = (2 * abs(nominal) + envelope) * envelope * normalization
    bound = roundoff(scale, n_ops=16, dtype=image.data.dtype)
    assert np.all(abs(np.asarray(image.data) - parent) <= bound)
    assert image.grid == field.grid
    assert image.measure == Measure.integral()
    assert image.quantity == "stellar_flux_fraction"
    # A tabulated field holds data in every cell of its grid.
    assert field.data_mask() is None


def test_tabulated_clock_origin_and_seconds(ingredients):
    """Fails if the table is read in days, at an epoch offset, or not held.

    Truth: the piecewise-linear table at elapsed seconds, written out; the
    values are dyadic, so the comparison is exact. The field sits at
    ``epoch_d = 100`` and must ignore it. The derivative between nodes is the
    tabulated slope per second. The endpoint differences are exact and the
    quotient rounds once in the derivative dtype.

    Args:
        ingredients: Synthetic tabulated drift and optical response.
    """
    field = TabulatedSpeckleField(**ingredients, epoch_d=100.0)
    elapsed = np.array([-1.0, 0.0, 5.0, 10.0, 15.0, 21.0])
    expected = np.array(
        [[0.0, 0.0], [0.0, 0.0], [0.5, -0.5], [1.0, -1.0], [1.5, -0.25], [2.0, 0.5]]
    )
    np.testing.assert_array_equal(field.coefficients(elapsed_s=elapsed), expected)
    derivative = jax.jacfwd(lambda t: field.coefficients(elapsed_s=t))(5.0)
    np.testing.assert_allclose(
        derivative,
        [0.1, -0.1],
        rtol=0,
        atol=float(roundoff(0.1, dtype=derivative.dtype)),
    )


def _tabulated(epoch_d):
    """A tabulated field in the active precision; coefficients at most 2 nm."""
    rng = np.random.default_rng(0)
    g = rng.standard_normal((2, 4, 4)) + 1j * rng.standard_normal((2, 4, 4))
    e = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    grid = PixelGrid.focal(4, 0.25, Frame(500.0, 1.0, "circumscribed"))
    return TabulatedSpeckleField(
        FocalResponse(e, g, grid, _e_in(3.0), 500.0, wavelength_law="opd"),
        [0.0, 10.0, 20.0],
        [[0.0, 0.0], [1.0, -1.0], [2.0, 0.5]],
        epoch_d=epoch_d,
    )


def _hwo_sim(epoch_d):
    """A two-component sinusoid drift on three dark-zone pixels.

    Amplitudes below 1 nm, frequencies below 1e-3 Hz and phases below 2 pi, so
    at 13.3 s each mode sums two terms whose cosine arguments are below 6.4 rad.
    """
    rng = np.random.default_rng(1)
    dark_zone = np.zeros((4, 4), bool)
    dark_zone[1, :3] = True
    shape = (1, 2, 2)
    return HwoSimSpeckleField(
        rng.standard_normal((1, 3)) + 1j * rng.standard_normal((1, 3)),
        rng.standard_normal((1, 3, 2)) + 1j * rng.standard_normal((1, 3, 2)),
        dark_zone,
        wavelengths_nm=[500.0],
        grid=PixelGrid.focal(4, 0.25, Frame(500.0, 1.0, "circumscribed")),
        epoch_d=epoch_d,
        contrast_to_flux_fraction=7e-3,
        amplitude=rng.random(shape),
        frequency_hz=rng.random(shape) * 1e-3,
        phase=rng.random(shape) * 2 * np.pi,
    )


@pytest.mark.parametrize("build", [_tabulated, _hwo_sim], ids=["tabulated", "hwo_sim"])
def test_subsecond_host_epoch_offset(build, precision):
    """Fails if absolute days collapse before subtraction or drift time widens.

    The dyadic day offset and its conversion to seconds are exact in binary64
    and binary32. At day 60000 the absolute times coincide in binary32.
    A one-day tangent gives exactly 86400 seconds on either clock.

    Args:
        build: Construct a speckle field anchored at the supplied epoch.
        precision: Real dtype of the current test lane.
    """
    from hwoutils.constants import d2s
    from hwoutils.leaves import canonical
    from hwoutils.testing import assert_precision_closed

    epoch = 60000.0
    delta_d = 2.0**-18
    field = canonical(build(epoch), precision="float32")
    expected_s = delta_d * d2s
    elapsed = field.elapsed_s(np.float64(epoch + delta_d))
    np.testing.assert_array_equal(elapsed, expected_s)
    coefficients = field.coefficients(elapsed_s=elapsed)
    assert coefficients.dtype == np.float32
    np.testing.assert_array_equal(
        coefficients, field.coefficients(elapsed_s=np.float32(expected_s))
    )
    if precision == np.float64:
        t32 = jnp.asarray(expected_s, dtype=jnp.float32)
        assert_precision_closed(lambda f, t: f.coefficients(elapsed_s=t), field, t32)
        _, from_days = jax.jvp(
            lambda d: field.coefficients(elapsed_s=field.elapsed_s(d)),
            (jnp.asarray(epoch + delta_d),),
            (jnp.asarray(1.0),),
        )
        _, from_seconds = jax.jvp(
            lambda t: field.coefficients(elapsed_s=t),
            (t32,),
            (jnp.asarray(d2s, dtype=t32.dtype),),
        )
        np.testing.assert_array_equal(from_days, from_seconds)


@pytest.mark.parametrize("hwo_sim", [False, True])
@pytest.mark.parametrize("host_knots", [False, True])
def test_tabulated_time_is_conditioned_before_narrowing(hwo_sim, host_knots, precision):
    """At 2**26 + 1 s the exact linear coefficient is one eighth.

    The spacing, offset and interpolated values are dyadic and exact.
    Host knots also resolve sub-ULP gaps before storage.

    Args:
        hwo_sim: Exercise the HWO-Sim reader instead of the linear response.
        host_knots: Supply closely spaced host knots when true.
        precision: Real floating dtype of the active test lane.
    """
    origin = 2.0**26
    gap = 2.0 if host_knots else 8.0
    knots = np.array([origin, origin + gap])
    if not host_knots:
        knots = jnp.asarray(knots, dtype=jnp.float32)
    if hwo_sim:
        field = HwoSimSpeckleField(
            np.zeros((1, 1)),
            np.ones((1, 1, 1)),
            np.ones((1, 1), dtype=bool),
            wavelengths_nm=[500.0],
            grid=PixelGrid.focal(1, 0.25, Frame(500.0, 1.0, "circumscribed")),
            contrast_to_flux_fraction=1.0,
            drift_time_s=knots,
            drift_coefficients=jnp.asarray([[[0.0]], [[1.0]]], dtype=jnp.float32),
        )
    else:
        field = TabulatedSpeckleField(
            _focal_response(np.zeros((1, 1)), np.ones((1, 1, 1))),
            knots,
            jnp.asarray([[0.0], [1.0]], dtype=jnp.float32),
        )

    def read(elapsed):
        return field.coefficients(elapsed_s=elapsed)

    elapsed = np.float64(origin + 1)
    expected = 1.0 / gap
    assert read(elapsed).item() == expected
    assert jax.jit(lambda: read(np.float64(origin + 1)))().item() == expected
    compiled = jax.jit(read)
    if precision == np.float64:
        query = jnp.asarray(elapsed, dtype=precision)
        assert read(query).item() == expected
        assert compiled(query).item() == expected
    for elapsed, expected in ((origin - 8, 0.0), (origin + 8, 1.0)):
        assert compiled(jnp.asarray(elapsed)).item() == expected


@pytest.mark.parametrize(
    ("build", "scale"),
    [(_tabulated, 2.0), (_hwo_sim, 2 * 1.0 * (1 + 6.4))],
    ids=["tabulated", "hwo_sim"],
)
def test_fields_run_on_their_own_clock(build, scale, precision):
    """Fails if a field reads ``epoch_d``, overrides the kernel or takes days.

    Truth: the same field anchored at zero. Two fields that differ only in
    ``epoch_d`` (0 and 60000.123) give bitwise equal coefficients and images
    at each elapsed second, eagerly and compiled, for a Python float, a NumPy
    array and a JAX array on both sides. The kernel is the
    interface's own, and a positional time or a ``time_d`` keyword raises.
    Eager against ``eqx.filter_jit``: tolerance basis floating-point, 8 eps of
    ``scale``, the sum over terms of amplitude times (1 + cosine argument),
    for the coefficients and 64 eps of the largest pixel for the images,
    because the compiled program contracts multiply-adds (one rounding fewer);
    an error only one execution mode makes (a traced-only image scaling)
    fails it.
    The times are off the float32 grid, so a branch that casts a JAX array
    through float32 fails the x64 comparison.
    """
    from optixstuff.speckle import AbstractSpeckleField

    zero, anchored = build(0.0), build(60000.123)
    assert type(anchored).elapsed_s is AbstractSpeckleField.elapsed_s
    coefficients = eqx.filter_jit(lambda field, t: field.coefficients(elapsed_s=t))
    realize = eqx.filter_jit(
        lambda field, t: field.realize(wavelength_nm=500.0, elapsed_s=t).data
    )
    eps = float(np.finfo(precision).eps)
    floor = 8 * eps * scale
    for elapsed in (0.0, 7.1, 13.3):
        reference = zero.coefficients(elapsed_s=elapsed)
        for form in (float, np.asarray, jnp.asarray):
            t = form(elapsed)
            eager = zero.coefficients(elapsed_s=t)
            np.testing.assert_array_equal(eager, reference)
            np.testing.assert_array_equal(anchored.coefficients(elapsed_s=t), eager)
            compiled = coefficients(zero, t)
            np.testing.assert_array_equal(coefficients(anchored, t), compiled)
            np.testing.assert_allclose(compiled, eager, rtol=0, atol=floor)
            image = zero.realize(wavelength_nm=500.0, elapsed_s=t).data
            np.testing.assert_array_equal(
                anchored.realize(wavelength_nm=500.0, elapsed_s=t).data, image
            )
            compiled_image = realize(zero, t)
            np.testing.assert_array_equal(realize(anchored, t), compiled_image)
            image_floor = 64 * eps * np.max(np.abs(image))
            np.testing.assert_allclose(compiled_image, image, rtol=0, atol=image_floor)
    with pytest.raises(TypeError):
        anchored.coefficients(1.0)
    with pytest.raises(TypeError):
        anchored.realize(wavelength_nm=500.0, time_d=1.0)


def test_density_normalization_number_is_unchanged():
    """Density e=2, G=3, eps=.5 gives (6+2.25)*.04/5 = .066.

    Allow eight operations for field scaling, modal and cross products,
    modulus square, summation and normalization.
    """
    grid = PixelGrid.regular(
        (1, 1), 0.2, "lod", frame=Frame(500.0, 1.0, "circumscribed")
    )
    field = TabulatedSpeckleField(
        FocalResponse(
            jnp.array([[2.0 + 0j]]),
            jnp.array([[[3.0 + 0j]]]),
            grid,
            5.0,
            500.0,
            wavelength_law="opd",
        ),
        jnp.array([0.0, 10.0]),
        jnp.array([[0.0], [1.0]]),
    )
    value = field.realize(wavelength_nm=500.0, elapsed_s=5.0).data[0, 0]
    assert abs(float(value) - 0.066) <= roundoff(0.066, n_ops=8, dtype=value.dtype)


def test_line_powers_nonuniform_and_discrete_line():
    """Affine PSD integrals agree to 2e-15 relative floating-point error.

    For f=(1,2,5), endpoint half widths are (0.5,1.5), and the
    interior half span is 2. The PSD 2*f+1 integrates to [f**2+f].
    A singleton retains its declared discrete line power exactly.
    """
    from tiptilt.speckle import _line_powers

    f = np.array([1.0, 2.0, 5.0])
    expected = np.array([0.5, 2.0, 1.5]) * (2 * f + 1)
    actual = _line_powers(f, 2 * f + 1)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)
    np.testing.assert_allclose(actual.sum(), (5**2 + 5) - (1**2 + 1), rtol=2e-15)
    np.testing.assert_array_equal(_line_powers([2.0], [7.0]), [7.0])
