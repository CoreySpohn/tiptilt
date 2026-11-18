"""Tests for the tiptilt speckle-field family."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import s2d
from optixstuff import FocalResponse
from optixstuff.grids import Frame, Measure, PixelGrid

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
    return FocalResponse(e_nom, g, grid, _e_in(1.0), 500.0)


def _ensemble_cov(covariance, weights, n_samples):
    def eps0(key):
        amplitudes, phases = _draw_correlated_spectrum(covariance, key, weights)
        return jnp.sum(amplitudes * jnp.cos(phases), axis=1)

    keys = jax.random.split(jax.random.PRNGKey(7), n_samples)
    samples = np.asarray(jax.vmap(eps0)(keys))
    return np.cov(samples.T)


class TestCorrelatedDriftField:
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
                [field.coefficients((float(t)) * s2d)[0] for t in [0.0, *lags]]
            )

        keys = jax.random.split(jax.random.PRNGKey(11), n)
        eps = np.asarray(jax.vmap(eps_at)(keys))  # (n, 1 + n_lags)
        return np.mean(eps[:, :1] * eps[:, 1:], axis=0) / np.var(eps[:, 0])

    def test_uniform_grid_changes_only_at_the_band_edges(self):
        """On a uniform grid the interior widths are a constant that
        normalizes away, so a uniform-grid caller sees no reweighting except
        at the two extreme lines, which a trapezoid rule half-weights because
        the band stops there rather than continuing.
        """
        from tiptilt.speckle import _line_powers

        freqs = np.linspace(1e-4, 1e-2, 16)
        psd = np.linspace(1.0, 0.1, 16)
        with_df = _line_powers(freqs, psd)
        ratio = with_df / psd
        np.testing.assert_allclose(ratio[1:-1], ratio[8], rtol=1e-12)
        for edge in (0, -1):
            assert np.isclose(ratio[edge], 0.5 * ratio[8], rtol=1e-12)

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
                [field.coefficients((0.0) * s2d), field.coefficients((lag) * s2d)]
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
            return field.coefficients((0.0) * s2d)

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
            FocalResponse(lin.e_nom, lin.G, lin.grid, lin.input_energy, 500.0),
            times,
            eps_table,
        )
        out = speckle.realize(wavelength_nm=500.0, time_d=(50.0) * s2d).data
        assert out.shape == (32, 32)
        assert jnp.all(jnp.isfinite(out))
        at_zero = speckle.realize(wavelength_nm=500.0, time_d=(0.0) * s2d).data
        np.testing.assert_allclose(np.asarray(at_zero), 0.0, atol=1e-30)


def test_shared_law_matches_density_parent_formula(ingredients):
    """Parent coherent density law; bound is 32 binary64 eps of peak."""
    from optixstuff.speckle import LinearSpeckleField

    field = TabulatedSpeckleField(**ingredients)
    assert isinstance(field, LinearSpeckleField)
    assert type(field).realize is LinearSpeckleField.realize
    assert type(field).broadened is LinearSpeckleField.broadened
    response = ingredients["response"]
    eps = np.array([0.7, -0.7])
    de = np.einsum("m,myx->yx", eps, response.G)
    nominal = np.asarray(response.e_nom)
    parent = 2 * np.real(nominal.conj() * de) + abs(de) ** 2
    parent *= response.grid.pixel_area / response.input_energy
    image = field.realize(wavelength_nm=500.0, time_d=7.0 * s2d)
    np.testing.assert_allclose(
        image.data, parent, rtol=0, atol=32 * np.finfo(float).eps * np.max(abs(parent))
    )
    assert image.grid == field.grid
    assert image.measure == Measure.integral()
    assert image.quantity == "stellar_flux_fraction"


def test_tabulated_clock_origin_and_seconds(ingredients):
    """Clock error bound: eight absolute-day ulps times the table slope."""
    from hwoutils.constants import d2s

    epoch = 100.0
    field = TabulatedSpeckleField(**ingredients, epoch_d=epoch)
    elapsed = np.array([-1.0, 0.0, 5.0, 10.0, 15.0, 21.0])
    expected = np.column_stack(
        [
            np.interp(elapsed, ingredients["times_s"], col)
            for col in np.asarray(ingredients["eps_table"]).T
        ]
    )
    bound = 8 * np.spacing(epoch) * d2s * 0.15
    np.testing.assert_allclose(
        field.coefficients(epoch + elapsed * s2d), expected, rtol=0, atol=bound
    )
    np.testing.assert_array_equal(field.coefficients(epoch), [0.0, 0.0])
    # Between nodes the exact derivative is the tabulated slope, per day.
    derivative = jax.jacfwd(field.coefficients)(epoch + 5.0 * s2d)
    np.testing.assert_allclose(
        derivative, np.array([0.1, -0.1]) * d2s, rtol=8 * np.finfo(float).eps
    )


def test_density_normalization_number_is_unchanged():
    """Density e=2, G=3, eps=.5 gives (6+2.25)*.04/5 = .066."""
    grid = PixelGrid.regular(
        (1, 1), 0.2, "lod", frame=Frame(500.0, 1.0, "circumscribed")
    )
    field = TabulatedSpeckleField(
        FocalResponse(
            jnp.array([[2.0 + 0j]]), jnp.array([[[3.0 + 0j]]]), grid, 5.0, 500.0
        ),
        jnp.array([0.0, 10.0]),
        jnp.array([[0.0], [1.0]]),
    )
    assert float(
        field.realize(wavelength_nm=500.0, time_d=5.0 * s2d).data[0, 0]
    ) == pytest.approx(0.066, rel=8 * np.finfo(float).eps)


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
