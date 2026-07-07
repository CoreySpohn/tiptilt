"""Tests for the wavefronts speckle-field family."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from optixstuff.speckle import AbstractSpeckleField

from wavefronts.speckle import (
    TabulatedSpeckleField,
    _draw_correlated_spectrum,
    correlated_drift_field,
)


@pytest.fixture
def ingredients():
    rng = np.random.default_rng(0)
    m, ny, nx = 2, 4, 4
    g = rng.standard_normal((m, ny, nx)) + 1j * rng.standard_normal((m, ny, nx))
    e = rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))
    return {
        "e_nom": jnp.asarray(e),
        "G": jnp.asarray(g),
        "times_s": jnp.asarray([0.0, 10.0, 20.0]),
        "eps_table": jnp.asarray([[0.0, 0.0], [1.0, -1.0], [2.0, 0.5]]),
        "normalization": 3.0,
    }


def _incoherent_delta(g, eps, norm):
    g_eps = jnp.tensordot(jnp.asarray(eps), g, axes=1)
    return np.asarray(jnp.abs(g_eps) ** 2 / norm)


class TestTabulatedSpeckleField:
    def test_is_a_speckle_field(self, ingredients):
        assert isinstance(TabulatedSpeckleField(**ingredients), AbstractSpeckleField)

    def test_realize_returns_a_nonnegative_contrast_map(self, ingredients):
        out = TabulatedSpeckleField(**ingredients).realize(
            wavelength_nm=500.0, time_s=5.0
        )
        assert out.shape == ingredients["G"].shape[1:]
        assert jnp.all(out >= 0.0)

    def test_matches_exact_eps_at_a_node(self, ingredients):
        out = TabulatedSpeckleField(**ingredients).realize(
            wavelength_nm=500.0, time_s=10.0
        )
        expected = _incoherent_delta(ingredients["G"], [1.0, -1.0], 3.0)
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12)

    def test_interpolates_eps_between_nodes(self, ingredients):
        out = TabulatedSpeckleField(**ingredients).realize(
            wavelength_nm=500.0, time_s=5.0
        )
        expected = _incoherent_delta(ingredients["G"], [0.5, -0.5], 3.0)
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12)

    def test_zero_eps_excludes_the_floor(self, ingredients):
        """At a node with eps = 0 the delta is exactly zero: the floor is not
        re-emitted (star_rate already carries it)."""
        out = TabulatedSpeckleField(**ingredients).realize(
            wavelength_nm=500.0, time_s=0.0
        )
        np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-30)

    def test_coherent_adds_the_pinning_cross_term(self, ingredients):
        out = TabulatedSpeckleField(**ingredients, coherent=True).realize(
            wavelength_nm=500.0, time_s=10.0
        )
        g, e_nom = ingredients["G"], ingredients["e_nom"]
        g_eps = jnp.tensordot(jnp.asarray([1.0, -1.0]), g, axes=1)
        expected = np.asarray((jnp.abs(e_nom + g_eps) ** 2 - jnp.abs(e_nom) ** 2) / 3.0)
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-10)

    def test_is_differentiable_in_time(self, ingredients):
        field = TabulatedSpeckleField(**ingredients)

        def total(t):
            return jnp.sum(field.realize(wavelength_nm=500.0, time_s=t))

        assert jnp.isfinite(jax.grad(total)(7.0))


def _synthetic_lin(seed=2, m=2, ny=3, nx=3):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((m, ny, nx)) + 1j * rng.standard_normal((m, ny, nx))
    e = rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))
    return jnp.asarray(e), jnp.asarray(g)


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
            e_nom,
            g,
            covariance,
            key=jax.random.PRNGKey(0),
            frequencies_hz=freqs,
            psd=psd,
            normalization=1.0,
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
        args = (e_nom, g, covariance)
        kw = dict(
            key=jax.random.PRNGKey(3),
            frequencies_hz=freqs,
            psd=psd,
            normalization=1.0,
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
                e_nom,
                g,
                cov,
                key=jax.random.PRNGKey(0),
                frequencies_hz=jnp.linspace(1e-4, 1e-2, 8),
                psd=psd,
                normalization=1.0,
            )

    def test_rejects_indefinite_covariance(self):
        e_nom, g = _synthetic_lin()
        indefinite = jnp.asarray([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues -1, 3
        with pytest.raises(ValueError, match="semidefinite"):
            correlated_drift_field(
                e_nom,
                g,
                indefinite,
                key=jax.random.PRNGKey(0),
                frequencies_hz=jnp.linspace(1e-4, 1e-2, 8),
                psd=jnp.ones(8),
                normalization=1.0,
            )

    def test_rejects_asymmetric_covariance(self):
        e_nom, g = _synthetic_lin()
        asymmetric = jnp.asarray([[4.0, 1.0], [0.5, 9.0]])
        with pytest.raises(ValueError, match="symmetric"):
            correlated_drift_field(
                e_nom,
                g,
                asymmetric,
                key=jax.random.PRNGKey(0),
                frequencies_hz=jnp.linspace(1e-4, 1e-2, 8),
                psd=jnp.ones(8),
                normalization=1.0,
            )

    def test_warns_when_neff_is_low(self):
        """A steeply red PSD concentrates power in one frequency, so a single
        realization is unrepresentative: the builder warns."""
        e_nom, g, cov, red = self._builder_kwargs(
            jnp.asarray([1.0, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
        )
        with pytest.warns(UserWarning, match="N_eff"):
            correlated_drift_field(
                e_nom,
                g,
                cov,
                key=jax.random.PRNGKey(0),
                frequencies_hz=jnp.linspace(1e-4, 1e-2, 8),
                psd=red,
                normalization=1.0,
            )


class TestPhysicaloptixIntegration:
    def test_tabulated_field_from_a_linearized_path(self):
        """A real (E_nom, G) from physicaloptix.linearize over a Zernike basis
        drives a tabulated drift into a finite, floor-excluded contrast map."""
        from physicaloptix import (
            Field,
            Fraunhofer,
            Grid,
            OpticalPath,
            PlaneKind,
            Stage,
            linearize,
            zernike_basis,
        )

        pupil = Grid.pupil(48)
        coords = np.asarray(pupil.coords)
        xx, yy = np.meshgrid(coords, coords)
        aperture = (xx**2 + yy**2 <= 0.25).astype(float)
        path = OpticalPath(
            stages=(
                Stage(
                    "science", Fraunhofer(grid_in=pupil, grid_out=Grid.focal(32, 0.5))
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
            lin.e_nom,
            lin.G,
            times,
            eps_table,
            normalization=float(jnp.abs(lin.e_nom).max() ** 2),
        )
        out = speckle.realize(wavelength_nm=500.0, time_s=50.0)
        assert out.shape == (32, 32)
        assert jnp.all(jnp.isfinite(out))
        at_zero = speckle.realize(wavelength_nm=500.0, time_s=0.0)
        np.testing.assert_allclose(np.asarray(at_zero), 0.0, atol=1e-30)
