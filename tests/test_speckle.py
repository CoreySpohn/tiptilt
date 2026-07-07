"""Tests for the wavefronts speckle-field family."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from optixstuff.speckle import AbstractSpeckleField

from wavefronts.speckle import TabulatedSpeckleField


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
