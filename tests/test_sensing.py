"""Tests for focal-plane wavefront estimation (pairwise probing)."""

import jax
import jax.numpy as jnp
import numpy as np
from physicaloptix import (
    Field,
    Fraunhofer,
    Grid,
    OpticalPath,
    PhaseScreen,
    PlaneKind,
    Stage,
    fourier_dm_basis,
)
from physicaloptix.stats import dark_zone_mask

from wavefronts.sensing import (
    estimate_field_pairwise,
    pairwise_estimate,
    probe_set,
)


def _simple_dm_path(npix=32, nfoc=64, pscale=0.25, wl=500.0):
    """A pupil -> Fourier-DM -> focal path plus its unaberrated model field."""
    pupil = Grid.pupil(npix)
    focal = Grid.focal(nfoc, pscale)
    coords = np.asarray(pupil.coords)
    xg, yg = np.meshgrid(coords, coords)
    aperture = (xg**2 + yg**2 <= 0.25).astype(complex)
    dm_basis = fourier_dm_basis(pupil, n_actuators=16, k_min=2.0, k_max=7.0)
    path = OpticalPath(
        stages=(
            Stage("dm", PhaseScreen(dm_basis, pupil, wavelength_nm=wl)),
            Stage("science", Fraunhofer(grid_in=pupil, grid_out=focal)),
        )
    )
    model_field = Field(data=jnp.asarray(aperture), grid=pupil, plane=PlaneKind.PUPIL)
    return path, dm_basis, aperture, (xg, yg), pupil, focal, model_field, wl


def _linear_diff_images(e_true, probe_fields):
    """Noiseless pairwise difference dI_j = 4 Re{conj(E) p_j} per pixel."""
    return 4.0 * jnp.real(jnp.conj(e_true)[None, :] * probe_fields)


class TestPairwiseEstimate:
    def test_recovers_known_field_from_two_probes(self):
        key = jax.random.PRNGKey(0)
        n_pix = 6
        kr, ki = jax.random.split(key)
        e_true = 1e-4 * (
            jax.random.normal(kr, (n_pix,)) + 1j * jax.random.normal(ki, (n_pix,))
        )
        p1 = 1e-3 * (1.0 + 0.2j) * jnp.ones(n_pix)
        p2 = 1e-3 * (0.1 + 1.0j) * jnp.ones(n_pix)
        probe_fields = jnp.stack([p1, p2])
        z = _linear_diff_images(e_true, probe_fields)
        e_hat = pairwise_estimate(probe_fields, z)
        np.testing.assert_allclose(
            np.asarray(e_hat), np.asarray(e_true), rtol=1e-8, atol=1e-12
        )

    def test_one_probe_cannot_recover_both_quadratures(self):
        n_pix = 4
        e_true = jnp.full(n_pix, 1e-4 + 1e-4j)
        p_real = jnp.full(n_pix, 1e-3 + 0j)  # senses only the real quadrature
        one = p_real[None, :]
        e_one = pairwise_estimate(
            one, _linear_diff_images(e_true, one), regularization=1e-12
        )

        p_imag = jnp.full(n_pix, 0 + 1e-3j)
        two = jnp.stack([p_real, p_imag])
        e_two = pairwise_estimate(two, _linear_diff_images(e_true, two))

        err_one = float(jnp.max(jnp.abs(e_one - e_true)))
        err_two = float(jnp.max(jnp.abs(e_two - e_true)))
        assert err_two < 1e-9  # two independent quadratures recover the field
        assert err_one > 0.5 * float(jnp.abs(e_true[0]))  # one pair misses a quadrature

    def test_shape_is_one_estimate_per_pixel(self):
        n_p, n_pix = 3, 10
        probe_fields = jnp.ones((n_p, n_pix), dtype=complex)
        z = jnp.zeros((n_p, n_pix))
        e_hat = pairwise_estimate(probe_fields, z)
        assert e_hat.shape == (n_pix,)
        assert jnp.iscomplexobj(e_hat)


class TestEstimateFieldThroughPath:
    def _aberrated_input(self, aperture, xg, yg, wl):
        opd = (
            3.0 * np.cos(2 * np.pi * 4 * xg)
            + 2.0 * np.cos(2 * np.pi * 3 * yg)
            + 2.0 * np.cos(2 * np.pi * 5 * (xg + yg) / np.sqrt(2))
        )
        return aperture * np.exp(1j * 2 * np.pi * opd / wl)

    def _setup(self):
        path, dm_basis, aperture, (xg, yg), pupil, focal, model_field, wl = (
            _simple_dm_path()
        )
        e_in = self._aberrated_input(aperture, xg, yg, wl)
        input_field = Field(data=jnp.asarray(e_in), grid=pupil, plane=PlaneKind.PUPIL)
        mask = dark_zone_mask(focal, iwa_lod=2.0, owa_lod=7.0)
        out, _ = path.propagate(input_field)
        return path, dm_basis, input_field, model_field, mask, out.data

    def test_oracle_model_recovers_to_linearization_floor(self):
        # An exact model (the true field) isolates the estimator + linearization.
        path, dm_basis, input_field, _, mask, focal_data = self._setup()
        e_true = focal_data[mask]
        probes = probe_set(dm_basis, amplitude_nm=2.0, n_probes=4)
        e_hat = estimate_field_pairwise(path, input_field, 0, probes, mask)
        err = float(jnp.max(jnp.abs(e_hat - e_true)))
        assert err < 0.03 * float(jnp.max(jnp.abs(e_true)))

    def test_honest_model_recovers_within_model_uncertainty(self):
        # The hardware-realistic case: the probe response is modelled on the
        # UNABERRATED pupil, so the single-shot estimate floors near the
        # aberration level (~0.09 rad here). Good enough to close the loop.
        path, dm_basis, input_field, model_field, mask, focal_data = self._setup()
        e_true = focal_data[mask]
        probes = probe_set(dm_basis, amplitude_nm=2.0, n_probes=4)
        e_hat = estimate_field_pairwise(
            path, input_field, 0, probes, mask, model_field=model_field
        )
        err = float(jnp.max(jnp.abs(e_hat - e_true)))
        assert err < 0.15 * float(jnp.max(jnp.abs(e_true)))

    def test_more_flux_reduces_error(self):
        from physicaloptix import read_detector

        path, dm_basis, input_field, _, mask, focal_data = self._setup()
        e_true = focal_data[mask]
        scale = float(jnp.max(jnp.abs(e_true)))
        peak = float(jnp.max(jnp.abs(focal_data) ** 2))
        probes = probe_set(dm_basis, amplitude_nm=3.0, n_probes=6)

        def mean_error(flux):
            def detector(image, key):
                counts = read_detector(
                    image / peak, key, flux=flux, exposure_time=1.0,
                    read_noise_e=0.0, method="poisson",
                )
                return peak * counts / flux  # unbiased, back to image units

            errs = []
            for s in range(8):
                e_hat = estimate_field_pairwise(
                    path, input_field, 0, probes, mask, key=jax.random.PRNGKey(s),
                    detector=detector,
                )
                errs.append(float(jnp.max(jnp.abs(e_hat - e_true))))
            return float(np.mean(errs))

        err_low = mean_error(1e5)
        err_high = mean_error(1e9)
        assert err_high < err_low  # more photons, better estimate
        assert err_high < 0.05 * scale
