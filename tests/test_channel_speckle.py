"""Tests for the correlated dual-channel speckle constructor and eps access."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from physicaloptix import ChannelLinearization, MultiChannelLinearization

from wavefronts.speckle import TabulatedSpeckleField, correlated_channel_fields

NPIX, M_SHARED = 8, 3


def _synthetic_mcl(key, names=("a", "b")):
    """Hand-built two-channel linearization: random blocks, no propagation."""
    channels = []
    for name in names:
        key, key_e, key_g = jax.random.split(key, 3)
        e_nom = (
            jax.random.normal(key_e, (NPIX, NPIX))
            + 1j * jax.random.normal(key_e, (NPIX, NPIX))
        ) * 1e-5
        g = (
            jax.random.normal(key_g, (M_SHARED, NPIX, NPIX))
            + 1j * jax.random.normal(key_g, (M_SHARED, NPIX, NPIX))
        ) * 1e-3
        channels.append(
            ChannelLinearization(
                name=name, e_nom=e_nom, g_shared=g, pixel_scale_lod=0.25
            )
        )
    return MultiChannelLinearization(
        channels=tuple(channels), wavelength_nm=500.0, kind="opd"
    )


def _white_psd(n=256):
    frequencies = jnp.linspace(1e-5, 1e-2, n)
    return frequencies, jnp.ones(n)


class TestCorrelatedChannelFields:
    def test_shared_draw_is_identical_across_channels(self):
        mcl = _synthetic_mcl(jax.random.PRNGKey(0))
        sigma = jnp.diag(jnp.asarray([4.0, 1.0, 0.25]))
        frequencies, psd = _white_psd()
        fields = correlated_channel_fields(
            mcl,
            sigma,
            key=jax.random.PRNGKey(1),
            frequencies_hz=frequencies,
            psd=psd,
            normalizations={"a": 1.0, "b": 2.0},
        )
        assert set(fields) == {"a", "b"}
        np.testing.assert_array_equal(
            np.asarray(fields["a"].amplitudes), np.asarray(fields["b"].amplitudes)
        )
        np.testing.assert_array_equal(
            np.asarray(fields["a"].phases), np.asarray(fields["b"].phases)
        )

    def test_per_channel_normalization_and_blocks(self):
        mcl = _synthetic_mcl(jax.random.PRNGKey(0))
        sigma = jnp.eye(M_SHARED)
        frequencies, psd = _white_psd()
        fields = correlated_channel_fields(
            mcl,
            sigma,
            key=jax.random.PRNGKey(1),
            frequencies_hz=frequencies,
            psd=psd,
            normalizations={"a": 1.0, "b": 2.0},
        )
        assert float(fields["a"].normalization) == 1.0
        assert float(fields["b"].normalization) == 2.0
        np.testing.assert_array_equal(
            np.asarray(fields["a"].G), np.asarray(mcl["a"].g_shared)
        )
        assert fields["a"].coherent  # the common-mode signal is a FIELD effect

    def test_cross_channel_field_covariance_matches_the_shared_block(self):
        """Time-averaged Cov(dE_a, dE_b) approaches G_a Sigma (G_b)^H."""
        mcl = _synthetic_mcl(jax.random.PRNGKey(0))
        sigma = jnp.diag(jnp.asarray([4.0, 1.0, 0.25]))
        frequencies, psd = _white_psd(512)
        fields = correlated_channel_fields(
            mcl,
            sigma,
            key=jax.random.PRNGKey(7),
            frequencies_hz=frequencies,
            psd=psd,
            normalizations={"a": 1.0, "b": 1.0},
        )
        # Reconstruct the (shared) eps stream from the stored spectrum and
        # accumulate the empirical cross-channel field covariance.
        amp = jnp.asarray(fields["a"].amplitudes)  # (m, f)
        phase = jnp.asarray(fields["a"].phases)
        freqs = jnp.asarray(frequencies)
        times = jnp.linspace(0.0, 3.0 / float(freqs[0]), 4096)

        def eps_at(t):
            return jnp.sum(
                amp * jnp.cos(2 * jnp.pi * freqs[None, :] * t + phase), axis=1
            )

        eps = jax.vmap(eps_at)(times)  # (T, m)
        g_a = mcl["a"].g_shared.reshape(M_SHARED, -1)
        g_b = mcl["b"].g_shared.reshape(M_SHARED, -1)
        de_a = eps @ g_a  # (T, pix)
        de_b = eps @ g_b
        emp = (de_a.T.conj() @ de_b) / times.shape[0]  # E[dE_a* dE_b]
        target = (g_a.T.conj() * jnp.diag(sigma)) @ g_b
        rel = float(jnp.linalg.norm(emp - target) / jnp.linalg.norm(target))
        # Statistical agreement: one frozen realization scatters ~1/sqrt(N_eff).
        assert rel < 0.25

    def test_local_blocks_are_independent_across_channels(self):
        mcl = _synthetic_mcl(jax.random.PRNGKey(0))
        sigma = jnp.eye(M_SHARED)
        frequencies, psd = _white_psd()
        key_local = jax.random.PRNGKey(3)
        g_local = {
            name: (
                jax.random.normal(jax.random.fold_in(key_local, i), (2, NPIX, NPIX))
                + 0j
            )
            * 1e-3
            for i, name in enumerate(("a", "b"))
        }
        fields = correlated_channel_fields(
            mcl,
            sigma,
            key=jax.random.PRNGKey(1),
            frequencies_hz=frequencies,
            psd=psd,
            normalizations={"a": 1.0, "b": 1.0},
            local={name: (g_local[name], 0.5 * jnp.eye(2)) for name in ("a", "b")},
        )
        # Shared rows identical; local rows differ (independent draws).
        amp_a = np.asarray(fields["a"].amplitudes)
        amp_b = np.asarray(fields["b"].amplitudes)
        np.testing.assert_array_equal(amp_a[:M_SHARED], amp_b[:M_SHARED])
        assert not np.allclose(amp_a[M_SHARED:], amp_b[M_SHARED:])
        assert fields["a"].G.shape[0] == M_SHARED + 2

    def test_requires_a_normalization_per_channel(self):
        mcl = _synthetic_mcl(jax.random.PRNGKey(0))
        frequencies, psd = _white_psd()
        with pytest.raises(ValueError, match="normalization"):
            correlated_channel_fields(
                mcl,
                jnp.eye(M_SHARED),
                key=jax.random.PRNGKey(1),
                frequencies_hz=frequencies,
                psd=psd,
                normalizations={"a": 1.0},  # missing "b"
            )


class TestTabulatedEpsAccessor:
    def test_public_eps_interpolates_the_trajectory(self):
        times = jnp.asarray([0.0, 1.0, 2.0])
        table = jnp.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
        field = TabulatedSpeckleField(
            e_nom=jnp.zeros((4, 4), dtype=complex),
            G=jnp.zeros((2, 4, 4), dtype=complex),
            times_s=times,
            eps_table=table,
            normalization=1.0,
        )
        np.testing.assert_allclose(
            np.asarray(field.eps(0.5)), np.asarray([0.5, 1.0]), atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(field.eps(5.0)), np.asarray([2.0, 4.0]), atol=1e-12
        )  # held at the endpoint
