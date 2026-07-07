"""Tests for the von Karman / Kolmogorov atmospheric phase-screen generator."""

import jax
import jax.numpy as jnp
import numpy as np

from wavefronts.turbulence import frozen_flow_sequence, von_karman_screen

WL = 500.0


def _structure_function(screens, dx_m, seps_px):
    """Ensemble phase structure function D(r) = <[phi(x+r) - phi(x)]^2> along x."""
    arr = np.asarray(screens)  # (n_real, npix, npix), OPD in nm
    phase = arr * (2 * np.pi / WL)  # radians
    out = []
    for sep in seps_px:
        diff = phase[:, :, sep:] - phase[:, :, :-sep]
        out.append(diff.var() + diff.mean() ** 2)  # <diff^2>
    return np.array(seps_px) * dx_m, np.array(out)


class TestVonKarmanScreen:
    def test_screen_is_real_and_zero_mean(self):
        screen = von_karman_screen(
            jax.random.PRNGKey(0), npix=256, dx_m=0.02, r0_m=0.2, wavelength_nm=WL
        )
        assert screen.shape == (256, 256)
        assert not jnp.iscomplexobj(screen)  # OPD is real
        assert abs(float(jnp.mean(screen))) < 1e-6 * float(jnp.std(screen))  # no piston

    def test_structure_function_has_five_thirds_slope(self):
        """The inertial-range structure function follows r^(5/3) (Kolmogorov)."""
        key = jax.random.PRNGKey(1)
        screens = jnp.stack(
            [
                von_karman_screen(k, 256, 0.01, 0.2, WL, l0_m=50.0)
                for k in jax.random.split(key, 40)
            ]
        )
        r, d = _structure_function(screens, 0.01, [2, 4, 8, 16])
        slope = np.polyfit(np.log(r), np.log(d), 1)[0]
        np.testing.assert_allclose(slope, 5.0 / 3.0, atol=0.2)

    def test_structure_function_scales_with_r0(self):
        """Halving r0 raises D(r) by 2^(5/3) (the r0^-5/3 dependence)."""
        key = jax.random.PRNGKey(2)
        keys = jax.random.split(key, 40)

        def ensemble_d(r0):
            screens = jnp.stack(
                [von_karman_screen(k, 256, 0.01, r0, WL, l0_m=50.0) for k in keys]
            )
            _, d = _structure_function(screens, 0.01, [8])
            return d[0]

        ratio = ensemble_d(0.1) / ensemble_d(0.2)
        np.testing.assert_allclose(ratio, 2.0 ** (5.0 / 3.0), rtol=0.15)


class TestFrozenFlow:
    def test_sequence_shape_and_frozen_flow_shift(self):
        """Frozen flow translates one screen: frame k+1 is frame k shifted by the
        wind, so the overlapping region matches."""
        shift = 3
        seq = frozen_flow_sequence(
            jax.random.PRNGKey(3),
            npix=64,
            dx_m=0.02,
            r0_m=0.2,
            wavelength_nm=WL,
            n_frames=5,
            shift_px=shift,
        )
        assert seq.shape == (5, 64, 64)
        a = np.asarray(seq[0])[:, shift:]
        b = np.asarray(seq[1])[:, :-shift]
        np.testing.assert_allclose(a, b, atol=1e-9)  # frame 1 = frame 0 shifted
