"""Tests for the dark-hole maintenance driver and the detector wrapper."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from physicaloptix import (
    Field,
    Fraunhofer,
    Grid,
    ModeBasis,
    OpticalPath,
    PhaseScreen,
    PlaneKind,
    Stage,
)

from wavefronts.control import close_dark_hole
from wavefronts.maintenance import maintain_dark_hole, make_detector
from wavefronts.sensing import probe_set

WL = 500.0
_KS = [(3, 1), (3, 0), (2, 1), (4, 1)]


def _fourier_basis(npix, freqs, amp_nm=4.0):
    x = np.asarray(Grid.pupil(npix).coords)
    xg, yg = np.meshgrid(x, x)
    modes = []
    for kx, ky in freqs:
        arg = 2 * np.pi * (kx * xg + ky * yg)
        modes.append(amp_nm * np.cos(arg))
        modes.append(amp_nm * np.sin(arg))
    stack = jnp.asarray(np.stack(modes))
    return ModeBasis(B=stack, coeffs=jnp.zeros(stack.shape[0]))


def _setup(npix=16, aberration_nm=0.0):
    """WFE screen (drift carrier) + DM + science focal; drift lies in DM span."""
    pupil = Grid.pupil(npix)
    focal = Grid.focal(32, 0.5)
    x = np.asarray(pupil.coords)
    xg, yg = np.meshgrid(x, x)
    aperture = (xg**2 + yg**2 <= 0.25).astype(float)
    opd = aberration_nm * np.cos(2 * np.pi * (3 * xg + yg))
    field = Field(
        data=jnp.asarray(aperture * np.exp(1j * 2 * np.pi * opd / WL)),
        grid=pupil,
        plane=PlaneKind.PUPIL,
    )
    drift_basis = _fourier_basis(npix, _KS[:2])  # 4 modes, inside the DM span
    dm_basis = _fourier_basis(npix, _KS)  # 8 modes
    path = OpticalPath(
        stages=(
            Stage("wfe", PhaseScreen(drift_basis, pupil, wavelength_nm=WL)),
            Stage("dm", PhaseScreen(dm_basis, pupil, wavelength_nm=WL)),
            Stage("science", Fraunhofer(grid_in=pupil, grid_out=focal)),
        )
    )
    fx = np.asarray(focal.coords)
    fxg, fyg = np.meshgrid(fx, fx)
    mask = jnp.asarray(
        ((np.abs(fxg - 3.0) < 0.6) & (np.abs(fyg - 1.0) < 0.6))
        | ((np.abs(fxg - 3.0) < 0.6) & (np.abs(fyg) < 0.6))
    )
    return path, field, mask


def _ramp_drift(n_steps, n_modes, scale_nm=0.15):
    """Deterministic monotone drift (seed-free): contrast grows as t^2."""
    directions = np.asarray([1.0, -0.7, 0.4, 0.9])[:n_modes]
    ramp = np.arange(1, n_steps + 1)[:, None] * scale_nm * directions[None, :]
    return jnp.asarray(ramp)


def _random_walk(n_steps, n_modes, scale_nm=0.15, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(scale=scale_nm, size=(n_steps, n_modes))
    return jnp.asarray(np.cumsum(steps, axis=0))


def _drift_fn(table, dt_s):
    table = jnp.asarray(table)

    def drift(time_s):
        index = jnp.clip(jnp.asarray(time_s / dt_s, dtype=int), 0, table.shape[0] - 1)
        return table[index]

    return drift


class TestMaintainDarkHole:
    @pytest.mark.slow
    def test_holds_contrast_a_static_command_loses(self):
        """Dig first, then drift: the maintenance regime. A frozen command
        loses the dug hole to the drift; the maintained loop holds it."""
        path, field, mask = _setup()
        command0, _ = close_dark_hole(
            path, field, 1, mask, n_steps=15, gain=0.6, regularization=1e-8
        )
        n_steps, dt_s = 12, 10.0
        table = _ramp_drift(n_steps, 4, scale_nm=0.3)
        drift = _drift_fn(table, dt_s)
        common = dict(
            drift=drift,
            drift_stage=0,
            n_steps=n_steps,
            dt_s=dt_s,
            regularization=1e-8,
            command0=command0,
        )
        _, held = maintain_dark_hole(path, field, 1, mask, gain=0.7, **common)
        _, static = maintain_dark_hole(path, field, 1, mask, gain=0.0, **common)
        assert float(static[-1]) > 10.0 * float(static[0])  # drift wrecks it
        assert float(held[-1]) < 0.1 * float(static[-1])  # the loop holds it

    @pytest.mark.slow
    def test_pairwise_maintenance_holds_honestly(self):
        path, field, mask = _setup()
        command0, _ = close_dark_hole(
            path, field, 1, mask, n_steps=15, gain=0.6, regularization=1e-8
        )
        n_steps, dt_s = 12, 10.0
        table = _ramp_drift(n_steps, 4)
        drift = _drift_fn(table, dt_s)
        dm_basis = path.stages[1].op.basis
        probes = probe_set(dm_basis, amplitude_nm=2.0, n_probes=3)
        common = dict(
            drift=drift,
            drift_stage=0,
            n_steps=n_steps,
            dt_s=dt_s,
            regularization=1e-8,
            command0=command0,
        )
        _, held = maintain_dark_hole(
            path,
            field,
            1,
            mask,
            gain=0.7,
            estimator="pairwise",
            model_field=field,
            probes=probes,
            **common,
        )
        _, static = maintain_dark_hole(path, field, 1, mask, gain=0.0, **common)
        assert float(held[-1]) < 0.3 * float(static[-1])

    @pytest.mark.slow
    def test_operating_point_command0_beats_cold_jacobian(self):
        """Maintaining a DUG hole: linearizing at the dug command tracks the
        drift better than the cold (zeros) Jacobian."""
        path, field, mask = _setup(aberration_nm=12.0)
        command0, dig_hist = close_dark_hole(
            path, field, 1, mask, n_steps=12, gain=0.6, regularization=1e-8
        )
        n_steps, dt_s = 10, 10.0
        table = _random_walk(n_steps, 4, scale_nm=0.2, seed=2)
        drift = _drift_fn(table, dt_s)
        common = dict(
            drift=drift,
            drift_stage=0,
            n_steps=n_steps,
            dt_s=dt_s,
            gain=0.7,
            regularization=1e-8,
            command0=command0,
        )
        _, warm = maintain_dark_hole(path, field, 1, mask, **common)
        _, cold = maintain_dark_hole(
            path, field, 1, mask, **{**common, "linearize_at_command0": False}
        )
        assert float(warm[-1]) <= float(cold[-1]) * 1.05
        assert float(warm[-1]) < 0.2 * float(dig_hist[0])  # still a dark hole


class TestMakeDetector:
    def test_mean_preserving(self):
        image = jnp.asarray([[1.0, 0.5], [0.1, 0.0]])
        detector = make_detector(flux=1e4, reference_peak=1.0, method="gaussian")
        keys = jax.random.split(jax.random.PRNGKey(0), 400)
        draws = jnp.stack([detector(image, k) for k in keys])
        np.testing.assert_allclose(
            np.asarray(draws.mean(axis=0)), np.asarray(image), atol=0.02
        )

    def test_more_flux_means_less_noise_not_more_gain(self):
        image = jnp.full((4, 4), 0.5)
        dim = make_detector(flux=1e3, method="gaussian")
        bright = make_detector(flux=1e6, method="gaussian")
        keys = jax.random.split(jax.random.PRNGKey(1), 200)
        std_dim = float(jnp.std(jnp.stack([dim(image, k) for k in keys])))
        std_bright = float(jnp.std(jnp.stack([bright(image, k) for k in keys])))
        assert std_bright < 0.1 * std_dim
