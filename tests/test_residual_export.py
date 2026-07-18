"""Tests for the maintained-residual export through the speckle seam."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from optixstuff.speckle import AbstractSpeckleField
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

from tiptilt.control import close_dark_hole
from tiptilt.maintenance import maintain_dark_hole, maintained_residual_field

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


def _setup(npix=16):
    pupil = Grid.pupil(npix)
    focal = Grid.focal(32, 0.5)
    x = np.asarray(pupil.coords)
    xg, yg = np.meshgrid(x, x)
    aperture = (xg**2 + yg**2 <= 0.25).astype(complex)
    field = Field(data=jnp.asarray(aperture), grid=pupil, plane=PlaneKind.PUPIL)
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
    mask = jnp.asarray((np.abs(fxg - 3.0) < 0.8) & (np.abs(fyg - 1.0) < 0.8))
    return path, field, mask


def _ramp_drift(scale_nm=0.3):
    direction = jnp.asarray([1.0, -0.6, 0.8, -0.4])

    def eps(time_s):
        return scale_nm * time_s * direction

    return eps


COMMON = dict(
    drift_stage=0,
    n_steps=6,
    dt_s=1.0,
    gain=0.7,
    regularization=1e-9,
)


def test_maintain_returns_trajectory_when_asked():
    path, field, mask = _setup()
    command, history, trajectory = maintain_dark_hole(
        path, field, 1, mask, drift=_ramp_drift(), return_trajectory=True, **COMMON
    )
    assert trajectory.shape == (COMMON["n_steps"], command.shape[0])
    # frame 0 is the pre-correction operating point (command0 = zeros here)
    assert jnp.allclose(trajectory[0], jnp.zeros_like(command))
    # the classic 2-tuple contract is unchanged
    command2, history2 = maintain_dark_hole(
        path, field, 1, mask, drift=_ramp_drift(), **COMMON
    )
    assert jnp.allclose(command2, command) and jnp.allclose(history2, history)


def _dug_command0(path, field, mask):
    # The export's contract: a maintenance loop about a DUG operating point,
    # so excursions stay drift-scale and the linearization is tight.
    command0, _ = close_dark_hole(
        path, field, 1, mask, n_steps=10, gain=0.6, regularization=1e-9
    )
    return command0


def test_realized_field_matches_direct_propagation():
    path, field, mask = _setup()
    command0 = _dug_command0(path, field, mask)
    residual, _history = maintained_residual_field(
        path,
        field,
        1,
        mask,
        drift=_ramp_drift(),
        normalization=1.0,
        command0=command0,
        **COMMON,
    )
    _, _, trajectory = maintain_dark_hole(
        path,
        field,
        1,
        mask,
        drift=_ramp_drift(),
        command0=command0,
        return_trajectory=True,
        **COMMON,
    )
    eps = _ramp_drift()
    for k in (0, 3, 5):
        drifted = eqx.tree_at(
            lambda p: (p.stages[0].op.basis.coeffs, p.stages[1].op.basis.coeffs),
            path,
            (eps(k * COMMON["dt_s"]), trajectory[k]),
        )
        out, _ = drifted.propagate(field)
        direct = jnp.abs(out.data) ** 2
        linear = jnp.abs(residual.e_nom) ** 2 + residual.realize(
            wavelength_nm=WL, time_s=k * COMMON["dt_s"]
        )
        err = jnp.linalg.norm(linear - direct) / jnp.linalg.norm(direct)
        assert float(err) < 1e-3, f"frame {k}: rel err {float(err):.2e}"


def test_seam_contract_and_closed_beats_open():
    path, field, mask = _setup()
    command0 = _dug_command0(path, field, mask)
    kwargs = dict(drift=_ramp_drift(), normalization=1.0, command0=command0)
    closed, _ = maintained_residual_field(path, field, 1, mask, **kwargs, **COMMON)
    open_kwargs = {**COMMON, "gain": 0.0}
    opened, _ = maintained_residual_field(path, field, 1, mask, **kwargs, **open_kwargs)
    assert isinstance(closed, AbstractSpeckleField)
    late = (COMMON["n_steps"] - 1) * COMMON["dt_s"]
    d_closed = closed.realize(wavelength_nm=WL, time_s=late)
    d_open = opened.realize(wavelength_nm=WL, time_s=late)
    assert d_closed.shape == d_open.shape and bool(jnp.all(jnp.isfinite(d_closed)))
    # the seam carries the controller conditioning: closed-loop residual
    # energy in the hole is far below open loop
    closed_mask = float(jnp.mean(jnp.abs(d_closed[mask])))
    open_mask = float(jnp.mean(jnp.abs(d_open[mask])))
    assert closed_mask < 0.2 * open_mask
