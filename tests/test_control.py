"""Tests for the single-DM electric-field-conjugation dark-hole loop."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from physicaloptix import (
    Field,
    Fraunhofer,
    Fresnel,
    Grid,
    ModeBasis,
    OpticalPath,
    PhaseScreen,
    PlaneKind,
    Spectrum,
    Stage,
    broadcast_to_spectrum,
    fourier_dm_basis,
)

from wavefronts.control import close_dark_hole
from wavefronts.sensing import probe_set

WL = 500.0
_KS = [(3, 1), (3, 0), (3, 2), (2, 1), (4, 1), (3, -1)]
# A richer symmetric set around the (3, 0) speckle for the two-DM demonstration.
_RELAY_KS = [(3, 0), (3, 1), (3, -1), (2, 0), (4, 0), (2, 1), (4, 1), (3, 2), (3, -2)]
DIAM_M = 0.02


def _fourier_dm(npix, amp_nm=5.0, freqs=None):
    x = np.asarray(Grid.pupil(npix).coords)
    x_grid, y_grid = np.meshgrid(x, x)
    modes = []
    for kx, ky in freqs or _KS:
        arg = 2 * np.pi * (kx * x_grid + ky * y_grid)
        modes.append(amp_nm * np.cos(arg))
        modes.append(amp_nm * np.sin(arg))
    stack = jnp.asarray(np.stack(modes))
    return ModeBasis(B=stack, coeffs=jnp.zeros(stack.shape[0]))


def _setup(npix=16):
    pupil = Grid.pupil(npix)
    focal = Grid.focal(32, 0.5)
    x = np.asarray(pupil.coords)
    x_grid, y_grid = np.meshgrid(x, x)
    aperture = (x_grid**2 + y_grid**2 <= 0.25).astype(float)
    aberration = 3.0 * np.cos(2 * np.pi * (3 * x_grid + 1 * y_grid))  # nm, freq (3,1)
    e_in = aperture * np.exp(1j * 2 * np.pi * aberration / WL)
    field = Field(data=jnp.asarray(e_in), grid=pupil, plane=PlaneKind.PUPIL)
    path = OpticalPath(
        stages=(
            Stage("dm", PhaseScreen(_fourier_dm(npix), pupil, wavelength_nm=WL)),
            Stage("science", Fraunhofer(grid_in=pupil, grid_out=focal)),
        )
    )
    fx = np.asarray(focal.coords)
    fx_grid, fy_grid = np.meshgrid(fx, fx)
    # A small one-sided dark zone around the aberration's speckle at (3, 1) lambda/D.
    mask = (np.abs(fx_grid - 3.0) < 0.4) & (np.abs(fy_grid - 1.0) < 0.4)
    return path, field, jnp.asarray(mask)


def _relay_setup(npix=16, alpha=0.0556):
    """A two-DM relay with an AMPLITUDE aberration: a real focal speckle a pupil
    phase DM cannot null (wrong quadrature), but two DMs can via Talbot amplitude.

    ``alpha`` is set near the Talbot peak for the (3, 0) speckle (pi alpha 3^2
    approximately pi/2), so the out-of-pupil DM has strong amplitude authority.
    """
    pupil = Grid.pupil(npix)
    focal = Grid.focal(32, 0.5)
    x = np.asarray(pupil.coords)
    x_grid, y_grid = np.meshgrid(x, x)
    aperture = (x_grid**2 + y_grid**2 <= 0.25).astype(float)
    amplitude_ripple = 1.0 + 0.15 * np.cos(2 * np.pi * 3 * x_grid)  # (3, 0) speckle
    e_in = (aperture * amplitude_ripple).astype(complex)
    field = Field(data=jnp.asarray(e_in), grid=pupil, plane=PlaneKind.PUPIL)
    z = alpha * DIAM_M**2 / (WL * 1e-9)

    def fresnel(distance_m, plane_in, plane_out):
        return Fresnel(
            grid=pupil,
            distance_m=distance_m,
            beam_diameter_m=DIAM_M,
            wavelength_nm=WL,
            plane_in=plane_in,
            plane_out=plane_out,
            on_undersampled="record",
        )

    dm_basis = _fourier_dm(npix, freqs=_RELAY_KS)
    path = OpticalPath(
        stages=(
            Stage("dm1", PhaseScreen(dm_basis, pupil, wavelength_nm=WL)),
            Stage("relay", fresnel(z, PlaneKind.PUPIL, PlaneKind.INTERMEDIATE)),
            Stage(
                "dm2",
                PhaseScreen(
                    dm_basis, pupil, wavelength_nm=WL, plane=PlaneKind.INTERMEDIATE
                ),
            ),
            Stage("relay_back", fresnel(-z, PlaneKind.INTERMEDIATE, PlaneKind.PUPIL)),
            Stage("science", Fraunhofer(grid_in=pupil, grid_out=focal)),
        )
    )
    fx = np.asarray(focal.coords)
    fx_grid, fy_grid = np.meshgrid(fx, fx)
    # A TWO-SIDED zone: a pupil phase DM cannot null a real symmetric speckle at
    # both +3 and -3 at once (its modes are imaginary-symmetric or real-anti-
    # symmetric); the out-of-pupil DM supplies the missing real-symmetric quadrature.
    both_sides = (np.abs(fx_grid - 3.0) < 0.6) | (np.abs(fx_grid + 3.0) < 0.6)
    mask = both_sides & (np.abs(fy_grid) < 0.6)
    return path, field, jnp.asarray(mask)


class TestTwoDeformableMirrors:
    def test_command_spans_both_mirrors(self):
        path, field, mask = _relay_setup()
        command, _ = close_dark_hole(
            path, field, (0, 2), mask, n_steps=2, gain=0.3, regularization=1e-4
        )
        n_each = path.stages[0].op.basis.n_modes
        assert command.shape == (2 * n_each,)

    def test_two_dm_nulls_amplitude_that_one_dm_cannot(self):
        """A single pupil phase DM cannot null the two-sided real-symmetric
        amplitude speckle (its modes are imaginary-symmetric or real-anti-
        symmetric), so it floors near the initial contrast; the out-of-pupil
        second DM supplies the missing real-symmetric quadrature, so two DMs dig
        far deeper."""
        path, field, mask = _relay_setup()
        _, hist_two = close_dark_hole(
            path, field, (0, 2), mask, n_steps=60, gain=0.5, regularization=1e-7
        )
        _, hist_one = close_dark_hole(
            path, field, (0,), mask, n_steps=60, gain=0.5, regularization=1e-7
        )
        assert hist_one[-1] > 0.3 * hist_one[0]  # one DM cannot null it
        assert hist_two[-1] < 1e-6 * hist_two[0]  # two DMs reach a deep null
        assert hist_two[-1] < 1e-3 * hist_one[-1]  # decisively deeper

    def test_is_differentiable_in_gain(self):
        path, field, mask = _relay_setup()

        def final_contrast(gain):
            _, history = close_dark_hole(
                path, field, (0, 2), mask, n_steps=6, gain=gain, regularization=1e-6
            )
            return history[-1]

        grad = jax.grad(final_contrast)(0.5)
        assert jnp.isfinite(grad)

    def test_two_dm_digs_a_broadband_dark_hole(self):
        """A chromatic input field digs a broadband hole: the dark-zone response
        is stacked over wavelengths. A single DM (chromatic phase only) floors on
        the band-averaged amplitude speckle; two DMs reduce it, though a
        limited-DOF broadband null floors above the monochromatic case."""
        path, mono, mask = _relay_setup()
        field = broadcast_to_spectrum(mono, Spectrum.tophat(WL, 0.15, 3))
        _, hist_two = close_dark_hole(
            path, field, (0, 2), mask, n_steps=60, gain=0.5, regularization=1e-7
        )
        _, hist_one = close_dark_hole(
            path, field, (0,), mask, n_steps=60, gain=0.5, regularization=1e-7
        )
        assert hist_one[-1] > 0.8 * hist_one[0]  # one DM cannot null it broadband
        assert hist_two[-1] < 0.6 * hist_two[0]  # two DMs reduce the band contrast
        assert hist_two[-1] < 0.6 * hist_one[-1]  # the broadband two-DM advantage


class TestCloseDarkHole:
    def test_reduces_dark_zone_intensity(self):
        """Best case: the aberration lies in the DM span and the null is
        underdetermined (fewer masked pixels than modes), so the loop reaches a
        deep null. A real dark hole floors at the out-of-DM-span residual."""
        path, field, mask = _setup()
        _, history = close_dark_hole(
            path, field, 0, mask, n_steps=20, gain=0.5, regularization=1e-6
        )
        assert history[-1] < 1e-3 * history[0]

    def test_rejects_empty_dark_zone(self):
        path, field, _ = _setup()
        empty = jnp.zeros((32, 32), dtype=bool)
        with pytest.raises(ValueError, match="dark_zone"):
            close_dark_hole(
                path, field, 0, empty, n_steps=3, gain=0.5, regularization=1e-6
            )

    def test_rejects_non_deformable_mirror_stage(self):
        path, field, mask = _setup()
        with pytest.raises(TypeError, match="PhaseScreen"):
            # stage 1 is the Fraunhofer, not a deformable mirror.
            close_dark_hole(
                path, field, 1, mask, n_steps=3, gain=0.5, regularization=1e-6
            )

    def test_rejects_nonpositive_regularization(self):
        path, field, mask = _setup()
        with pytest.raises(ValueError, match="regularization"):
            close_dark_hole(
                path, field, 0, mask, n_steps=3, gain=0.5, regularization=0.0
            )

    def test_is_differentiable_in_gain(self):
        path, field, mask = _setup()

        def final_contrast(gain):
            _, history = close_dark_hole(
                path, field, 0, mask, n_steps=8, gain=gain, regularization=1e-6
            )
            return history[-1]

        grad = jax.grad(final_contrast)(0.5)
        assert jnp.isfinite(grad)


def _estimated_setup(npix=32, nfoc=64, pscale=0.25):
    """A pupil -> Fourier-DM -> focal path with a phase aberration and a
    one-sided (single-DM-nullable) dark zone, plus its unaberrated model."""
    pupil = Grid.pupil(npix)
    focal = Grid.focal(nfoc, pscale)
    coords = np.asarray(pupil.coords)
    xg, yg = np.meshgrid(coords, coords)
    aperture = (xg**2 + yg**2 <= 0.25).astype(complex)
    dm = fourier_dm_basis(pupil, n_actuators=16, k_min=2.0, k_max=7.0)
    path = OpticalPath(
        stages=(
            Stage("dm", PhaseScreen(dm, pupil, wavelength_nm=WL)),
            Stage("sci", Fraunhofer(grid_in=pupil, grid_out=focal)),
        )
    )
    opd = 4.0 * np.cos(2 * np.pi * 4 * xg) + 3.0 * np.cos(2 * np.pi * 5 * xg + 0.7)
    input_field = Field(
        data=jnp.asarray(aperture * np.exp(1j * 2 * np.pi * opd / WL)),
        grid=pupil,
        plane=PlaneKind.PUPIL,
    )
    model_field = Field(data=jnp.asarray(aperture), grid=pupil, plane=PlaneKind.PUPIL)
    fx = np.asarray(focal.coords)
    fxg, fyg = np.meshgrid(fx, fx)
    r = np.hypot(fxg, fyg)
    mask = jnp.asarray((r >= 2.0) & (r <= 7.0) & (fxg > 1.0))  # one-sided D-shape
    return path, dm, input_field, model_field, mask


class TestEstimatedLoop:
    def test_pairwise_loop_digs_with_an_honest_model(self):
        path, dm, input_field, model_field, mask = _estimated_setup()
        probes = probe_set(dm, amplitude_nm=2.0, n_probes=4)
        _, history = close_dark_hole(
            path, input_field, 0, mask, n_steps=20, gain=0.4, regularization=1e-6,
            estimator="pairwise", model_field=model_field, probes=probes,
        )
        # Digs about an order of magnitude; depth floors on the model mismatch
        # (the probe response is modelled without the unknown aberration).
        assert float(history[-1]) < 0.2 * float(history[0])

    def test_kalman_loop_digs_with_one_probe_per_step(self):
        path, dm, input_field, model_field, mask = _estimated_setup()
        probes = probe_set(dm, amplitude_nm=2.0, n_probes=2)
        _, history = close_dark_hole(
            path, input_field, 0, mask, n_steps=30, gain=0.4, regularization=1e-6,
            estimator="kalman", model_field=model_field, probes=probes,
        )
        assert float(history[-1]) < 0.2 * float(history[0])

    def test_oracle_still_digs_deeper_than_estimated(self):
        path, dm, input_field, model_field, mask = _estimated_setup()
        probes = probe_set(dm, amplitude_nm=2.0, n_probes=4)
        _, oracle = close_dark_hole(
            path, input_field, 0, mask, n_steps=20, gain=0.4, regularization=1e-6,
        )
        _, est = close_dark_hole(
            path, input_field, 0, mask, n_steps=20, gain=0.4, regularization=1e-6,
            estimator="pairwise", model_field=model_field, probes=probes,
        )
        assert float(oracle[-1]) < float(est[-1])  # perfect knowledge digs deeper

    def test_pairwise_requires_probes(self):
        path, _dm, input_field, model_field, mask = _estimated_setup()
        with pytest.raises(ValueError, match="probes"):
            close_dark_hole(
                path, input_field, 0, mask, n_steps=3, gain=0.4,
                regularization=1e-6, estimator="pairwise", model_field=model_field,
            )

    def test_rejects_unknown_estimator(self):
        path, _dm, input_field, _model_field, mask = _estimated_setup()
        with pytest.raises(ValueError, match="estimator"):
            close_dark_hole(
                path, input_field, 0, mask, n_steps=3, gain=0.4,
                regularization=1e-6, estimator="bogus",
            )
