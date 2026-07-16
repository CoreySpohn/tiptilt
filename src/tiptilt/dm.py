"""The programmable deformable mirror: an actuator grid with influence functions.

The modal bases (Fourier, Zernike, segment piston/tip/tilt) command abstract
modes; a real deformable mirror is commanded per ACTUATOR. This module builds
that device: a square actuator lattice across the pupil, one Gaussian
influence function per actuator with the standard nearest-neighbor coupling
parameterization (``f(r) = coupling^((r/pitch)^2)``, so the surface at the
adjacent actuator is ``coupling`` of the poke -- the ~10-15 percent of
electrostrictive and MEMS devices), peak-normalized so a unit coefficient is
a 1 nm OPD poke at the actuator.

Because the result is an ordinary ``ModeBasis`` inside an ordinary
``PhaseScreen``, EVERYTHING built on the control seams works on it unchanged
-- ``linearize``, ``close_dark_hole``, ``maintain_dark_hole``, the estimators,
the testbed -- but the command vector is now in actuator space, the language
published control algorithms speak. Actuator count sets the correctable field
of view (a dark hole reaches ``n_actuators / 2`` lambda/D); per-actuator
stroke limits live on the device (``clip``) and as the harness stroke cap.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from physicaloptix import ModeBasis, PhaseScreen, PlaneKind


def dm_influence_basis(grid, *, n_actuators, coupling=0.15, margin_actuators=1.0):
    """The actuator-grid influence-function basis on a pupil grid.

    Actuator centers form an ``n_actuators x n_actuators`` square lattice
    across the pupil diameter (pitch ``1 / n_actuators`` in pupil-diameter
    units); actuators whose centers fall more than ``margin_actuators``
    pitches outside the aperture edge are dropped (they would only add null
    modes). Each kept actuator contributes one Gaussian influence function
    with value ``coupling`` at its nearest neighbor, peak-normalized to a
    1 nm OPD poke per unit coefficient.

    Args:
        grid: The pupil ``Grid`` the modes are sampled on.
        n_actuators: Actuators across the pupil diameter.
        coupling: Influence at the adjacent actuator as a fraction of the
            poke (sets the Gaussian width).
        margin_actuators: How many pitches beyond the aperture edge (radius
            0.5) an actuator center may sit and still be kept.

    Returns:
        ``(basis, centers)``: a ``ModeBasis`` with ``B`` shape
        ``(n_active, npix, npix)`` in nm and zero coefficients, and the kept
        actuator centers, shape ``(n_active, 2)`` in pupil-diameter units.

    Raises:
        ValueError: If ``coupling`` is not in (0, 1) or no actuator survives.
    """
    if not 0.0 < coupling < 1.0:
        raise ValueError(f"coupling must be in (0, 1), got {coupling}")
    pitch = 1.0 / n_actuators
    lattice = (np.arange(n_actuators) + 0.5) / n_actuators - 0.5
    xc, yc = np.meshgrid(lattice, lattice)
    centers = np.stack([xc.ravel(), yc.ravel()], axis=1)
    keep = np.hypot(centers[:, 0], centers[:, 1]) <= 0.5 + margin_actuators * pitch
    centers = centers[keep]
    if centers.shape[0] == 0:
        raise ValueError("no actuator centers survive the aperture margin")

    coords = np.asarray(grid.coords)
    xg, yg = np.meshgrid(coords, coords)
    # f(r) = coupling^((r/pitch)^2): Gaussian with f(0)=1, f(pitch)=coupling.
    log_c = np.log(coupling)
    r2 = (xg[None, :, :] - centers[:, 0, None, None]) ** 2 + (
        yg[None, :, :] - centers[:, 1, None, None]
    ) ** 2
    modes = np.exp(log_c * r2 / pitch**2)
    basis = ModeBasis(B=jnp.asarray(modes), coeffs=jnp.zeros(centers.shape[0]))
    return basis, jnp.asarray(centers)


@jax.custom_jvp
def _round_straight_through(x):
    """``round(x)`` whose gradient is the identity (straight-through).

    The true derivative of rounding is zero almost everywhere, which would
    silently null every Jacobian column built by differentiating through a
    quantized mirror. The straight-through convention keeps the model
    Jacobian at the ideal slope -- exactly what a linearized controller
    assumes about its DAC -- while the staircase still bites in every
    propagated image.
    """
    return jnp.round(x)


@_round_straight_through.defjvp
def _round_straight_through_jvp(primals, tangents):
    (x,) = primals
    (t,) = tangents
    return jnp.round(x), t


class HardwareDM(PhaseScreen):
    """A deformable mirror that realizes its command imperfectly.

    A drop-in ``PhaseScreen``: every driver, estimator, and ``jacfwd``-built
    Jacobian accepts it unchanged (the ``isinstance`` seams see a
    ``PhaseScreen``). At propagation time the commanded coefficients pass
    through the hardware transfer before becoming an OPD::

        realized = gains * quantize(command) + offsets

    - ``dac_step_nm``: DAC least-significant-bit quantization (straight-
      through gradient, so model Jacobians keep the ideal slope while every
      image sees the staircase). Probe amplitudes must exceed the step or
      pairwise probing loses its signal -- real hardware physics.
    - ``actuator_gains``: per-actuator response (1 = perfect, 0 = dead).
      Gains flow into ``jacfwd`` Jacobians, so the model is gain-CALIBRATED:
      dead columns vanish and Tikhonov-regularized control works around
      them. An uncalibrated (model does not know) device needs a model-path
      vs truth-path split, which is a driver seam, not a device property.
    - ``actuator_offsets_nm``: additive surface offsets; a stuck actuator is
      gain 0 plus its stuck value here.

    Attributes:
        actuator_gains: Optional ``(n_modes,)`` response gains.
        actuator_offsets_nm: Optional ``(n_modes,)`` additive offsets in nm.
        dac_step_nm: Optional DAC step in nm (``None`` = continuous).
    """

    actuator_gains: Array | None
    actuator_offsets_nm: Array | None
    dac_step_nm: float | None = eqx.field(static=True)

    def __init__(
        self,
        basis,
        grid,
        *,
        wavelength_nm,
        plane=PlaneKind.PUPIL,
        actuator_gains=None,
        actuator_offsets_nm=None,
        dac_step_nm=None,
    ):
        """Build the imperfect mirror.

        Args:
            basis: The influence-function ``ModeBasis`` (coeffs = command).
            grid: The pupil ``Grid``.
            wavelength_nm: Design wavelength of the phase screen.
            plane: The plane the mirror sits in.
            actuator_gains: Optional per-actuator response gains.
            actuator_offsets_nm: Optional per-actuator offsets in nm.
            dac_step_nm: Optional DAC quantization step in nm.
        """
        super().__init__(basis, grid, wavelength_nm=wavelength_nm, plane=plane)
        self.actuator_gains = (
            None if actuator_gains is None else jnp.asarray(actuator_gains)
        )
        self.actuator_offsets_nm = (
            None if actuator_offsets_nm is None else jnp.asarray(actuator_offsets_nm)
        )
        self.dac_step_nm = None if dac_step_nm is None else float(dac_step_nm)

    def __check_init__(self):
        """Validate the hardware fields against the basis."""
        if self.dac_step_nm is not None and self.dac_step_nm <= 0.0:
            raise ValueError(f"dac_step_nm must be positive, got {self.dac_step_nm}")
        n_modes = self.basis.B.shape[0]
        for name, values in (
            ("actuator_gains", self.actuator_gains),
            ("actuator_offsets_nm", self.actuator_offsets_nm),
        ):
            if values is not None and values.shape != (n_modes,):
                raise ValueError(
                    f"{name} must have shape ({n_modes},), got {tuple(values.shape)}"
                )

    def realized_command(self):
        """The command the hardware actually applies, in nm."""
        command = self.basis.coeffs
        if self.dac_step_nm is not None:
            command = self.dac_step_nm * _round_straight_through(
                command / self.dac_step_nm
            )
        if self.actuator_gains is not None:
            command = self.actuator_gains * command
        if self.actuator_offsets_nm is not None:
            command = command + self.actuator_offsets_nm
        return command

    def __call__(self, field):
        """Apply the phase of the REALIZED (not commanded) surface."""
        realized = eqx.tree_at(lambda s: s.basis.coeffs, self, self.realized_command())
        return PhaseScreen.__call__(realized, field)


class DeformableMirror(eqx.Module):
    """A programmable actuator-grid mirror, ready to drop into a path.

    ``screen`` is the ``PhaseScreen`` element to place in a ``Stage``; its
    coefficients ARE the per-actuator OPD pokes in nm, so every existing
    driver commands this device unchanged and returns actuator-space
    commands. The device carries the geometry (``centers``) and the
    per-actuator stroke limit (``clip``); pass ``stroke_limit_nm`` as the
    harness ``stroke_cap_nm`` to enforce it inside a loop.

    Attributes:
        screen: The commandable ``PhaseScreen`` element.
        centers: Kept actuator centers, ``(n_active, 2)``.
        n_actuators: Actuators across the pupil diameter.
        coupling: Nearest-neighbor influence fraction.
        stroke_limit_nm: Per-actuator OPD limit (``None`` = unlimited).
    """

    screen: PhaseScreen
    centers: Array
    n_actuators: int = eqx.field(static=True)
    coupling: float = eqx.field(static=True)
    stroke_limit_nm: float | None = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        grid,
        *,
        n_actuators,
        wavelength_nm,
        coupling=0.15,
        margin_actuators=1.0,
        stroke_limit_nm=None,
        plane=PlaneKind.PUPIL,
        dac_step_nm=None,
        actuator_gains=None,
        actuator_offsets_nm=None,
    ):
        """Build the device on a pupil grid.

        Args:
            grid: The pupil ``Grid``.
            n_actuators: Actuators across the pupil diameter.
            wavelength_nm: Design wavelength of the phase screen.
            coupling: Nearest-neighbor influence fraction.
            margin_actuators: Kept margin beyond the aperture edge, in
                pitches.
            stroke_limit_nm: Optional per-actuator OPD limit.
            plane: The plane the mirror sits in (an out-of-pupil mirror uses
                ``PlaneKind.INTERMEDIATE`` behind a Fresnel relay).
            dac_step_nm: Optional DAC quantization step in nm; any hardware
                knob makes ``screen`` a ``HardwareDM``.
            actuator_gains: Optional per-actuator response gains (0 = dead).
            actuator_offsets_nm: Optional per-actuator offsets in nm (a
                stuck actuator is gain 0 plus its value here).

        Returns:
            A ``DeformableMirror``.
        """
        basis, centers = dm_influence_basis(
            grid,
            n_actuators=n_actuators,
            coupling=coupling,
            margin_actuators=margin_actuators,
        )
        if (
            dac_step_nm is None
            and actuator_gains is None
            and actuator_offsets_nm is None
        ):
            screen = PhaseScreen(basis, grid, wavelength_nm=wavelength_nm, plane=plane)
        else:
            screen = HardwareDM(
                basis,
                grid,
                wavelength_nm=wavelength_nm,
                plane=plane,
                actuator_gains=actuator_gains,
                actuator_offsets_nm=actuator_offsets_nm,
                dac_step_nm=dac_step_nm,
            )
        return cls(
            screen=screen,
            centers=centers,
            n_actuators=n_actuators,
            coupling=coupling,
            stroke_limit_nm=stroke_limit_nm,
        )

    @property
    def n_active(self):
        """Number of kept (commandable) actuators."""
        return self.centers.shape[0]

    def clip(self, command):
        """Per-actuator stroke clipping (identity when unlimited).

        Args:
            command: Actuator strokes in nm.

        Returns:
            The command, clipped to ``[-stroke_limit_nm, stroke_limit_nm]``.
        """
        if self.stroke_limit_nm is None:
            return command
        return jnp.clip(command, -self.stroke_limit_nm, self.stroke_limit_nm)

    def surface(self, command):
        """The OPD map a command produces, shape ``(npix, npix)`` in nm.

        Args:
            command: Actuator strokes in nm.

        Returns:
            The summed influence-function surface.
        """
        return jnp.tensordot(command, self.screen.basis.B, axes=1)


__all__ = ["DeformableMirror", "HardwareDM", "dm_influence_basis"]
