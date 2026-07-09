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

        Returns:
            A ``DeformableMirror``.
        """
        basis, centers = dm_influence_basis(
            grid,
            n_actuators=n_actuators,
            coupling=coupling,
            margin_actuators=margin_actuators,
        )
        screen = PhaseScreen(basis, grid, wavelength_nm=wavelength_nm, plane=plane)
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


__all__ = ["DeformableMirror", "dm_influence_basis"]
