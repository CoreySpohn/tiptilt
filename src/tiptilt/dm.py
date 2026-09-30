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


class ActuatorLattice(eqx.Module):
    """An actuator-lattice OPD basis evaluated by Fourier convolution.

    The dense influence basis stores one ``(npix, npix)`` map per actuator,
    which stops being possible long before flight scale (two 96 x 96 lattices
    on a 2048-pixel pupil are ~600 GB). A lattice of identical influence
    functions is a convolution, so the surface is evaluated instead as

        surface = ifft2( G(f) * S(f) ),   S(f) = sum_a c_a exp(-2 pi i f . x_a)

    with ``G`` the analytic Fourier transform of the Gaussian influence
    function and ``S`` a matrix Fourier transform of the ``(n, n)`` command
    lattice onto the pupil grid's FFT frequencies (two ``(npix, n)``
    matrices). Nothing per actuator is ever stored, the map is linear in the
    command, and it differentiates like any other pytree leaf. The lattice is
    square with a uniform pitch and every actuator is commandable; ``coeffs``
    is the full ``(n * n,)`` command in row-major ``(y, x)`` order. Each unit
    coefficient is a 1 nm OPD poke at its actuator, the same contract as
    :func:`dm_influence_basis`. A lattice wider than the array is allowed, and
    the periodic FFT then wraps the outside rows to the opposite edge (the same
    behavior as any FFT-convolution mirror model); whether those
    wrapped rows land inside the illuminated pupil is the caller's geometry to
    check, since a controller would otherwise be handed unphysical authority.

    Quacks like a ``ModeBasis`` for the control seams (``coeffs``, ``n_modes``,
    ``opd``, ``kind``); ``B`` is an EMPTY stack so that the ``PhaseScreen``
    grid check passes while consumers that need dense modes refuse it
    (``probe_set`` raises; ``linearize``'s analytic route fails on the shape).

    Attributes:
        coeffs: The command, ``(n_across * n_across,)`` nm OPD pokes.
        transfer: ``(npix, npix)`` complex frequency response of one poke,
            including the half-pixel-offset grid phase and the FFT scale.
        fx: ``(npix, n_across)`` complex MFT kernel along x.
        fy: ``(npix, n_across)`` complex MFT kernel along y.
        n_across: Actuators across the lattice.
        npix: Pupil grid size the surface is evaluated on.
        pitch: Actuator pitch in pupil-diameter units.
        coupling: Nearest-neighbor influence fraction.
        centers: Actuator centers, ``(n_across * n_across, 2)`` in
            pupil-diameter units, row-major ``(y, x)`` order.
        kind: Always ``"opd"``.
    """

    coeffs: Array
    transfer: Array
    fx: Array
    fy: Array
    centers: Array
    n_across: int = eqx.field(static=True)
    npix: int = eqx.field(static=True)
    pitch: float = eqx.field(static=True)
    coupling: float = eqx.field(static=True)
    kind: str = eqx.field(static=True, default="opd")

    @classmethod
    def build(cls, grid, *, n_across, pitch, coupling=0.15, center=(0.0, 0.0)):
        """Build the lattice on a pupil grid.

        Args:
            grid: The pupil ``Grid`` (half-pixel-offset coordinates in
                pupil-diameter units).
            n_across: Actuators across the lattice.
            pitch: Actuator pitch in pupil-diameter units.
            coupling: Influence at the adjacent actuator as a fraction of the
                poke (sets the Gaussian width, ``sigma = pitch /
                sqrt(-ln coupling)``).
            center: ``(x, y)`` of the lattice center in pupil-diameter units.

        Returns:
            An ``ActuatorLattice`` with zero command.

        Raises:
            ValueError: If ``coupling`` is not in (0, 1) or ``pitch`` is not
                positive.
        """
        if not 0.0 < coupling < 1.0:
            raise ValueError(f"coupling must be in (0, 1), got {coupling}")
        if pitch <= 0.0:
            raise ValueError(f"pitch must be positive, got {pitch}")
        npix = grid.npix
        dx = grid.dx
        sigma = pitch / np.sqrt(-np.log(coupling))
        lattice = (np.arange(n_across) - (n_across - 1) / 2.0) * pitch
        cx = lattice + center[0]
        cy = lattice + center[1]
        xc, yc = np.meshgrid(cx, cy)
        centers = np.stack([xc.ravel(), yc.ravel()], axis=1)

        freq = np.fft.fftfreq(npix, d=dx)
        # Gaussian exp(-r^2 / sigma^2) -> pi sigma^2 exp(-pi^2 sigma^2 f^2); the
        # ramp evaluates the inverse FFT on the half-pixel-offset coordinates
        # x_j = (j - npix/2 + 1/2) dx, and 1/dx^2 is the Riemann df^2 times the
        # npix^2 that ifft2 divides out.
        offset = (0.5 - npix / 2.0) * dx
        ramp = np.exp(2j * np.pi * freq * offset)
        gauss_1d = np.sqrt(np.pi) * sigma * np.exp(-((np.pi * sigma * freq) ** 2))
        transfer = np.outer(gauss_1d * ramp, gauss_1d * ramp) / dx**2
        fx = np.exp(-2j * np.pi * np.outer(freq, cx))
        fy = np.exp(-2j * np.pi * np.outer(freq, cy))
        return cls(
            coeffs=jnp.zeros(n_across * n_across),
            transfer=jnp.asarray(transfer),
            fx=jnp.asarray(fx),
            fy=jnp.asarray(fy),
            centers=jnp.asarray(centers),
            n_across=int(n_across),
            npix=int(npix),
            pitch=float(pitch),
            coupling=float(coupling),
        )

    @property
    def n_modes(self):
        """Number of commandable actuators."""
        return self.n_across * self.n_across

    @property
    def B(self):
        """An EMPTY mode stack ``(0, npix, npix)``: modes are never stored."""
        return jnp.zeros((0, self.npix, self.npix))

    def opd(self):
        """The OPD map of the current command, ``(npix, npix)`` in nm."""
        return self.surface(self.coeffs)

    def surface(self, command):
        """The OPD map of an arbitrary command, ``(npix, npix)`` in nm."""
        lattice = command.reshape(self.n_across, self.n_across)
        spectrum = self.fy @ lattice.astype(self.fy.dtype) @ self.fx.T
        return jnp.real(jnp.fft.ifft2(self.transfer * spectrum))


class ActuatorDM(PhaseScreen):
    """A deformable mirror on an :class:`ActuatorLattice`: flight-scale ready.

    A drop-in ``PhaseScreen`` whose ``basis`` is the FFT-evaluated lattice, so
    every driver that swaps ``stage.op.basis.coeffs`` commands it unchanged
    and the ``jacfwd`` / matrix-free Jacobians differentiate through the
    convolution. Use it where :class:`DeformableMirror` would materialize an
    impossible mode stack; use the dense device where probe generation or
    the analytic linearization need explicit modes.
    """

    basis: ActuatorLattice

    def __init__(self, lattice, grid, *, wavelength_nm, plane=PlaneKind.PUPIL):
        """Wrap a lattice as a phase stage.

        Args:
            lattice: The ``ActuatorLattice`` (its ``coeffs`` are the command).
            grid: The pupil ``Grid`` the lattice was built on.
            wavelength_nm: Design wavelength of the phase screen.
            plane: The plane the mirror sits in (``INTERMEDIATE`` for an
                out-of-pupil mirror behind a Fresnel relay).
        """
        super().__init__(lattice, grid, wavelength_nm=wavelength_nm, plane=plane)

    def __check_init__(self):
        """The lattice must have been built on this grid."""
        if self.basis.npix != self.grid.npix:
            raise ValueError(
                f"lattice built on {self.basis.npix} pixels does not match grid "
                f"({self.grid.npix})"
            )

    @classmethod
    def build(
        cls,
        grid,
        *,
        n_across,
        pitch,
        wavelength_nm,
        coupling=0.15,
        center=(0.0, 0.0),
        plane=PlaneKind.PUPIL,
    ):
        """Build the mirror on a pupil grid (see :meth:`ActuatorLattice.build`)."""
        lattice = ActuatorLattice.build(
            grid, n_across=n_across, pitch=pitch, coupling=coupling, center=center
        )
        return cls(lattice, grid, wavelength_nm=wavelength_nm, plane=plane)

    def surface(self, command):
        """The OPD map a command produces, ``(npix, npix)`` in nm."""
        return self.basis.surface(command)


__all__ = [
    "ActuatorDM",
    "ActuatorLattice",
    "DeformableMirror",
    "HardwareDM",
    "dm_influence_basis",
]
