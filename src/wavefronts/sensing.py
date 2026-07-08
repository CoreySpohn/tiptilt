"""Focal-plane wavefront sensing: pairwise-probe electric-field estimation.

A coronagraph detector measures intensity, not the complex focal field the
control loop needs. Pairwise probing recovers the field by applying equal
positive and negative deformable-mirror probes: the difference image cancels
the quadratic terms and the unknown incoherent bias, leaving the linear
cross-term ``dI_j = 4 Re{conj(E) p_j}`` (Give'on 2011, Groff 2016). Stacking a
few probes with independent quadratures and solving the per-pixel least squares
gives the complex field, replacing the ``jax.jacfwd``-on-truth shortcut that a
real system cannot take.
"""

import equinox as eqx
import jax
import jax.numpy as jnp


def probe_set(basis, *, amplitude_nm, n_probes=3, seed=0):
    """Deformable-mirror probe commands for pairwise field estimation.

    Each probe is a fixed pseudo-random combination of the deformable mirror's
    modes, scaled so its wavefront error is ``amplitude_nm`` RMS. A random
    combination of the band-limited control modes spreads the probe field over
    the whole controllable region, and several probes give the per-pixel
    quadrature diversity the batch solve needs (``n_probes >= 2``, three or more
    to condition the inverse). This is the simplest probe that covers an annular
    dark hole uniformly; localized sinc probes (Give'on 2011) are a later
    signal-to-noise refinement.

    Args:
        basis: The deformable mirror's ``ModeBasis`` (its ``B`` sets the OPD
            scale of a unit coefficient).
        amplitude_nm: Per-probe RMS wavefront amplitude in nanometres. Small
            enough to stay in the linear regime (a few nanometres) yet bright
            enough for signal to noise.
        n_probes: Number of probes (each applied as a positive and a negative
            pair).
        seed: Base PRNG seed, so a probe set is deterministic.

    Returns:
        A list of ``n_probes`` command vectors (each shape ``(n_modes,)``).
    """
    modes = basis.B
    n_modes = modes.shape[0]
    probes = []
    for key in jax.random.split(jax.random.PRNGKey(seed), n_probes):
        coeffs = jax.random.normal(key, (n_modes,))
        opd = jnp.tensordot(coeffs, modes, axes=1)
        rms = jnp.sqrt(jnp.mean(opd**2))
        probes.append(coeffs * (amplitude_nm / rms))
    return probes


def _with_command(path, dm_index, command):
    """A copy of ``path`` with deformable-mirror ``dm_index`` set to ``command``."""
    return eqx.tree_at(
        lambda p: p.stages[dm_index].op.basis.coeffs, path, command
    )


def _focal_field(path, field, dm_index, command):
    """The complex focal field for a given deformable-mirror command."""
    out, _ = _with_command(path, dm_index, command).propagate(field)
    return out.data


def estimate_field_pairwise(
    path,
    input_field,
    dm_index,
    probes,
    dark_zone_mask,
    *,
    model_field=None,
    command=None,
    detector=None,
    key=None,
    regularization=0.0,
):
    """Pairwise-probe estimate of the dark-zone field through an optical path.

    Applies each probe as a positive and negative deformable-mirror pair, reads
    the two focal images off the TRUE ``input_field`` (optionally through a
    ``detector`` for noise), and computes each probe's added field from the
    ``model_field`` (the design pupil, no aberration -- keeping the estimate
    honest, since a real system has only the model, not the true field). The
    per-pixel batch least squares then returns the complex dark-zone field.

    Args:
        path: The ``OpticalPath`` ending at the focal plane; ``dm_index`` is the
            probe deformable mirror (a ``PhaseScreen``).
        input_field: The true entrance field (carries the unknown aberration).
        dm_index: Stage index of the probe deformable mirror.
        probes: Probe command vectors (see :func:`probe_set`).
        dark_zone_mask: Boolean focal-plane region to estimate.
        model_field: The design entrance field for the model probe response;
            defaults to ``input_field`` (an oracle model).
        command: Current deformable-mirror command the probes are applied
            around; defaults to zeros.
        detector: Optional ``callable(image, key) -> image`` applying noise (in
            normalized-intensity units) to each focal image.
        key: PRNG key for the detector (split per image); required if
            ``detector`` is given.
        regularization: Tikhonov term for the per-pixel solve.

    Returns:
        Complex field estimate over the masked pixels, shape ``(n_dark,)``.
    """
    if command is None:
        command = path.stages[dm_index].op.basis.coeffs  # probe around current state
    if model_field is None:
        model_field = input_field
    keys = (
        jax.random.split(key, 2 * len(probes))
        if detector is not None
        else [None] * (2 * len(probes))
    )

    probe_fields, diff_images = [], []
    for j, probe in enumerate(probes):
        e_plus = _focal_field(path, input_field, dm_index, command + probe)
        e_minus = _focal_field(path, input_field, dm_index, command - probe)
        i_plus = jnp.abs(e_plus) ** 2
        i_minus = jnp.abs(e_minus) ** 2
        if detector is not None:
            i_plus = detector(i_plus, keys[2 * j])
            i_minus = detector(i_minus, keys[2 * j + 1])
        diff_images.append((i_plus - i_minus)[dark_zone_mask])
        # Symmetric probe field (f(+) - f(-))/2 cancels the even (quadratic)
        # deformable-mirror nonlinearity, matching the symmetric difference image.
        model_plus = _focal_field(path, model_field, dm_index, command + probe)
        model_minus = _focal_field(path, model_field, dm_index, command - probe)
        probe_fields.append(0.5 * (model_plus - model_minus)[dark_zone_mask])

    return pairwise_estimate(
        jnp.stack(probe_fields), jnp.stack(diff_images), regularization=regularization
    )


def pairwise_estimate(probe_fields, diff_images, *, regularization=0.0):
    """Batch pairwise-probe estimate of the complex focal field per pixel.

    Solves ``dI_j = 4 Re{conj(E) p_j}`` for ``E`` at every pixel. With
    ``x = [Re E; Im E]`` and observation matrix ``H = 4 [Re p_j, Im p_j]`` the
    per-pixel normal equations are ``x = (H^T H + reg I)^{-1} H^T z``. At least
    two probes with linearly independent ``(Re p, Im p)`` directions are needed
    for ``H`` to have rank two; a single probe recovers only one quadrature.

    Args:
        probe_fields: Complex model probe fields, shape ``(n_probes, n_pixels)``
            (the field each probe adds at the estimated pixels).
        diff_images: Measured probe-difference intensities ``I_+ - I_-``, shape
            ``(n_probes, n_pixels)``.
        regularization: Non-negative Tikhonov term stabilizing the per-pixel
            inverse (helps where the probe directions are near-degenerate).

    Returns:
        Complex field estimate, shape ``(n_pixels,)``.
    """
    # H per pixel: (n_pixels, n_probes, 2), stacking the real and imag rows.
    h_rows = 4.0 * jnp.stack([jnp.real(probe_fields), jnp.imag(probe_fields)], axis=-1)
    h = jnp.moveaxis(h_rows, 1, 0)  # (n_pixels, n_probes, 2)
    z = jnp.moveaxis(diff_images, 1, 0)[..., jnp.newaxis]  # (n_pixels, n_probes, 1)

    gram = jnp.einsum("kna,knb->kab", h, h)  # (n_pixels, 2, 2)
    rhs = jnp.einsum("kna,knb->kab", h, z)  # (n_pixels, 2, 1)
    reg = regularization * jnp.eye(2)
    x = jnp.linalg.solve(gram + reg, rhs)[..., 0]  # (n_pixels, 2)
    return x[:, 0] + 1j * x[:, 1]
