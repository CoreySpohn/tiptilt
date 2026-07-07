"""Single-deformable-mirror wavefront control: a differentiable EFC dark-hole loop."""

import equinox as eqx
import jax
import jax.numpy as jnp


def _dm_coeffs(path, dm_index):
    return path.stages[dm_index].op.basis.coeffs


def close_dark_hole(
    path,
    input_field,
    dm_index,
    dark_zone_mask,
    *,
    n_steps,
    gain,
    regularization,
):
    """Dig a dark hole with one deformable mirror by electric-field conjugation.

    Linearizes the focal field with respect to the DM command ONCE (the control
    Jacobian is constant to first order in the small-signal dark-hole regime),
    builds a regularized real control matrix, then runs a differentiable
    ``lax.scan`` that RE-PROPAGATES for the measurement and updates the command
    each step by swapping the DM coefficients with ``eqx.tree_at`` -- never a
    reconstruction, which would re-run the propagator's construction-time gates.
    Because every propagation is a pure function, the loop differentiates through
    the feedback (e.g. the final contrast with respect to the loop gain).

    Args:
        path: An ``OpticalPath`` ending at the focal plane whose stage
            ``dm_index`` is a ``PhaseScreen`` deformable mirror.
        input_field: The entrance field carrying the aberration to correct.
        dm_index: Index of the DM ``PhaseScreen`` stage in ``path.stages``.
        dark_zone_mask: Boolean ``(y, x)`` focal-plane region to null.
        n_steps: Number of control iterations.
        gain: Loop gain (the fraction of the computed correction applied).
        regularization: Tikhonov regularization for the control-matrix inverse.

    Returns:
        ``(command, dark_zone_history)``: the final DM command and the mean
        dark-zone intensity at each iteration.
    """
    mask = jnp.asarray(dark_zone_mask)
    n_modes = path.stages[dm_index].op.basis.n_modes

    def focal_field(command):
        commanded = eqx.tree_at(lambda p: _dm_coeffs(p, dm_index), path, command)
        out, _ = commanded.propagate(input_field)
        return out.data

    # Hoist: the control Jacobian d(E_focal)/d(command), computed once.
    jacobian = jax.jacfwd(focal_field)(jnp.zeros(n_modes))  # (y, x, m) complex
    g_dz = jacobian[mask]  # (n_dz, m) complex
    # Real electric-field conjugation: a real command cancels a complex field, so
    # stack the real and imaginary response rows.
    response = jnp.concatenate([jnp.real(g_dz), jnp.imag(g_dz)], axis=0)  # (2 n_dz, m)
    gram = response.T @ response + regularization * jnp.eye(n_modes)
    control_matrix = jnp.linalg.solve(gram, response.T)  # (m, 2 n_dz)

    def step(command, _):
        field_dz = focal_field(command)[mask]  # exact re-propagation each step
        residual = jnp.concatenate([jnp.real(field_dz), jnp.imag(field_dz)])
        command_next = command - gain * (control_matrix @ residual)
        return command_next, jnp.mean(jnp.abs(field_dz) ** 2)

    return jax.lax.scan(step, jnp.zeros(n_modes), None, length=n_steps)
