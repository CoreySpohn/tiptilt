"""Cross-channel wavefront control: sense one arm, correct them all.

The shared-segment physics gives every channel the SAME shared-mode
coefficients through its OWN sensitivity block, so a field measured in one
arm determines a correction valid in every arm -- up to the sensing arm's
private (non-common-path) error, which the solve cannot distinguish from
shared drift and therefore INJECTS into the other arms (the aliasing failure
mode this module makes measurable, not just avoidable).

The machinery is three pieces on the existing seams:

- ``MultiChannelModel``: the dark-zone views of a
  ``physicaloptix.MultiChannelLinearization`` -- per channel, the masked
  shared-mode block ``g = d(E_dz)/d(shared mode)``.
- ``shared_dm_command`` / ``FeedForwardController``: the analytic map from a
  sensed dark-zone field to shared-mode coefficients (a real least squares
  over the sensing arms' stacked blocks) and its negation as the shared-DM
  command. Exact matmuls; no re-propagation.
- ``run_multichannel``: the co-run driver -- advance the drift on the TRUE
  system, propagate the container ONCE per frame, sense the chosen arms
  against their calibration reference frames, and command the shared
  corrector. Histories are reported both ABSOLUTE and as EXCESS over the
  reference (the static Airy + NCPA floor), the axis maintenance cares
  about.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from physicaloptix import PhaseScreen, linearize_shared

from tiptilt.control import AbstractController


def _stage_index(path, name, context):
    names = [stage.name for stage in path.stages]
    if name not in names:
        raise ValueError(f"unknown {context} stage {name!r}; stages are {names}")
    return names.index(name)


class MultiChannelModel(eqx.Module):
    """Masked dark-zone views of the per-channel shared-mode blocks.

    Attributes:
        names: Channel names, in linearization order.
        g_blocks: Per channel, the ``(n_dark, m)`` complex shared-mode block.
        e_refs: Per channel, the nominal (reference) dark-zone field.
        masks: Per channel, the boolean focal mask.
    """

    names: tuple = eqx.field(static=True)
    g_blocks: tuple
    e_refs: tuple
    masks: tuple

    @classmethod
    def build(cls, mcl, masks):
        """Mask each channel's shared block onto its dark zone.

        Args:
            mcl: A ``physicaloptix.MultiChannelLinearization``.
            masks: Dict mapping every channel name to its boolean focal mask.

        Returns:
            A ``MultiChannelModel``.
        """
        missing = [name for name in mcl.names if name not in masks]
        if missing:
            raise ValueError(f"masks missing for channel(s) {missing}")
        g_blocks, e_refs, mask_list = [], [], []
        for name in mcl.names:
            channel = mcl[name]
            mask = jnp.asarray(masks[name])
            g_blocks.append(channel.g_shared[:, mask].T)  # (n_dark, m)
            e_refs.append(channel.e_nom[mask])
            mask_list.append(mask)
        return cls(
            names=tuple(mcl.names),
            g_blocks=tuple(g_blocks),
            e_refs=tuple(e_refs),
            masks=tuple(mask_list),
        )

    def _index(self, name):
        if name not in self.names:
            raise KeyError(f"unknown channel {name!r}; channels are {self.names}")
        return self.names.index(name)

    def g_block(self, name):
        """The named channel's ``(n_dark, m)`` shared-mode dark-zone block."""
        return self.g_blocks[self._index(name)]

    def e_ref(self, name):
        """The named channel's reference (nominal) dark-zone field."""
        return self.e_refs[self._index(name)]

    def mask(self, name):
        """The named channel's boolean focal mask."""
        return self.masks[self._index(name)]


def shared_dm_command(model, sense, e_hat, *, regularization, gain=1.0):
    """Estimate the shared coefficients from a sensed field; negate them.

    Solves the real least squares ``min_eps || [Re; Im](G_sense eps - e) ||``
    over the sensing arm's shared block and returns the shared-DM command
    that cancels the estimate (the corrector shares the drift basis, so the
    command is the negated estimate).

    Args:
        model: The ``MultiChannelModel``.
        sense: The sensing channel name.
        e_hat: The sensed dark-zone field DELTA (reference-subtracted),
            shape ``(n_dark,)`` complex.
        regularization: Tikhonov term of the modal solve.
        gain: Fraction of the correction applied.

    Returns:
        ``(command, eps_hat)``: the shared-DM command and the raw estimate.
    """
    g = model.g_block(sense)
    h = jnp.concatenate([jnp.real(g), jnp.imag(g)], axis=0)  # (2n, m)
    z = jnp.concatenate([jnp.real(e_hat), jnp.imag(e_hat)])
    gram = h.T @ h + regularization * jnp.eye(h.shape[1])
    eps_hat = jnp.linalg.solve(gram, h.T @ z)
    return -gain * eps_hat, eps_hat


class FeedForwardController(AbstractController):
    """Cross-channel feed-forward: sensed fields in, a shared command out.

    Stacks the sensing arms' shared blocks (weighted) into one real least
    squares whose solution is the shared-mode estimate; the delta is its
    negation times the gain. Because the sensing arms' PRIVATE errors are
    indistinguishable from shared drift in this solve, they are injected
    into every corrected arm -- measure it with the aliasing scenario before
    trusting a feed-forward loop.

    Attributes:
        solve_matrix: ``(m, 2 n_stacked)`` least-squares operator.
        gain: Fraction of the correction applied per step.
    """

    solve_matrix: Array
    gain: Array

    @classmethod
    def build(cls, model, *, sense, gain, regularization, weights=None):
        """The stacked-arm feed-forward law.

        Args:
            model: The ``MultiChannelModel``.
            sense: Tuple of sensing channel names (their estimates are
                concatenated in this order).
            gain: Loop gain.
            regularization: Tikhonov term of the modal solve.
            weights: Optional per-arm weights (equal by default).

        Returns:
            A ``FeedForwardController``.
        """
        sense = tuple(sense)
        if weights is None:
            weights = jnp.ones(len(sense))
        weights = jnp.asarray(weights)
        blocks = [weights[i] * model.g_block(name) for i, name in enumerate(sense)]
        g = jnp.concatenate(blocks, axis=0)  # (n_stacked, m)
        h = jnp.concatenate([jnp.real(g), jnp.imag(g)], axis=0)
        gram = h.T @ h + regularization * jnp.eye(h.shape[1])
        solve_matrix = jnp.linalg.solve(gram, h.T)
        return cls(solve_matrix=solve_matrix, gain=jnp.asarray(gain))

    def command_delta(self, estimate):
        """The negated weighted-least-squares shared-mode estimate."""
        flat = estimate.reshape(-1)
        z = jnp.concatenate([jnp.real(flat), jnp.imag(flat)])
        eps_hat = self.solve_matrix @ z
        return self, -self.gain * eps_hat


def run_multichannel(
    system,
    input_field,
    *,
    shared_dm_stage,
    drift_stage,
    sense,
    science,
    masks,
    wavelength_nm,
    drift_table,
    n_steps,
    gain,
    regularization,
    local_drift=None,
    weights=None,
    substeps=1,
):
    """Co-run a forked system under drift with cross-channel feed-forward.

    Each frame: advance the shared drift (and any per-branch private drift)
    on the TRUE system, propagate the container ONCE, read every channel's
    dark zone, sense the chosen arms against their CALIBRATION REFERENCE
    frames (the clean system's dark-zone fields), and apply the feed-forward
    shared command. The sensing weights are a plain vector (equal by
    default); ``substeps`` repeats the sense-and-correct inner loop per
    drift frame (the v1 stand-in for a faster sensing cadence).

    Args:
        system: The ``OpticalSystem`` (trunk carries the drift screen and
            the shared corrector).
        input_field: The true entrance field.
        shared_dm_stage: Trunk stage name of the shared corrector screen.
        drift_stage: Trunk stage name of the drift carrier screen.
        sense: Tuple of channel names whose dark zones feed the solve.
        science: The science channel name (reported first; all channels are
            reported).
        masks: Dict mapping every channel name to its focal mask.
        wavelength_nm: Design wavelength of the linearization.
        drift_table: ``(n_steps, m_shared)`` drift coefficients per frame.
        n_steps: Number of frames.
        gain: Feed-forward gain (0 = open loop).
        regularization: Tikhonov term of the modal solve.
        local_drift: Optional dict mapping a branch name to
            ``(stage_name, table)`` -- that branch's private drift.
        weights: Optional per-sensing-arm weights.
        substeps: Sense-and-correct iterations per drift frame.

    Returns:
        Dict with ``command`` (final shared command), ``history`` (absolute
        dark-zone contrast per channel), and ``excess`` (contrast of the
        reference-subtracted field per channel -- the drift-tracking axis).
    """
    del science  # all channels are reported; kept for call-site clarity
    trunk_drift = _stage_index(system.trunk, drift_stage, "drift")
    trunk_dm = _stage_index(system.trunk, shared_dm_stage, "shared corrector")
    for stage_idx, label in ((trunk_drift, "drift"), (trunk_dm, "shared_dm")):
        if not isinstance(system.trunk.stages[stage_idx].op, PhaseScreen):
            raise TypeError(f"{label} stage is not a PhaseScreen")

    mcl = linearize_shared(
        system,
        input_field,
        wavelength_nm=wavelength_nm,
        shared_stage=shared_dm_stage,
    )
    model = MultiChannelModel.build(mcl, masks)
    controller = FeedForwardController.build(
        model,
        sense=tuple(sense),
        gain=gain,
        regularization=regularization,
        weights=weights,
    )

    local = {}
    if local_drift:
        branch_names = [branch.name for branch in system.branches]
        for name, (stage_name, table) in local_drift.items():
            branch_idx = branch_names.index(name)
            stage_idx = _stage_index(
                system.branches[branch_idx].path, stage_name, f"{name} local"
            )
            local[name] = (branch_idx, stage_idx, jnp.asarray(table))

    def with_state(command, frame):
        updated = eqx.tree_at(
            lambda s: (
                s.trunk.stages[trunk_drift].op.basis.coeffs,
                s.trunk.stages[trunk_dm].op.basis.coeffs,
            ),
            system,
            (jnp.asarray(drift_table[frame]), command),
        )
        for _name, (branch_idx, stage_idx, table) in local.items():
            updated = eqx.tree_at(
                lambda s, b=branch_idx, st=stage_idx: (
                    s.branches[b].path.stages[st].op.basis.coeffs
                ),
                updated,
                table[frame],
            )
        return updated

    command = jnp.zeros(model.g_blocks[0].shape[1])
    history = {name: [] for name in model.names}
    excess = {name: [] for name in model.names}
    for frame in range(n_steps):
        for _ in range(substeps):
            outputs, _ = with_state(command, frame).propagate(input_field)
            sensed = jnp.concatenate(
                [
                    outputs[name].data[model.mask(name)] - model.e_ref(name)
                    for name in sense
                ]
            )
            controller, delta = controller.command_delta(sensed)
            command = command + delta
        outputs, _ = with_state(command, frame).propagate(input_field)
        for name in model.names:
            dz = outputs[name].data[model.mask(name)]
            history[name].append(jnp.mean(jnp.abs(dz) ** 2))
            excess[name].append(jnp.mean(jnp.abs(dz - model.e_ref(name)) ** 2))
    return {
        "command": command,
        "history": {name: jnp.stack(vals) for name, vals in history.items()},
        "excess": {name: jnp.stack(vals) for name, vals in excess.items()},
    }
