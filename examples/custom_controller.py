"""Bring your own control algorithm: a worked example.

The control loop factors onto two seams -- ``AbstractEstimator`` (images in,
field estimate out) and ``AbstractController`` (estimate in, DM command delta
out). Implementing ONE method on either seam plugs a new algorithm into every
driver (``close_dark_hole``, ``maintain_dark_hole``, the benchmark harness)
and scores it on the same scenarios and metrics as the built-in laws.

This example defines a leaky-integrator EFC variant (a STATEFUL law: the
carried state is the accumulated command), registers it in the testbed, and
benchmarks it against plain EFC and stroke minimization on a programmable
actuator-grid deformable mirror -- commands in actuator space, exactly how
published algorithms are expressed. Because every propagation is differentiable
JAX, a custom law can also be TRAINED by gradient (differentiate the final
contrast with respect to its parameters).

Run:  python examples/custom_controller.py
"""

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)  # deep contrast needs float64

import jax.numpy as jnp  # noqa: E402

from tiptilt.control import AbstractController, EFCController  # noqa: E402
from tiptilt.testbed import ALGORITHMS, dig_from_cold, run, sweep  # noqa: E402


class LeakyIntegratorEFC(AbstractController):
    """EFC with a leaky-integrator memory.

    Each step applies the EFC correction MINUS a small leak of the total
    command applied so far, the classic guard against slow command runaway
    under model error. The carried state (``accumulated``) demonstrates the
    stateful half of the controller seam.
    """

    efc: EFCController
    accumulated: jnp.ndarray
    leak: float = eqx.field(static=True)

    def command_delta(self, estimate):
        """One correction: the EFC delta with a leak on the accumulated command."""
        _, raw = self.efc.command_delta(estimate)
        delta = raw - self.leak * self.accumulated
        new_self = eqx.tree_at(lambda c: c.accumulated, self, self.accumulated + delta)
        return new_self, delta


def build_leaky(model, params, gain):
    """The registry hook: (dark-zone model, scenario params, gain) -> law."""
    return LeakyIntegratorEFC(
        efc=EFCController.build(
            model, gain=gain, regularization=params["regularization"]
        ),
        accumulated=jnp.zeros(model.n_total),
        leak=0.02,
    )


def main():
    """Register the custom law and benchmark it on the actuator-grid mirror."""
    ALGORITHMS["leaky-efc"] = {
        "estimator": ALGORITHMS["oracle-efc"]["estimator"],
        "controller": build_leaky,
        "gain": None,
    }
    scenario = dig_from_cold(npix=24, actuator_dm=True)
    table = sweep(
        {"dig(actuator DM)": scenario},
        ["oracle-efc", "oracle-strokemin", "leaky-efc"],
    )
    print(f"{'algorithm':<20}{'final contrast':>16}{'dig':>10}{'stroke rms':>12}")
    for (_scene, algorithm), metrics in table.items():
        print(
            f"{algorithm:<20}{metrics.final_contrast:>16.3e}"
            f"{metrics.dig_factor:>10.4f}{metrics.stroke_rms:>12.3f}"
        )
    result = run(scenario, "leaky-efc")
    print(
        f"\ncommand is actuator-space: {result.command.shape[0]} actuators; "
        f"saturated={result.metrics.saturated}"
    )


if __name__ == "__main__":
    main()
