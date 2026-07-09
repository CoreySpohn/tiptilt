"""Tests for the WFSC algorithm testbed (registries, scenarios, metrics)."""

import jax.numpy as jnp
import numpy as np
import pytest

from wavefronts.testbed import (
    ALGORITHMS,
    compute_metrics,
    dig_from_cold,
    dual_science_common_mode,
    hold_against_drift,
    one_dm_vs_two_dm,
    run,
    sweep,
    wfs_private_aliasing,
)


class TestMetrics:
    def test_convergence_rate_and_floors(self):
        history = jnp.asarray([1.0, 0.1, 0.01, 0.001, 0.001, 0.001])
        metrics = compute_metrics(history, jnp.asarray([0.5, -0.2]))
        assert metrics.initial_contrast == 1.0
        assert metrics.final_contrast == pytest.approx(0.001)
        assert metrics.dig_factor == pytest.approx(0.001)
        assert metrics.convergence_rate < 0.0  # log10 falls per iteration
        assert metrics.stroke_rms > 0.0
        assert not metrics.saturated

    def test_stroke_cap_saturation_flag(self):
        history = jnp.asarray([1.0, 0.5])
        metrics = compute_metrics(history, jnp.asarray([2.0, 0.1]), stroke_cap_nm=1.0)
        assert metrics.saturated


class TestDigScenarios:
    @pytest.mark.slow
    def test_oracle_digs_deeper_than_pairwise(self):
        scenario = dig_from_cold()
        oracle = run(scenario, "oracle-efc")
        pairwise = run(scenario, "pairwise-efc")
        assert oracle.metrics.final_contrast < pairwise.metrics.final_contrast
        assert pairwise.metrics.dig_factor < 0.5  # the honest loop still digs

    @pytest.mark.slow
    def test_two_dm_beats_one_dm_on_the_two_sided_zone(self):
        scenarios = one_dm_vs_two_dm()
        one = run(scenarios["one_dm"], "oracle-efc")
        two = run(scenarios["two_dm"], "oracle-efc")
        assert one.metrics.dig_factor > 0.3  # one DM floors
        assert two.metrics.final_contrast < 1e-3 * one.metrics.final_contrast

    @pytest.mark.slow
    def test_stroke_cap_is_enforced(self):
        scenario = dig_from_cold(stroke_cap_nm=0.05)
        result = run(scenario, "oracle-efc")
        assert float(jnp.max(jnp.abs(result.command))) <= 0.05 + 1e-12
        assert result.metrics.saturated


class TestMaintainScenario:
    @pytest.mark.slow
    def test_maintenance_holds_what_open_loop_loses(self):
        scenario = hold_against_drift()
        held = run(scenario, "oracle-efc")
        lost = run(scenario, "open-loop")
        assert held.metrics.final_contrast < 0.1 * lost.metrics.final_contrast


class TestMultichannelScenarios:
    @pytest.mark.slow
    def test_aliasing_scenario_measures_the_injection(self):
        scenario = wfs_private_aliasing()
        aliased = run(scenario, "feedforward")
        clean = run(scenario, "open-loop")
        assert float(aliased.excess["sci"][-1]) > 5.0 * float(
            clean.excess["sci"][-1] + 1e-16
        )

    @pytest.mark.slow
    def test_common_mode_rejection(self):
        scenario = dual_science_common_mode()
        held = run(scenario, "feedforward")
        lost = run(scenario, "open-loop")
        for name in held.excess:
            assert float(held.excess[name][-1]) < 0.1 * float(lost.excess[name][-1])


class TestSweep:
    @pytest.mark.slow
    def test_sweep_produces_a_metrics_table(self):
        table = sweep(
            {"dig": dig_from_cold()},
            ["oracle-efc", "oracle-strokemin", "pairwise-efc"],
        )
        assert set(table) == {
            ("dig", "oracle-efc"),
            ("dig", "oracle-strokemin"),
            ("dig", "pairwise-efc"),
        }
        for metrics in table.values():
            assert np.isfinite(metrics.final_contrast)
            assert np.isfinite(metrics.convergence_rate)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(KeyError, match="algorithm"):
            run(dig_from_cold(), "does-not-exist")
        assert "oracle-efc" in ALGORITHMS


class TestUserExtension:
    @pytest.mark.slow
    def test_actuator_dm_scenario_runs_in_actuator_space(self):
        from wavefronts.dm import DeformableMirror  # noqa: F401 (the device path)

        scenario = dig_from_cold(npix=24, actuator_dm=True)
        result = run(scenario, "oracle-efc")
        # Actuator-space command vector; digs to the fitting-error floor.
        assert result.command.shape[0] > 50  # many actuators, not 8 modes
        assert result.metrics.dig_factor < 0.1

    @pytest.mark.slow
    def test_user_registered_controller_via_a_callable_builder(self):
        """The extension point: register a CUSTOM stateful control law and
        run it against the same scenarios and metrics."""
        import equinox as eqx
        import jax.numpy as jnp

        from wavefronts.control import AbstractController, EFCController

        class LeakyIntegratorEFC(AbstractController):
            """EFC with a leaky-integrator memory (a genuinely stateful law)."""

            efc: EFCController
            accumulated: jnp.ndarray
            leak: float = eqx.field(static=True)

            def command_delta(self, estimate):
                _, raw = self.efc.command_delta(estimate)
                delta = raw - self.leak * self.accumulated
                new_self = eqx.tree_at(
                    lambda c: c.accumulated, self, self.accumulated + delta
                )
                return new_self, delta

        def build_leaky(model, params, gain):
            return LeakyIntegratorEFC(
                efc=EFCController.build(
                    model, gain=gain, regularization=params["regularization"]
                ),
                accumulated=jnp.zeros(model.n_total),
                leak=0.02,
            )

        ALGORITHMS["leaky-efc"] = {
            "estimator": ALGORITHMS["oracle-efc"]["estimator"],
            "controller": build_leaky,
            "gain": None,
        }
        try:
            result = run(dig_from_cold(), "leaky-efc")
            assert result.metrics.dig_factor < 0.1  # the custom law digs
        finally:
            del ALGORITHMS["leaky-efc"]
