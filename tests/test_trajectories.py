"""Tests for the mode-coefficient trajectory builders.

These generate the ``eps_table`` a :class:`TabulatedSpeckleField` replays. The
statistical assertions run a vmapped ensemble of independent trajectories, so
the tolerances are set by the ensemble size (a variance estimated from ``n``
samples scatters by ``sqrt(2/n)``), not by any discretization: the OU recursion
is exact at every step size, and that exactness is what the deterministic tests
below pin.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tiptilt.speckle import (
    TabulatedSpeckleField,
    compose_trajectories,
    creep_trajectory,
    ou_covariance,
    ou_exposure_neff,
    ou_lag_covariance,
    ou_trajectory,
    random_walk_trajectory,
)

N_ENS = 8000

# An autocorrelation estimated from n independent trajectories has standard
# error sqrt((1 + rho^2) / n) <= sqrt(2 / N_ENS) = 0.016, so this is a >3 sigma
# band -- loose enough to be seed-stable, tight enough that the artifact it
# exists to exclude (a cosine sum's kernel wandering by +/- 0.2 at long lag)
# fails it by an order of magnitude.
RHO_ATOL = 0.05


def _ensemble(builder, n_samples=N_ENS, seed=11):
    """Stack ``n_samples`` independent trajectories, shape ``(n, t, m)``."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n_samples)
    return np.asarray(jax.vmap(builder)(keys))


class TestOUCovariance:
    """The realized equal-time covariance, in closed form."""

    def test_diagonal_is_the_target_exactly(self):
        covariance = jnp.asarray([[4.0, 1.5, 0.5], [1.5, 9.0, 2.0], [0.5, 2.0, 1.0]])
        timescales = jnp.asarray([10.0, 100.0, 1000.0])
        realized = ou_covariance(covariance, timescales)
        np.testing.assert_allclose(
            np.diag(np.asarray(realized)), np.diag(np.asarray(covariance)), rtol=1e-14
        )

    def test_shared_timescale_reproduces_the_target(self):
        """One timescale for every mode: no spectral mismatch, no damping."""
        covariance = jnp.asarray([[4.0, 1.5], [1.5, 9.0]])
        realized = ou_covariance(covariance, jnp.asarray([42.0, 42.0]))
        np.testing.assert_allclose(
            np.asarray(realized), np.asarray(covariance), rtol=1e-14
        )

    def test_cross_terms_are_damped_by_the_spectral_overlap(self):
        """Modes that decorrelate on different schedules cannot be perfectly
        correlated: the off-diagonal carries the Lorentzian overlap factor
        ``2 sqrt(tau_k tau_l) / (tau_k + tau_l)``."""
        covariance = jnp.asarray([[1.0, 1.0], [1.0, 1.0]])  # perfectly correlated
        timescales = jnp.asarray([10.0, 1000.0])
        realized = np.asarray(ou_covariance(covariance, timescales))
        overlap = 2.0 * np.sqrt(10.0 * 1000.0) / 1010.0
        assert realized[0, 1] == pytest.approx(overlap, rel=1e-12)
        assert realized[0, 1] < 1.0

    def test_stays_positive_semidefinite(self):
        """The overlap matrix is a Gram matrix, so the Hadamard product with a
        PSD covariance is PSD (Schur product theorem) -- a drawable process."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((6, 6))
        covariance = jnp.asarray(a @ a.T)
        timescales = jnp.asarray([1.0, 10.0, 100.0, 1e3, 1e4, 1e5])
        eigvals = np.linalg.eigvalsh(np.asarray(ou_covariance(covariance, timescales)))
        assert eigvals.min() > -1e-12

    def test_scalar_timescale_broadcasts(self):
        covariance = jnp.asarray([[4.0, 1.5], [1.5, 9.0]])
        np.testing.assert_allclose(
            np.asarray(ou_covariance(covariance, 30.0)),
            np.asarray(covariance),
            rtol=1e-14,
        )


class TestOUIsAlwaysRealizable:
    """Why the equal-time covariance is damped rather than imposed.

    A stationary AR(1) is only a process if its innovation covariance
    ``Sigma - D_a Sigma D_a`` is positive semidefinite, so a covariance and a
    set of per-mode timescales are NOT a free pair. Deriving the damping from
    the timescales satisfies that constraint identically; imposing the
    covariance instead violates it for most inputs.
    """

    @staticmethod
    def _random_case(rng):
        n_modes = int(rng.integers(2, 6))
        a = rng.standard_normal((n_modes, n_modes))
        covariance = a @ a.T
        tau = 10.0 ** rng.uniform(0.0, 4.0, size=n_modes)
        step_s = 10.0 ** rng.uniform(-1.0, 3.0)
        decay = np.exp(-step_s / tau)
        return covariance, tau, 1.0 - np.outer(decay, decay)

    def test_the_realized_covariance_always_gives_a_valid_innovation(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            covariance, tau, one_minus = self._random_case(rng)
            sigma = np.asarray(ou_covariance(jnp.asarray(covariance), jnp.asarray(tau)))
            innovation = sigma * one_minus
            floor = -1e-12 * np.abs(covariance).max()
            assert np.linalg.eigvalsh(innovation).min() >= floor

    def test_imposing_the_target_covariance_usually_would_not(self):
        rng = np.random.default_rng(0)
        invalid = 0
        for _ in range(300):
            covariance, _, one_minus = self._random_case(rng)
            innovation = covariance * one_minus  # the undamped alternative
            if np.linalg.eigvalsh(innovation).min() < -1e-12 * np.abs(covariance).max():
                invalid += 1
        assert invalid > 150  # a majority, not an edge case


class TestOULagCovariance:
    """The two-time law ``Sigma_kl exp(-lag/tau_l)``, and its asymmetry."""

    COV = jnp.asarray([[4.0, 5.4], [5.4, 9.0]])
    TAU = jnp.asarray([50.0, 500.0])

    def test_reduces_to_the_equal_time_covariance_at_zero_lag(self):
        np.testing.assert_allclose(
            np.asarray(ou_lag_covariance(self.COV, self.TAU, 0.0)),
            np.asarray(ou_covariance(self.COV, self.TAU)),
            rtol=1e-14,
        )

    def test_is_asymmetric_under_per_mode_timescales(self):
        """How correlated mode k now is with mode l later is not the same as
        the reverse: the lagged mode's own timescale does the decaying. A
        symmetric surrogate would describe a different process."""
        lag = ou_lag_covariance(self.COV, self.TAU, 200.0)
        assert not np.allclose(np.asarray(lag), np.asarray(lag).T)
        sigma = np.asarray(ou_covariance(self.COV, self.TAU))
        tau = np.asarray(self.TAU)
        np.testing.assert_allclose(
            np.asarray(lag), sigma * np.exp(-200.0 / tau)[None, :], rtol=1e-14
        )

    def test_is_symmetric_when_the_timescales_agree(self):
        lag = np.asarray(ou_lag_covariance(self.COV, 100.0, 250.0))
        np.testing.assert_allclose(lag, lag.T, rtol=1e-14)

    def test_negative_lag_is_the_transpose(self):
        forward = np.asarray(ou_lag_covariance(self.COV, self.TAU, 200.0))
        backward = np.asarray(ou_lag_covariance(self.COV, self.TAU, -200.0))
        np.testing.assert_allclose(backward, forward.T, rtol=1e-14)

    def test_matches_the_generated_ensemble(self):
        """The closed form and the generator agree, including the asymmetry --
        so an analytic two-time prediction and a realization can be compared
        against the same matrix."""
        lag = 200.0
        times = jnp.asarray([0.0, lag])
        traj = _ensemble(
            lambda key: ou_trajectory(self.COV, self.TAU, key=key, times_s=times),
            n_samples=40000,
        )
        measured = (traj[:, 0, :, None] * traj[:, 1, None, :]).mean(axis=0)
        expected = np.asarray(ou_lag_covariance(self.COV, self.TAU, lag))
        # Off-diagonal standard error is sqrt((s_kk s_ll + s_kl^2) / n) ~ 0.03.
        np.testing.assert_allclose(measured, expected, atol=0.15)
        assert abs(measured[0, 1] - measured[1, 0]) > 0.5  # the asymmetry is real


class TestOUTrajectory:
    # Strongly correlated on purpose: the off-diagonal of a sample covariance
    # has relative standard error sqrt((s_11 s_22 + s_12^2) / n) / |s_12|, so a
    # weakly correlated pair is intrinsically noisy and would only be testable
    # behind a tolerance loose enough to hide a real error.
    COV = jnp.asarray([[4.0, 5.4], [5.4, 9.0]])
    TAU = jnp.asarray([50.0, 500.0])

    def _build(self, times_s, covariance=None, timescales=None):
        covariance = self.COV if covariance is None else covariance
        timescales = self.TAU if timescales is None else timescales
        return lambda key: ou_trajectory(
            covariance, timescales, key=key, times_s=times_s
        )

    def test_shape_and_dtype(self):
        times = jnp.arange(0.0, 200.0, 10.0)
        table = ou_trajectory(
            self.COV, self.TAU, key=jax.random.PRNGKey(0), times_s=times
        )
        assert table.shape == (times.size, 2)
        assert jnp.isrealobj(table)

    def test_is_deterministic_given_a_key(self):
        times = jnp.arange(0.0, 200.0, 10.0)
        kw = dict(key=jax.random.PRNGKey(4), times_s=times)
        first = ou_trajectory(self.COV, self.TAU, **kw)
        second = ou_trajectory(self.COV, self.TAU, **kw)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_equal_time_covariance_matches_the_closed_form(self):
        """Including the off-diagonal damping: the realized covariance is
        ``ou_covariance``, not the driving covariance."""
        times = jnp.arange(0.0, 1000.0, 25.0)
        traj = _ensemble(self._build(times))
        expected = np.asarray(ou_covariance(self.COV, self.TAU))
        assert expected[0, 1] < 0.7 * float(self.COV[0, 1])  # damping is in play
        for index in (0, traj.shape[1] // 2, -1):
            measured = np.cov(traj[:, index, :].T)
            np.testing.assert_allclose(measured, expected, rtol=0.08)

    def test_is_stationary_from_the_first_sample(self):
        """The initial draw comes from the stationary distribution, so there is
        no burn-in transient: the variance at t=0 already equals the variance at
        the end of the record."""
        times = jnp.arange(0.0, 4000.0, 50.0)
        traj = _ensemble(self._build(times))
        first = traj[:, 0, :].var(axis=0)
        last = traj[:, -1, :].var(axis=0)
        np.testing.assert_allclose(first, last, rtol=0.08)

    def test_autocorrelation_is_lorentzian_per_mode(self):
        """Each mode decorrelates on its OWN timescale, exactly ``exp(-lag/tau)``
        -- the property the finite cosine sum cannot hold past a few tau."""
        dt = 25.0
        times = jnp.arange(0.0, 2000.0, dt)
        traj = _ensemble(self._build(times))
        sigma = np.asarray(ou_covariance(self.COV, self.TAU))
        tau = np.asarray(self.TAU)
        for lag_steps in (1, 4, 20):
            lag = lag_steps * dt
            rho = (traj[:, 0, :] * traj[:, lag_steps, :]).mean(axis=0) / np.diag(sigma)
            np.testing.assert_allclose(rho, np.exp(-lag / tau), atol=RHO_ATOL)

    def test_long_lag_decay_has_no_revival(self):
        """The cosine-sum synthesis wanders in +/- 0.2 past a few tau; the OU
        kernel is monotone and dead by 10 tau, which is what long-baseline
        scheduling work needs."""
        dt = 100.0
        times = jnp.arange(0.0, 12000.0, dt)
        traj = _ensemble(self._build(times, timescales=jnp.asarray([500.0, 500.0])))
        sigma = np.asarray(ou_covariance(self.COV, jnp.asarray([500.0, 500.0])))
        rho = np.array(
            [
                (traj[:, 0, :] * traj[:, k, :]).mean(axis=0) / np.diag(sigma)
                for k in range(traj.shape[1])
            ]
        )
        lags = np.asarray(times)
        assert np.abs(rho[lags >= 10 * 500.0]).max() < 0.06

    def test_exact_at_any_step_size(self):
        """No discretization error: ONE step of 50 s reproduces the same
        autocorrelation as five steps of 10 s, and both match the closed form.

        A Euler-Maruyama discretization would miss here; the transition is the
        exact one, so the step size is a sampling choice and not an accuracy
        knob. Each grid is compared to the analytic value rather than to the
        other, which is both the stronger claim and the stabler test (two
        independent estimates scatter by sqrt(2) times one).
        """
        lag = 50.0
        expected = np.exp(-lag / np.asarray(self.TAU))
        sigma = np.diag(np.asarray(ou_covariance(self.COV, self.TAU)))
        for times in (jnp.asarray([0.0, lag]), jnp.arange(0.0, lag + 1.0, 10.0)):
            traj = _ensemble(self._build(times))
            rho = (traj[:, 0, :] * traj[:, -1, :]).mean(axis=0) / sigma
            np.testing.assert_allclose(rho, expected, atol=RHO_ATOL)

    def test_non_uniform_grid_honors_every_gap(self):
        times = jnp.asarray([0.0, 10.0, 500.0, 520.0, 3000.0])
        traj = _ensemble(self._build(times))
        sigma = np.diag(np.asarray(ou_covariance(self.COV, self.TAU)))
        tau = np.asarray(self.TAU)
        for index in (1, 2, 4):
            lag = float(times[index] - times[0])
            rho = (traj[:, 0, :] * traj[:, index, :]).mean(axis=0) / sigma
            np.testing.assert_allclose(rho, np.exp(-lag / tau), atol=RHO_ATOL)
        # Stationary throughout, despite the wildly uneven spacing.
        np.testing.assert_allclose(traj[:, -1, :].var(axis=0), sigma, rtol=0.08)

    def test_rank_deficient_covariance_is_handled(self):
        """A singular covariance must not crash: the square root is an
        eigendecomposition, not a Cholesky factorization.

        Under a SHARED timescale there is no overlap damping, so the realized
        covariance is the singular target itself and the degenerate direction
        is exercised end to end. (With per-mode timescales the Hadamard damping
        would pull a singular input back to full rank.)
        """
        v = jnp.asarray([1.0, 2.0])
        singular = jnp.outer(v, v)
        traj = _ensemble(
            self._build(
                jnp.arange(0.0, 500.0, 25.0), covariance=singular, timescales=100.0
            ),
            n_samples=4000,
        )
        measured = np.cov(traj[:, -1, :].T)
        np.testing.assert_allclose(measured, np.asarray(singular), rtol=0.09)
        assert abs(np.linalg.det(measured)) < 1e-8

    def test_zero_covariance_is_exactly_zero(self):
        table = ou_trajectory(
            jnp.zeros((2, 2)),
            self.TAU,
            key=jax.random.PRNGKey(0),
            times_s=jnp.arange(0.0, 100.0, 10.0),
        )
        np.testing.assert_array_equal(np.asarray(table), 0.0)

    def test_scalar_timescale_broadcasts(self):
        table = ou_trajectory(
            self.COV, 100.0, key=jax.random.PRNGKey(0), times_s=jnp.arange(5.0)
        )
        assert table.shape == (5, 2)

    def test_rejects_asymmetric_covariance(self):
        with pytest.raises(ValueError, match="symmetric"):
            ou_trajectory(
                jnp.asarray([[4.0, 1.0], [0.5, 9.0]]),
                self.TAU,
                key=jax.random.PRNGKey(0),
                times_s=jnp.arange(5.0),
            )

    def test_rejects_indefinite_covariance(self):
        with pytest.raises(ValueError, match="semidefinite"):
            ou_trajectory(
                jnp.asarray([[1.0, 2.0], [2.0, 1.0]]),
                self.TAU,
                key=jax.random.PRNGKey(0),
                times_s=jnp.arange(5.0),
            )

    def test_rejects_mismatched_timescales(self):
        with pytest.raises(ValueError, match="timescales_s"):
            ou_trajectory(
                self.COV,
                jnp.asarray([1.0, 2.0, 3.0]),
                key=jax.random.PRNGKey(0),
                times_s=jnp.arange(5.0),
            )

    def test_rejects_non_positive_timescales(self):
        with pytest.raises(ValueError, match="positive"):
            ou_trajectory(
                self.COV,
                jnp.asarray([50.0, 0.0]),
                key=jax.random.PRNGKey(0),
                times_s=jnp.arange(5.0),
            )

    def test_rejects_unsorted_times(self):
        with pytest.raises(ValueError, match="ascending"):
            ou_trajectory(
                self.COV,
                self.TAU,
                key=jax.random.PRNGKey(0),
                times_s=jnp.asarray([0.0, 10.0, 5.0]),
            )


class TestRandomWalkTrajectory:
    # Strongly correlated, for the reason given on TestOUTrajectory.COV.
    DIFFUSION = jnp.asarray([[4.0, 5.0], [5.0, 9.0]])

    def _build(self, times_s):
        return lambda key: random_walk_trajectory(
            self.DIFFUSION, key=key, times_s=times_s
        )

    def test_starts_at_the_origin(self):
        table = random_walk_trajectory(
            self.DIFFUSION,
            key=jax.random.PRNGKey(0),
            times_s=jnp.arange(0.0, 50.0, 5.0),
        )
        np.testing.assert_array_equal(np.asarray(table[0]), 0.0)

    def test_variance_grows_linearly_in_elapsed_time(self):
        times = jnp.asarray([0.0, 1.0, 4.0, 16.0])
        traj = _ensemble(self._build(times))
        for index in (1, 2, 3):
            measured = np.cov(traj[:, index, :].T)
            expected = np.asarray(self.DIFFUSION) * float(times[index])
            np.testing.assert_allclose(measured, expected, rtol=0.08)

    def test_increments_are_independent_of_the_past(self):
        """Successive increments are uncorrelated, so the walk is Markov and
        its variance accumulates rather than saturating."""
        times = jnp.asarray([0.0, 10.0, 20.0])
        traj = _ensemble(self._build(times))
        first = traj[:, 1, :] - traj[:, 0, :]
        second = traj[:, 2, :] - traj[:, 1, :]
        # Normalized to a correlation, whose standard error is 1/sqrt(n).
        scale = np.sqrt(np.outer(first.var(axis=0), second.var(axis=0)))
        cross = (first[:, :, None] * second[:, None, :]).mean(axis=0) / scale
        np.testing.assert_allclose(cross, 0.0, atol=5.0 / np.sqrt(N_ENS))

    def test_start_offset_shifts_the_whole_record(self):
        times = jnp.arange(0.0, 50.0, 5.0)
        key = jax.random.PRNGKey(2)
        base = random_walk_trajectory(self.DIFFUSION, key=key, times_s=times)
        start = jnp.asarray([3.0, -2.0])
        shifted = random_walk_trajectory(
            self.DIFFUSION, key=key, times_s=times, start_nm=start
        )
        np.testing.assert_allclose(
            np.asarray(shifted - base), np.broadcast_to(np.asarray(start), base.shape)
        )

    def test_non_uniform_spacing_scales_each_increment(self):
        times = jnp.asarray([0.0, 1.0, 101.0])
        traj = _ensemble(self._build(times))
        step = traj[:, 2, :] - traj[:, 1, :]
        np.testing.assert_allclose(
            np.cov(step.T), np.asarray(self.DIFFUSION) * 100.0, rtol=0.08
        )

    def test_rejects_indefinite_diffusion(self):
        with pytest.raises(ValueError, match="semidefinite"):
            random_walk_trajectory(
                jnp.asarray([[1.0, 2.0], [2.0, 1.0]]),
                key=jax.random.PRNGKey(0),
                times_s=jnp.arange(5.0),
            )


class TestCreepTrajectory:
    RATES = jnp.asarray([0.02, -0.005])

    def test_deterministic_ramp_is_exact(self):
        times = jnp.asarray([0.0, 10.0, 250.0])
        table = creep_trajectory(self.RATES, times_s=times)
        expected = np.outer(np.asarray(times), np.asarray(self.RATES))
        np.testing.assert_allclose(np.asarray(table), expected, rtol=1e-14)

    def test_measures_elapsed_time_from_the_first_sample(self):
        times = jnp.asarray([100.0, 110.0])
        table = creep_trajectory(self.RATES, times_s=times)
        np.testing.assert_array_equal(np.asarray(table[0]), 0.0)
        np.testing.assert_allclose(
            np.asarray(table[1]), 10.0 * np.asarray(self.RATES), rtol=1e-14
        )

    def test_drawn_rates_keep_the_sign_of_the_mean(self):
        """One-sided by construction: every realization creeps the same way, so
        the composed marginal is skewed rather than symmetric."""
        times = jnp.asarray([0.0, 100.0])
        traj = _ensemble(
            lambda key: creep_trajectory(
                self.RATES, times_s=times, key=key, rate_shape=2.0
            ),
            n_samples=2000,
        )
        end = traj[:, -1, :]
        assert np.all(end[:, 0] > 0.0)
        assert np.all(end[:, 1] < 0.0)

    def test_drawn_rates_have_the_requested_mean_and_skewness(self):
        """A gamma rate of shape ``k`` has mean ``rate`` and skewness
        ``2/sqrt(k)``, so ``rate_shape`` is a direct modal-skewness knob."""
        times = jnp.asarray([0.0, 1.0])
        shape = 4.0
        traj = _ensemble(
            lambda key: creep_trajectory(
                self.RATES, times_s=times, key=key, rate_shape=shape
            ),
            n_samples=20000,
        )
        end = traj[:, -1, :]
        np.testing.assert_allclose(end.mean(axis=0), np.asarray(self.RATES), rtol=0.05)
        centred = end - end.mean(axis=0)
        skew = (centred**3).mean(axis=0) / centred.std(axis=0) ** 3
        expected = 2.0 / np.sqrt(shape)
        np.testing.assert_allclose(skew, [expected, -expected], atol=0.15)

    def test_large_shape_approaches_the_deterministic_ramp(self):
        times = jnp.asarray([0.0, 100.0])
        table = creep_trajectory(
            self.RATES, times_s=times, key=jax.random.PRNGKey(0), rate_shape=1e8
        )
        expected = np.outer(np.asarray(times), np.asarray(self.RATES))
        np.testing.assert_allclose(np.asarray(table), expected, rtol=1e-3)

    def test_rate_shape_needs_a_key(self):
        with pytest.raises(ValueError, match="key"):
            creep_trajectory(self.RATES, times_s=jnp.arange(3.0), rate_shape=2.0)

    def test_rejects_non_positive_shape(self):
        with pytest.raises(ValueError, match="rate_shape"):
            creep_trajectory(
                self.RATES,
                times_s=jnp.arange(3.0),
                key=jax.random.PRNGKey(0),
                rate_shape=0.0,
            )


class TestComposeTrajectories:
    def test_sums_the_tables(self):
        times = jnp.asarray([0.0, 10.0, 20.0])
        stationary = ou_trajectory(
            jnp.eye(2), 100.0, key=jax.random.PRNGKey(0), times_s=times
        )
        creep = creep_trajectory(jnp.asarray([0.01, 0.02]), times_s=times)
        total = compose_trajectories(stationary, creep)
        np.testing.assert_allclose(
            np.asarray(total), np.asarray(stationary) + np.asarray(creep), rtol=1e-14
        )

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="shape"):
            compose_trajectories(jnp.zeros((3, 2)), jnp.zeros((3, 4)))

    def test_rejects_an_empty_call(self):
        with pytest.raises(ValueError, match="at least one"):
            compose_trajectories()


class TestFeedsATabulatedField:
    """The builders exist to fill ``TabulatedSpeckleField``: check the seam."""

    def test_two_regime_drift_drives_a_field(self):
        rng = np.random.default_rng(3)
        m, ny, nx = 2, 3, 3
        g = jnp.asarray(rng.standard_normal((m, ny, nx)) * (1 + 1j))
        e_nom = jnp.asarray(rng.standard_normal((ny, nx)) * (1 + 1j))
        times = jnp.arange(0.0, 3600.0, 60.0)
        eps_table = compose_trajectories(
            ou_trajectory(
                jnp.asarray([[0.04, 0.0], [0.0, 0.01]]),
                jnp.asarray([300.0, 3000.0]),
                key=jax.random.PRNGKey(0),
                times_s=times,
            ),
            creep_trajectory(jnp.asarray([1e-5, 0.0]), times_s=times),
        )
        field = TabulatedSpeckleField(e_nom, g, times, eps_table, 1.0)
        early = field.realize(wavelength_nm=500.0, time_s=0.0)
        late = field.realize(wavelength_nm=500.0, time_s=3000.0)
        assert early.shape == (ny, nx)
        assert jnp.all(jnp.isfinite(late))
        assert not jnp.allclose(early, late)


class TestOUExposureNeff:
    """Closed-form exposure averaging for the OU process.

    Exact at EVERY exposure length, which is what separates it from the
    spectral synthesis' version: the kernel being integrated is exact at
    every lag rather than only over the window a finite line sum spans.
    """

    def test_a_frozen_field_averages_over_one_realization(self):
        neff = np.asarray(ou_exposure_neff(1000.0, 1e-6))
        np.testing.assert_allclose(neff, 1.0, rtol=1e-9)

    def test_zero_exposure_is_exactly_one(self):
        np.testing.assert_allclose(
            np.asarray(ou_exposure_neff(jnp.asarray([10.0, 100.0]), 0.0)),
            1.0,
            rtol=1e-14,
        )

    def test_matches_the_closed_form(self):
        tau = 500.0
        for fraction in (0.01, 0.5, 1.0, 10.0, 100.0):
            u = fraction
            expected = u**2 / (2.0 * (u - 1.0 + np.exp(-u)))
            got = float(np.asarray(ou_exposure_neff(tau, fraction * tau))[0])
            assert got == pytest.approx(expected, rel=1e-10)

    def test_long_exposure_approaches_half_the_timescale_ratio(self):
        tau = 200.0
        exposure = 1e4 * tau
        got = float(np.asarray(ou_exposure_neff(tau, exposure))[0])
        assert got == pytest.approx(exposure / (2.0 * tau), rel=1e-3)

    def test_matches_a_generated_ensemble(self):
        """The predicted suppression is what averaging real trajectories
        delivers."""
        tau = 300.0
        exposure = 6.0 * tau
        times = jnp.asarray(np.linspace(0.0, exposure, 600))
        traj = _ensemble(
            lambda key: ou_trajectory(
                jnp.eye(2), jnp.asarray([tau, tau]), key=key, times_s=times
            ),
            n_samples=6000,
        )
        measured = traj.mean(axis=1).var(axis=0) / traj[:, 0, :].var(axis=0)
        predicted = 1.0 / np.asarray(ou_exposure_neff(tau, exposure))[0]
        np.testing.assert_allclose(measured, predicted, rtol=0.10)

    def test_per_mode_and_broadcast_shapes(self):
        tau = jnp.asarray([10.0, 1000.0])
        neff = ou_exposure_neff(tau, jnp.asarray([100.0, 500.0, 2000.0]))
        assert neff.shape == (2, 3)
        # The fast mode averages over more realizations at every exposure.
        assert np.all(np.asarray(neff)[0] > np.asarray(neff)[1])

    def test_rejects_non_positive_timescales(self):
        with pytest.raises(ValueError, match="positive"):
            ou_exposure_neff(jnp.asarray([100.0, 0.0]), 10.0)

    def test_traces_over_the_exposure(self):
        """The exposure is what varies inside a simulation loop, so it must
        jit and differentiate -- including at zero, where the closed form is
        a 0/0 that only the guarded branches keep finite."""
        tau = jnp.asarray([50.0, 500.0])

        def total(exposure_s):
            return jnp.sum(ou_exposure_neff(tau, exposure_s))

        assert jnp.isfinite(jax.jit(total)(100.0))
        assert jnp.isfinite(jax.grad(total)(100.0))
        assert jnp.isfinite(jax.grad(total)(0.0))

    def test_small_exposures_keep_their_precision(self):
        """Below u ~ 1e-4 the direct form cancels to noise; the series branch
        holds the exact N_eff - 1 = u/3 behaviour down to u = 1e-12."""
        tau = 1.0
        for u in (1e-12, 1e-9, 1e-6, 1e-4, 1e-3):
            got = float(np.asarray(ou_exposure_neff(tau, u))[0])
            assert got - 1.0 == pytest.approx(u / 3.0, rel=1e-3)

    def test_the_two_branches_agree_at_the_crossover(self):
        tau = 1.0
        below = float(np.asarray(ou_exposure_neff(tau, 1e-4 * (1 - 1e-9)))[0])
        above = float(np.asarray(ou_exposure_neff(tau, 1e-4 * (1 + 1e-9)))[0])
        assert below == pytest.approx(above, rel=1e-11)
