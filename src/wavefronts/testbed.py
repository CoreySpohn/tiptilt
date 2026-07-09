"""The WFSC algorithm testbed: scenarios x algorithms, scored on one axis set.

A benchmark harness is the cartesian product {algorithm} x {scenario} run on
fixed seams and scored on a fixed metric set, so digging laws, maintenance
filters, and estimators are compared apples-to-apples. Three regimes share
the drivers built on the estimator/controller seams:

- ``dig``: dark hole from cold (the seam loop; oracle-EFC is the reference).
- ``maintain``: hold a dug hole against injected drift
  (``maintain_dark_hole``).
- ``multichannel``: forked-system co-runs with cross-channel feed-forward
  (``run_multichannel``), including the NCPA-aliasing stress case.

Scenario builders return self-contained ``Scenario`` objects (path/system,
fields, masks, drift, probes, stroke cap); ``ALGORITHMS`` maps a name to the
estimator/controller construction; ``run`` dispatches by regime and returns
histories plus ``Metrics``; ``sweep`` tabulates the product. Commands are
CLIPPED to ``stroke_cap_nm`` when set, and saturation is flagged rather than
silently exceeded.
"""

from dataclasses import dataclass
from dataclasses import field as dataclass_field

import jax.numpy as jnp
import numpy as np
from physicaloptix import (
    BeamSplitter,
    Branch,
    Field,
    Fraunhofer,
    Fresnel,
    Grid,
    ModeBasis,
    OpticalPath,
    OpticalSystem,
    PhaseScreen,
    PlaneKind,
    Stage,
)

from wavefronts.control import (
    DarkZoneModel,
    EFCController,
    PredictiveController,
    StrokeMinController,
)
from wavefronts.maintenance import maintain_dark_hole
from wavefronts.multichannel import run_multichannel
from wavefronts.sensing import (
    KalmanEstimator,
    OracleEstimator,
    PairwiseEstimator,
    probe_set,
)

WL = 500.0
DIAM_M = 0.02


@dataclass
class Scenario:
    """One benchmark configuration (regime + optics + drive + budget)."""

    regime: str  # "dig" | "maintain" | "multichannel"
    params: dict = dataclass_field(default_factory=dict)
    stroke_cap_nm: float | None = None


@dataclass
class Metrics:
    """The comparison axes every run reports."""

    initial_contrast: float
    final_contrast: float
    dig_factor: float
    convergence_rate: float
    stroke_rms: float
    stroke_peak: float
    saturated: bool


@dataclass
class RunResult:
    """A run's histories, final command, and metrics."""

    history: jnp.ndarray
    command: jnp.ndarray
    metrics: Metrics
    excess: dict | None = None


def compute_metrics(history, command, *, stroke_cap_nm=None):
    """Score a contrast history and a final command.

    ``convergence_rate`` is pinned as the log10-contrast slope over the
    pre-floor dig phase (iterations until within 2x of the final floor).

    Args:
        history: Dark-zone contrast per iteration.
        command: The final stacked command (nm coefficients).
        stroke_cap_nm: Optional actuator range; at or beyond it the run is
            flagged saturated.

    Returns:
        A ``Metrics``.
    """
    history = jnp.asarray(history)
    initial = float(history[0])
    final = float(history[-1])
    floor = max(final, 1e-300)
    above = np.asarray(history) > 2.0 * floor
    k = int(above.sum()) if bool(above.any()) else 1
    k = max(min(k, history.shape[0] - 1), 1)
    rate = (
        np.log10(max(float(history[k]), 1e-300)) - np.log10(max(initial, 1e-300))
    ) / k
    peak = float(jnp.max(jnp.abs(command))) if command.size else 0.0
    saturated = stroke_cap_nm is not None and peak >= 0.999 * stroke_cap_nm
    return Metrics(
        initial_contrast=initial,
        final_contrast=final,
        dig_factor=final / max(initial, 1e-300),
        convergence_rate=float(rate),
        stroke_rms=float(jnp.sqrt(jnp.mean(command**2))) if command.size else 0.0,
        stroke_peak=peak,
        saturated=bool(saturated),
    )


# ---------------------------------------------------------------------------
# Algorithms: name -> (estimator builder, controller builder) on the seams.


def _oracle(scenario, model):
    del scenario, model
    return lambda p: OracleEstimator(input_field=p["input_field"])


def _pairwise(scenario, model):
    del model

    def build(p):
        return PairwiseEstimator(
            input_field=p["input_field"],
            model_field=p.get("model_field") or p["input_field"],
            probes=tuple(p["probes"]),
            probe_dm=p["probe_dm"],
            detector=p.get("detector"),
            regularization=p["regularization"],
        )

    return build


def _kalman(scenario, model):
    del scenario

    def build(p):
        return KalmanEstimator.build(
            model,
            input_field=p["input_field"],
            model_field=p.get("model_field") or p["input_field"],
            probes=tuple(p["probes"]),
            probe_dm=p["probe_dm"],
            detector=p.get("detector"),
        )

    return build


ALGORITHMS = {
    "oracle-efc": {"estimator": _oracle, "controller": "efc", "gain": None},
    "pairwise-efc": {"estimator": _pairwise, "controller": "efc", "gain": None},
    "kalman-efc": {"estimator": _kalman, "controller": "efc", "gain": None},
    "oracle-strokemin": {
        "estimator": _oracle,
        "controller": "strokemin",
        "gain": None,
    },
    "oracle-predictive": {
        "estimator": _oracle,
        "controller": "predictive",
        "gain": None,
    },
    "feedforward": {"estimator": None, "controller": None, "gain": None},
    "open-loop": {"estimator": _oracle, "controller": "efc", "gain": 0.0},
}


def _controller_for(kind, model, p, gain):
    if kind == "efc":
        return EFCController.build(model, gain=gain, regularization=p["regularization"])
    if kind == "strokemin":
        return StrokeMinController.build(model, target_contrast=p["target_contrast"])
    if kind == "predictive":
        return PredictiveController.build(
            model, gain=gain, regularization=p["regularization"], alpha=0.5
        )
    raise KeyError(f"unknown controller kind {kind!r}")


def _clip(command, cap):
    return command if cap is None else jnp.clip(command, -cap, cap)


def _run_dig(scenario, spec):
    p = scenario.params
    model = DarkZoneModel.build(
        p["path"],
        p["dm_indices"],
        p["mask"],
        jacobian_field=p.get("model_field") or p["input_field"],
    )
    gain = spec["gain"] if spec["gain"] is not None else p["gain"]
    controller = _controller_for(spec["controller"], model, p, gain)
    sensor = spec["estimator"](scenario, model)(p)
    cap = scenario.stroke_cap_nm
    command = jnp.zeros(model.n_total)
    history = []
    for _ in range(p["n_steps"]):
        history.append(model.contrast(model.focal_of(command, p["input_field"])))
        sensor, e_hat = sensor.estimate(model, command)
        controller, delta = controller.command_delta(e_hat)
        command = _clip(command + delta, cap)
    history = jnp.stack(history)
    return RunResult(
        history=history,
        command=command,
        metrics=compute_metrics(history, command, stroke_cap_nm=cap),
    )


def _run_maintain(scenario, spec):
    p = scenario.params
    gain = spec["gain"] if spec["gain"] is not None else p["gain"]
    command, history = maintain_dark_hole(
        p["path"],
        p["input_field"],
        p["dm_indices"],
        p["mask"],
        drift=p["drift"],
        drift_stage=p["drift_stage"],
        n_steps=p["n_steps"],
        dt_s=p["dt_s"],
        gain=gain,
        regularization=p["regularization"],
        command0=p.get("command0"),
    )
    return RunResult(
        history=history,
        command=command,
        metrics=compute_metrics(history, command, stroke_cap_nm=scenario.stroke_cap_nm),
    )


def _run_multichannel(scenario, spec):
    p = scenario.params
    gain = spec["gain"] if spec["gain"] is not None else p["gain"]
    result = run_multichannel(
        p["system"],
        p["input_field"],
        shared_dm_stage=p["shared_dm_stage"],
        drift_stage=p["drift_stage"],
        sense=p["sense"],
        science=p["science"],
        masks=p["masks"],
        wavelength_nm=WL,
        drift_table=p["drift_table"],
        n_steps=p["n_steps"],
        gain=gain,
        regularization=p["regularization"],
        local_drift=p.get("local_drift"),
    )
    science = p["science"]
    history = result["history"][science]
    return RunResult(
        history=history,
        command=result["command"],
        metrics=compute_metrics(
            history, result["command"], stroke_cap_nm=scenario.stroke_cap_nm
        ),
        excess=result["excess"],
    )


def run(scenario, algorithm):
    """Run one algorithm on one scenario.

    Args:
        scenario: A ``Scenario`` from a builder below.
        algorithm: A name in ``ALGORITHMS``.

    Returns:
        A ``RunResult``.
    """
    if algorithm not in ALGORITHMS:
        raise KeyError(
            f"unknown algorithm {algorithm!r}; registered: {sorted(ALGORITHMS)}"
        )
    spec = ALGORITHMS[algorithm]
    dispatch = {
        "dig": _run_dig,
        "maintain": _run_maintain,
        "multichannel": _run_multichannel,
    }
    if scenario.regime not in dispatch:
        raise ValueError(f"unknown regime {scenario.regime!r}")
    return dispatch[scenario.regime](scenario, spec)


def sweep(scenarios, algorithms):
    """The cartesian product: metrics per (scenario, algorithm) pair.

    Args:
        scenarios: Dict of name -> ``Scenario``.
        algorithms: Iterable of algorithm names.

    Returns:
        Dict mapping ``(scenario_name, algorithm)`` to ``Metrics``.
    """
    return {
        (scenario_name, algorithm): run(scenario, algorithm).metrics
        for scenario_name, scenario in scenarios.items()
        for algorithm in algorithms
    }


# ---------------------------------------------------------------------------
# Scenario builders (small, fast, deterministic).


def _fourier_basis(npix, freqs, amp_nm=5.0):
    x = np.asarray(Grid.pupil(npix).coords)
    xg, yg = np.meshgrid(x, x)
    modes = []
    for kx, ky in freqs:
        arg = 2 * np.pi * (kx * xg + ky * yg)
        modes.append(amp_nm * np.cos(arg))
        modes.append(amp_nm * np.sin(arg))
    stack = jnp.asarray(np.stack(modes))
    return ModeBasis(B=stack, coeffs=jnp.zeros(stack.shape[0]))


def _aperture(npix):
    grid = Grid.pupil(npix)
    x = np.asarray(grid.coords)
    xg, yg = np.meshgrid(x, x)
    return grid, xg, yg, (xg**2 + yg**2 <= 0.25).astype(float)


def dig_from_cold(npix=16, n_steps=15, stroke_cap_nm=None):
    """A single-DM dig from a static phase aberration (the digging baseline).

    Args:
        npix: Pupil sampling.
        n_steps: Control iterations.
        stroke_cap_nm: Optional actuator range.

    Returns:
        A ``Scenario`` (regime ``dig``).
    """
    grid, xg, yg, aperture = _aperture(npix)
    focal = Grid.focal(32, 0.5)
    dm = _fourier_basis(npix, [(3, 1), (3, 0), (2, 1), (4, 1)])
    path = OpticalPath(
        stages=(
            Stage("dm", PhaseScreen(dm, grid, wavelength_nm=WL)),
            Stage("science", Fraunhofer(grid_in=grid, grid_out=focal)),
        )
    )
    opd = 3.0 * np.cos(2 * np.pi * (3 * xg + yg))
    input_field = Field(
        data=jnp.asarray(aperture * np.exp(1j * 2 * np.pi * opd / WL)),
        grid=grid,
        plane=PlaneKind.PUPIL,
    )
    model_field = Field(
        data=jnp.asarray(aperture).astype(complex), grid=grid, plane=PlaneKind.PUPIL
    )
    fx = np.asarray(focal.coords)
    fxg, fyg = np.meshgrid(fx, fx)
    mask = jnp.asarray((np.abs(fxg - 3.0) < 0.6) & (np.abs(fyg - 1.0) < 0.8))
    initial = 1e-2  # a modest target the stroke-min law can aim for
    return Scenario(
        regime="dig",
        stroke_cap_nm=stroke_cap_nm,
        params=dict(
            path=path,
            input_field=input_field,
            model_field=model_field,
            dm_indices=0,
            mask=mask,
            n_steps=n_steps,
            gain=0.6,
            regularization=1e-8,
            probes=probe_set(dm, amplitude_nm=2.0, n_probes=3),
            probe_dm=0,
            target_contrast=initial,
        ),
    )


def one_dm_vs_two_dm(npix=16, n_steps=25):
    """The two-sided amplitude-speckle pair: one DM floors, two DMs dig.

    Args:
        npix: Pupil sampling.
        n_steps: Control iterations.

    Returns:
        Dict with ``one_dm`` and ``two_dm`` ``Scenario`` objects.
    """
    grid, xg, _yg, aperture = _aperture(npix)
    focal = Grid.focal(32, 0.5)
    ripple = 1.0 + 0.15 * np.cos(2 * np.pi * 3 * xg)
    input_field = Field(
        data=jnp.asarray(aperture * ripple).astype(complex),
        grid=grid,
        plane=PlaneKind.PUPIL,
    )
    z = 0.0556 * DIAM_M**2 / (WL * 1e-9)

    def fresnel(dist, pin, pout):
        return Fresnel(
            grid=grid,
            distance_m=dist,
            beam_diameter_m=DIAM_M,
            wavelength_nm=WL,
            plane_in=pin,
            plane_out=pout,
            on_undersampled="record",
        )

    dm = _fourier_basis(
        npix,
        [(3, 0), (3, 1), (3, -1), (2, 0), (4, 0), (2, 1), (4, 1), (3, 2), (3, -2)],
    )
    path = OpticalPath(
        stages=(
            Stage("dm1", PhaseScreen(dm, grid, wavelength_nm=WL)),
            Stage("relay", fresnel(z, PlaneKind.PUPIL, PlaneKind.INTERMEDIATE)),
            Stage(
                "dm2",
                PhaseScreen(dm, grid, wavelength_nm=WL, plane=PlaneKind.INTERMEDIATE),
            ),
            Stage("back", fresnel(-z, PlaneKind.INTERMEDIATE, PlaneKind.PUPIL)),
            Stage("science", Fraunhofer(grid_in=grid, grid_out=focal)),
        )
    )
    fx = np.asarray(focal.coords)
    fxg, fyg = np.meshgrid(fx, fx)
    both = (np.abs(fxg - 3.0) < 0.6) | (np.abs(fxg + 3.0) < 0.6)
    mask = jnp.asarray(both & (np.abs(fyg) < 0.6))
    common = dict(
        path=path,
        input_field=input_field,
        mask=mask,
        n_steps=n_steps,
        gain=0.5,
        regularization=1e-7,
        probes=probe_set(dm, amplitude_nm=2.0, n_probes=3),
        probe_dm=0,
        target_contrast=1e-8,
    )
    return {
        "one_dm": Scenario(regime="dig", params={**common, "dm_indices": (0,)}),
        "two_dm": Scenario(regime="dig", params={**common, "dm_indices": (0, 2)}),
    }


def hold_against_drift(npix=16, n_steps=12):
    """Dig, then hold a ramp drift: the maintenance benchmark.

    Args:
        npix: Pupil sampling.
        n_steps: Maintenance frames.

    Returns:
        A ``Scenario`` (regime ``maintain``) with the dug ``command0``.
    """
    from wavefronts.control import close_dark_hole

    grid, _xg, _yg, aperture = _aperture(npix)
    focal = Grid.focal(32, 0.5)
    drift_basis = _fourier_basis(npix, [(3, 1), (3, 0)], amp_nm=4.0)
    dm = _fourier_basis(npix, [(3, 1), (3, 0), (2, 1), (4, 1)], amp_nm=4.0)
    path = OpticalPath(
        stages=(
            Stage("wfe", PhaseScreen(drift_basis, grid, wavelength_nm=WL)),
            Stage("dm", PhaseScreen(dm, grid, wavelength_nm=WL)),
            Stage("science", Fraunhofer(grid_in=grid, grid_out=focal)),
        )
    )
    input_field = Field(
        data=jnp.asarray(aperture).astype(complex), grid=grid, plane=PlaneKind.PUPIL
    )
    fx = np.asarray(focal.coords)
    fxg, fyg = np.meshgrid(fx, fx)
    mask = jnp.asarray(
        ((np.abs(fxg - 3.0) < 0.6) & (np.abs(fyg - 1.0) < 0.6))
        | ((np.abs(fxg - 3.0) < 0.6) & (np.abs(fyg) < 0.6))
    )
    command0, _ = close_dark_hole(
        path, input_field, 1, mask, n_steps=15, gain=0.6, regularization=1e-8
    )
    directions = np.asarray([1.0, -0.7, 0.4, 0.9])
    table = jnp.asarray(np.arange(1, n_steps + 1)[:, None] * 0.3 * directions[None, :])
    dt_s = 10.0

    def drift(time_s):
        index = jnp.clip(jnp.asarray(time_s / dt_s, dtype=int), 0, n_steps - 1)
        return table[index]

    return Scenario(
        regime="maintain",
        params=dict(
            path=path,
            input_field=input_field,
            dm_indices=1,
            mask=mask,
            drift=drift,
            drift_stage=0,
            n_steps=n_steps,
            dt_s=dt_s,
            gain=0.7,
            regularization=1e-8,
            command0=command0,
        ),
    )


def _forked_system(npix=16, sci_ncpa_nm=0.0):
    grid, _xg, _yg, aperture = _aperture(npix)
    focal = Grid.focal(32, 0.5)
    shared = _fourier_basis(npix, [(3, 1), (3, 0)], amp_nm=4.0)
    corrector = _fourier_basis(npix, [(3, 1), (3, 0)], amp_nm=4.0)
    private = _fourier_basis(npix, [(2, 1)], amp_nm=4.0)

    def screen(basis, coeffs=None):
        b = basis if coeffs is None else ModeBasis(B=basis.B, coeffs=coeffs)
        return PhaseScreen(b, grid, wavelength_nm=WL)

    trunk = OpticalPath(
        stages=(
            Stage("drift", screen(shared)),
            Stage("shared_dm", screen(corrector)),
        )
    )
    split = BeamSplitter.energy(0.5, grid=grid, plane=PlaneKind.PUPIL)
    sci = OpticalPath(
        stages=(
            Stage("ncpa", screen(private, jnp.asarray([sci_ncpa_nm / 4.0, 0.0]))),
            Stage("science", Fraunhofer(grid_in=grid, grid_out=focal)),
        )
    )
    wfs = OpticalPath(
        stages=(
            Stage("wfs_ncpa", screen(private)),
            Stage("sensorcam", Fraunhofer(grid_in=grid, grid_out=focal)),
        )
    )
    system = OpticalSystem(
        trunk=trunk,
        split=split,
        branches=(Branch("sci", "transmit", sci), Branch("wfs", "reflect", wfs)),
    )
    input_field = Field(
        data=jnp.asarray(aperture).astype(complex), grid=grid, plane=PlaneKind.PUPIL
    )
    fx = np.asarray(focal.coords)
    fxg, fyg = np.meshgrid(fx, fx)
    mask = jnp.asarray((np.abs(fxg - 3.0) < 0.6) & (np.abs(np.abs(fyg) - 0.5) < 0.7))
    return system, input_field, {"sci": mask, "wfs": mask}


def _ramp(n_steps, n_modes, scale_nm):
    directions = np.asarray([1.0, -0.6, 0.8, -0.4])[:n_modes]
    return jnp.asarray(
        np.arange(1, n_steps + 1)[:, None] * scale_nm * directions[None, :]
    )


def wfs_plus_science(npix=16, n_steps=8, sci_ncpa_nm=2.0):
    """Sense on the WFS arm, correct the shared DM, score the science arm.

    Args:
        npix: Pupil sampling.
        n_steps: Frames.
        sci_ncpa_nm: The science arm's private NCPA (the floor).

    Returns:
        A ``Scenario`` (regime ``multichannel``).
    """
    system, input_field, masks = _forked_system(npix, sci_ncpa_nm=sci_ncpa_nm)
    return Scenario(
        regime="multichannel",
        params=dict(
            system=system,
            input_field=input_field,
            shared_dm_stage="shared_dm",
            drift_stage="drift",
            sense=("wfs",),
            science="sci",
            masks=masks,
            drift_table=_ramp(n_steps, 4, 0.3),
            n_steps=n_steps,
            gain=1.0,
            regularization=1e-10,
        ),
    )


def ncpa_limited(npix=16, n_steps=8, sci_ncpa_nm=3.0):
    """The science-side blindness floor: shared drift held, NCPA remains.

    Args:
        npix: Pupil sampling.
        n_steps: Frames.
        sci_ncpa_nm: The science arm's private NCPA.

    Returns:
        A ``Scenario`` (regime ``multichannel``).
    """
    return wfs_plus_science(npix, n_steps, sci_ncpa_nm=sci_ncpa_nm)


def wfs_private_aliasing(npix=16, n_steps=8):
    """Drift in the SENSING arm's private modes only: the injection stress.

    Args:
        npix: Pupil sampling.
        n_steps: Frames.

    Returns:
        A ``Scenario`` (regime ``multichannel``).
    """
    system, input_field, masks = _forked_system(npix)
    return Scenario(
        regime="multichannel",
        params=dict(
            system=system,
            input_field=input_field,
            shared_dm_stage="shared_dm",
            drift_stage="drift",
            sense=("wfs",),
            science="sci",
            masks=masks,
            drift_table=jnp.zeros((n_steps, 4)),
            n_steps=n_steps,
            gain=1.0,
            regularization=1e-10,
            local_drift={"wfs": ("wfs_ncpa", _ramp(n_steps, 2, 0.5))},
        ),
    )


def dual_science_common_mode(npix=16, n_steps=6):
    """Two arms sense jointly (equal weights); both hold the shared drift.

    Args:
        npix: Pupil sampling.
        n_steps: Frames.

    Returns:
        A ``Scenario`` (regime ``multichannel``).
    """
    system, input_field, masks = _forked_system(npix)
    return Scenario(
        regime="multichannel",
        params=dict(
            system=system,
            input_field=input_field,
            shared_dm_stage="shared_dm",
            drift_stage="drift",
            sense=("sci", "wfs"),
            science="sci",
            masks=masks,
            drift_table=_ramp(n_steps, 4, 0.3),
            n_steps=n_steps,
            gain=1.0,
            regularization=1e-10,
        ),
    )
