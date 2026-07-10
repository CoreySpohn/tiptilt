# tiptilt

Differentiable wavefront-error generation and wavefront control for the HWO
direct imaging simulation suite.

> Named for tip and tilt, the humblest wavefront correction.

## What it is

`tiptilt` owns the two halves of the wavefront program that the propagation
engine (`physicaloptix`) deliberately leaves out:

1. **Aberration generation.** A physically meaningful mode basis (segment
   piston/tip/tilt, deformable-mirror influence functions, Zernikes) plus
   per-mode temporal statistics produce a drifting wavefront-error realization.
   It is delivered as a time-varying speckle residual implementing the
   `optixstuff.AbstractSpeckleField` contract, built on the `(E_nom, G)`
   linearization `physicaloptix` provides, so the image and yield layers consume
   it unchanged.
2. **Wavefront control.** Wavefront-sensor estimators, a reconstructor over the
   same `(E_nom, G)` product (electric-field conjugation in space, a modal loop
   on the ground), and a differentiable control loop that commands the
   deformable mirror.

Space wavefront control is the priority; atmospheric turbulence shares the loop,
the mode basis, and the residual interface.

## What is here

- **Aberration sources** (`speckle`, `turbulence`): a stationary
  spectrally-factorized drift field, a tabulated replay field, correlated
  multi-channel realizations, and von Karman / frozen-flow screens.
- **Sensing** (`sensing`): pairwise probe estimation, Zernike wavefront
  sensor calibration and reconstruction, and Kalman field estimators behind
  a common `AbstractEstimator` seam.
- **Control** (`control`): electric-field conjugation, stroke minimization,
  and a predictive law behind a common `AbstractController` seam, with
  `close_dark_hole` as the driver.
- **Operations** (`maintenance`, `multichannel`): dark-hole maintenance
  under injected drift with honest model/truth separation, and cross-channel
  feed-forward on a shared deformable mirror.
- **Hardware** (`dm`): a programmable actuator-grid deformable mirror
  (Gaussian influence functions, per-actuator stroke limits) whose command
  vector is actuator-space, the language published control algorithms speak.
- **Benchmarking** (`testbed`): an algorithm registry, scenario builders
  (dig from cold, hold against drift, NCPA-limited, dual-channel), metrics,
  and `run`/`sweep` drivers. Custom laws plug in by implementing one method;
  see `examples/custom_controller.py`.

The mode-basis constructors it builds on (`zernike_basis`,
`segment_ptt_basis`, `fourier_dm_basis`) live in `physicaloptix`.

## Layout

- `src/tiptilt/` -- the package.
- `tests/` -- the test suite (pytest, CPU/x64-pinned like `physicaloptix`).
- `examples/` -- worked examples (bring-your-own control algorithm).
