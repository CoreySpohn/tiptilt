# wavefronts

Differentiable wavefront-error generation and wavefront control for the HWO
direct imaging simulation suite.

> **Working name.** `wavefronts` is a temporary distribution name chosen to get
> the library moving; the final name (and PyPI availability) is still to be
> decided.

## What it is

`wavefronts` owns the two halves of the wavefront program that the propagation
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

## Status

Early scaffold. The mode-basis constructors it builds on
(`zernike_basis`, `segment_ptt_basis`) live in `physicaloptix`. The first
component here is the aberration-source speckle-field family (a stationary
spectrally-factorized field and a tabulated replay field for non-stationary
drift).

## Layout

- `src/wavefronts/` -- the package.
- `tests/` -- the test suite (pytest, CPU/x64-pinned like `physicaloptix`).
