# tiptilt

Differentiable wavefront-error generation and wavefront control for the
HWO direct imaging simulation suite.

> Named for tip and tilt, the humblest wavefront correction.

`tiptilt` owns the two halves of the wavefront program that the
propagation engine (`physicaloptix`) deliberately leaves out:

1. **Aberration generation.** A physically meaningful mode basis
   (segment piston/tip/tilt, deformable-mirror influence functions,
   Zernikes) plus per-mode temporal statistics produce a drifting
   wavefront-error realization, delivered as a time-varying speckle
   residual implementing the `optixstuff.AbstractSpeckleField` contract
   on top of the `(E_nom, G)` linearization the engine provides.
2. **Wavefront control.** Wavefront-sensor estimators, a reconstructor
   over the same `(E_nom, G)` product (electric-field conjugation in
   space, a modal loop on the ground), and a differentiable control
   loop that commands the deformable mirror.

The library keeps its devices and control loop private and exposes only
the resulting residual through the speckle seam, so the image and yield
layers pick it up without any change. Space wavefront control is the
priority; atmospheric turbulence shares the loop, the mode basis, and
the residual interface.

```{toctree}
:maxdepth: 1

API reference <autoapi/index>
```
