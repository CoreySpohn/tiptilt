"""wavefronts -- differentiable wavefront-error generation and control.

A downstream consumer of physicaloptix (the differentiable propagation engine)
and optixstuff (the hardware and speckle-field contracts), parallel to
coronagraphoto (image simulation). It owns the two halves of the wavefront
program that the propagation engine deliberately does not:

- **Aberration generation.** A physically meaningful mode basis (segment
  piston/tip/tilt, deformable-mirror influence functions, Zernikes) plus
  per-mode temporal statistics produce a drifting wavefront-error realization,
  delivered as a time-varying speckle residual that implements the
  ``optixstuff.AbstractSpeckleField`` contract on top of the ``(E_nom, G)``
  linearization the engine provides.
- **Wavefront control.** Wavefront-sensor estimators, a reconstructor over the
  same ``(E_nom, G)`` product (electric-field conjugation in space, a modal
  loop on the ground), and the differentiable control loop that commands the
  deformable mirror.

The library keeps its devices and control loop private and exposes only the
resulting residual through the speckle seam, so the image and yield layers pick
it up without any change. Space wavefront control is the priority; atmospheric
turbulence shares the loop, the mode basis, and the residual interface.

The distribution name is a working placeholder and may change before release.
"""

from wavefronts.speckle import TabulatedSpeckleField

__version__ = "0.0.1"

__all__ = ["TabulatedSpeckleField", "__version__"]
