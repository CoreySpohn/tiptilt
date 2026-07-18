# Changelog

## 0.1.0 (2026-07-18)


### Features

* **control,sensing:** symmetric estimator/controller seams -- DarkZoneModel, EFC/stroke-min/predictive controllers, oracle/pairwise/Kalman estimators; close_dark_hole a thin driver ([c94ee5d](https://github.com/CoreySpohn/tiptilt/commit/c94ee5d4ff183375ea6c7e02f2be897537fd5ff2))
* **control:** broadband estimated dark-hole control via per-sub-band probing ([3db547b](https://github.com/CoreySpohn/tiptilt/commit/3db547b438a3e6bdec4ec0f3ef0b4dc649a65970))
* **control:** close_dark_hole -- differentiable single-DM EFC loop (linearize once & hoist, tree_at command updates, propagate per step); digs a deep dark hole and differentiates through the scan ([a80c053](https://github.com/CoreySpohn/tiptilt/commit/a80c0534edeccf35c4af8b38b97464cd44b59557))
* **control:** estimated (pairwise-probe) dark-hole loop alongside oracle ([fe8cb7d](https://github.com/CoreySpohn/tiptilt/commit/fe8cb7de209f89fa8ff7254e2f6ec3447cbab885))
* **control:** generalize close_dark_hole to N deformable mirrors and broadband (chromatic) dark holes ([08d670f](https://github.com/CoreySpohn/tiptilt/commit/08d670f594da7d6e20f7d024d9547b95e3065312))
* **control:** kalman estimated loop + shared probe_measurement helper ([f4180d7](https://github.com/CoreySpohn/tiptilt/commit/f4180d746eb1609f1156e2e1d117af0e61ccb09d))
* **dm:** HardwareDM hardware-honesty layer -- DAC quantization with straight-through Jacobians, per-actuator gains and dead columns, stuck-actuator offsets, hardware_dm testbed scenario ([91e0136](https://github.com/CoreySpohn/tiptilt/commit/91e01369e608563fd859e167ba0f984330564a9e))
* **dm:** programmable actuator-grid DeformableMirror (Gaussian influence functions, coupling parameterization, per-actuator stroke limits) + callable-controller registry extension point and a bring-your-own-algorithm worked example ([5a505f3](https://github.com/CoreySpohn/tiptilt/commit/5a505f32d8a02686dd229a58e8da65efe53ec537))
* **lowfs:** pickoff low-order sensor and pointing loop (defocus-biased response matrix, integrator on the fork, pointing testbed regime) ([0158bc2](https://github.com/CoreySpohn/tiptilt/commit/0158bc2fdf7029b8f4ac6b9cf3e79fecbca483fc))
* **maintenance:** maintain_dark_hole (drift-injected hold at the dug operating point, honest model side) + mean-preserving make_detector ([3bfcef7](https://github.com/CoreySpohn/tiptilt/commit/3bfcef742916bff9d792520a31bfa95577789ee3))
* **maintenance:** maintained_residual_field exports the closed-loop residual as a TabulatedSpeckleField (drift+command sensitivities about a dug operating point); maintain_dark_hole grows return_trajectory ([05f746b](https://github.com/CoreySpohn/tiptilt/commit/05f746ba8ce565a70a08a10cd66009ffb3de6744))
* **multichannel:** cross-channel feed-forward -- MultiChannelModel dark-zone blocks, shared_dm_command, FeedForwardController, run_multichannel with reference frames and measurable NCPA aliasing ([fca42ee](https://github.com/CoreySpohn/tiptilt/commit/fca42eedb166f8aba2ebfeddad844a9b57b6cb7c))
* scaffold wavefronts -- differentiable wavefront-error generation and control (working name) ([c263d98](https://github.com/CoreySpohn/tiptilt/commit/c263d984c9d180149a41f564e0e18c49bf84320a))
* **sensing:** pairwise-probe focal-plane field estimator ([372e2f7](https://github.com/CoreySpohn/tiptilt/commit/372e2f7bdfa8db0bbe760ce68cf004bc81786f3a))
* **sensing:** recursive Kalman field estimator (one probe pair per step) ([3688eb1](https://github.com/CoreySpohn/tiptilt/commit/3688eb1aa9bfd0be090cfbd425187b5ec1baeecc))
* **sensing:** ZWFS low-order reconstruction (interaction-matrix inverse) ([746f522](https://github.com/CoreySpohn/tiptilt/commit/746f52259931915ebc8386a2a20d9f80fa9d3a33))
* **speckle:** correlated_channel_fields (one shared draw through per-channel g_shared blocks, independent local NCPA blocks) + public TabulatedSpeckleField.eps ([fdb7c40](https://github.com/CoreySpohn/tiptilt/commit/fdb7c4052c3a079f749cdde957e5ee061ae250f3))
* **speckle:** correlated_drift_field -- stationary AnalyticSpeckleField with a target cross-mode covariance via eigendecomposition spectral factorization; adversarial-review hardening (ensemble-covariance docs, input validation, low-N_eff warning) ([6b7ffd3](https://github.com/CoreySpohn/tiptilt/commit/6b7ffd38fc6de4e9c349c5ca9d12b3ce7d017702))
* **speckle:** TabulatedSpeckleField -- replay a precomputed WFE trajectory as a floor-excluded AbstractSpeckleField (the non-stationary/AR escape hatch), with physicaloptix (E_nom,G) integration ([a7c883f](https://github.com/CoreySpohn/tiptilt/commit/a7c883f06126c59352c9c4340a9bda006b8e9061))
* **testbed:** the WFSC benchmark harness -- algorithm registry, Scenario/Metrics with stroke caps, run/sweep, seven scenario builders (dig, one-vs-two DM, hold, ncpa_limited, aliasing, dual-science, wfs+science) ([99afbd9](https://github.com/CoreySpohn/tiptilt/commit/99afbd9c6378dea15f0a2956af60cded4d455fc1))
* **turbulence:** von Karman/Kolmogorov phase screens with frozen flow (ground-turbulence aberration generation) ([28d0756](https://github.com/CoreySpohn/tiptilt/commit/28d07567ad1868f9506202acfb7060c9dc8b94cc))
* wire release automation and docs -- release-please, PyPI trusted publishing, hatch-vcs versioning, sphinx + ReadTheDocs scaffold ([3fcd2e8](https://github.com/CoreySpohn/tiptilt/commit/3fcd2e8c2b9fe5f51c4b685bc01d552ec08238f9))


### Bug Fixes

* **control:** guard empty dark zone, non-PhaseScreen dm_index, and non-positive regularization; document single-DM/mono/one-sided scope + dense-jacfwd scaling (adversarial-review hardening) ([53fa069](https://github.com/CoreySpohn/tiptilt/commit/53fa06949e4d137d59fb17f2253f5d4e8120072f))
