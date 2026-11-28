# Changelog

## [2.0.0](https://github.com/CoreySpohn/tiptilt/compare/v1.0.0...v2.0.0) (2026-11-28)


### ⚠ BREAKING CHANGES

* **tiptilt:** decide drift admission, HWO-Sim storage and plane switches identically in both lanes, condition replay clocks, refuse unresolved float32 line grids
* **control:** StrokeMinController holds its DarkZoneModel (the n_dark and stack_weights fields are gone) and tests candidates with DarkZoneModel.weighted_contrast, the pixel-weighted mean intensity, instead of dividing by the pixel count; target_contrast is in the units of DarkZoneModel.contrast, so re-derive a target set against non-uniform pixel weights. New DarkZoneModel.weighted_contrast and StrokeMinController.contrast.
* **multichannel:** FeedForwardController.build and run_multichannel weights are now inverse variances of each arm's estimate (up to a common factor) applied once in the solve (H^T W H + reg I)^-1 H^T W, instead of scaling each arm's response; unequal weights give different commands than before. Weights must be one finite, nonnegative value per sensing arm with a positive sum.
* **speckle:** correlated drift and channel fields refuse a covariance whose size does not match the response modes; correlated_channel_fields requires an opd or amplitude multi-channel linearization and passes its kind as the wavelength law.
* **speckle:** TabulatedSpeckleField.coefficients and HwoSimSpeckleField.coefficients, realize and decompose_delta take keyword-only elapsed_s, seconds since the field's clock origin, in place of time_d; positional coefficients(t) raises TypeError. Neither reads epoch_d any more: HwoSimSpeckleField no longer subtracts it on the host for a concrete time, so convert scene days with field.elapsed_s(time_d), which refuses a float32 time at a nonzero epoch.
* **hwo_sim:** HwoSimSpeckleField.epoch_d is a static Python float, subtracted from a concrete time_d in binary64 before the cast; a non-finite or traced epoch_d is refused at construction, and eqx.tree_at cannot replace it (use with_drift or with_tabulated_drift, which rebuild through the constructor). wavelengths_nm, amplitude, frequency_hz, phase, drift_time_s and drift_coefficients are strong float leaves of the active dtype.
* **leaves:** field kinds change from Python float or weakly typed array to a strong jax.Array: EFCController, MatrixFreeEFCController, StrokeMinController, FeedForwardController gain; StrokeMinController target_contrast; PredictiveController alpha; HwoSimSpeckleField epoch_d and contrast_to_flux_fraction (were Python floats); TabulatedSpeckleField times_s; KalmanEstimator step_index (strong integer).
* build detector rates on the single optixstuff Rates type
* read named quadrature weights
* import grids and spectral types from optixstuff
* update commands through with_command and build speckle drift on FocalResponse
* **speckle:** take ou_exposure_neff from physicaloptix
* integrate every speckle PSD with trapezoid widths and take DM centres from hwoutils
* **speckle:** build tabulated speckle fields on the shared linear speckle field with days and real grids

### Features

* **hwo_sim:** HwoSimSpeckleField reports its dark zone as data_mask, so support ends at the delivery's pixel list ([8ce2fa8](https://github.com/CoreySpohn/tiptilt/commit/8ce2fa8abed3c99927404bed76911918831101f9))
* **speckle:** build tabulated speckle fields on the shared linear speckle field with days and real grids ([77bc4f2](https://github.com/CoreySpohn/tiptilt/commit/77bc4f26ce0188727715a70750c40891cbe6df57))
* **speckle:** declare wavelength laws and check covariance modes in drift fields ([9f76f0a](https://github.com/CoreySpohn/tiptilt/commit/9f76f0a93441ecfc4805744d75200acc7399b949))
* **speckle:** tabulated and hwo_sim speckle fields take elapsed seconds ([dd24b45](https://github.com/CoreySpohn/tiptilt/commit/dd24b45490252eb3b821cd3e2134243e2d75a1ac))


### Bug Fixes

* build detector background with a band-aligned frame plan ([95eda05](https://github.com/CoreySpohn/tiptilt/commit/95eda05366f4f20ded5d36bc383cecb3233ed174))
* **control:** evaluate stroke-min feasibility in the dark-zone model's weighted contrast ([c5ab812](https://github.com/CoreySpohn/tiptilt/commit/c5ab8122185295e0557729c4e683e8147d9f42e4))
* **control:** score a single wavelength slice with pixel weights only ([ffd67ed](https://github.com/CoreySpohn/tiptilt/commit/ffd67ed5f636bccc2ec353c805415dc9695523a2))
* **hwo_sim:** keep epoch_d a static host float and store the drift arrays as strong leaves ([a724347](https://github.com/CoreySpohn/tiptilt/commit/a7243472549c90e7a10d68c0812b6ffbda6db2e2))
* **hwo_sim:** re-anchor a speckle field converting only the supplied values ([2b8ef96](https://github.com/CoreySpohn/tiptilt/commit/2b8ef9624bed0448dd71aed309fa8d869497b891))
* **hwo_sim:** re-anchoring always returns a new field and leaves the source unchanged ([1d8d5e5](https://github.com/CoreySpohn/tiptilt/commit/1d8d5e5517bca3d47814e7536a9b26b063a6ca48))
* **leaves:** store controller gains, speckle scalars, times and probe index as strong leaves ([a1094a7](https://github.com/CoreySpohn/tiptilt/commit/a1094a784502767505038042a1c4aefded34355d))
* **multichannel:** apply feed-forward arm weights as statistical weights on each arm's squared residual ([c925b5e](https://github.com/CoreySpohn/tiptilt/commit/c925b5ef43f3dcf5b93ec942b80a1a3d1b7e1a47))
* **speckle:** store the tabulated eps table as a strong float leaf ([87cc5d2](https://github.com/CoreySpohn/tiptilt/commit/87cc5d2d322c70d3dd95415be65bdbdf42b374da))
* **tiptilt:** decide drift admission, HWO-Sim storage and plane switches identically in both lanes, condition replay clocks, refuse unresolved float32 line grids ([0beb2da](https://github.com/CoreySpohn/tiptilt/commit/0beb2da837dfcf9f872fa9bc25dbacd8219b83cc))
* **tiptilt:** read speckle coefficient series with interp_linear for jax 0.12.1 ([4a33885](https://github.com/CoreySpohn/tiptilt/commit/4a338854425017e49f9ed5a5bba5cae69d1c7e9f))


### Code Refactoring

* build detector rates on the single optixstuff Rates type ([9a47aa2](https://github.com/CoreySpohn/tiptilt/commit/9a47aa2d8e1ad05c40d1872a4c7a459c302994f8))
* import grids and spectral types from optixstuff ([fcb2b61](https://github.com/CoreySpohn/tiptilt/commit/fcb2b61a402d3d76cfee46b7e765f5f22a3b7a90))
* integrate every speckle PSD with trapezoid widths and take DM centres from hwoutils ([ab2c974](https://github.com/CoreySpohn/tiptilt/commit/ab2c97486389e6ef20101236496d4c887d501542))
* read named quadrature weights ([8e4e5d8](https://github.com/CoreySpohn/tiptilt/commit/8e4e5d852d281b332ecc765e2186d6436bad75d7))
* **speckle:** take ou_exposure_neff from physicaloptix ([2f45638](https://github.com/CoreySpohn/tiptilt/commit/2f45638ee6c513da48aaecc7f2d44da5f12dcf97))
* update commands through with_command and build speckle drift on FocalResponse ([3afc526](https://github.com/CoreySpohn/tiptilt/commit/3afc52668ac18a9a56f416049f3fa2f7d17adf46))

## [1.0.0](https://github.com/CoreySpohn/tiptilt/compare/v0.3.0...v1.0.0) (2026-11-16)


### ⚠ BREAKING CHANGES

* **speckle:** the speckle clock origin is days from the scene epoch

### Features

* **speckle:** the speckle clock origin is days from the scene epoch ([9877d50](https://github.com/CoreySpohn/tiptilt/commit/9877d5065108283ad92649a110e2308fe8bf4265))

## [0.3.0](https://github.com/CoreySpohn/tiptilt/compare/v0.2.0...v0.3.0) (2026-11-05)


### Features

* **control:** add MatrixFreeEFCController and a matrix-free close_dark_hole that solve EFC by conjugate gradients on autodiff products ([d6d65c0](https://github.com/CoreySpohn/tiptilt/commit/d6d65c063597aa872d478ec5cbb33a86456b7b18))
* **control:** per-pixel weighted dark-zone objectives through the metric, Jacobian, and solve ([806a1d5](https://github.com/CoreySpohn/tiptilt/commit/806a1d5b9fa19cae51ab48061c30d720d35db127))
* **dm:** add ActuatorLattice and ActuatorDM, a flight-scale deformable mirror evaluated by FFT convolution without per-actuator maps ([f1c180a](https://github.com/CoreySpohn/tiptilt/commit/f1c180a7dd48036e6f3a0b3f7220135381101025))
* **hwo_sim:** add load_hwo_sim, HwoSimSpeckleField and segment_zernike_series, a reader and speckle field for the hwo_sim linear E-field format ([390b6cd](https://github.com/CoreySpohn/tiptilt/commit/390b6cdb7dad7c0c41cb5d9dac96b707061dbc7c))
* **speckle:** closed-form exposure averaging count for OU drift ([5387676](https://github.com/CoreySpohn/tiptilt/commit/538767652952a150fbfeb7b355757879bf71b7f5))
* **speckle:** exact two-time law for OU drift, and the realizability it rests on ([d3384a0](https://github.com/CoreySpohn/tiptilt/commit/d3384a0dc7b3655dda59335f550ed3e1228ca1c2))
* **speckle:** grouped_drift_field -- mode blocks drifting on their own timescales ([9a59579](https://github.com/CoreySpohn/tiptilt/commit/9a595795122a764e704ecb57e449f33e5b59e592))
* **speckle:** OU innovation covariance -- what a step cannot predict ([1310c75](https://github.com/CoreySpohn/tiptilt/commit/1310c75cf55cfe02ea0b566ff573cb37ccf46394))
* **speckle:** replace normalization with input-energy primitives across fields and builders ([4ef1ca8](https://github.com/CoreySpohn/tiptilt/commit/4ef1ca8ccabf3dc028d6097917bf45d5c3c0f289))
* **speckle:** trajectory builders -- exact OU drift, random walk, one-sided creep ([6847ab1](https://github.com/CoreySpohn/tiptilt/commit/6847ab184db0a14f13d568848ce56fd45f68da1d))


### Bug Fixes

* American spelling throughout (nanometres, metre, centred, colour, behaviour) ([6289144](https://github.com/CoreySpohn/tiptilt/commit/6289144d30367d2e3dad4d911304e8a96e1aff00))
* **speckle:** treat psd as a density, weighting lines by psd*df on the caller's grid ([fc5fbb9](https://github.com/CoreySpohn/tiptilt/commit/fc5fbb9d949d1da7550a2cf288a8a036e26c849c))

## [0.2.0](https://github.com/CoreySpohn/tiptilt/compare/v0.1.0...v0.2.0) (2026-07-20)


### Features

* **speckle:** chromatic wavelength channels for TabulatedSpeckleField -- nearest-channel realize + lambda-scaled broadened() sharing the mode trajectory ([8f9bff2](https://github.com/CoreySpohn/tiptilt/commit/8f9bff2cdf4b731b24095508d725a6ac1a760f91))

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
