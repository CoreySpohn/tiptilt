"""Scale and boundary checks for user-supplied drift covariances."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.numerics import COARSEST, eps, roundoff
from hwoutils.precision import PrecisionError
from optixstuff import FocalResponse
from optixstuff.grids import Frame, PixelGrid
from physicaloptix import ChannelLinearization, MultiChannelLinearization

from tiptilt.speckle import (
    _line_powers,
    _psd_sqrt,
    correlated_channel_fields,
    correlated_drift_field,
    grouped_drift_field,
    ou_covariance,
    random_walk_trajectory,
)


@pytest.fixture(scope="module")
def response():
    """Make a monochromatic response with two modes and one pixel.

    Returns:
        The shared focal response for checking drift field construction.
    """
    return FocalResponse(
        np.ones((1, 1)),
        np.ones((2, 1, 1)),
        PixelGrid.focal(1, 0.25, Frame(500.0, 1.0, "circumscribed")),
        1.0,
        500.0,
        wavelength_law="opd",
    )


@pytest.mark.parametrize(
    "builder", ["correlated", "grouped", "shared", "local", "walk"]
)
def test_tiny_indefinite_block_is_rejected(builder, response):
    """A correlation of two is impossible at every variance scale.

    Args:
        builder: Public drift builder or channel block receiving the covariance.
        response: A two-mode focal response.
    """
    invalid = 1e-12 * np.array([[1, 2], [2, 1]])
    key = jax.random.PRNGKey(0)
    powers = np.ones(16)
    frequencies = np.arange(1, 17)
    kwargs = dict(key=key, frequencies_hz=frequencies, psd=powers)
    if builder == "correlated":
        with pytest.raises(ValueError, match="semidefinite"):
            correlated_drift_field(response, invalid, **kwargs)
    elif builder == "grouped":
        with pytest.raises(ValueError, match="semidefinite"):
            grouped_drift_field(response, [(invalid, frequencies, powers)], key=key)
    elif builder == "walk":
        with pytest.raises(ValueError, match="semidefinite"):
            random_walk_trajectory(invalid, key=key, times_s=[0, 1])
    else:
        channel = ChannelLinearization(
            name="a", e_nom=response.e_nom, g_shared=response.G, grid=response.grid
        )
        mcl = MultiChannelLinearization(
            channels=(channel,), wavelength_nm=500.0, kind="opd", input_energy=1.0
        )
        shared = invalid if builder == "shared" else np.eye(2)
        local = {"a": (response.G, invalid)} if builder == "local" else None
        with pytest.raises(ValueError, match="semidefinite"):
            correlated_channel_fields(mcl, shared, local=local, **kwargs)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("kind", ["positive", "symmetric", "opposite signs"])
def test_covariance_boundary_uses_common_resolution(compiled, kind):
    """Admit the coarsest-grid boundary and reject an exact excess above it.

    Symmetry allows one entry epsilon and the eigensolver allows two
    spectral epsilons for two modes. Opposite signs can hide an exact excess
    even when both entries lie on the float32 grid.

    Args:
        compiled: Test both a host closure and a traced array if true.
        kind: Covariance admission boundary to probe.
    """

    def evaluate(matrix):
        return ou_covariance(matrix, 1.0)

    array_read = jax.jit(evaluate) if compiled else evaluate

    def host_read(matrix):
        if compiled:
            return jax.jit(lambda: evaluate(matrix))()
        return evaluate(matrix)

    threshold = np.float32(-2 * eps(COARSEST) if kind == "positive" else eps(COARSEST))
    direction = np.copysign(np.inf, float(threshold))
    matrix = np.eye(2)
    index = (1, 1) if kind == "positive" else (0, 1)
    boundary = (float(threshold),)
    if kind != "opposite signs":
        boundary += (np.nextafter(float(threshold), direction),)
    for value in boundary:
        matrix[index] = value
        assert np.isfinite(host_read(matrix)).all()
        assert np.isfinite(array_read(jnp.asarray(matrix))).all()
    if kind == "opposite signs":
        matrix[0, 1] = threshold
        matrix[1, 0] = -(2.0**-50)
    else:
        matrix[index] = np.nextafter(threshold, np.float32(direction))
    with pytest.raises(ValueError, match="semidefinite"):
        host_read(matrix)
    with pytest.raises((RuntimeError, ValueError), match="semidefinite"):
        np.asarray(array_read(jnp.asarray(matrix)))


def test_covariance_root_does_not_overflow_large_finite_entries(precision):
    """The diagonal and its square root are dyadic and exactly representable.

    Args:
        precision: Real floating dtype of the active test lane.
    """
    root = 1.5 * 2.0 ** (np.finfo(precision).maxexp // 2 - 1)
    matrix = jnp.asarray(np.eye(2) * root**2)
    factor = _psd_sqrt(matrix)
    np.testing.assert_array_equal(factor @ factor.T, matrix)


def test_subnormal_indefinite_covariance_is_rejected(precision):
    """The smallest subnormal matrix with correlation two is indefinite.

    Args:
        precision: Real floating dtype of the active test lane.
    """
    matrix = (
        np.array([[1, 2], [2, 1]], dtype=precision)
        * np.finfo(precision).smallest_subnormal
    )
    with pytest.raises(ValueError, match="semidefinite"):
        ou_covariance(jnp.asarray(matrix), 1.0)


@pytest.mark.float32_only(reason="overflow on storage in the float32 construction lane")
@pytest.mark.parametrize("compiled", [False, True])
def test_static_covariance_cannot_overflow_working_precision(compiled):
    """Twice the largest float32 variance is finite host data.

    Args:
        compiled: Compile a closure over the host covariance when true.
    """
    matrix = np.array([[2 * float(np.finfo(COARSEST).max)]])

    def build():
        return ou_covariance(matrix, 1.0)

    evaluate = jax.jit(build) if compiled else build
    with pytest.raises(ValueError, match="do not fit float32"):
        evaluate()


def test_compiled_single_sample_walk_checks_its_diffusion():
    """An indefinite diffusion cannot define a walk, even at its origin."""

    @jax.jit
    def start(diffusion):
        return random_walk_trajectory(diffusion, key=jax.random.key(0), times_s=[0])

    with pytest.raises(RuntimeError, match="semidefinite"):
        np.asarray(start(jnp.asarray([[1.0, 2.0], [2.0, 1.0]])))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("distinct", [False, True])
def test_grouped_grid_layout_uses_the_supplied_frequencies(
    compiled, distinct, response
):
    """Only equal host grids share a frequency axis.

    Args:
        compiled: Compile the build over the random key with static metadata.
        distinct: Move the second grid by one host ULP when true.
        response: A two-mode focal response.
    """
    frequencies = np.array([1.0, 2.0, 3.0])
    other = frequencies.copy()
    if distinct:
        other[1] = np.nextafter(other[1], np.inf)
    np.testing.assert_array_equal(frequencies.astype(COARSEST), other.astype(COARSEST))
    groups = [(np.eye(1), f, np.ones(3)) for f in (frequencies, other)]

    def build(key):
        return grouped_drift_field(response, groups, key=key).frequencies_hz

    evaluate = jax.jit(build) if compiled else build
    expected_shape = (2, 3) if distinct else (3,)
    assert evaluate(jax.random.key(0)).shape == expected_shape


@pytest.mark.parametrize("compiled", [False, True])
def test_large_carrier_lines_are_resolved_in_the_synthesized_field(
    response, precision, compiled
):
    """Unit density gives exact widths; float32 cannot synthesize these lines.

    The float64 oracle uses the delivered Hz and the frozen draw. At 0.5 s
    the time multiply is exact. Four roundings cover the phase and cosine
    products per line, plus two additions on the absolute sum scale.

    Args:
        response: A two-mode focal response.
        precision: Real floating dtype of the active test lane.
        compiled: Compile the build and coefficient read if true.
    """
    frequencies = 2.0**26 + np.array([0.0, 1.0, 3.0])
    assert np.unique(frequencies.astype(COARSEST)).size == 1
    np.testing.assert_array_equal(
        _line_powers(frequencies, np.ones(3)), [0.5, 1.5, 1.0]
    )

    def build(key):
        field = grouped_drift_field(
            response, [(np.eye(2), frequencies, np.ones(3))], key=key
        )
        return field.coefficients(elapsed_s=0.5), field.amplitudes, field.phases

    evaluate = jax.jit(build) if compiled else build
    key = jax.random.key(0)
    if precision == np.float32:
        with pytest.raises(PrecisionError, match=r"frequency lines.*float64"):
            evaluate(key)
    else:
        coefficients, amplitudes, phases = map(np.asarray, evaluate(key))
        angle = np.pi * frequencies + phases
        reference = np.sum(amplitudes * np.cos(angle), axis=-1)
        scale = np.sum(np.abs(amplitudes) * (1 + np.abs(angle)), axis=-1)
        assert np.all(np.abs(coefficients - reference) <= roundoff(scale, n_ops=6))


@pytest.mark.float32_only(reason="overflow on storage in the float32 construction lane")
def test_compiled_grouped_grid_cannot_store_infinite_frequencies(response):
    """A finite host frequency beyond float32 range must fail on storage.

    Args:
        response: A two-mode focal response.
    """
    too_large = 2 * float(np.finfo(COARSEST).max)
    groups = [(np.eye(2), [too_large], [1.0])]

    @jax.jit
    def build(key):
        return grouped_drift_field(response, groups, key=key).frequencies_hz

    with pytest.raises(ValueError, match="do not fit float32"):
        build(jax.random.key(0))
