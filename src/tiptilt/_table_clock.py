"""Condition the sample coordinates before storing JAX drift tables."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import as_float_leaf, canonical, is_traced
from hwoutils.precision import like
from jaxtyping import Array


class _TableClock(eqx.Module):
    """A drift grid referenced to its first sample.

    Attributes:
        offsets_s: Seconds since the first knot, at table precision.
    """

    offsets_s: Array
    _host_origin_s: float | None = eqx.field(static=True)
    _traced_origin_s: Array | None

    def __init__(self, times_s, table):
        """Rebase a concrete or already-staged time grid.

        Args:
            times_s: Nonempty ascending knot times in seconds.
            table: Coefficient table supplying the storage precision.

        Raises:
            ValueError: If the grid is invalid or has unresolved knots.
        """
        shape = np.shape(times_s)
        if len(shape) != 1 or not shape[0]:
            raise ValueError("table times must be a nonempty 1D array")
        message = "table times must be finite and ascending at table precision"
        if any(is_traced(x) for x in jax.tree.leaves(times_s)):
            times = as_float_leaf(times_s)
            self._host_origin_s = None
            self._traced_origin_s = times[0]
            offsets = like(times - times[0], table)
            valid = jnp.all(jnp.isfinite(offsets)) & jnp.all(offsets[1:] > offsets[:-1])
            self.offsets_s = eqx.error_if(offsets, ~valid, message)
        else:
            times = np.asarray(times_s, dtype=np.float64)
            if not np.all(np.isfinite(times)) or np.any(times[1:] <= times[:-1]):
                raise ValueError(message)
            self._host_origin_s = float(times[0])
            self._traced_origin_s = None
            with jax.ensure_compile_time_eval():
                offsets = canonical(times - self._host_origin_s, precision=table.dtype)
                host_offsets = np.asarray(offsets)
                if np.any(host_offsets[1:] <= host_offsets[:-1]):
                    raise ValueError(message)
                self.offsets_s = offsets

    @property
    def times_s(self) -> Array:
        """Reconstruct the knot times for inspection.

        Returns:
            Knot times in seconds at the storage precision.
        """
        origin = (
            self._host_origin_s
            if self._traced_origin_s is None
            else self._traced_origin_s
        )
        return self.offsets_s + like(origin, self.offsets_s)

    def offset(self, elapsed_s) -> Array:
        """Subtract the table origin before casting to interpolation precision.

        Args:
            elapsed_s: Query time in seconds since the field's epoch.

        Returns:
            Query time relative to the first knot.
        """
        if self._traced_origin_s is not None:
            return like(elapsed_s - self._traced_origin_s, self.offsets_s)
        origin = self._host_origin_s
        if is_traced(elapsed_s):
            query = jnp.asarray(elapsed_s)
            if not jnp.issubdtype(query.dtype, jnp.floating):
                query = like(query, self.offsets_s)
            high = query.dtype.type(origin)
            # A rounded origin can be inaccurate even for an exact query.
            residual = float(origin - float(high))
            offset = (query - high) - residual
        else:
            offset = np.asarray(elapsed_s, dtype=np.float64) - origin
        return like(offset, self.offsets_s)
