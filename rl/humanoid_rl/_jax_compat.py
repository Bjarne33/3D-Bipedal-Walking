"""Restore JAX APIs that Brax and older Flax still call but newer JAX has removed.

Each shim is installed only when the attribute is missing, so it is a no-op on older JAX.
The replacements follow https://docs.jax.dev/en/latest/migrate_pmap.html.
"""

import jax
import jax.core
import numpy as np


def _missing(module, name):
    try:
        getattr(module, name)
    except AttributeError:
        return True
    return False


def _stacked_sharding(devices):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    return NamedSharding(Mesh(np.array(devices), ('x',)), PartitionSpec('x'))


def _device_put_replicated(x, devices):
    import jax.numpy as jnp

    sharding = _stacked_sharding(devices)
    return jax.tree.map(lambda y: jax.device_put(jnp.stack([y] * len(devices)), sharding), x)


def _device_put_sharded(shards, devices):
    import jax.numpy as jnp

    sharding = _stacked_sharding(devices)
    return jax.tree.map(lambda *xs: jax.device_put(jnp.stack(xs), sharding), *shards)


def install():
    # Removed in JAX 0.11; Flax < 0.12.7 calls it when initializing networks.
    if _missing(jax.core, 'get_opaque_trace_state'):
        from jax.extend import core as jex_core
        jax.core.get_opaque_trace_state = jex_core.get_opaque_trace_state
    # Removed after the pmap/shard_map merge; Brax PPO still calls them.
    if _missing(jax, 'device_put_replicated'):
        jax.device_put_replicated = _device_put_replicated
    if _missing(jax, 'device_put_sharded'):
        jax.device_put_sharded = _device_put_sharded
    # The flag was removed once the pmap/shard_map merge became permanent; Brax still reads it.
    if _missing(jax.config, 'jax_pmap_shmap_merge'):
        object.__setattr__(jax.config, 'jax_pmap_shmap_merge', True)
