import jax
import pytest


@pytest.fixture(autouse=True, scope='module')
def _clear_jax_caches():
    """Free compiled executables between modules (XLA CPU compiles of MJX are large)."""
    yield
    jax.clear_caches()
