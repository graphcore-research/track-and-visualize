_libname = 'nvis'
_TORCH_EXTRA = False
_JAX_EXTRA = False

try:
    import torch
    _TORCH_EXTRA = True
except ImportError:
    ...


try:
    import jax,flax,optax
    _JAX_EXTRA = True
except ImportError:
    ...