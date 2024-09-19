# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
_libname = "tandv"
_TORCH_EXTRA = False
_JAX_EXTRA = False
_WANDB_EXTRA = False

try:
    import torch  # NOQA: F401 F403

    _TORCH_EXTRA = True
except ImportError:
    ...


try:
    import flax  # NOQA: F401 F403
    import jax  # NOQA: F401 F403
    import optax  # NOQA: F401 F403

    _JAX_EXTRA = True
except ImportError:
    ...


try:
    import wandb  # NOQA: F401 F403

    _WANDB_EXTRA = True
except ImportError:
    ...
