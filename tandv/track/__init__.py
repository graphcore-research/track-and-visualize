# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
from .. import _config
from .common import *  # NOQA: F401 F403
from .common import __all__

if _config._TORCH_EXTRA and not _config._JAX_EXTRA:
    from . import torch

    __all__ = ["torch", *__all__]  # type: ignore
elif not _config._TORCH_EXTRA and _config._JAX_EXTRA:
    from . import jax

    __all__ = ["jax", *__all__]  # type: ignore
elif _config._TORCH_EXTRA and _config._JAX_EXTRA:
    from . import jax, torch  # NOQA: F401 F403

    __all__ = ["torch", "jax", *__all__]  # type: ignore
else:
    __all__ = [*__all__]  # type: ignore
