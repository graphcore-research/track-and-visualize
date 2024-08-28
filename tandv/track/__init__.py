# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
from .. import _config

if _config._TORCH_EXTRA and not _config._JAX_EXTRA:
    from . import common, torch

    __all__ = ["torch", "common"]  # type: ignore
elif not _config._TORCH_EXTRA and _config._JAX_EXTRA:
    from . import common, jax

    __all__ = ["jax", "common"]  # type: ignore
elif _config._TORCH_EXTRA and _config._JAX_EXTRA:
    from . import common, jax, torch  # NOQA: F401 F403

    __all__ = ["torch", "jax", "common"]  # type: ignore
else:
    from . import common

    __all__ = ["common"]  # type: ignore
