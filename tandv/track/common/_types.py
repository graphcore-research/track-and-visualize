# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NewType,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np

from ... import _config

if _config._TORCH_EXTRA and not _config._JAX_EXTRA:
    import torch
    from torch import nn

    ModuleType = NewType("ModuleType", Union[Type[nn.Module], str, None])
    DType = NewType("DType", Union[torch.dtype, str])


elif not _config._TORCH_EXTRA and _config._JAX_EXTRA:

    import flax.linen as lnn

    ModuleType = NewType("ModuleType", Union[Type[lnn.Module], str, None])
    DType = NewType("DType", Union[np.dtype, str])

elif _config._TORCH_EXTRA and _config._JAX_EXTRA:
    import flax.linen as lnn
    import torch
    from torch import nn

    ModuleType = NewType(
        "ModuleType", Union[Type[nn.Module], Type[lnn.Module], str, None]
    )
    DType = NewType("DType", Union[torch.dtype, np.dtype, str])

else:
    ModuleType = NewType("ModuleType", Union[str, None])
    DType = NewType("DType", Union[str])


"""
    Data Structures for logging
"""

# The can be used for generating queries, (all data stored in log)


class WildCard:
    ...

    def __str__(self):
        return "*"


class WCAL1(WildCard):
    min: int = 1
    max: int = 2**32


@dataclass
class WildCardTracker:
    dtype: Any
    count: int = 0


@dataclass
class SchemaMap:
    """
    For mapping from the Table format of your Logs to the \
        format expected by the downstream tools in the library.
    """

    metadata: Dict
    scalar_stats: Union[Dict[str, str], None]
    exponent_counts: Union[Dict[str, str], None]
    # tl_index_map:

    def __post_init__(self):
        # both scalar_stats and exponent_counts can't be None ->
        assert (
            self.scalar_stats is not None or self.exponent_counts is not None
        ), "A mapping must be provided for \
            either scalar_stats or exponent_counts"

        # assert that keys provided exist in the logframe schema keys
        sp = set(self.metadata.keys())
        ss = set(LogFrame.schema["metadata"].keys())
        assert sp.issubset(
            ss
        ), f'The keys: {",".join(list(sp.difference()))} , are not \
            in the allowed set: {",".join(list(ss))}'


# Logging Types
TT = Literal[
    "Activation",
    "Gradient",
    "Weights",
    r"Optimiser_State\.[a-zA-Z_\-]+",
    "Weight_Gradients",
]


class LogFrame:
    # Need to attach additional metadata that is per iteration (i.e. train/val
    # loss, scheduled lr, etc..)
    # Also model / mp config

    """
    table_schema: {
        # metadata which describes the logged tensor
        'metadata' : {
            'name' : str # name of the layer
            'layer_type' : str # type of layer
            'tensor_type' : str|TensorType # type of tensor stats are for
            'it' : int # training iteration
            'dtype' : DType # piggy back of some library for this
            'dim' : Tuple[int] # Tensor Dimensions
        },
        # single value summaries of the tensor - what they are
        # named is irrelevant (can be user provided)
        # e.g. mean,std,mean_abs,rm2,ofc,ufc,...
        # ofc (over flow count), ufc (underflow count)
        'scalar_stats : {
            * : int | float
        }
        # what other special numbers??
        'exponent_counts: {
            -inf: int
            +inf: int
            * : int
        }
    }
    """
    _toplevels = ("metadata", "scalar_stats", "exponent_counts")
    # Use Optional for a column whether the column is optional?
    schema: Dict[str, Dict[Any, Any]] = {
        _toplevels[0]: {
            "name": str,
            "type": str,
            "tensor_type": Union[str, TT],
            "step": int,
            "dtype": str,
            # 'dim' : Any
        },
        _toplevels[1]: {WCAL1: Union[int, float]},
        _toplevels[2]: {float("-inf"): int, float("inf"): int, WCAL1: int},
    }

    @staticmethod
    def get_flat_schema():
        flat_schema, wilcards = {}, {}
        for k, v in LogFrame.schema.items():
            if type(v) is dict:
                for k_, v_ in v.items():
                    # slightly hacky as this doesn't differentiate between \
                    # classes and variables (that aren't subclasses of
                    # \WildCard)
                    if inspect.isclass(k_) and issubclass(k_, WildCard):
                        # could make this dict into a class?
                        wilcards[(k, k_)] = WildCardTracker(dtype=v_)
                    else:
                        flat_schema[(k, k_)] = v_

        return flat_schema, wilcards


@dataclass
class Event:
    name: str
    type: ModuleType  # type: ignore
    tensor_type: TT
    value: Any
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


@dataclass
class Stash:
    name: str
    type: ModuleType  # type: ignore
    tensor_type: TT
    dtype: DType
    value: Any

    @property
    def first_value(self) -> Any:
        def _value(v: Any) -> Any:
            if isinstance(v, (tuple, list)) and len(v) >= 1:
                return _value(v[0])
            return v

        return _value(self.value)


StashFn = Callable[[Event], Stash]


@dataclass
class TrainingStats:
    steps: List[int]
    train_loss: List[float]
    val_loss: Optional[List[float]] = None
