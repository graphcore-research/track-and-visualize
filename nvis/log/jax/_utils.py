from typing import Any, Callable, Dict, Sequence
from functools import partial
import jax
from optax._src.transform import NamedTuple
from optax._src.base import EmptyState
from ..common._types import Event

# largely copied/apapted from https://github.com/graphcore-research/jax-scalify/blob/32f81b951176e0085fec5c88be95f113d9cb2177/jax_scalify/ops/utils.py#L8
@partial(jax.custom_vjp, nondiff_argnums=(0,))
def forward_callback(f, *args):
    """Custom callback, called on gradients."""
    return args


def debug_callback_fwd(f, *args):
    f(*args)
    return args, None


def debug_callback_bwd(f, _, args_grad):
    return args_grad


forward_callback.defvjp(debug_callback_fwd, debug_callback_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def backward_callback(f, *args):
    """Custom callback, called on gradients."""
    return args


def backward_callback_fwd(f, *args):
    return args, None


def backward_callback_bwd(f, _, args_grad):
    f(*args_grad)
    return args_grad


backward_callback.defvjp(backward_callback_fwd, backward_callback_bwd)