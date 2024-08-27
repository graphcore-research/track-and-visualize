# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
from typing import Dict, List, Union

import jax
import jax.numpy as jnp


def _concretize_as_list(value: Dict[str, jax.Array]) -> Dict[str, List]:
    value = jax.device_get(value)
    return {k: v.tolist() for k, v in value.items()}


def _concretize_as_scalar(value: Dict[str, jax.Array]) -> Dict[
        str, Union[int, float]]:
    value = jax.device_get(value)
    return {k: v.item() for k, v in value.items()}


def exp_histogram(tensor: jax.Array, min_exp=-16, max_exp=16) -> Dict[
        str, jax.Array]:
    """
    Gets the histogram of the exponents for the values in the jax.Array, \
        any thing over/under max_exp/min_exp will be set to +/-inf

    Args:
        t (jax.Array): The jax array you're getting the histogram from
        min_exp (int): the exponent underwhich you consider the values \
            to be underflowing
        max_exp (int): the exponent over which you consider the values \
            to be overflowing

    Returns:
        Dict : A dictionary containing the lists for \
            histogram counts & bin edges

    """

    # moving from gpu to cpu?
    tensor = jax.lax.stop_gradient(tensor)

    e = jnp.frexp(tensor)[1].flatten()
    e = jnp.where(e < min_exp, (min_exp - 1) * jnp.ones_like(e), e)
    e = jnp.where(e > max_exp, (max_exp + 1) * jnp.ones_like(e), e)
    bins = jnp.asarray([i for i in range(min_exp - 1, max_exp + 3)])
    hist = jnp.histogram(e, bins=bins, range=(min_exp - 1, max_exp + 1))

    # Import to call `jax.device_get`, to move to host an end jit trace
    return dict(hist=hist[0].astype(int), bins=hist[1].astype(int)[:-1])


def _stash_scalar_stats(tensor: jax.Array) -> Dict[str, jax.Array]:
    """
    Default stash value fn for gathering scalar stats, gets the mean, std, \
        abs_mean, abs_min, abs_max, rms (called rm2), rm4 & rm8

    Args:
        tensor (jax.Array): The jax array you're getting the stats for.

    Returns:
        Dict : A dictionary of the containing the scalar values for \
            various statistics
        ```python
        {'stat1' : 0.0,...}
        ```


    """
    tensor = jax.lax.stop_gradient(tensor)
    rm2 = jnp.pow(jnp.mean(jnp.pow(tensor, 2)), 1 / 2)
    abs_t = jnp.abs(tensor)
    # Import to call `jax.device_get`, to move to host an end jit trace
    return {
        "mean": tensor.mean(),
        "std": tensor.std(),
        "mean_abs": abs_t.mean(),
        "max_abs": abs_t.max(),
        "min_abs": abs_t.min(),
        "rm2": rm2,
        "rm4": jnp.pow(jnp.pow(tensor.__div__(rm2),
                               4).mean(), 1 / 4).__mul__(rm2),  # type: ignore
        "rm8": jnp.pow(jnp.pow(tensor.__div__(rm2),
                               8).mean(), 1 / 8).__mul__(rm2),  # type: ignore
    }


def stash_scalar_stats(tensor: jax.Array) -> Dict[str, Union[int, float]]:
    return _concretize_as_scalar(_stash_scalar_stats(tensor))


def stash_hist(tensor: jax.Array, min_exp=-16, max_exp=16) -> Dict:
    """
    Get the exponent histogram for the Tensor, values under min_exp and \
        over max_exp will be set to +/- infitity.

    Args:
        tensor (jax.Array): The jax Array you're getting the histogram

    Returns:
        Dict : A dictionary of the containing the scalar statistics \
            and the Exponent Histogram for the tensor.
        ```python
        {'exp_hist' : {'hist' : [...], 'bins' : [... ]}}
        ```

    """
    return {"exp_hist": _concretize_as_list(exp_histogram(
        tensor, min_exp, max_exp))}


def stash_all_stats_and_hist(tensor: jax.Array) -> Dict[str, Dict]:
    """
    The default stash value function, and the one which is most compatible \
        with the visualisation library.

    Args:
        tensor (jax.Array): The jax Array you're getting the the stastics for.

    Returns:
        Dict : A dictionary of the containing the scalar statistics \
            and the Exponent Histogram for the tensor.
        ```python
        {'scalar_stats' {'stat1' : 0.0,...}, \
            'exp_hist' : {'hist' : [...], 'bins' : [... ]}}
        ```
    """
    return {
        "scalar_stats": stash_scalar_stats(tensor=tensor),
        "exp_hist": _concretize_as_list(exp_histogram(tensor=tensor)),
    }


def stash_full_tensor(tensor: jax.Array) -> Dict[str, jax.Array]:
    tensor = jax.device_get(jax.lax.stop_gradient(tensor).copy())

    return {"full_tensor": tensor}
