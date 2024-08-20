from typing import Dict, List
import jax
import jax.numpy as jnp


# isinstance(x,jax.Array)

def exp_histogram(tensor: jax.Array, min_exp=-16, max_exp=16) -> Dict[str,List]:
    # moving from gpu to cpu?

    # jnp.frexp(tensor)
    e = jnp.frexp(tensor)[1].flatten()
    e = jnp.where(e < min_exp, (min_exp - 1) * jnp.ones_like(e), e)
    e = jnp.where(e > max_exp, (max_exp + 1) * jnp.ones_like(e), e)
    bins = jnp.asarray([i for i in range(min_exp-1, max_exp+2)],dtype=e.dtype)
    hist = jnp.histogram(e,bins=bins,range=(min_exp-1,max_exp+1))

    return dict(
        hist=hist[0].tolist(),
        bins=hist[1].tolist()
    )

def stash_full_tensor(tensor: jax.Array) -> Dict[str,jax.Array]:
    # moving from gpu to cpu?
    return {'full_tensor' : jnp.ones((1,1))}