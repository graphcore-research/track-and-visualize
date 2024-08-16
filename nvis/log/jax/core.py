from typing import Any, Callable, Dict, Literal
from ..common._tracker import BaseTracker
from ..common._types import Event, StashFn,Stash
from ..common._write import lf_to_pickle
import jax
import jax.numpy as jnp
import numpy as np





class JaxTracker(BaseTracker):
    def __init__(self, 
                 stash: Callable[[Event], Stash], 
                 async_offload: bool, 
                 offload_inc: int,
                 offload_fn: Callable, 
                 use_wandb:bool, name: str | None = None, 
                 init_step: int | None = None):
        super().__init__(
            stash=stash,
            name=name,
            init_step=init_step,
            async_offload=async_offload,
            offload_inc=offload_inc,
            offload_fn=offload_fn,
            use_wandb=use_wandb)
        ...
        
    # capture and store intermediate values, fwd/bwd pass

    # def stash_opt_state():
    #     ...

def default_stash(event: Event) -> Stash:

    # event.name,event.tensor_type,event.type,event.value
    
    return Stash(
        name = '',
        type = '',
        tensor_type='Activation',
        dtype=jnp.float16,
        value = ()
    )

def track(
        module: Any,
        track_gradients: bool = True,
        model_state: Dict = {},
        optimizer_state:  Union[Dict,None] = None, #type: ignore
        track_weights: bool = True,
        # include: NamePattern = None,
        # exclude: NamePattern = None,
        # stash_value: Optional[StashValueFn] = None,
        async_offload: bool = False,
        offload_inc: int = 10,
        offload_type: Literal['.pkl'] = '.pkl',
        use_wandb: bool = False,
        only_stash_during_training = True,
        init_step =None,

) -> JaxTracker:
    
    offload_fn: Dict[str,Callable] = {'.pkl' : lf_to_pickle}
    
    tracker = JaxTracker(stash=default_stash,async_offload=async_offload,offload_inc=offload_inc,)

    # implement tracking hooks, etc..

    # can pass in optimiser state, model_state.
    # just need to capture activations, gradients..

    return tracker