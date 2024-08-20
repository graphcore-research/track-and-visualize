from typing import Any, Callable, Dict, Literal
from ..common._tracker import BaseTracker
from ..common._types import Event, StashFn,Stash
from ..common._write import lf_to_pickle
import jax
import jax.numpy as jnp
import numpy as np
from ._utils import forward_callback,backward_callback





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


    def data_interceptor(self,next_fun, args, kwargs, context):
        """
            Captures the output of each linen module.
        
        """

        # self.stash_event()

        assert len(args) == 1
        
        output = next_fun(*args, **kwargs)

        path = "/".join(context.module.path)
        forward_callback(lambda v: print(f"Tensor fwd '{path}': {v}"), *output)
        output = backward_callback(lambda v: print(f"Tensor bwd '{path}': {v}"), output)
        
        assert len(output) == 1
        # backward callback returns a tuple of len 1, where as the module is expecting what's in that tuple (index it outß)
        return output[0]
        
    def step(self, model_state, optimizer_state) -> None:

        # capture model_state

        # capture opt_state

        self._internal_step()
        ...

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
    
    tracker = JaxTracker(
        stash=default_stash,
        async_offload=async_offload,
        offload_inc=offload_inc,
        offload_fn=offload_fn[offload_type],
        use_wandb=use_wandb,
        init_step=init_step
        )

    # implement tracking hooks, etc..

    # can pass in optimiser state, model_state.
    # just need to capture activations, gradients..

    return tracker