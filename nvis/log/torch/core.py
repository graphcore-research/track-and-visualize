# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
# Code adapated from tensor_tracker

"""Utility for tracking activations and gradients at `nn.Module` outputs.

Use `track` to start tracking a module & submodules. Then use the original module
as usual. Your `Tracker` will be filled with a list of `Stash`es, containing
copies of fwd/bwd tensors at (sub)module outputs. (Beware, this can consume
a lot of memory.)

Usage ([notebook](usage.html)):

```
with tensor_tracker.track(model) as tracker:
    model(inputs).backward()

print(list(tracker))
# => [Stash(name="0.linear", type=nn.Linear, grad=False, value=tensor(...)),
#     ...]

display(tracker.to_frame())  # requires 'pandas'
```

Advanced usage:

 - Filter modules based on name:
   `track(include="<regex>", exclude="<regex>")`

 - Pre-transform tracked tensors to save memory:
   `track(stash_value=lambda t: t.std().detach().cpu())`

 - Customise tracked state:
   `track(stash=lambda event: ...)`

 - Manually register/unregister hooks:
  `tracker = Tracker(); tracker.register(...); tracker.unregister()`

See also: [example of
visualising transformer activations & gradients using UMAP](example.html).
"""

import dataclasses
import logging
import re
from dataclasses import dataclass
from functools import partial
from types import TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Pattern,
    Tuple,
    Type,
    Union,
    Literal
)

import torch.utils.hooks
from torch import Tensor, nn
from copy import deepcopy

from ..common._tracker import BaseTracker
from .stash_functions import stash_full_tensor
from ..common._types import Stash, Event, StashFn
import randomname
from pathlib import Path
import pickle


TT = Literal['Activation', 'Gradient', 'Weights', 'Optimiser_State']


StashValueFn = Callable[[torch.Tensor], Any]

def rmap_tensor(value: Any, fn: Callable[[Tensor], Any]) -> Any:
    if isinstance(value, (tuple, list)):
        return type(value)(rmap_tensor(a, fn) for a in value)
    if isinstance(value, dict):
        return {rmap_tensor(k, fn): rmap_tensor(a, fn) for k, a in value.items()}
    # if dataclasses.is_dataclass(value): TORCH COMPILE DOES NOT LIKE THIS
    #     return type(value)(**{k: rmap_tensor(v, fn) for k, v in value.__dict__.items()})
    if isinstance(value, Tensor):
        return fn(value)
    return value


def tensor_dtype(tensor: torch.Tensor) -> torch.dtype:
    return tensor.dtype

def default_stash(event: Event, stash_value: StashValueFn) -> Stash:
    return Stash(
        name=event.name, 
        type=event.type, 
        tensor_type=event.tensor_type,
        dtype=rmap_tensor(event.value, tensor_dtype),
        value=rmap_tensor(event.value, stash_value)
    )


def get_stash_fn(
    stash_value: Optional[StashValueFn] = None, stash: Optional[StashFn] = None
) -> StashFn:
    if stash_value and stash:
        raise ValueError("Cannot provide StashValueFn and StashFn to get_stash_fn()")
    if stash:
        return stash
    return partial(default_stash, stash_value=stash_value or stash_full_tensor)


NamePattern = Union[None, Pattern[str], str]

class TorchTracker(BaseTracker):
    def __init__(self, stash: Callable[[Event], Stash], name: str | None = None, init_step: int | None = None,async_offload:bool = True, offload_inc: int = 10):
        super().__init__(stash, name, init_step, async_offload, offload_inc)
        self._handles: List[torch.utils.hooks.RemovableHandle] = [] # torch specific
        self._model: Union[torch.nn.Module,None] = None # torch specific

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        # logging.warning('TRACKER GONE OUT OF SCOPE')
        # # check if global_stash is empty, if it isn't offload stash to disk/wandb
        # if self._global_stash:
        #     self.offload_global_stash()
        self.unregister()
        super().__exit__(exc_type,exc,traceback)

    # REGISTERING ENTITIES TO BE TRACKED
    def register(self, module: nn.Module, name: str = "", grad: bool = True) -> None:
        self._handles.append(
            module.register_forward_hook(
                partial(self._forward_hook, name=name), with_kwargs=True
            )
        )
        if grad:
            self._handles.append(
                module.register_full_backward_pre_hook(
                    partial(self._backward_hook, name=name)
                )
            )

    def register_optimiser(self,optimizer: torch.optim.Optimizer, param_names: List[str]) -> None:
        self._handles.append(
            optimizer.register_step_pre_hook(partial(self._optim_step_hook,p_names=param_names))
            )
    
    def register_weights(self,model: nn.Module):
        self._model = model

    def register_all(
        self,
        module: nn.Module,
        grad: bool = True,
        include: NamePattern = None,
        exclude: NamePattern = None,
    ) -> None:
        include = re.compile(include) if isinstance(include, str) else include
        exclude = re.compile(exclude) if isinstance(exclude, str) else exclude
        for name, child in module.named_modules():
            if ((not include) or include.search(name)) and not (
                exclude and exclude.search(name)
            ):
                self.register(child, name, grad=grad)

    def unregister(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # HOOKS WHICH ARE USED TO CAPTURE TENSOR STATS
    def _forward_hook(
        self,
        module: nn.Module,
        args: Tuple[Any],
        kwargs: Dict[str, Any],
        output: Any,
        *,
        name: str,) -> None:
        self.stashes.append(
            self._stash(Event(name, str(type(module)), 'Activation', output, (), {}))
        )

    def _backward_hook(self, 
                       module: nn.Module, 
                       grad_output: Any, 
                       *, 
                       name: str) -> None:
        self.stashes.append(
            self._stash(Event(name, str(type(module)), 'Gradient', grad_output, (), {}))
        )

    def _optim_step_hook(
            self,
            optimizer: torch.optim.Optimizer, 
            *args, 
            **kwargs):

        for pn, state in zip(kwargs.get('p_names',[]),optimizer.state_dict()['state'].values()):
            for k,v in state.items():
                if k != 'step':
                    self.stashes.append(self._stash(Event(pn.removesuffix('.weight'),None,f'Optimiser_State.{k}',v,(),{}))) #type: ignore

    def _model_weights_hook(self):
        if self._model:
            for name,params in self._model.named_parameters():
                self.stashes.append(self._stash(Event(name.removesuffix('.weight'),None,'Weights',params.data,(),{})))


    def step(self):
        # write stats to file?
        # clear stashes
        if self._model:
            self._model_weights_hook()
        
        self._internal_step()

            
def track(
    module: nn.Module,
    grad: bool = True,
    optimiser: Union[torch.optim.Optimizer,None] = None,
    track_weights: bool = True,
    include: NamePattern = None,
    exclude: NamePattern = None,
    stash_value: Optional[StashValueFn] = None,
    stash: Optional[StashFn] = None,
    async_offload: bool = False,
    use_wandb: bool = False,
    wandb_kws: Optional[Dict] = None,
    ) -> TorchTracker:

    assert not (use_wandb and wandb_kws!= None), 'Must provide wandb_kws use_wandb==True to init the wandb run'

    tracker = TorchTracker(get_stash_fn(stash_value=stash_value, stash=stash),async_offload=async_offload)
    tracker.register_all(module, grad=grad, include=include, exclude=exclude)
    if optimiser:
        tracker.register_optimiser(optimiser, param_names = [m for m,p in module.named_parameters()])
    if track_weights:
        tracker.register_weights(model=module)
    return tracker


track.__doc__ = __doc__

__all__ = [
    "Event",
    "Stash",
    "StashFn",
    "StashValueFn",
    "rmap_tensor",
    "get_stash_fn",
    "TorchTracker",
    "track",
]
