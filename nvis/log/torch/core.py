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

from stash_functions import stash_full_tensor

TT = Literal['Activation', 'Gradient', 'Weights', 'Optimiser_State']


@dataclass
class Event:
    name: str
    type: Type[nn.Module]
    tensor_type: TT
    value: Any
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

@dataclass
class Stash:
    name: str
    type: Type[nn.Module]
    tensor_type: TT
    dtype: torch.dtype
    value: Any  # output(s) or grad_output(s)

    @property
    def first_value(self) -> Any:
        def _value(v: Any) -> Any:
            if isinstance(v, (tuple, list)) and len(v) >= 1:
                return _value(v[0])
            return v

        return _value(self.value)



StashFn = Callable[[Event], Stash]
StashValueFn = Callable[[Tensor], Any]


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


class Tracker:
    def __init__(self, stash: StashFn):
        self.stashes: List[Stash] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._stash = stash
        self._model = None
        self._step = 0
        self._global_stash = {}

    # Registration/tracking

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.unregister()

    def clear(self) -> None:
        self.stashes.clear()

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

    def _forward_hook(
        self,
        module: nn.Module,
        args: Tuple[Any],
        kwargs: Dict[str, Any],
        output: Any,
        *,
        name: str,
    ) -> None:
        # Check if this fixes bwd pass
        self.stashes.append(
            self._stash(Event(name, None, 'Activation ', output, (), {}))
        )

        # exp_histogram(output)

    def _backward_hook(self, module: nn.Module, grad_output: Any, *, name: str) -> None:
        self.stashes.append(
            self._stash(Event(name, None, 'Gradient', grad_output, (), {}))
        )

    # Read results

    def __str__(self) -> str:
        return f"Tracker(stashes={len(self)}, tracking={len(self._handles)})"

    def __iter__(self) -> Iterator[Stash]:
        return iter(self.stashes)

    def __getitem__(self, index: int) -> Stash:
        return self.stashes[index]

    def __len__(self) -> int:
        return len(self.stashes)

    def _optim_step_hook(self,optimizer: torch.optim.Optimizer, *args, **kwargs):
        for pn, state in zip(kwargs.get('p_names'),optimizer.state_dict()['state'].values()):
            
            for k,v in state.items():
                if k != 'step':
                    self.stashes.append(self._stash(Event(pn.removesuffix('.weight'),None,f'Optimiser_State.{k}',v,(),{})))

    def register_optimiser(self,optimizer: torch.optim.Optimizer, param_names: List[str]) -> None:

        self._handles.append(optimizer.register_step_pre_hook(partial(self._optim_step_hook,p_names=param_names)))
    
    def register_weights(self,model: nn.Module):
        self._model = model

    def evict_stash(self):
        self.stashes = []


    def offload_global_stash(self):
        # unimplemented as of yet, but where the stashes are converted to DF/Dict and offloaded to disk or sent to wandb
        ...


    def move_to_global_stash(self):
        self._global_stash[self._step] = deepcopy(self.stashes)


    def step(self, call_item = True):
        # write stats to file?
        # clear stashes
        if self._model:
            for name,params in self._model.named_parameters():
                self.stashes.append(self._stash(Event(name.removesuffix('.weight'),None,f'Weights',params.data,(),{})))

        # (so should offload logs to file over wandb?)
        self.move_to_global_stash()
        # need to clear stashes at the end of every iteration (to keep torch compile happy as the hooks depend on it)
        self.evict_stash()
        # increment step
        self._step += 1

            
def track(
    module: nn.Module,
    grad: bool = True,
    optimiser: torch.optim.Optimizer = None,
    track_weights: bool = True,
    include: NamePattern = None,
    exclude: NamePattern = None,
    stash_value: Optional[StashValueFn] = None,
    stash: Optional[StashFn] = None,) -> Tracker:

    tracker = Tracker(get_stash_fn(stash_value=stash_value, stash=stash))
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
    "default_stash_value",
    "default_stash",
    "get_stash_fn",
    "Tracker",
    "track",
]
