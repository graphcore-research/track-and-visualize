# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
# Code adapated from tensor_tracker

import re
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

from ..common._tracker import BaseTracker
from ..common._write import lf_to_pickle
from .stash_values import stash_all_stats_and_hist
from ..common._types import Stash, Event, StashFn

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


def tensor_dtype(tensor: torch.Tensor) -> str:
    return str(tensor.dtype)

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
    return partial(default_stash, stash_value=stash_value or stash_all_stats_and_hist)


NamePattern = Union[None, Pattern[str], str]



class TorchTracker(BaseTracker):
    def __init__(self, 
                 stash: Callable[[Event], Stash], 
                 async_offload: bool, 
                 only_stash_during_training: bool, 
                 offload_inc: int,offload_fn: Callable, 
                 use_wandb:bool, name: str | None = None, 
                 init_step: int | None = None):
        super().__init__(stash, name, init_step, async_offload, offload_inc, offload_fn, use_wandb)
        self._handles: List[torch.utils.hooks.RemovableHandle] = [] # torch specific
        self._model: Union[torch.nn.Module,None] = None # torch specific
        self.only_stash_during_training = only_stash_during_training
        self.track_gradients: bool = False # torch specific

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.unregister()
        super().__exit__(exc_type,exc,traceback)

    # REGISTERING ENTITIES TO BE TRACKED
    def register(self, module: nn.Module, name: str = "", grad: bool = True) -> None:
        self._handles.append(
            module.register_forward_hook(
                partial(self._forward_hook_v2 if self.only_stash_during_training else self._forward_hook_v1, name=name), with_kwargs=True
            )
        )
        if grad:
            self._handles.append(
                module.register_full_backward_pre_hook(
                    partial(self._backward_hook, name=name)
                )
            )

    def register_optimiser(self,optimizer: torch.optim.Optimizer, param_names: List[str]) -> None: #type: ignore
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
        self.track_gradients = grad
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
    def _forward_hook_v1(
        self,
        module: nn.Module,
        args: Tuple[Any],
        kwargs: Dict[str, Any],
        output: Any,
        *,
        name: str,) -> None:
        # only do stashes when training?
        self.stashes.append(
                self._stash(Event(name, str(type(module)), 'Activation', output, (), {}))
            )

    def _forward_hook_v2(
        self,
        module: nn.Module,
        args: Tuple[Any],
        kwargs: Dict[str, Any],
        output: Any,
        *,
        name: str,) -> None:
        # only do stashes when training?
        if module.training:
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
            optimizer: torch.optim.Optimizer, #type: ignore
            *args, 
            **kwargs):

        for pn, state in zip(kwargs.get('p_names',[]),optimizer.state_dict()['state'].values()):
            for k,v in state.items():
                if k != 'step':
                    self.stashes.append(self._stash(Event(pn.removesuffix('.weight'),None,f'Optimiser_State.{k}',v,(),{}))) #type: ignore
                    

    def _stash_model_weights(self):

        if self.only_stash_during_training:
            if self._model and self._model.training:
                for name,params in self._model.named_parameters():
                    self.stashes.append(self._stash(Event(name.removesuffix('.weight'),None,'Weights',params.data,(),{})))
                    if self.track_gradients and params.grad != None:
                        self.stashes.append(self._stash(Event(name.removesuffix('.weight'),None,'Weight_Gradients',params.grad,(),{})))
                            
        else:
            if self._model:
                for name,params in self._model.named_parameters():
                    self.stashes.append(self._stash(Event(name.removesuffix('.weight'),None,'Weights',params.data,(),{})))
                    if self.track_gradients and params.grad != None:
                        # If grads aren't zeroed before the evaluation loop, this could store several duplicates of the weight grad
                        # perhaps should check train flag?
                        self.stashes.append(self._stash(Event(name.removesuffix('.weight'),None,'Weight_Gradients',params.grad,(),{})))


    def step(self):
        if self._model:
            self._stash_model_weights()
        
        self._internal_step()

            
def track(
    module: nn.Module,
    track_gradients: bool = True,
    optimizer: Union[torch.optim.Optimizer,None] = None, #type: ignore
    track_weights: bool = True,
    include: NamePattern = None,
    exclude: NamePattern = None,
    stash_value: Optional[StashValueFn] = None,
    async_offload: bool = False,
    offload_inc: int = 10,
    offload_type: Literal['.pkl'] = '.pkl',
    use_wandb: bool = False,
    only_stash_during_training = True,
    init_step =None,
    ) -> TorchTracker:
    """
        Function for initialising the Pytorch Tensor Tracker context manager.

        By default it tracks the stastics for the what we refer to as the Activations (i.e. the outputs of the forward method in the nn module). However it can also track 
        gradients, weights and optimiser state. At the end of training it will write all the logs to a LogFrame (a `pd.DataFrame` which conforms to the schema outline in our docs) 


        ```python
            with track(...) as tracker:
                #Your training loop goes here
                for i in range(max_its):
                    # fwd & bwd pass
                    tracker.step() # at the end of your loop
        
        ```

        Args:
            module (nn.Module): The top level module you wish to track, this would typically be the root class your use to define your model, the tracker will then recursively find all the submodules and add tracking hooks to their respective tensors
            track_gradients (bool): Whether or not you wish to track the gradients.
            optimizer (torch.optim.Optimizer | None): If you wish to track the optimiser state, pass it as an argument.
            track_weights (bool): Whether or not to track the models weights/parameters.
            include (None | Pattern[str] | str) : A module or modules (via regex) you wish to track.
            exclude (None | Pattern[str] | str) : A module or modules (via regex) you wish not to track.
            stash_value (StashValueFn): This is the statistics you wish to track, it defaults to `stash_all_stats_and_hist`, you can provide a custom fn here however inspect the other stash_fns to see the required args/returns values.
            async_offload (bool): If true the set of stashes since last offloaded are serialised and passed to a seperate python process to be converted to a Logframe (currently a very limited reduction in overhead, but working on improving it)
            offload_inc (int): How frequently you wish to the stashes from memory to disk, i.e. offload more frequently to minimise Torch Tracker's memory usage. If using wandb, this value should be the same (or a multiple) of the increment being used to call `wandb.log'
            offload_type (Literal['.pkl']): The file format you wish to write the LogFrame(s) to disk as.
            use_wandb (bool): If you wish to push the Logframes as artifacts and get summary numerics statistic in wandb. (`offload_inc` should be the same (or a multiple) of the increment being used to call `wandb.log')
            only_stash_during_training (bool): Torch Tracker can track statistics for Activations when the model is in eval mode (the default behaviour is for this not to be the case and only track activations during training)
            init_step (int): The tracker has an internal step property for assigning statistics to the correct iteration, if `init_step == None`, defaults to zero, else init_step (if you are continuing from a checkpoint for exampkle)


        Returns:
            TorchTracker (The context manger)
    
    
    """

    # assert not (use_wandb and wandb_kws!= None), 'Must provide wandb_kws use_wandb==True to init the wandb run'
    # Check if wandb is logged in and a run has been initialised

    offload_fn: Dict[str,Callable] = {'.pkl' : lf_to_pickle}

    tracker = TorchTracker(
        get_stash_fn(stash_value=stash_value, stash=None),
        async_offload=async_offload,
        only_stash_during_training=only_stash_during_training,
        offload_fn=offload_fn[offload_type],
        init_step=init_step,
        use_wandb=use_wandb,
        offload_inc=offload_inc)
    
    tracker.register_all(module, grad=track_gradients, include=include, exclude=exclude)
    if optimizer:
        tracker.register_optimiser(optimizer, param_names = [m for m,p in module.named_parameters()])
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
