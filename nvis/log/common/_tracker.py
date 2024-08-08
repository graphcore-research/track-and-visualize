import logging
import concurrent.futures
from pathlib import Path
import pickle
from types import TracebackType

from nvis.log.common._log_handler import global_stash_to_logframe

from ._types import StashFn,Stash
from typing import Any, Callable, Dict, Iterator, List, Optional, Type, Union, ByteString
import randomname
from copy import deepcopy
import multiprocessing
import os
import tempfile
logger = logging.getLogger(__name__)
import msgpack


def async_wrapper(f: Callable,name: str, step: int, object: ByteString):
    # function for writing to disk  
    f(name,step,msgpack.unpackb(object,strict_map_key=False))

def pickle_to_disk(name: str, step: int, object: Any):
    object = global_stash_to_logframe(object)
    p = Path("./nvis-logs")
    p.mkdir(parents=True, exist_ok=True)
    out= p / f'{name}-{step}.pkl'
    with open(out, 'wb') as f:
        pickle.dump(object,f)

class BaseTracker:
    def __init__(self, 
                 stash: StashFn, 
                 name: Union[str,None] = None,
                 init_step: Union[int,None]=None,
                 async_offload: bool = False,
                 offload_inc: int = 10):
        self.stashes: List[Stash] = []
        self._stash = stash
        self._global_stash: Dict[int,List[Dict]] = {}
        self._name = name if name != None else randomname.get_name() # run name
        self._offload_inc: int = offload_inc
        self._step: int = init_step if init_step else 0
        self.async_offload = async_offload
        self.offload_fn: Callable = pickle_to_disk # could make this an init argument?

        # Redundant not that I'm using concurrent.futures
        
        if async_offload:
            # spawn is preferable here as it eliminates race condition on _global_stash
            self._executor = concurrent.futures.ProcessPoolExecutor()


    def __enter__(self) -> "BaseTracker":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        # check if global_stash is empty, 
        # if it isn't offload stash to disk/wandb
        if self._global_stash:
            self.offload_global_stash()
        
        # wait for all background processes 
        # to end before leaving tracker
        if self.async_offload:
            self._executor.shutdown()

    def __str__(self) -> str:
        return f"Tracker(stashes={len(self)})"

    def __iter__(self) -> Iterator[Stash]:
        return iter(self.stashes)

    def __getitem__(self, index: int) -> Stash:
        return self.stashes[index]

    def __len__(self) -> int:
        return len(self.stashes)
    
    def evict_global_stash(self):
        self._global_stash.clear()

    def evict_stash(self):
        self.stashes.clear()

    def move_to_global_stash(self):
        self._global_stash[self._step] = [stash.__dict__ for stash in self.stashes]

    def offload_global_stash(self):
        # unimplemented as of yet, but where the stashes are converted to DF/Dict and offloaded to disk or sent to wandb
        # this should be configurable?
        
        if self.async_offload:
                self.launch_background_process()
        else:
            self.offload_fn(self._name,self._step,self._global_stash)
            self.evict_global_stash()

    def launch_background_process(self):
        # serialise and offload to background thread
        future = self._executor.submit(async_wrapper,self.offload_fn, self._name, self._step, msgpack.packb(self._global_stash)) #type: ignore

        self.evict_global_stash()


    def _internal_step(self):
        # write stats to file?
        # clear stashes
        # (so should offload logs to file over wandb?)
        self.move_to_global_stash()
        # need to clear stashes at the end of every iteration (to keep torch compile happy as the hooks depend on it)
        logging.warning(f'Global_Stash: {len(self._global_stash)}, Local Stash: {len(self.stashes)}')
        self.evict_stash()

        

        if self._step % self._offload_inc == 0 and self._step > 0:
            self.offload_global_stash()

        # increment step
        self._step += 1

    def step(self):
        raise NotImplementedError('This must be implemented for each framework (i.e. capture model \
                                  weight stats when step is called for torch.nn, etc..)\n \
                                  Also the of step must call _internal_step')