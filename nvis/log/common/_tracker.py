import logging
from mimetypes import init
import multiprocessing.context
import multiprocessing.context
from pathlib import Path
import pickle
from types import TracebackType

from click import launch
from ._types import StashFn,Stash
from typing import Any, Callable, Dict, Iterator, List, Optional, Type, Union
import randomname
from copy import deepcopy
import multiprocessing

def pickle_to_disk(name: str, step: int, object: Any):
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
        self._global_stash: Dict[int,List[Stash]] = {}
        self._name = name if name != None else randomname.get_name() # run name
        self._offload_inc: int = offload_inc
        self._step: int = init_step if init_step else 0
        self.async_offload = async_offload # requires if __name__ == '__main__'
        self.offload_fn: Callable = pickle_to_disk # could make this an init argument?
        if async_offload:
            # spawn is preferable here as it eliminates race condition on _global_stash
            self._mp_context: multiprocessing.context.SpawnContext = multiprocessing.get_context('spawn')
            self._processes: List[multiprocessing.context.SpawnProcess] = []

    def __enter__(self) -> "BaseTracker":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        logging.warning('TRACKER GONE OUT OF SCOPE')
        # check if global_stash is empty, 
        # if it isn't offload stash to disk/wandb
        if self._global_stash:
            self.offload_global_stash()
        
        # wait for all background processes 
        # to end before leaving tracker
        if self.async_offload:
            for p in self._processes:
                while p.is_alive():
                    ...

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
        self._global_stash[self._step] = deepcopy(self.stashes)

    def offload_global_stash(self):
        # unimplemented as of yet, but where the stashes are converted to DF/Dict and offloaded to disk or sent to wandb
        # this should be configurable?
        
        if self.async_offload:
            self.launch_background_process()
        else:
            self.offload_fn(self._name,self._step,self._global_stash)
            self.evict_global_stash()

    def launch_background_process(self):
        # should check that args for offload_fn match (self._name,self._step,self._global_stash)
        logging.warning(f'Background Processes{[_p for _p in self._processes if _p.is_alive()]}')
        p = self._mp_context.Process(
            target=self.offload_fn, 
            args=(self._name,self._step,self._global_stash))
        p.start()
        self._processes.append(
            p
        )
        self.evict_global_stash()


    def _internal_step(self):
        # write stats to file?
        # clear stashes
        # if self._model:
            # self._model_weights_hook()

        # (so should offload logs to file over wandb?)
        self.move_to_global_stash()
        # need to clear stashes at the end of every iteration (to keep torch compile happy as the hooks depend on it)
        self.evict_stash()

        if self._step % self._offload_inc == 0:
            self.offload_global_stash()

        # increment step
        self._step += 1

    def step(self):
        raise NotImplementedError('This must be implemented for each framework (i.e. capture model \
                                  weight stats when step is called for torch.nn, etc..)\n \
                                  Also the of step must call _internal_step')