import logging
import concurrent.futures
from pathlib import Path
import pickle
from types import TracebackType

from ._types import StashFn,Stash
from typing import Any, Callable, Dict, Iterator, List, Optional, Type, Union
import randomname
from copy import deepcopy
import multiprocessing
import os
import tempfile

def async_wrapper(name: str, step: int, object: Any):
    object.seek(0)
    temp = pickle.loads(object.read())
    object.close()
    pickle_to_disk(name,step,temp)

def pickle_to_disk(name: str, step: int, object: Any):
    logging.warning(f'Offload PID: {os.getpid()}')
    p = Path("./nvis-logs")
    p.mkdir(parents=True, exist_ok=True)
    out= p / f'{name}-{step}.pkl'
    with open(out, 'wb') as f:
        pickle.dump(df,f)

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
        self.async_offload = async_offload


        # Redundant not that I'm using concurrent.futures
        if multiprocessing.current_process().name != "MainProcess" and async_offload:
            logging.warning('It would appear that your code is not main protected (i.e. if __name__ == "__main__":...) \n \
                            async_offload uses multiprocessing and if your main function is not protected, it will simply keep re-executing it')

        self.offload_fn: Callable = pickle_to_disk # could make this an init argument?
        if async_offload:
            # spawn is preferable here as it eliminates race condition on _global_stash
            self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)


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
            # do I need to wait for the future to be finished?
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
        # offload to temporary file before doing anything else
        tfile = tempfile.TemporaryFile()
        pickle.dump(self._global_stash,tfile)
        
        self._executor.submit(async_wrapper, self._name, self._step, tfile)
        self.evict_global_stash()


    def _internal_step(self):
        # write stats to file?
        # clear stashes
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