
import logging
import concurrent.futures
import os
from pathlib import Path
import pickle
import traceback
from types import TracebackType
import wandb
# rel imports
from ._log_handler import combine_incremental_dfs, global_stash_to_logframe, nuke_intermediate_logframes,summarise_logframe
from ..._config import _libname
from ._types import StashFn,Stash,Event
from ._write import lf_to_pickle,write_summary_bin_log

from typing import Any, Callable, Dict, Iterator, List, Optional, Type, Union, ByteString, Tuple
import randomname
logger = logging.getLogger(__name__)
import msgpack


def async_wrapper(f: Callable,name: str, step: int, object: ByteString):
    # function for writing to disk  
    try:
        object = msgpack.unpackb(object,strict_map_key=False)
    except Exception:
        logger.warning(traceback.format_exc())
    try:
        df = global_stash_to_logframe(object) #type: ignore
    except Exception as e:
        logger.warning(object)
        logger.warning(e)
    # writes to disk
    f(name,step,df)

    # only if wandb is being used..
    summary_dict = summarise_logframe(df)
    write_summary_bin_log(name=name, summary_dict=summary_dict)



class BaseTracker:
    def __init__(self, 
                 stash: StashFn, 
                 name: Union[str,None] = None,
                 init_step: Union[int,None]=None,
                 async_offload: bool = False,
                 offload_inc: int = 10,
                 offload_fn: Callable = lf_to_pickle,
                 use_wandb: bool = False):
        self.stashes: List[Stash] = []
        self._stash = stash
        self._global_stash: Dict[int,List[Dict]] = {}
        self._name = name if name != None else randomname.get_name() # run name
        self._offload_inc: int = offload_inc
        self._step: int = init_step if init_step else 0
        self.async_offload = async_offload
        self.offload_fn = offload_fn# could make this an init argument?
        self.use_wandb = use_wandb
        self.out_path = None

        if self.use_wandb:

            self.wandb_run = wandb.run
            self._artifact = wandb.Artifact(name=self._name, type=f"{_libname}-logframe")

            if not self.wandb_run: 
                raise Exception('No wandb run has been initialised and you have selected to use_wandb == True. Please initialise wandb run.')

        # Redundant not that I'm using concurrent.futures
        
        if async_offload:
            # spawn is preferable here as it eliminates race condition on _global_stash
            self._executor = concurrent.futures.ProcessPoolExecutor()

            if self.use_wandb:
                # create an empty binary file for logs
                p = Path(f"./{_libname}/{self._name}/")
                p.mkdir(parents=True, exist_ok=True)
                out = p / 'binlog.pkl'
                self.summary_bytes = self._name
                with open(out,'wb') as f:
                    pickle.dump(self.summary_bytes,f)


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
            self.offload_global_stash(final=True)

        # wait for all background processes 
        # to end before leaving tracker
        if self.async_offload:
            self._executor.shutdown()
            if self.use_wandb and self.wandb_run:
                p = Path(f"./{_libname}/{self._name}/lf")
                self._artifact.add_dir((str(p)))

        if self.use_wandb and self.wandb_run:
            self.wandb_run.log_artifact(self._artifact)

        output_path = combine_incremental_dfs(self._name)
        self.out_path = output_path
        print(f'The output LogFrame is available at: {output_path}')
        nuke_intermediate_logframes(self._name)
        

    def __str__(self) -> str:
        return f"Tracker(stashes={len(self)})"

    def __iter__(self) -> Iterator[Stash]:
        return iter(self.stashes)

    def __getitem__(self, index: int) -> Stash:
        return self.stashes[index]

    def __len__(self) -> int:
        return len(self.stashes)
    
    def stash_event(self, event: Event):
        self.stashes.append(
                self._stash(event)
        )
    
    def evict_global_stash(self):
        self._global_stash.clear()

    def evict_stash(self):
        self.stashes.clear()

    def move_to_global_stash(self):
        self._global_stash[self._step] = [stash.__dict__ for stash in self.stashes]


    def offload_global_stash(self,final: bool = False):
        # unimplemented as of yet, but where the stashes are converted to DF/Dict and offloaded to disk or sent to wandb
        # this should be configurable?
        
        if self.async_offload:
            self.launch_background_process()

        else:
            df = global_stash_to_logframe(self._global_stash)
            out = self.offload_fn(self._name,self._step,df)
            self.evict_global_stash()
            if self.use_wandb and self.wandb_run:
                
                summary_dict = summarise_logframe(df)
                # Log numerics stats without incremeting step
                if final:
                    wandb.log(
                        summary_dict[self._step-1],
                        step=self.wandb_run.step-1)
                    
                else:
                    wandb.log(
                        summary_dict[self.wandb_run.step],
                        step=self.wandb_run.step)
                    
                self._artifact.add_file(str(out))
                    
                
                
                

    def launch_background_process(self):
        # serialise and offload to background thread
        future = self._executor.submit(async_wrapper,self.offload_fn, self._name, self._step, msgpack.packb(self._global_stash)) #type: ignore
        self.evict_global_stash()

    def _read_summary_bin_log(self) -> Tuple[bool,Union[Dict[Any,Any],None]]:
        # May replace this with a shared memory buffer, if I can think of a good way to set the size effectly
        p = Path(f"./{_libname}/{self._name}/")
        out = p / 'binlog.pkl'
        with open(out,'rb') as f:
            summary_bytes = pickle.load(f)

        if summary_bytes != self.summary_bytes:
            # update summary bytes & deserialize
            self.summary_bytes = summary_bytes
            summary_dict: Dict = summary_bytes 

            return (True,summary_dict)

        else:
            return (False,None)
        

    def _internal_step(self,*args):
        # write stats to file?
        # clear stashes
        # (so should offload logs to file over wandb?)
        self.move_to_global_stash()
        # need to clear stashes at the end of every iteration (to keep torch compile happy as the hooks depend on it)

        self.evict_stash()

        if self._step % self._offload_inc == 0 and self._step >= 0:
            self.offload_global_stash()


        if self.async_offload and self.use_wandb and self.wandb_run:
            if self._step % self._offload_inc == self._offload_inc // 2:
                a, b = self._read_summary_bin_log()
                if a and b:
                    if self.wandb_run.step in b.keys():
                    # Log numerics stats without incremeting step
                        wandb.log(
                        b[self.wandb_run.step],
                        step=self.wandb_run.step)

                
                
        # increment step
        self._step += 1

    def step(self):
        raise NotImplementedError('This must be implemented for each framework (i.e. capture model \
                                  weight stats when step is called for torch.nn, etc..)\n \
                                  Also the of step must call _internal_step')