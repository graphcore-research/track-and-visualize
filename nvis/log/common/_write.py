from typing import Dict
from ..._config import _libname
import pandas as pd
from pathlib import Path
import pickle
import time
import msgpack
import logging
logger = logging.getLogger(__name__)
import os
def lf_to_pickle(name: str, step: int, object: pd.DataFrame) -> Path:
    """
        Write the intermediate logrames to disk
    """
    
    p = Path(f"./{_libname}/{name}/lf")
    p.mkdir(parents=True, exist_ok=True)
    out= p / f'{step}-{hex(time.time_ns())}.pkl'
    with open(out, 'wb') as f:
        pickle.dump(object,f)
    
    return out



def write_summary_bin_log(name: str, summary_dict: Dict) -> None:
    # May replace this with a shared memory buffer, if I can think of a good way to set the size effectly
    p = Path(f"./{_libname}/{name}/")
    p.mkdir(parents=True, exist_ok=True)
    out = p / 'binlog.pkl'
    with open(out,'wb') as f:
        pickle.dump(summary_dict, f) #type: ignore
    