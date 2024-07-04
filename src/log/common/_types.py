from enum import Enum
import pandas as _pd
from typing import Optional, Dict, Any, Union, TypeVar
from dataclasses import dataclass
import inspect
# from pandas._typing import 
# from abc import ABC

"""
    Data Structures for logging
"""

# The can be used for generating queries, (all data stored in log)

class WildCard:
    ...
    def __str__(self):
        return '*'

class WCAL1(WildCard):
    min: int =  0
    max: int = 2**32

@dataclass
class WildCardTracker:
    dtype: Any
    count: int = 0

class TensorType(str, Enum):
    Activation = 'Activation'
    Weight = 'Weight'
    Gradient = 'Gradient'
    OptimiserState = 'OptimiserState'

@dataclass
class SchemaMap:
    """
        For mapping from the Table format of your Logs to the format expected by the downstream tools in the library.
    """
    metadata: Dict
    scalar_stats: Union[Dict[str,str], None]
    exponent_counts: Union[Dict[str,str],None]
    # tl_index_map:

    def __post_init__(self):
        # both scalar_stats and exponent_counts can't be None -> 
        assert self.scalar_stats != None or self.exponent_counts != None, 'A mapping must be provided for either scalar_stats or exponent_counts'
        
        # assert that keys provided exist in the logframe schema keys
        sp = set(self.metadata.keys())
        ss = set(LogFrame.schema['metadata'].keys())
        assert sp.issubset(ss), f'The keys: {",".join(list(sp.difference()))} , are not in the allowed set: {",".join(list(ss))}'

        

class LogFrame:
    # Need to attach additional metadata that is per iteration (i.e. train/val 
    # loss, scheduled lr, etc..)
    # Also model / mp config

    """
        table_schema: {
            # metadata which describes the logged tensor
            'metadata' : {
                'name' : str # name of the layer
                'layer_type' : str # type of layer
                'tensor_type' : str|TensorType # type of tensor stats are for
                'it' : int # training iteration
                'dtype' : DType # piggy back of some library for this
                'dim' : Tuple[int] # Tensor Dimensions
            },
            # single value summaries of the tensor - what they are
            # named is irrelevant (can be user provided)
            # e.g. mean,std,mean_abs,rm2,ofc,ufc,...
            # ofc (over flow count), ufc (underflow count)
            'scalar_stats : {
                * : int | float 
            }
            # what other special numbers??
            'exponent_counts: {
                -inf: int
                +inf: int
                * : int
            }
        }
    """
    _toplevels = ('metadata','general_stats','exponent_count')
    # Use Optional for a column whether the column is optional?
    schema: Dict[str, Dict[Any,Any]] = {
        _toplevels[0] : {
            'name': str,
            'type' : str,
            'grad' : Union[str,TensorType],
            'step': int,
            # 'dtype': Any,
            # 'dim' : Any
        },
        _toplevels[1]  : {
            WCAL1 : Union[int, float]
        },
        _toplevels[2]  : {
            float('-inf') : int,
            float('inf') : int,
            WCAL1 : int 
        }
    }


    @staticmethod
    def get_flat_schema():
        flat_schema,wilcards = {},{}
        for k,v in LogFrame.schema.items():
            if type(v) == dict:
                for k_, v_ in v.items():
                    # slightly hacky as this doesn't differentiate between classes and variables (that aren't subclasses of WildCard)
                    if inspect.isclass(k_) and issubclass(k_,WildCard):
                        # could make this dict into a class?
                        wilcards[(k,k_)] = WildCardTracker(dtype=v_)
                    else:
                        flat_schema[(k,k_)] = v_



        return flat_schema, wilcards
    
