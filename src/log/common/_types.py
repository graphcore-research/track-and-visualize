from enum import Enum
import pandas as _pd
from typing import Optional, Dict, Any
# from abc import ABC

"""
    Data Structures for logging
"""

# The can be used for generating queries, (all data stored in log)


class TensorType(str, Enum):
    Activation = 'Activation'
    Weight = 'Weight'
    Gradient = 'Gradient'
    OptimiserState = 'OptimiserState'


class _BaseFrame:
    def __init__(self, df):
        self._df = df

    def _ipython_display_(self):
        # just want to display internal dataframe...
        print(self._df)

    @staticmethod
    def _assert_schema(schema: Dict, df: _pd.DataFrame):
        'implement schema checking logic..'
        return True


class LogFrame(_BaseFrame):
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
            # naming is irrelevant (can be user provided)
            'scalar_stats : {
                * : int | float # e.g. mean,std,mean_abs,rm2,ofc,ufc,...
                # ofc (over flow count), ufc (underflow count)
            }
            # what other special numbers??
            'exponent_counts: {
                -inf: int
                +inf: int
                * : int
            }
        }
    """
    # static schema variable..
    _schema: Dict[str, Any] = {}

    def __init__(self, df: _pd.DataFrame):
        # Could be replaced with schema mismatch exception in the base class
        assert LogFrame._assert_schema(
            schema=LogFrame._schema,
            df=df), f'Provided DataFrame does not match {LogFrame.__name__} \
                Schema: {LogFrame._schema}'
        
        super().__init__(df=df)

    # could use generics?
    @staticmethod
    def from_pickle(path: str, map: Optional[Dict] = None) -> "LogFrame":
        df = _pd.read_pickle(path)
        LogFrame._assert_schema(
            LogFrame._schema,
            df=df
        )
        return LogFrame(df=df)


class MasterView(_BaseFrame):
    """
        A view of 
        index: layer-name
        column: n_{it}

    """
    _schema: Dict[str, Any] = {}

    def __init__(self, df: _pd.DataFrame, tt: TensorType, metric: str):
        self.tt = tt
        self.metric = metric
        # Could be replaced with schema mismatch exception in the base class
        assert MasterView._assert_schema(
            schema=MasterView._schema,
            df=df), f'Provided DataFrame does not match {MasterView.__name__} \
                Schema: {MasterView._schema}'
        
        super().__init__(df=df)