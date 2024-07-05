import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src._transform import _flatten_multi_index  # type: ignore
from ..log.common import TensorType
from ..log.common import _q
from src.log.common._utils import _validate_df_hash
from typing import List, Tuple, Optional, Union


def _get_fig():
    ...


"""
    Generates a heatmap of some scalar metric (w.r.t to the chosen Tensor \
        Type) for each layer.
        Y-axis:  Layer Name
        X-axis:  Time Step


    Args:
        mv (MasterView): the MV you wish to visualise
        title (str): Optional Plot title, 
        inc (int): the increment between training iterations to include
        figsize (Tuple[int, int]): Tuple of width, height for the size you \
              want the plot to be

    Returns:
        Figure: A Heatmap of the provided MasterView
"""


class _PlotPrepper:
    def __init__(self, df):
        _validate_df_hash(df)
        self.df = df
    

def scalar_global_heatmap(
        df: pd.DataFrame,
        tt: Union[TensorType, List[TensorType], None],
        scalar_metric: Union[str, List[str]],
        inc: int = 1,
        x=_q.IT,
        y=_q.NAME,
        col_wrap: int = None,
        **kwargs):
    """
    Plots a global view a single tensor type or a facet across multiple tensor types with respect to some scalar metrics.
    If tt = None or List[TensorType] it returns a faceted plot, if tt = TensorType then a single figure plot.

    Args:
         df (pd.DataFrame): the logs for the entire training run
         tt (TensorType | List[TensorType] | None): the type of tensor for the view to include (if not plots a facet of all tensor types)
         inc (int): the increment between training iterations to include
         scalar_metric (str | List[str]): which scalar metric to plot
         x : what to plot on x-axis
         y : what to plot on y-axis
         col_wrap: if faceting on tt or scalar metric set max-col width before wrapping
         **kwargs: all kwargs from sns.heatmap

         One of tt or scalar_metric must be a single value.
       

     Returns:
         Figure

    """

    assert (type(tt) != list or tt != None) and (type(scalar_metric)!= list or scalar_metric != None), 'Cannot Facet across both TensorType and scalar_metric, Please choose a single value for one'
    assert inc >= 1, 'inc must be a positive integer value'

    _validate_df_hash(df) 
    
    # Implement Logic for faceting on tt or scalar_metric
    if (type(tt) == list or tt == None) or (type(scalar_metric)== list or scalar_metric == None):
        raise NotImplementedError('Faceting not yet implemented!')

    df = df.query(
        f'@df.metadata.grad == "{tt.name}" & \
            @df.metadata.step % {inc} == 0')

    df = df[[x, y, _q.SCA(scalar_metric)]]

    df = _flatten_multi_index(df=df)

    fig = sns.heatmap(
    data=df.pivot(
        index=y[1],
        columns=x[1],
        values=scalar_metric),
        **kwargs)

    return fig



def scalar_line(
        df: pd.DataFrame,
        tt: Union[TensorType, List[TensorType], None],
        scalar_metric: Union[str,List[str],None] = None, # By Default
        x=_q.IT,
        **kwargs 
    ):
    """
    Plot a line of a scalar metric or metric(s) for a Tensor type or (types). 
    
    
    """




# HistoGram Plots

# Heatmaps 

# Kde (need to translate from bins counts to format for kde)