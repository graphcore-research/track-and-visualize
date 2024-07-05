import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src._transform import _flatten_multi_index  # type: ignore
from ..log.common import TensorType
from ..log.common import _q
from ._errors import FacetException
from src.log.common._utils import _validate_df_hash
from typing import List, Tuple, Optional, Union, Pattern


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
         tt (TensorType | List[TensorType] | None): the type of tensor for the view to include (if None plots a facet of all tensor types)
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

    if ((type(tt) == list or tt == None) and (type(scalar_metric) == list or scalar_metric == None)):
        raise FacetException('Cannot Facet across both TensorType and scalar_metric, Please choose a single value for one')
    if inc <= 1:
        raise ValueError('inc must be a positive integer value')

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
        layer: Union[str, List[str], Pattern],
        tt: Union[TensorType, List[TensorType], None],
        scalar_metric: Union[str,List[str]], # By Default
        x=_q.IT,
        col_wrap: int = None,
        kind='line',
        facet_kws = {'sharey': True, 'sharex': True},
        **kwargs 
    ):
    """
    Plot a line for a scalar metric or metric(s) for a Tensor type or (types: facted plot) of a single layer or set of layer(s).

    One of tt or layer_name must be a single value.

    Args:
         df (pd.DataFrame): the logs for the entire training run
         layer (str | List[str] | Pattern): a str (single layer), set of str or a compiled regex of a set of names to plot (each layer is a facet)
         tt (TensorType | List[TensorType] | None): the type of tensor for the view to include (if None plots a facet of all tensor types)
         scalar_metric (str | List[str]): which scalar metric or metrics to plot
         x : what to plot on x-axis (defaults to training step)
         col_wrap (int | None): if faceting on tt or scalar metric set max-col width before wrapping
         kind: (str): Typle of plot to display accepts args "line" or "scatter"
         facet_kws (Dict): Arguements for sns.FacetGrad, if faceting on TensorType,  sharey is automatically set to False (as grads and activations typically on very different scales)
         **kwargs: all kwargs from sns.relplot (if faceting) or sns.lineplot/sns.scatterplot if

         One of tt or scalar_metric must be a single value.
       

     Returns:
         Figure

    """

    if ((type(layer) == list or isinstance(layer,Pattern)) and (type(tt) == list or tt == None)):
        FacetException('Cannot Facet or both Tensor Type and layer, one must be a single value')

    _validate_df_hash(df)

    facet = None
    # Logic for faceting by layer
    if type(layer) == list:
        lq = f' in {layer}'
        facet = _q.NAME
    elif isinstance(layer,Pattern):
        lq = f'.str.match("{layer.pattern}")'
        facet = _q.NAME
    # single layer
    else:
        lq = f' == "{layer}"'

    # logic for faceting by Tensor
    if type(tt) == list:
        # convert List[TT] to list string and format for query
        tq = f' in {[t.name for t in tt]}'
        facet = _q.TTYPE
    elif tt == None:
        # Retrieve the set of unique TT's in DF and format query (this is rendudant)
        tq = f' in {[t.name for t in df.metadata.grad.unique()]}'
        facet = _q.TTYPE
    elif type(tt) == TensorType:
        tq =  f' == "{tt.name}"'

    df = df.query(
        f'@df.metadata.name{lq} & @df.metadata.grad{tq}')
    
    
    scalar_metric = [scalar_metric] if type(scalar_metric) == str else scalar_metric
    scalar_metric = list(df.general_stats.columns) if scalar_metric == None else scalar_metric

    cols = [x] if facet == None else [x,facet]
    cols.extend([_q.SCA(sm) for sm in scalar_metric])
    
    df = df[cols]
    _df = _flatten_multi_index(df)


    if facet:

        fig = sns.relplot(
            pd.melt(_df,[x[1], facet[1]]).rename(columns={"variable" : "Metric"}),
            x=x[1],
            y='value', 
            hue='Metric',
            col=facet[1],
            col_wrap= col_wrap,
            facet_kws={'sharey': False, **{k:v for k,v in facet_kws.items() if k != 'sharey'}} if facet == _q.TTYPE else facet_kws, # slightly convoluted but had a weird var scope issue
            kind=kind,
            **kwargs
        )
    else:
        if kind == 'line':
            fig = sns.lineplot(
                pd.melt(_df,[x[1]]).rename(columns={"variable" : "Metric"}),
                x=x[1], 
                y='value', 
                hue='Metric',
                **kwargs
            )
        elif kind == 'scatter':
            fig = sns.scatterplot(
                pd.melt(_df,[x[1]]).rename(columns={"variable" : "Metric"}),
                x=x[1], 
                y='value', 
                hue='Metric',
                **kwargs
            )
        else:
            err = f"Plot kind {kind} not recognized, 'line' or 'scatter' are valid arguements"
            raise ValueError(err)

    return fig



# HistoGram Plots

# Heatmaps 

# Kde (need to translate from bins counts to format for kde)