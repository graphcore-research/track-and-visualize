from functools import partial
import matplotlib.pyplot as plt
import ml_dtypes
import numpy as np
import seaborn as sns
import pandas as pd

from src._transform import _flatten_multi_index  # type: ignore
from ..log.common import TensorType
from ..log.common import _q
from ._errors import FacetException
from src.log.common._utils import _validate_df_hash
from typing import Any, Callable, Dict, List, Literal, Tuple, Optional, Union, Pattern


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


def _base2_format(l,value, tick_number):
    value = l[tick_number]
    if value == float('inf') or value == -float('inf'):
        return value
    # slightly hacky here as having an issue with double '{{}}' using f-string
    return f"$2^{{{value}}}$"

def _getformatter(list) -> Callable:
    return partial(_base2_format, list)

def _gen_facet_query(layer, tt, df) -> Tuple[Any, str]:
    """
        fn for generating df query for faceting across TensorType or Layername.


        Args:
            layer
            tt
            df

        Returns:
            (facet (Union[COLUMN,None]),query (str)) : (which column to facet on, df query string)
    """
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

    query = f'@df.metadata.name{lq} & @df.metadata.grad{tq}'

    return facet, query

    

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
        facet_kws: Dict[Any,Any] = None,
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
         **kwargs: all kwargs from `sns.relplot` (if faceting) or `sns.lineplot` (if not faceting & kind='line') | `sns.scatterplot` (if not faceting & kind='scatter')

         One of tt or scalar_metric must be a single value.
       

     Returns:
         Figure

    """

    if ((type(layer) == list or isinstance(layer,Pattern)) and (type(tt) == list or tt == None)):
        raise FacetException('Cannot Facet or both Tensor Type and layer, one must be a single value')

    _validate_df_hash(df)

    # Filter rows in DF to those of interest
    facet, query = _gen_facet_query(layer=layer,tt=tt,df=df)
    df = df.query(query)
    
    # handling scalar metric input -> should check that the provided metric is in the DF?
    scalar_metric = [scalar_metric] if type(scalar_metric) == str else scalar_metric
    scalar_metric = list(df.general_stats.columns) if scalar_metric == None else scalar_metric

    # columns to use from the provided DF
    cols = [x] if facet == None else [x,facet]
    cols.extend([_q.SCA(sm) for sm in scalar_metric])
    df = df[cols]

    # Removing multi-index
    _df = _flatten_multi_index(df)


    if facet:
        if facet_kws == None:
            facet_kws = {'sharey': True, 'sharex': True}

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
        plot_df = pd.melt(_df,[x[1]]).rename(columns={"variable" : "Metric"}) 
        if kind == 'line':
            fig = sns.lineplot(
                plot_df,
                x=x[1], 
                y='value', 
                hue='Metric',
                **kwargs
            )
        elif kind == 'scatter':
            fig = sns.scatterplot(
                plot_df,
                x=x[1], 
                y='value', 
                hue='Metric',
                **kwargs
            )
        else:
            err = f"Plot kind {kind} not recognized, 'line' or 'scatter' are valid arguements"
            raise ValueError(err)

    return fig


def _annotate_nf_details(ax: plt.Axes, x_values: List[str], fp_dtype: Union[str,List[str]], color_map: List[str] = None):

    # if single value is passed in, wrap in list so branch in logic
    if type(fp_dtype) == str:
        fp_dtype = [fp_dtype]
    
    # just a quick soln for having different annotation colours
    if color_map == None:
        color_map = plt.colormaps.get_cmap('Set1').resampled(len(fp_dtype)).colors

    # need some logic for changing colour for different ftypes..
    for fpdt,color in zip(fp_dtype,color_map):
        # parse dtype
        fp_info = ml_dtypes.finfo(fpdt)

        ax.axvline(
            x_values.index(str(fp_info.maxexp)), 
            ls="-",
            color=color,
            alpha=1, 
            label=f"{str(fp_info.dtype).upper()} - Max: {r'$2^{{0}}$'.replace('{0}',str(fp_info.maxexp))} (exp) | {fp_info.max} (rv)")
        
        ax.axvline(
            x_values.index(str(np.log2(fp_info.smallest_normal))), 
            ls="--",
            color=color,
            alpha=1, 
            label=f"{str(fp_info.dtype).upper()} - Smallest Normal: {r'$2^{{0}}$'.replace('{0}',str(np.log2(fp_info.smallest_normal)))} (exp) | {fp_info.smallest_normal} (rv)")

    return ax

# HistoGram Plots - BarPlot
def exp_hist(
        df: pd.DataFrame,
        layer: Union[str, List[str], Pattern],
        tt: Union[TensorType, List[TensorType], None],
        step: int,
        kind: Literal['bar','line','kde'] = 'bar',
        dtype_annotation: Union[bool, str, List[str]]= True,
        col_wrap: int = None,
        figsize: Tuple[int,int] = (10,10),
        xtick_labelsize: int = None,
        fig_title: str = None,
        facet_kws: Dict = None,
        sp_kws: Dict = None,
        legend_kws: Dict = None,
        **kwargs):
    """
    Bar Plot, Line Plot or resampled kde plot (based on resampling from histogram) of a Tensor type or (types: facted plot) of a single layer or (set of layers: facted plot), for a single training step.

    One of tt or layer_name must be a single value.

    Args:
         df (pd.DataFrame): the logs for the entire training run
         layer (str | List[str] | Pattern): a str (single layer), set of str or a compiled regex of a set of names to plot (each layer is a facet)
         tt (TensorType | List[TensorType] | None): the type of tensor for the view to include (if None plots a facet of all tensor types)
         step (int): the training step you wish to plot
         kind: (Literal['bar','line','kde']): Typle of plot to display accepts args "bar", "line" or "kde", kde is a resampled kde (from the exponent counts histogram)
         dtype_annotation (Union[bool, str, List[str]]): 
            False: No Annotations
            True: will draw numerical format annotations for the dtype logged as metadata
            str: provide the string of that dtype you wish to annotate e.g. 'float8_e4m3fn'
            List[str]: provide a list of strings of the dtypes you wish to annotate, e.g. ['float8_e4m3fn', 'float8_e5m2]

         col_wrap (int | None): if faceting on tt or scalar metric set max-col width before wrapping
         figsize (Tuple[int,int]): size of the figure
         fig_title (str): Custom title of the figure, on faceted plots this equates to the overall figure title as sub plot titles are autopopulated.
         facet_kws (Dict): Arguments for sns.FacetGrad
         sp_kws (Dict): **kwargs for plt.plot, plt.bar or sns.kdeplot (depending on value chosen for 'kind')
         legend_kws (Dict): **kwargs for FacetGrid.add_legend (if faceting) or plt.legend if not
         **kwargs: tbd

         One of tt or scalar_metric must be a single value.
       

     Returns:
         Figure


    TO-DO, facet on step (), col == step, (row == tt or layer)

    """
    
    
    if ((type(layer) == list or isinstance(layer,Pattern)) and (type(tt) == list or tt == None)):
        raise FacetException('Cannot Facet or both Tensor Type and layer, one must be a single value')
    
    # initialising kws if not provided
    if sp_kws == None: sp_kws = dict()
    if legend_kws == None: legend_kws = dict(fontsize=10,loc='upper right')
    if facet_kws == None: facet_kws = dict(legend_out=False)

    

    facet_dtypes = None
    # Internal Functions ...
    def _facet_plot_wrapper(*args,**kwargs):
        nonlocal facet_dtypes # Set scope to outer fn scope
        # store dtype for this facet
        if facet_dtypes == None:
            facet_dtypes = [kwargs['data'].metadata.dtype.item()]
        else:
            facet_dtypes.append(kwargs['data'].metadata.dtype.item())

        _plot_single(df_ = kwargs['data'], ax= None)


    def _plot_single(df_, ax):
        # normalize, convert to long format & sort ascending
        _df = pd.melt(df_.exponent_count.div(df_.exponent_count.sum(axis=1), axis=0)).rename(columns={'variable':'Exponent'}).sort_values('Exponent')
        # make string for categorical plot
        x = _df.Exponent.astype(str).tolist()
        if kind == 'bar':
            # plot bar
            plt.bar(
            x=x,
            height=_df.value,
            align='edge', # default
            **sp_kws
            )
        elif kind == 'line':
            # plot line
            plt.plot(
                x,
                _df.value,
                **sp_kws
            )

        elif kind == 'kde':
            raise NotImplementedError('The feature has not yet been implmented')
        
        # Only do this for non-facet plots
        if ax != None:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(_getformatter(x)))
            if xtick_labelsize:
                ax.xaxis.set_tick_params(labelsize=xtick_labelsize)
            # Draw dtype annotations
            if dtype_annotation:
                _annotate_nf_details(
                    ax=ax,
                    x_values=x,
                    fp_dtype=df_.metadata.dtype.item() if type(dtype_annotation) == bool else dtype_annotation)
                # legend location -> outter fn argument?
                plt.legend(**legend_kws)


    _validate_df_hash(df)
    # Filter rows in DF to those of interest (w.r.t. tt and layer type)
    facet, query = _gen_facet_query(layer=layer,tt=tt,df=df)
    # query and get the specific step (may need to change this if allowing for faceting on steps)
    df = df.query(query).pipe(lambda x: x[x.metadata.step == step])


    if facet:

        g = sns.FacetGrid(
            df,
            col=facet,
            # may move to the height / aspect approach for our api, currently sticking to figsize
            height=figsize[1],
            aspect=figsize[0]/figsize[1],
            col_wrap=col_wrap,
            **facet_kws # Need to find a solution to place the legend appropriately
        )

        g.map_dataframe(_facet_plot_wrapper)
        # get the exponent column names and sort
        l = df.exponent_count.columns.to_list()
        l.sort()

        
        # For storing legend data.
        all_leg_handles = {}
        for ind,ax in enumerate(g.axes.flat):
            ax.xaxis.set_major_formatter(plt.FuncFormatter(_getformatter(l)))
            if xtick_labelsize:
                ax.xaxis.set_tick_params(labelsize=xtick_labelsize)


            # if metadata dtype is being plotted, different dtypes will use the same colour, need to fix this
            if dtype_annotation:
                _annotate_nf_details(
                ax=ax,
                x_values=list(map(str,l)),
                # 
                fp_dtype=facet_dtypes[ind] if type(dtype_annotation) == bool else dtype_annotation )

                # For creating overall legend for dtype annotations.
                han_artists, han_keys = ax.get_legend_handles_labels()
                # As there is likely to be duplicates in annotations
                for hk,ha in zip(han_keys,han_artists):
                    if hk not in all_leg_handles.keys():
                        all_leg_handles[hk] = ha


        # Add the legend - Want to overlap this some how.
        if dtype_annotation:
            g.add_legend(legend_data=all_leg_handles,**legend_kws)
        # g.tight_layout()
        g.set_titles("{col_var[1]} = '{col_name}'")
        
        # Overall Plot title
        if fig_title:
            g.figure.suptitle(fig_title)
        return g
    
    # No faceting -> Single plot 
    # Could just use facet grid for everything?
    else:
        # create figure
        fig, ax = plt.subplots(figsize=figsize)
        # plot figure
        _plot_single(df_=df,ax=ax)
        # format axis to log2

        if fig_title:
            fig.suptitle(fig_title)
        else:
            fig.suptitle(f'Name = "{layer}"')

        # fig.tight_layout()

        
        return fig


# Heatmaps 

# Kde (need to translate from bins counts to format for kde)
