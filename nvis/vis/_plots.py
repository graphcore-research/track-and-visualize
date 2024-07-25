from functools import partial
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib
import ml_dtypes
import numpy as np
import seaborn as sns
import pandas as pd

from ._transform import _flatten_multi_index  # type: ignore
from ..log.common import TensorType, _q
from ._facet import FacetGrid, relplot

from ._errors import FacetException
from ..log.common._utils import _validate_df_hash
from typing import Any, Callable, Dict, List, Literal, Tuple, Optional, Union, Pattern


################################### Helper Methods #############################################


def _base2_format(l,value, tick_number):
    
    # the categorical case
    if type(value) == int:
        value = l[tick_number]
        if value == float('inf') or value == -float('inf'):
            return value
        # slightly hacky here as having an issue with double '{{}}' using f-string
        return f"$2^{{{value}}}$"
    # ... continuous x-axis
    else:
        if value <= float(l[0]):
            return -float('inf')
        elif value >= float(l[-1]):
            return float('inf')
        return f"$2^{{{int(np.log2(value))}}}$"


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
        tq = f' in {[t.name if type(t) != str else t for t in tt]}'
        facet = _q.TTYPE
    elif tt == None:
        # Retrieve the set of unique TT's in DF and format query (this is rendudant)
        tq = f' in {[t.name if type(t) != str else t  for t in df.metadata.tensor_type.unique()]}'
        facet = _q.TTYPE
    elif type(tt) == TensorType or type(tt) == str:
        tq =  f' == "{tt.name if type(tt) != str else tt}"'

    query = f'@df.metadata.name{lq} & @df.metadata.tensor_type{tq}'

    return facet, query

def _annotate_nf_details(
        ax: plt.Axes, 
        x_values: List[str], 
        fp_dtype: Union[str,List[str]], 
        dtype_info: Tuple[bool,bool,bool], 
        color_map: List[str] = None, 
        logged_dtypes: List[str] = None):
    """
        This implementation is based on using categorical values for the x-axis, which is not the case in a kdeplot

        !Need to keep track of dtype for faceting (i.e. gradients logged in e5m2, activations e4m3)
    """
    offset = 0
    # will either be a list of strings or a np array

    if logged_dtypes and len(logged_dtypes) > 1:
        offset = logged_dtypes.index(fp_dtype)

    # if single value is passed in, wrap in list so branch in logic
    if type(fp_dtype) == str:
        fp_dtype = [fp_dtype]




    # just a quick soln for having different annotation colours
    if color_map == None:
        color_map = plt.colormaps.get_cmap('Set1').resampled(len(fp_dtype) + offset).colors

    # need some logic for changing colour for different ftypes..
    for fpdt,color in zip(fp_dtype,color_map[offset:]):
        # parse dtype
        fp_info = ml_dtypes.finfo(fpdt)
        line_styles = (
        (x_values.index(str(fp_info.maxexp)) if type(x_values) == list else fp_info.max,'-', f"{str(fp_info.dtype).upper()} - Max: {f'$2^{{{str(fp_info.maxexp)}}}$'} (exp) | {fp_info.max} (rv)",),
        (x_values.index(str(np.log2(fp_info.smallest_normal))) if type(x_values) == list else fp_info.smallest_normal,'--', f"{str(fp_info.dtype).upper()} - Smallest Normal: {f'$2^{{{str(np.log2(fp_info.smallest_normal))}}}$'} (exp) | {fp_info.smallest_normal} (rv)"),
        (x_values.index(str(np.log2(fp_info.smallest_subnormal))) if type(x_values) == list else fp_info.smallest_subnormal,':', f"{str(fp_info.dtype).upper()} - Smallest SubNormal: {f'$2^{{{str(np.log2(fp_info.smallest_subnormal))}}}$'} (exp) | {fp_info.smallest_subnormal} (rv)")
        )

        for ls,d_if in zip(line_styles,dtype_info):

            if d_if:
                ax.axvline(
                    ls[0], 
                    ls=ls[1],
                    color=color,
                    alpha=1, 
                    label=ls[2])
                
    return ax


def _swap_infities(ed):
    """
        As x-axis is not categorical (like in bar & line plot) infinities break the plot. 
        Therefore they are swapped out for 2^{max_exp +1} / 2^{min_exp -1} for the data being plotted.
        The axis tickets are the original hist edges.
    """
    ed_copy = np.copy(ed)
    ed_copy[0] = ed[1] -1 # make -inf the min non inf exponent -1
    ed_copy[-1] = ed[-2] + 1 # make inf into max non inf exponent +1
    return ed_copy

def _generate_underlying_data(h: np.ndarray,e: np.ndarray, n : int = 1000000) -> np.ndarray:
    """
        Need underlying data for kde plot, slightly hacky way of generating an approximation of it.
        Will refine this in due course.. 
    """
    e.sort() # to ensure that it is -inf ... inf,
    e2 = _swap_infities(e) 
    # this formula isn't exactly correct ... but will do for now 
    act = np.array([2**exp for exp in e2],dtype='float64')
    # iterate over each 2^e value and sample from a normal distribution with mean (2^n + 2^n/2 and std 2^n/4)
    empty = []
    for e_,h_ in zip(act,h):
        empty.append(np.random.normal(e_ + e_ / 2 , scale = e_ / 4,  size=int(n*h_)))

    return np.concatenate(empty), act

################################### Plotting Methods #############################################


class _GlobalHeatmapPlotter:
    def __init__(self, x: Tuple[Any,Any], y: Tuple[Any,Any], scalar_metric: str, **kwargs) -> None:
        self.x = x
        self.y = y
        self.scalar_metric = scalar_metric
        self.kwargs = kwargs


    def _query(self,df :pd.DataFrame, tt: TensorType, inc: int) -> pd.DataFrame:
        return df.query(f'@df.metadata.tensor_type == "{tt.name if type(tt) != str else tt }" & @df.metadata.step % {inc} == 0')
        

    def _plot_single(self,df: pd.DataFrame, ax: matplotlib.axes.Axes):

        # print(self.x,self.y,self.scalar_metric)
        df = df[[self.x, self.y, _q.SCA(self.scalar_metric)]]

        df = _flatten_multi_index(df=df)

        dfp = df.pivot(
                index=self.y[1],
                columns=self.x[1],
                values=self.scalar_metric)
        
        
        with plt.ioff():
            
            ax = sns.heatmap(
            data=dfp,
                ax=ax,
                yticklabels=True,
                **self.kwargs)
            
            # print(dfp.columns)
            ytick_labelsize = 4
            ytick_rotation = 0
            ax.yaxis.set_tick_params(labelsize=ytick_labelsize,rotation=ytick_rotation,)

def scalar_global_heatmap(
        df: pd.DataFrame,
        tt: Union[TensorType, List[TensorType], None],
        scalar_metric: Union[str, List[str]],
        inc: int = 1,
        x=_q.IT,
        y=_q.NAME,
        figsize: Tuple[float,float] = (10,10),
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
    if inc < 1:
        raise ValueError('inc must be a positive integer value greater than zero')
    # plt.switch_backend('inline')
    _validate_df_hash(df) 
    
    # Implement Logic for faceting on tt or scalar_metric
    if (type(tt) == list or tt == None) or (type(scalar_metric)== list or scalar_metric == None):
        raise NotImplementedError('Faceting not yet implemented!')
    
    # Create Plotter
    plotter = _GlobalHeatmapPlotter(
        x=x,
        y=y,
        scalar_metric=scalar_metric,
        **kwargs
    )

    # filter relevent rows
    df = plotter._query(
        df,tt,inc
    )

    # create figure
    fig, ax = plt.subplots(figsize=figsize) 
    # plot on axes
    plotter._plot_single(
        df=df, ax=ax
    )

    fig.suptitle(scalar_metric.upper())
    


    return fig


class _ScalarLinePlotter:
    def __init__(self,kind: str, x: Tuple[Any,Any], scalar_metric: Union[str,List[str]],col_wrap,facet_kws, **kwargs) -> None:
        self.kind = kind
        self.kwargs = {**kwargs}
        self.x = x
        self.col_wrap = col_wrap
        self.scalar_metric = [scalar_metric] if type(scalar_metric) == str else scalar_metric
        self.facet_kws = facet_kws

    def _query(self, df: pd.DataFrame, layer: str, tt: TensorType) -> pd.DataFrame:
        self.facet, query = _gen_facet_query(layer=layer,tt=tt,df=df)
        # handling scalar metric input -> should check that the provided metric is in the DF?
        self.scalar_metric = list(df.general_stats.columns) if self.scalar_metric == None else self.scalar_metric
        df = df.query(query)
        
        # columns to use from the provided DF
        cols = [self.x] if self.facet == None else [self.x,self.facet]
        cols.extend([_q.SCA(sm) for sm in self.scalar_metric])
        df = df[cols]

        # Removing multi-index
        return _flatten_multi_index(df)

    def _plot_single(self,df: pd.DataFrame, ax: matplotlib.axes.Axes):
        df =  pd.melt(df,[self.x[1]]).rename(columns={"variable" : "Metric"}) 
        if self.kind == 'line':
            ax = sns.lineplot(
                df,
                x=self.x[1], 
                y='value', 
                hue='Metric',
                ax=ax,
                **self.kwargs
            )
        elif self.kind == 'scatter':
            ax = sns.scatterplot(
                df,
                x=self.x[1], 
                y='value', 
                hue='Metric',
                ax=ax,
                **self.kwargs
            )

        else:
            err = f"Plot kind {self.kind} not recognized, 'line' or 'scatter' are valid arguements"
            raise ValueError(err)
        

    def _plot_facet(self,df, figure: matplotlib.figure.Figure):

        if self.facet_kws == None:
            self.facet_kws = {'sharey': True, 'sharex': True}

        g = relplot(
            pd.melt(df,[self.x[1], self.facet[1]]).rename(columns={"variable" : "Metric"}),
            x=self.x[1],
            y='value', 
            hue='Metric',
            col=self.facet[1],
            col_wrap= self.col_wrap,
            facet_kws={'sharey': False, **{k:v for k,v in self.facet_kws.items() if k != 'sharey'}} if self.facet == _q.TTYPE else self.facet_kws, # slightly convoluted but had a weird var scope issue
            kind=self.kind,
            figure=figure,
            **self.kwargs
        )

        g.figure.legend(loc=self.kwargs.pop('loc','lower center'), 
                bbox_to_anchor=self.kwargs.pop('bbox_to_anchor',(0.9, 0.7)))

        # print(fig._legend_data)

def scalar_line(
        df: pd.DataFrame,
        layer: Union[str, List[str], Pattern],
        tt: Union[TensorType, List[TensorType], None],
        scalar_metric: Union[str,List[str]], # By Default
        x=_q.IT,
        col_wrap: int = None,
        kind='line',
        figsize: Tuple[float,float] = (8,6),
        facet_kws: Union[Dict[Any,Any],None] = None,
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
    # plt.switch_backend('inline')
    if ((type(layer) == list or isinstance(layer,Pattern)) and (type(tt) == list or tt == None)):
        raise FacetException('Cannot Facet or both Tensor Type and layer, one must be a single value')

    _validate_df_hash(df)

    plotter = _ScalarLinePlotter(kind=kind,
                                x=x,
                                scalar_metric=scalar_metric,
                                facet_kws=facet_kws,
                                col_wrap = col_wrap,
                                **kwargs)
    

    df = plotter._query(df=df,layer=layer,tt=tt)


    if plotter.facet:
        fig = plt.figure(figsize=figsize)
        plotter._plot_facet(df=df,figure=fig)
        
    else:

        fig, ax = plt.subplots(figsize=figsize)
        plotter._plot_single(df=df,ax=ax)

    return fig


class _ExpHistPlotter:
    def __init__(self, 
                 kind, 
                 sp_kws,
                 xtick_labelsize,
                 xtick_rotation,
                 dtype_annotation,
                 dtype_info,
                 legend_kws,
                 **kwargs
                 ) -> None:
        self.kind = kind
        self.sp_kws = sp_kws
        self.xtick_labelsize = xtick_labelsize
        self.xtick_rotation = xtick_rotation
        self.dtype_annotation = dtype_annotation
        self.dtype_info = dtype_info 
        self.legend_kws = legend_kws
        # for faceted plots
        self.all_leg_handles = {}
        self.kwargs = {**kwargs}

    def _query(self,df: pd.DataFrame, layer: str, tt: TensorType, step: int):

        # Filter rows in DF to those of interest (w.r.t. tt and layer type)
        self.facet, query = _gen_facet_query(layer=layer,tt=tt,df=df)
        # query and get the specific step (may need to change this if allowing for faceting on steps)
        df = df.query(query).pipe(lambda x: x[x.metadata.step == step])
        # set logged dtypes
        self.logged_dtypes = df.metadata.dtype.unique().tolist()

        return df 

    def _plot_single(self,df_, ax: Union[matplotlib.axes.Axes, None]):   
        # normalize, convert to long format & sort ascending
        fp = False
        _df = pd.melt(df_.exponent_count.div(df_.exponent_count.sum(axis=1), axis=0)).rename(columns={'variable':'Exponent'}).sort_values('Exponent')
        # when using seaborn facet grid
        if ax == None:
            # internally sns fg sets the current axis -> therefore retrieve it
            ax = plt.gca()
            fp = True
        # make string for categorical plot
        x = _df.Exponent.astype(str).tolist()
        if self.kind == 'bar':
            # plot bar
            ax.bar(
            x=x,
            height=_df.value,
            align='edge', # default
            **self.sp_kws
            )
        elif self.kind == 'line':
            # plot line
            ax.plot(
                x,
                _df.value,
                **self.sp_kws
            )

        elif self.kind == 'kde':
            # extract counts,edges
            h,e = _df.value.to_numpy(),_df.Exponent.to_numpy()
            n = self.sp_kws.pop('n') 
            x, act = _generate_underlying_data(h,e,n=n)
            # store act values for formating x-axis outside
            # if self.facet_dtypes != None:
                # self.kde_facet_ax_ticks.append(act)

            sns.kdeplot(x,ax=ax,**self.sp_kws)
            self.sp_kws['n'] = n # n is re-used for each facet so need to reset it
            

            # raise NotImplementedError('The feature has not yet been implmented')
        else:
            err = f"Plot kind {self.kind} not recognized, 'line', 'bar' or 'kde' are valid arguements"
            raise ValueError(err)

        
        
        # 
        if self.kind == 'kde':
            ax.set_xticks(act)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(_getformatter(act))) # custom format for 2^n & inf
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            ax.set_xbound(lower=act[0],upper=act[-1])
        else:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(_getformatter(x)))
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        if self.xtick_labelsize:
            ax.xaxis.set_tick_params(labelsize=self.xtick_labelsize,rotation=self.xtick_rotation)
            # Draw dtype annotations
        if self.dtype_annotation:
            _annotate_nf_details(
                ax=ax,
                x_values=x,
                fp_dtype=df_.metadata.dtype.item() if type(self.dtype_annotation) == bool else self.dtype_annotation,
                dtype_info=self.dtype_info
                )
                # legend location -> outter fn argument?
            

            # if metadata dtype is being plotted, different dtypes will use the same colour, need to fix this
            if fp:
                # For creating overall legend for dtype annotations.
                han_artists, han_keys = ax.get_legend_handles_labels()
                # As there is likely to be duplicates in annotations
                for hk,ha in zip(han_keys,han_artists):
                    if hk not in self.all_leg_handles.keys():
                        self.all_leg_handles[hk] = ha
            else:
                ax.legend(**self.legend_kws)

    def _plot_facet(self, df: pd.DataFrame, figure: matplotlib.figure.Figure = None):

        # Internal Functions ...
        def _facet_plot_wrapper(*args,**kwargs):
            # pass the DF (from)
            self._plot_single(df_ = kwargs['data'], ax= None)


        g = FacetGrid(
            df,
            col=self.facet,
            # may move to the height / aspect approach for our api, currently sticking to figsize
            figure=figure,
            col_wrap=self.kwargs.pop('col_wrap', None),
            **self.kwargs.get('facet_kws', {}) # Need to find a solution to place the legend appropriately
            )
        
        g = g.map_dataframe(_facet_plot_wrapper)
        
        # # Add the legend - Want to overlap this some how.
        # print(self.legend_kws, self.all_leg_handles)
        # TO DO: 
            # Add Legend 

        if self.dtype_annotation:
            # create legend
            self.legend = g.figure.legend(
                list(self.all_leg_handles.values()),
                list(self.all_leg_handles.keys()),
                loc=self.legend_kws.pop('loc','lower center'), 
                bbox_to_anchor=self.legend_kws.pop('bbox_to_anchor',(0.9, 0.9)),
                **self.legend_kws)
            
        g.tight_layout()
        g.set_titles("{col_var[1]} = '{col_name}'")
        # print(g.figure.get_figwidth())
        ...


# HistoGram Plots - BarPlot
def exp_hist(
        df: pd.DataFrame,
        layer: Union[str, List[str], Pattern],
        tt: Union[TensorType, List[TensorType], None],
        step: int,
        kind: Literal['bar','line','kde'] = 'bar',
        dtype_annotation: Union[bool, str, List[str]]= True,
        dtype_info: Tuple[bool,bool,bool] = (True,True,False), # Will figure out a way to make this slightly nicer!
        col_wrap: int = None,
        figsize: Tuple[int,int] = (10,10),
        xtick_labelsize: int = 12,
        xtick_rotation: int = 45, # can I make these kwargs will defaults??
        fig_title: str = None,
        facet_kws: Union[Dict,None] = None,
        sp_kws: Union[Dict,None] = None,
        legend_kws: Union[Dict,None] = None,
        ) -> matplotlib.figure.Figure:
    """
    Bar Plot, Line Plot or resampled kde plot (based on resampling from histogram) of a Tensor type or (types: facted plot) of a single layer or (set of layers: facted plot), for a single training step.

    Always uses log2 for x-axis.
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
            str: provide the string of that dtype you wish to annotate e.g. `'float8_e4m3fn'`
            List[str]: provide a list of strings of the dtypes you wish to annotate, e.g. `['float8_e4m3fn', 'float8_e5m2]`
         dtype_info (Tuple[bool,bool,bool]): 
            Which dtype info to annotate (Max Representable Value, Smallest Normal, Smallest Subnormal), defaults to MRV & SN
         col_wrap (int | None): if faceting on tt or scalar metric set max-col width before wrapping
         figsize (Tuple[int,int]): size of the figure
         xtick_labelsize (int) : size of x-axis tick labels 
         xtick_rotation (int) : rotation of x-axis tick labels
         fig_title (str): Custom title of the figure, on faceted plots this equates to the overall figure title as sub plot titles are autopopulated.
         facet_kws (Dict): Arguments for `sns.FacetGrad`
         sp_kws (Dict): **kwargs for `plt.plot`, `plt.bar` or `sns.kdeplot` (depending on value chosen for 'kind')
                        in the case of sns.kdeplot n=int (> 10000) is required for resampling from histogram.
         legend_kws (Dict): **kwargs for `FacetGrid.add_legend` (if faceting) or `plt.legend` if not
         **kwargs: tbd

         One of tt or scalar_metric must be a single value.
       

     Returns:
         Figure


    TO-DO, facet on step (), col == step, (row == tt or layer)

    """
    
    # plt.switch_backend('inline')
    if ((type(layer) == list or isinstance(layer,Pattern)) and (type(tt) == list or tt == None)):
        raise FacetException('Cannot Facet or both Tensor Type and layer, one must be a single value')
    
    # initialising kws if not provided
    if sp_kws == None: sp_kws = dict() if kind != 'kde' else dict(n=10000,log_scale=2, bw_adjust=4, fill=True) # sub kde also use sp_kws or be a seperate arg??
    # if sp_kws & kde need to assert n is in dict, as it will throw a key error when pop is called?
    if legend_kws == None: legend_kws = dict(fontsize=10,loc='upper right')
    if facet_kws == None: facet_kws = dict(legend_out=False)

    
    _validate_df_hash(df)
    

    # get the set of logged dtypes


    plotter = _ExpHistPlotter(
        kind=kind,
        sp_kws=sp_kws,
        xtick_labelsize=xtick_labelsize,
        xtick_rotation=xtick_rotation,
        dtype_annotation=dtype_annotation,
        dtype_info=dtype_info,
        facet_kws = facet_kws,
        legend_kws = legend_kws,
        col_wrap = col_wrap
    )

    df = plotter._query(
        df=df,
        layer=layer,
        tt=tt, 
        step=step)
    

    if plotter.facet:
        fig = plt.figure(figsize=figsize)
        
        plotter._plot_facet(
            df=df,
            figure=fig
        )

        # Overall Plot title
        if fig_title: fig.suptitle(fig_title) 

        return fig
    
    else:
        # create figure
        fig, ax = plt.subplots(figsize=figsize)
        df_ = plotter._query(df,layer,tt,step)
        # plot figure
        plotter._plot_single(df_=df,ax=ax)
        # Overall plot title ...
        fig.suptitle(fig_title) if fig_title else fig.suptitle(f'Name = "{layer}"')

        return fig


# Heatmaps 