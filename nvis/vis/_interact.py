from dataclasses import asdict
from enum import Enum
import itertools
import sys
from ipywidgets import widgets
from typing import Dict, Callable, List, Union
import matplotlib.axes
import matplotlib.figure
import pandas as pd
from requests import get
from torch import NoneType
import traitlets
import matplotlib
from functools import partial


from ..log.common import TensorType
from ..log.common._types import TrainingStats
from ._plots import _ExpHistPlotter, _GlobalHeatmapPlotter, _ScalarLinePlotter
from._toolbars import get_toolbar, _ExponentHistogramToolbar, _ScalarLineToolbar
from ._widget_holder import WidgetHolder
from ._crosshair import SnappingCrossHair

from IPython import display
import matplotlib.pyplot as plt 
import warnings

import numpy as np



class NotebookType(str,Enum):
    colab = 'colab'
    vscode = 'vscode'
    any = 'any'


def what_nb_frontend():
    # What NB front-end
    if 'google.colab' in sys.modules:
        NB = NotebookType.colab
    elif 'vscode' in sys.modules:
        NB = NotebookType.vscode
    else:
        NB = NotebookType.any

    return NB


NoneType = type(None)




def _exp_hist_redraw(fig: matplotlib.figure.Figure, df: pd.DataFrame, layer: str, tt: TensorType, step: int, **kwargs):
    """
        Essesntially the same as z `nvis.vis.exp_hist` except it doesn't generate a new figure, which is the required behaviour for interactive plots.

        Could save some redundant code here by refactoring `nvis.vis.exp_hist` into a public and private fn, and reuse the private fn here? \
        But likely wouldn't save a huge amount.

    """
    # clear all axes from the figure
    fig.clear()
    
    
    kind = kwargs.pop('kind',_ExponentHistogramToolbar.kind[0])
    # Get required data & draw plot
    plotter = _ExpHistPlotter(
                    kind=kind,
                    sp_kws=kwargs.pop('sp_kws',dict(n=10000,log_scale=2, bw_adjust=4, fill=True) if kind == 'kde' else {}),
                    xtick_labelsize=kwargs.pop('xtick_labelsize',6),
                    xtick_rotation=kwargs.pop('xtick_rotation',45),
                    dtype_annotation=kwargs.pop('dtype_annotation',True),
                    dtype_info=kwargs.pop('dtype_info',(True,True,True)),
                    legend_kws = kwargs.pop('legend_kws',dict(fontsize=10,loc='upper right')),
                    **kwargs
                )
    


    _df = plotter._query(df,layer,tt, step)
    if not _df.empty:
        if plotter.facet:
            with plt.ioff():
                plotter._plot_facet(
                df=_df,
                figure=fig
                )
        else:
            # create new axes
            new_ax = fig.gca()
            with plt.ioff():
                plotter._plot_single(_df,new_ax)
        # X-axis isn't functioning correctly
            fig.suptitle(f'Layer: {layer}, Step: {step}, TT: {tt}')
    else:
        warnings.warn('The input query return no results, displaying an empty figure')
    fig.canvas.draw_idle()

    



def _global_scalar_redraw(fig: matplotlib.figure.Figure, df: pd.DataFrame, scalar_metric: str, tt: TensorType, inc: int,  **kwargs):
    """
        Essesntially the same as z `nvis.vis.global_scalar_heatmap` except it doesn't generate a new figure, which is the required behaviour for interactive plots.
    """

    # clear figure
    fig.clear()
    # create new axes
    new_ax = fig.gca()

    plotter = _GlobalHeatmapPlotter(
            x = kwargs.pop('x', ('metadata', 'step')),
            y = kwargs.pop('y', ('metadata', 'name')),
            scalar_metric=scalar_metric,
            **kwargs
            )
    
    _df = plotter._query(df,tt,inc)
    if not _df.empty:
        # create new axes
        plotter._plot_single(
            df=_df,
            ax=new_ax
        )

        fig.suptitle(scalar_metric.upper())
    else:
        warnings.warn('The input query return no results, displaying an empty figure')
    fig.canvas.draw_idle()


def _scalar_line_redraw(fig: matplotlib.figure.Figure, df: pd.DataFrame, layer: str, tt: TensorType, scalar_metric: Union[str, List[str]], **kwargs):
    """
        Essesntially the same as z `nvis.vis.scalar_line` except it doesn't generate a new figure, which is the required behaviour for interactive plots.
    """
    # clear figure
    fig.clear()
    ch_callback = kwargs.pop('ch_callback',None)
    plotter = _ScalarLinePlotter(kind=kwargs.pop('kind',_ScalarLineToolbar.kind[0]),
                                x = kwargs.pop('x', ('metadata', 'step')),
                                scalar_metric=scalar_metric,
                                facet_kws=kwargs.pop('facet_kws', {}),
                                col_wrap = kwargs.pop('col_wrap', None),
                                **kwargs)
    
    _df = plotter._query(df=df,
                         layer=layer,
                         tt=tt)
    
    if not _df.empty:

        if plotter.facet:
            with plt.ioff():
                plotter._plot_facet(
                df=_df,
                figure=fig
                )
                
        else:
            # create new axes
            new_ax = fig.gca()
            with plt.ioff():
                plotter._plot_single(_df,new_ax)

                fig.suptitle(f'Layer: {layer}, TT: {tt}')
    else:
        warnings.warn('The input query return no results, displaying an empty figure')

    if not isinstance(ch_callback, NoneType):
        ch_callback(other_ax=fig.axes)
    fig.canvas.draw()

# TODO: Need to check that querys are in the DF, both w.r.t to dtype annotations (i.e. logged dtype is out of range of logged hists)
# TODO: Need to ensure check that the tensors stats exists (i.e. step 0 optimiser state)


def interactive(f: Callable, train_stats: Union[Dict,TrainingStats,None] = None,  width: int =1500, mouse_sensitivity: int = 10000 ,**kwargs) -> None:
    """
        This function makes the visualisation interactive in a jupyter notebook.

        
        Args:
        f (Callable) : The plotting function you wish to make interactive
        train_stats (Dict | TrainingStats | None ) = None: If you wish to cross reference your training statistics with the numerics stats \
        pass it in and the interactive plot will display a line plot with the training stats to the left of the numerics visualisation.
        width (float) : The value in pixels for how wide you wish the visualisation to be.
        mouse_sensitivity (int) : Only required if train_stats is passed in. If you find the cross hairs lagging increase the value, it reduces the \
        frequency with which redraws triggered and therefore reducing lag at the expense of transition smoothness.
        kwargs: The arguments for provided vis function `f` 

        Returns:
        None
    
    """
    # change backend
    plt.switch_backend('ipympl')


    assert isinstance(f,Callable), f'{f} is not Callable'
    # Accessible with-in the function scope
    TSTATS_ENABLED = f.__name__ in ['exp_hist','scalar_line']
    if not isinstance(train_stats, NoneType) and not TSTATS_ENABLED:
        warnings.warn(f'Cross Referencing with Training Stats is not implemented for {f.__name__} and therefore providng train_stats will do nothing.')
    WH : Union[WidgetHolder,None] = None
    CROSSHAIR : Union[SnappingCrossHair,None] = None
    ch_mne_id: Union[int,None] = None
    ch_bpe_id: Union[int,None] = None
    STACK_NUM = 0
    APP = widgets.Output(layout={'width': f'{width}px','justify-content': 'space-around', 'overflow': 'scroll hidden'})
    TOOLBAR = widgets.Output(layout={'width': '100%','justify-content': 'space-around'})


    # DF for accessing in closures (for event handling, etc..)
    DF = kwargs['df'] # not optional ever
    # initial state of plot (These are set here, so if you need them the plot can be returned to the init state)
    TT = kwargs.get('tt',None)
    INC = kwargs.get('inc',None)
    SCALAR_METRIC = kwargs.get('scalar_metric',None)
    LAYER = kwargs.get('layer',None)
    STEP = kwargs.get('step', None)

    def tstats_onpress(event: matplotlib.backend_bases.MouseEvent, step: Callable, button_press: bool) -> None: #type: ignore
        """
            This is the event handling function for button press events on the training steps figure.

            Currently just calls step, and passes the result in as the arguement for the _redraw method in the `WidgetHolder` object, \
            which overides the value in the widget with that value.

            Args:
                event (matplotlib.backend_bases.MouseEvent): The matplotlib mouse press event
                step (Callable): The getter for the step property of the SnappingCrossHair object (i.e. what training step is the cross hair at),\
                for use in querying the numerics logs.
                button_press (bool): Not all visualisations (i.e. scalar_line) need to support querying numerics data from the training stats \
                this is simply a flag to make the event a no-op (need to listen for the event either way as it seems that if button press isn't listened for \
                other event listeners get garbage collected)

            Returns:
                None
        """
        if not isinstance(WH, NoneType) and button_press:
            WH._redraw(step=step())

    def set_up_crosshair(training_ax, other_ax, lines, button_press: bool = False):
        """
            This closure, initialises a new SnappingCrossHair state management object (in the outer function scope) and disconnecting any old event listeners, etc.
        
            Args:
                training_ax (...): The matplotlib axis for the training statistics figure (currenly only supports single axis figures)
                other_ax (...): The (axis or axes) of the numerics figure, only used currently in scalar_line which overlays the cross hair on the numerics axes.
                lines (...): The line(s) in the training axis, used to snap to samples and displaying the info text.
                button_press (...): Where or not the 'button_press_event' handler is a no-op or not.

            Returns:
                None
        """
        # Disconnect old events, before adding new ones
        nonlocal ch_mne_id,ch_bpe_id,CROSSHAIR
        if ch_mne_id:
            training_fig.canvas.mpl_disconnect(ch_mne_id)
        if ch_bpe_id:
            training_fig.canvas.mpl_disconnect(ch_bpe_id)
        CROSSHAIR = SnappingCrossHair(training_ax, other_ax, lines, sensitivity=mouse_sensitivity)
        
        ch_bpe_id = training_fig.canvas.mpl_connect('button_press_event', partial(tstats_onpress,step=CROSSHAIR.get_step_in_crosshairs, button_press=button_press))
        ch_mne_id = training_fig.canvas.mpl_connect('motion_notify_event', CROSSHAIR.on_mouse_move)
        
        training_fig.canvas.draw()
    
    ###########################################################################################################################
    # Event Handler(s) for ScalarGlobalHeatmap
    ###########################################################################################################################
    # Onclick 
    def sgh_onclick(event: matplotlib.backend_bases.MouseEvent): # type: ignore
        """
            The event handler for the scalar global heatmap 'button_press_event'. Facilitates the interaction of clicking on a patch on the heatmap on \
            navigating to the exponent histogram of that plot and conversely navigating back from the exponent histogram plot back to heatmap.

            Args:
                event (matplotlib.backend_bases.MouseEvent): The matplotlib mouse press event

            Returns:
                None
        """
        assert type(WH) == WidgetHolder, 'No Widget Holder initalised'
        nonlocal STACK_NUM # for tracking plot state
        # nonlocal APP
       
        # where is the mouse w.r.t to x-tickets / y-ticks (this will form the query)
        if event.button == 1 and STACK_NUM == 0:
            STACK_NUM = 1
            tf, ind =  event.canvas.figure.axes[0].collections[0].contains(event)
            if tf:
                
                # from the current fig get the x,y labels so that clicking can 
                # this breaks if not all values are displayed on the x or y-axis ...
                x_vals = [int(l.get_text()) for l in  event.canvas.figure.axes[0].get_xticklabels()]
                y_vals = [l.get_text() for l in event.canvas.figure.axes[0].get_yticklabels()]
                prod_xy = list(itertools.product(y_vals,x_vals))
                # use the index of the object in the mesh grid to get the query params (l_name, Step)
                layer, step = prod_xy[ind['ind'].item()]
                # clear all axes from the figure

                drargs = dict(fig=event.canvas.figure,
                    df=DF,
                    layer=layer,
                    step=step,
                    **WH.state())
                # could add **kwargs here...
                drf = _exp_hist_redraw

                drf(**drargs)

                # Update ToolBar
                WH.nuke()
                WH.set_current_redraw_function(drf, **drargs)
                WH.rebuild(
                    **get_toolbar(kind=_ExponentHistogramToolbar.kind)
                )
                
        elif event.button == 3 and STACK_NUM > 0 :
            STACK_NUM = 0
            # Redraw Figure
            drf = _global_scalar_redraw
            drargs = dict(fig=event.canvas.figure,
                df=DF,
                tt=TT,
                scalar_metric=SCALAR_METRIC,
                inc=INC)
            drf(**drargs)
                
            WH.nuke()
            WH.set_current_redraw_function(drf, **drargs) # need to include the kwargs in here ..
            WH.rebuild(
                    **get_toolbar(
                        df = kwargs['df'],
                        scalar_metric = SCALAR_METRIC,
                        tt = TT
                    )
            )
    
    ###########################################################################################################################
    # Event Handler(s) for Exponent Hist
    ###########################################################################################################################
    # Onclick 
    def eh_onclick(event: matplotlib.backend_bases.MouseEvent): # type: ignore
        ...

    ###########################################################################################################################
    # Event Handler(s) for Global Scalar Line
    ###########################################################################################################################
    # Onclick 
    def sl_onclick(event: matplotlib.backend_bases.MouseEvent): # type: ignore
        ...

    # Generate Training Stats Figure (Not implemented for Scalar Global Heatmap as of yet)
    if not isinstance(train_stats,NoneType) and TSTATS_ENABLED:
        # need to have something for handling dictionaries
        tstats_dict: Dict = asdict(train_stats, dict_factory=lambda x: {k: v for (k, v) in x if v is not None}) # type: ignore
        tsteps = tstats_dict.pop('steps', None)
        assert tsteps != None, 'Steps must be provided to use Training stats visualisation'
        tdf = kwargs.get('df',None)
        assert not isinstance(tdf,NoneType), 'Must provide a DataFrame'
        assert set(tsteps)==(set(tdf.metadata.step.unique())), 'The number of steps in train_stats does not match the numerics logging'
        lines = dict()
        with plt.ioff():
            training_fig, training_ax = plt.subplots()
            for k,v in tstats_dict.items():
                lines[k], = training_ax.plot(tsteps,v, picker=True, pickradius=5)
        
        training_fig.canvas.toolbar_visible = False # type: ignore
        training_fig.canvas.figure_title = False # type: ignore
        training_fig.canvas.header_visible = False # type: ignore
        training_fig.canvas.footer_visible = False # type: ignore
        training_fig.canvas.resizable = False # type: ignore
        


    
    # Create figure
    with plt.ioff():
        fig: matplotlib.figure.Figure = f(**kwargs)
    
    # General Plot Formating
    fig.canvas.toolbar_visible = False # type: ignore
    fig.canvas.figure_title = False # type: ignore
    fig.canvas.header_visible = False # type: ignore
    fig.canvas.footer_visible = False # type: ignore
    fig.canvas.resizable = False # type: ignore

    N2TRAINRATION = 4

    # adjust here if using training_stats
    if not isinstance(train_stats,NoneType) and TSTATS_ENABLED:
        # Canvas output & Figure set to same size
        fig_width = ((width / (N2TRAINRATION + 1)) * N2TRAINRATION).__ceil__()
        tfig_width = width - fig_width
        fig.figure.set_figwidth(fig_width/fig.dpi)
        training_fig.figure.set_figwidth(tfig_width/training_fig.dpi)
        training_fig.figure.set_figheight(fig.figure.get_figheight())

    else:
        # Canvas output & Figure set to same size
        fig.figure.set_figwidth(width/fig.dpi)
    
    
    
    # Set Initial State of the interactive visualisation..
    ###### SCALAR GLOBAL HEATMAP ######
    if f.__name__ == 'scalar_global_heatmap':
        cid = fig.figure.canvas.mpl_connect('button_press_event', sgh_onclick)

        # FN specific 
        WH = WidgetHolder(parent=TOOLBAR,
                          **get_toolbar(
                              df=kwargs['df'],
                              scalar_metric = kwargs['scalar_metric'],
                              tt = kwargs['tt']
                          )
                          )
        WH.observe()
        WH.set_current_redraw_function(_global_scalar_redraw, fig=fig.figure, **kwargs)
        WH.display()
    ###### EXPONENT HISTOGRAM ######
    elif f.__name__ == 'exp_hist':
        cid = fig.figure.canvas.mpl_connect('button_press_event', eh_onclick)

        # Need to add support for 
        WH = WidgetHolder(parent=TOOLBAR,
                          **get_toolbar(
                              df=kwargs['df'],
                              kind=_ExponentHistogramToolbar.kind,
                              tt=kwargs['tt'],
                              step=kwargs['step'],
                              layer=kwargs['layer']
                              )
                          )
        WH.observe()
        # For using cross hairs with training stats
        if not isinstance(train_stats,NoneType) and TSTATS_ENABLED:
            set_up_crosshair(training_ax, None, lines, button_press=True)
            WH.set_current_redraw_function(
                _exp_hist_redraw,
                fig=fig.figure,  
                **kwargs)
        else:
            WH.set_current_redraw_function(_exp_hist_redraw, fig=fig.figure, **kwargs)
        WH.display()

    ###### SCALAR GLOBAL LINE ######
    elif f.__name__ == 'scalar_line':
        cid = fig.figure.canvas.mpl_connect('button_press_event', sl_onclick)

        WH = WidgetHolder(parent=TOOLBAR,
                          **get_toolbar(
                              df=kwargs['df'],
                              kind = _ScalarLineToolbar.kind,
                              tt = kwargs['tt'],
                              scalar_metric = kwargs['scalar_metric'],
                              layer = kwargs['layer'])
                          )
        
        WH.observe()
        # For using cross hairs with training stats
        if not isinstance(train_stats,NoneType) and TSTATS_ENABLED:
            set_up_crosshair(training_ax, fig.axes, lines)
            WH.set_current_redraw_function(
                _scalar_line_redraw,fig=fig.figure, 
                ch_callback=partial(set_up_crosshair,training_ax=training_ax,lines=lines), 
                **kwargs)
        else:
            WH.set_current_redraw_function(_scalar_line_redraw, fig=fig.figure, **kwargs)
        WH.display()

    else:
        raise Exception(f'{f.__name__} is not a valid vis function')
        
        
    # Add to master output...
    if not isinstance(train_stats,NoneType) and TSTATS_ENABLED:
        APP.append_display_data(widgets.HBox(children=[training_fig.canvas,fig.canvas]))
    else:
        APP.append_display_data(fig.canvas)
    display.display(TOOLBAR,APP)

    if what_nb_frontend() == NotebookType.colab:
        plt.show()
        fig.canvas.resizable = False # type: ignore
