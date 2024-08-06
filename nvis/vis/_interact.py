from enum import Enum
import itertools
import sys
from ipywidgets import widgets
from typing import Dict, Callable, List, Union
import matplotlib.figure
import pandas as pd
from requests import get
import traitlets
import matplotlib
from functools import partial

from nvis.log.common._types import ExponentHistogram

from ..log.common import TensorType
from ._plots import _ExpHistPlotter, _GlobalHeatmapPlotter, _ScalarLinePlotter
from._toolbars import get_toolbar, _ExponentHistogramToolbar, _ScalarLineToolbar

from IPython import display
import matplotlib.pyplot as plt 

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



class WidgetHolder:
        
        def __init__(self, parent : widgets.Output, **kwargs) -> None:
            self.parent = parent
            self.widgets: Dict[str,widgets.Widget] = {
                **kwargs
            }
            self.hbox_layout = widgets.Layout(flex_flow='row',display='flex',**{'width': '100%','justify-content': 'space-around'})
            self.container = widgets.HBox(
                children=list(self.widgets.values()),
                layout=self.hbox_layout
            )

        def _redraw(self):
            redraw = True
        # overwrite redraw_fn args
            for k,v in self.widgets.items():
                if (type(v.value) == list or type(v.value) == tuple) and len(v.value) <= 1: # type: ignore
                    # fixing bug doug identified
                    if len(v.value) == 0: # type: ignore
                        redraw = False
                    else:
                        self.redraw_fn_args[k] = v.value[0] # type: ignore
                else:
                    # handling select multiple
                    if type(v.value) == tuple: # type: ignore
                        self.redraw_fn_args[k] = list(v.value) # type: ignore
                    else:
                        self.redraw_fn_args[k] = v.value # type: ignore
            if redraw:
                self.redraw_fn(**self.redraw_fn_args)

            
        def _f(self,*args,**kwargs):
            # so it only updates on the final point (for dropdown)
            if args[0]['name'] == '_property_lock' and type(args[0]['old']) != traitlets.utils.sentinel.Sentinel: # type: ignore

                if isinstance(args[0]['owner'],widgets.widget_selection.Dropdown):
                    if 'index' in args[0]['old'].keys():
                        self._redraw()
                    ...
                if isinstance(args[0]['owner'], widgets.widget_tagsinput.TagsInput):
                    self._redraw()
                
                if isinstance(args[0]['owner'],widgets.widget_selection.SelectMultiple):
                    self._redraw()


            else:
                
                ...
                

        def observe(self):
            for k,v in self.widgets.items():
                v.observe(self._f)

        def __getitem__(self,key) -> widgets.Widget:
            return self.widgets[key]
        
        def __setitem__(self,key: str, value: widgets.Widget) -> None:
            # add widget to key
            self.widgets[key] = value
            # observe widget
            self.widgets[key].observe(self._f)

        def __delitem__(self,key: str) -> None:
            # self.widgets[key].unobserve()
            self.widgets[key].close()
            self.widgets.__delitem__(key)


        def set_current_redraw_function(self, f: Callable, fig: matplotlib.figure.Figure, **kwargs) -> None:
            self.redraw_fn = partial(f,fig=fig)
            self.redraw_fn_args = kwargs

        def display(self) -> None:
            self.parent.append_display_data(self.container)
            self.parent.outputs = self.parent.outputs[-1:]

        def nuke(self) -> None:
            """Kill all widgets in WH"""
            for k,v in self.widgets.items():
                v.close()
            self.widgets = {}
            self.container.close()
            del self.container

            # Also needs to kill Hboxes...

        def rebuild(self,**kwargs) -> None:
            self.widgets: Dict[str,widgets.Widget] = {
                **kwargs
            }
            self.container = widgets.HBox(
                children=list(self.widgets.values()),
                layout=self.hbox_layout
            )
            self.observe()
            self.display()


        def state(self) -> Dict:
            return {k:v.value for k,v in self.widgets.items()} # type: ignore
        


def _exp_hist_redraw(fig: matplotlib.figure.Figure, df: pd.DataFrame, layer: str, tt: TensorType, step: int, **kwargs):

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
    fig.canvas.draw_idle()

    



def _global_scalar_redraw(fig: matplotlib.figure.Figure, df: pd.DataFrame, scalar_metric: str, tt: TensorType, inc: int,  **kwargs):

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
    
    # create new axes
    plotter._plot_single(
        df=_df,
        ax=new_ax
    )

    fig.suptitle(scalar_metric.upper())

    fig.canvas.draw_idle()


def _scalar_line_redraw(fig: matplotlib.figure.Figure, df: pd.DataFrame, layer: str, tt: TensorType, scalar_metric: Union[str, List[str]], **kwargs):
    # clear figure
    fig.clear()


    plotter = _ScalarLinePlotter(kind=kwargs.pop('kind',_ScalarLineToolbar.kind[0]),
                                x = kwargs.pop('x', ('metadata', 'step')),
                                scalar_metric=scalar_metric,
                                facet_kws=kwargs.pop('facet_kws', {}),
                                col_wrap = kwargs.pop('col_wrap', None),
                                **kwargs)
    
    _df = plotter._query(df=df,
                         layer=layer,
                         tt=tt
                         )

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
    fig.canvas.draw_idle()
    


def interactive(f: Callable,width: int =1500 ,**kwargs) -> None:
    """
        This function makes the visualisation interactive in a jupyter notebook.

        
        Args:
        f (Callable) : The plotting function you wish to make interactive
        width (float) : The value in pixels for how wide you wish the visualisation to be.
        kwargs: The arguments for provided vis function f 

        Returns:
        None
    
    """
    # change backend
    plt.switch_backend('ipympl')


    assert isinstance(f,Callable), f'{f} is not Callable'
    # Accessible with-in the function scope
    WH : Union[WidgetHolder,None] = None
    stack_num = 0
    APP = widgets.Output(layout={'width': f'{width}px','justify-content': 'space-around', 'overflow': 'scroll hidden'})
    TOOLBAR = widgets.Output(layout={'width': '100%','justify-content': 'space-around'})
    
    ###########################################################################################################################
    # Event Handler(s) for ScalarGlobalHeatmap
    ###########################################################################################################################
    # Onclick 
    def sgh_onclick(event: matplotlib.backend_bases.MouseEvent): # type: ignore
        assert type(WH) == WidgetHolder, 'No Widget Holder initalised'
        nonlocal stack_num # for tracking plot state
        nonlocal APP
        # initial state of plot
        DF = kwargs['df'] # not optional ever
        TT = kwargs.get('tt',None)
        INC = kwargs.get('inc',None)
        SCALAR_METRIC = kwargs.get('scalar_metric',None)
        LAYER = kwargs.get('layer',None)
        STEP = kwargs.get('step', None)
        # where is the mouse w.r.t to x-tickets / y-ticks (this will form the query)
        if event.button == 1 and stack_num == 0:
            stack_num = 1
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
                
        elif event.button == 3 and stack_num > 0 :
            stack_num = 0
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

    
    # Create figure
    with plt.ioff():
        fig = f(**kwargs)
    
    # General Plot Formating
    fig.canvas.toolbar_visible = False
    fig.canvas.figure_title = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    fig.canvas.resizable = False


    # Canvas output & Figure set to same size
    fig.figure.set_figwidth(width/fig.dpi)
    
    # Initial State of the interactive visualisation..
    if f.__name__ == 'scalar_global_heatmap':
        cid = fig.figure.canvas.mpl_connect('button_press_event', sgh_onclick)

        # FN specific 
        WH = WidgetHolder(parent=TOOLBAR,
                          **get_toolbar(
                              df=kwargs['df'],
                              scalar_metric = kwargs['scalar_metric'],
                              tt = kwargs['tt']
                          )
                        #   scalar_metric=widgets.Dropdown(options=kwargs['df'].scalar_stats.columns.tolist() , value=kwargs['scalar_metric']),
                        #   tt=widgets.Dropdown(options=kwargs['df'].metadata.tensor_type.unique().tolist(), value=kwargs['tt'])
                          )
        WH.observe()
        WH.set_current_redraw_function(_global_scalar_redraw, fig=fig.figure, **kwargs)
        WH.display()
    
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
        WH.set_current_redraw_function(_exp_hist_redraw, fig=fig.figure, **kwargs)
        WH.display()

        # raise NotImplementedError(f'{f.__name__} is not currently supported')
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
        WH.set_current_redraw_function(_scalar_line_redraw, fig=fig.figure, **kwargs)
        WH.display()

    else:
        raise Exception(f'{f.__name__} is not a valid vis function')
        
        
    # Add to master output...
    # APP.append_display_data(TOOLBAR)
    # APP.append_display_data(TOOLBAR)
    APP.append_display_data(fig.canvas)
    display.display(TOOLBAR,APP)

    if what_nb_frontend() == NotebookType.colab:
        plt.show()
        fig.canvas.resizable = False
