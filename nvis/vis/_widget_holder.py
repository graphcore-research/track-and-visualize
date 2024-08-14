from functools import partial
from ipywidgets import widgets
from typing import Dict, Callable, List, Union
import matplotlib
import traitlets

class WidgetHolder:
    """
        The WidgetHolder manages the state of the interactive visualisations toolbar.

        Args
            parent (widgets.Output): The Output widget which the widget holder belongs too.
            **kwargs (widgets): The keyword args are the set of widgets that are manager by the WidgetHolder, with keyword being the parameter the widget is reponsible for.\
            for example step = widget(), the value from that widget will be used to query w.r.t. the step column in the LogFrame.
        

    """
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

    def _redraw(self,**kwargs):
        redraw = True
    # overwrite redraw_fn args
        for k,v in self.widgets.items():
            if (type(v.value) == list or type(v.value) == tuple) and len(v.value) <= 1: # type: ignore
                # fixing bug doug identified
                if len(v.value) == 0: # type: ignore
                    redraw = False
                else:
                    if kwargs.get(k,None):
                        self.redraw_fn_args[k] = kwargs.get(k) # type: ignore
                        v.value = (kwargs.get(k),) # type: ignore
                    else:
                        self.redraw_fn_args[k] = v.value[0] # type: ignore
            else:
                # handling select multiple
                if type(v.value) == tuple: # type: ignore
                    self.redraw_fn_args[k] = list(v.value) # type: ignore
                else:
                    if kwargs.get(k,None):
                        self.redraw_fn_args[k] = kwargs.get(k) 
                        v.value = kwargs.get(k) # type: ignore
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


    def set_current_redraw_function(self, f: Callable, fig: matplotlib.figure.Figure, **kwargs) -> None: #type: ignore
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
        