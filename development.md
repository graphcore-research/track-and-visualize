# Development

For users who wish to develop using this codebase, the following setup is required:

**First-time setup**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt  # Or requirements-dev-ipu.txt for the ipu
```

**Subsequent setup**:

```bash
source .venv/bin/activate
```

**Run pre-flight checks** (or run `./dev --help` to see supported commands):

```bash
./dev
```

## Adding a Static Plot
### Plotters
Implement a plotter class which inherits from either and conforms to one of the interface(s) from `nvis.vis._plots`:

```python
_BasePlotter #(Single Axes plot) 
```
or
```python
_BaseFacetPlotter #(Single or Multi-Axes plot)
```


`__init__`
Arguments should be the names of any columns that are required to generate the plot along with any arguments for the underlying plotting function(s)

`_query` 
Arguments are; `pd.DataFrame` and the set of values which to filter out the sets of rows query for the plot and/or the set of columns required for the visualisation

`_plot_single`
Arguments are; `pd.DataFrame` and a `matplotlib.axes.Axes`
This is where the plotting function from Matplotlib or Seaborn (or any other visualisation library built on top of MPL) lives. 

No querying of data should take place here, however you can transform the DF from wide to long format, etc, normalise values before visualisation etc..

`_plot_facet` (only for `_BaseFacetPlotter`)
Arguments are;  `pd.DataFrame` and a `matplotlib.figure.Figure`
This is for generating a multi-axes plot, faceting around some column or column(s)

Should filter the `pd.DataFrame` appropriately for each `Axes` and call then call the plot single function for that axes.

### Plotting Functions
The `Plotters` are private and exposed to the public api via a plotting function, the rationale for this is that in order to make `Plotters` work in a interactive setting they cannot generate a `matplotlib.figure.Figure` internally, it is with-in the public plot function that the figure is generated, a `Plotter` is instantiated and the `Figure` or `Axes` is passed into that plotter.

All plotting functions return a `matplotlib.figure.Figure`

### Design Principles
The goal of the plotting function is to offer as much of the power and customisability of the underlying visualisation tools (`matplotlib`, `seaborn`, etc..) as reasonably possible, therefore the plot functions should allow the user to pass arguments via \*\*kwargs or explicit keyword arguments which are plumbed down to the underlying visualisation functions.


## Making a Static Plot Interactive
The design of the API for making plot interactive, is you pass the vis function into `interactive` function as an argument, and pass in the required arguments for the vis function as `kwargs`, therefore there is near zero additional learning curve for users to go from static to interactive plots.

*If you have implemented a new vis function, here are some key things to note for making it interactive*

### State Variables you may need

*Some variables with-in the `interactive` fn scope which are relevant, for state data which isn't accessible from the `matplotlib` objects alone and needs to be persisted*. They are defined at the top of the `interactive` function (please keep to this convention if adding additional variables).

`APP` (`widgets.Output`) -> This is where the figure(s) lives.  
`TOOLBAR` (`widgets.Output`) -> This is where the widgets for the toolbar live.

The reason for having two separate output widgets, is that when you support an interaction which transitions from one plot type to another, the toolbar will change and need to be cleared and redrawn. This clearing process, deletes all widgets which up to a shared common ancestor (if the figure and the toolbar widgets were in the same output, the figure would then also be cleared, which lead to problems).

`WH` (`WidgetHolder`)
This manages the state of the all the widgets in the toolbar. When a value is change in any of the widgets, it will pass the current value of all of the widgets into the redraw function for the figure.

Also contains some methods for managing the toolbar.

All the widgets are stored in a Dictionary. Also an important note is that the `key` of the widget in the `WH` dictionary is the keyword argument used to pass the widget value into the vis function, therefore it must be the same as the `kwarg` the vis function is expecting or it will result in an `AttributeError`.

`STACK_NUM` (`int`)
If your plot transitions from one plot type to another change the value `STACK_NUM`, so it handles the interaction in the appropriate way.

*The need for `STACK_NUM` is due to the fact that event handler is the same as the figure hasn't changed (this could also be implemented by disconnecting the current and attaching a new event handler, but this the way it's implemented currently)*, Also it's a little bit cleaner if you're listening for multiple event types.

`CROSSHAIR` (`SnappingCrossHair)
The `CROSSHAIR` manages interactions the state of interactions with the training_stats figure (if used).

`DF` The pd.DataFrame variable, so that it can be accessed from any `matplotlib` event handling closure.

`TT, INC, SCALAR_METRIC, etc...` These are the initial query parameters and are required if you wish to be able to return your visualisation to its initial state.
### Helper function(s) you may need

`get_toolbar` is a helper function that simply creates the various widgets for the toolbar, if your `vis` function uses a query argument other than those already implemented, just add it to `get_toolbar` function.
```python
def get_toolbar(**kwargs) -> Dict[str,widgets.Widget]:
	...
	if 'new_query_argument' in kwargs.keys():
		# code to create widget
		# add initial value, etc..
	
	
		toolbar_components['new_query_argument'] = YourWidget()

	return toolbar_components

```

Note order is important here, if your widget can expand in size (i.e. `widgets.TagsInput`) put it at the end, if it doesn't then place it anywhere.

### MPL Event Handler(s)

[From the MPL Docs](https://matplotlib.org/stable/users/explain/figure/event_handling.html): *The canvas retains only weak references to instance methods used as callbacks**. Therefore, you need to retain a reference to instances owning such methods. Otherwise the instance will be garbage-collected and the callback will vanish. This does not affect free functions used as callbacks.* 

The way we have found so far to ensure that these references are not garbage collected is by implementing them as closures with-in the `interactive` function. With any global state being managed by `nonlocal` variables with-in the closure. Any interactions with the plot directly i.e. clicking on the figure, mouse over, etc.. are handled using the `matplotlib` events API.

`fig.figure.canvas.mpl_connect('button_press_event', HANDLER_FUNCTION)`, to attach an event handler to the figure, matplotlib will listen for it and call the `HANDLER_FUNCTION` and pass the `matplotlib.backend_bases.Event` (or any of it's subclasses) to the Handler function is its first argument  (as per the [`matplotlib` docs]()).

For any state values you may need which aren't provided see **State Variables you may need** section below.

If you are transitioning from vis type to another (i.e. `scalar_global_heatmap` to `exp_hist` via a `mpl` event handler), it is important to rebuild the Toolbar and change the state of the `WidgetHolder`:

```python
	WidgetHolder.nuke()
	WidgetHolder.set_current_redraw_function(...)
	WH.rebuild(
		**get_toolbar(...) # kwargs required to build your toolbar
	) 
```

`
**Naming Convention for MPL event handlers**
`plotabbreviation_eventtype`, e.g. `sgh_onclick` for the event handler for the`scalar_global_heatmap` `onclick` event handler`

### Implementing a redraw function
The redraw function is quite similar to the public vis function for your creating your static plot. 
The only difference is you do not create a new figure, and call `fig.clear()` and `fig.canvas.redraw()`.

```python
	### ... denotes query args for your plot
	def _your_vis_fn_name_redraw(fig: matplotlib.figure.Figure, 
								df: pd.DataFrame,...,**kwargs):
		# clear figure
		fig.clear()

		# create your plotter
		plotter = YourPlotter(...)
		# query from your plotter
		_df = plotter._query(...)

		# check if plotters empty and create plot
		if not _df.empty:
			...
			

		fig.canvas.draw()
```


### Initialising your interactive figure
In the control branch of the `interactive` function add some code like this
```python
###### YOUR VIS FUNCTION ######

elif f.__name__ == 'your_vis_fn_name':
	#connect event handler(s) to figure
	cid = fig.figure.canvas.mpl_connect('button_press_event', your_handler_fn)
	# create WidgetHolder
	WH = WidgetHolder(parent=TOOLBAR,
						**get_toolbar(
							df=kwargs['df'],
							... # kwargs specific to your 
							)
						)
	# Listen for changes from widgets (figure on change)
	WH.observe()
	
	# For using cross hairs with training stats 
	# (don't need this if your_vis_fn_name doesn't support)
	# cross referencing with training stats
	if not isinstance(train_stats,NoneType):
		# fig.axes or None for 2nd positional arg
		# if your crosshair isn't also drawn on the
		# numerics figure
		set_up_crosshair(training_ax, fig.axes, lines)
		WH.set_current_redraw_function(
			_scalar_line_redraw,fig=fig.figure,
			# add a call back to set_up_crosshair
			# with partial arguments (the training figure axes)
			# and the training figure lines
			ch_callback=partial(
			set_up_crosshair,
			training_ax=training_ax,
			lines=lines),
			**kwargs)
	else:
		# Set the redraw function for the widget holder,
		# so it redraws the correct plot (from widget interactions)
		WH.set_current_redraw_function(
		_your_vis_fn_name_redraw, 
		fig=fig.figure, 
		**kwargs)
	# displays the toolbar in the `TOOLBAR` output widget	
	WH.display()
```

**T.D.L.R.**
attach `matplotlib` event handler, initialise `WidgetHolder`,  call `.observe` on `WidgetHolder`,
attach a redraw function (`_your_vis_fn_name_redraw`) to the `WidgetHolder`
