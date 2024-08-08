from contextlib import contextmanager
from functools import partial
from typing import Union
import matplotlib.figure
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product
from textwrap import dedent
import warnings

#  Seaborn imports
from seaborn.axisgrid import Grid
from seaborn._base import categorical_order
from seaborn._core.data import handle_data_source
from seaborn import utils
from seaborn.relational import scatterplot, lineplot, _ScatterPlotter,_LinePlotter


@contextmanager
def _disable_autolayout():
    """Context manager for preventing rc-controlled auto-layout behavior."""
    # This is a workaround for an issue in matplotlib, for details see
    # https://github.com/mwaskom/seaborn/issues/2914
    # The only affect of this rcParam is to set the default value for
    # layout= in plt.figure, so we could just do that instead.
    # But then we would need to own the complexity of the transition
    # from tight_layout=True -> layout="tight". This seems easier,
    # but can be removed when (if) that is simpler on the matplotlib side,
    # or if the layout algorithms are improved to handle figure legends.
    orig_val = mpl.rcParams["figure.autolayout"]
    try:
        mpl.rcParams["figure.autolayout"] = False
        yield
    finally:
        mpl.rcParams["figure.autolayout"] = orig_val

# Code adapted from Seborn FacetGrid
# Ability to pass in a pre-existing figure rather than generate a new one
class FacetGrid(Grid):
    """Multi-plot grid for plotting conditional relationships."""

    def __init__(
        self, data, *,
        row=None, col=None, hue=None, col_wrap=None,
        sharex=True, sharey=True, height=3, aspect=1, palette=None,
        row_order=None, col_order=None, hue_order=None, hue_kws=None,
        dropna=False, legend_out=True, despine=True,
        margin_titles=False, xlim=None, ylim=None, subplot_kws=None,
        gridspec_kws=None, figure: Union[matplotlib.figure.Figure,None] = None
    ):

        super().__init__()
        data = handle_data_source(data)

        # Determine the hue facet layer information
        hue_var = hue
        if hue is None:
            hue_names = None
        else:
            hue_names = categorical_order(data[hue], hue_order)

        colors = self._get_palette(data, hue, hue_order, palette)

        # Set up the lists of names for the row and column facet variables
        if row is None:
            row_names = []
        else:
            row_names = categorical_order(data[row], row_order)

        if col is None:
            col_names = []
        else:
            col_names = categorical_order(data[col], col_order)

        # Additional dict of kwarg -> list of values for mapping the hue var
        hue_kws = hue_kws if hue_kws is not None else {}

        # Make a boolean mask that is True anywhere there is an NA
        # value in one of the faceting variables, but only if dropna is True
        none_na = np.zeros(len(data), bool)
        if dropna:
            row_na = none_na if row is None else data[row].isnull()
            col_na = none_na if col is None else data[col].isnull()
            hue_na = none_na if hue is None else data[hue].isnull()
            not_na = ~(row_na | col_na | hue_na)
        else:
            not_na = ~none_na

        # Compute the grid shape
        ncol = 1 if col is None else len(col_names)
        nrow = 1 if row is None else len(row_names)
        self._n_facets = ncol * nrow

        self._col_wrap = col_wrap
        if col_wrap is not None:
            if row is not None:
                err = "Cannot use `row` and `col_wrap` together."
                raise ValueError(err)
            ncol = col_wrap
            nrow = int(np.ceil(len(col_names) / col_wrap))
        self._ncol = ncol
        self._nrow = nrow

        # Calculate the base figure size
        # This can get stretched later by a legend
        # TODO this doesn't account for axis labels
        figsize = (ncol * height * aspect, nrow * height)

        # Validate some inputs
        if col_wrap is not None:
            margin_titles = False

        # Build the subplot keyword dictionary
        subplot_kws = {} if subplot_kws is None else subplot_kws.copy()
        gridspec_kws = {} if gridspec_kws is None else gridspec_kws.copy()
        if xlim is not None:
            subplot_kws["xlim"] = xlim
        if ylim is not None:
            subplot_kws["ylim"] = ylim

        # --- Initialize the subplot grid

        if figure == None:
            with _disable_autolayout():
                fig = plt.figure(figsize=figsize)
        else:
            # with _disable_autolayout():
            fig = figure

        if col_wrap is None:

            kwargs = dict(squeeze=False,
                          sharex=sharex, sharey=sharey,
                          subplot_kw=subplot_kws,
                          gridspec_kw=gridspec_kws)
            axes = fig.subplots(nrow, ncol, **kwargs)

            if col is None and row is None:
                axes_dict = {}
            elif col is None:
                axes_dict = dict(zip(row_names, axes.flat))
            elif row is None:
                axes_dict = dict(zip(col_names, axes.flat))
            else:
                facet_product = product(row_names, col_names)
                axes_dict = dict(zip(facet_product, axes.flat))
        else:

            # If wrapping the col variable we need to make the grid ourselves
            if gridspec_kws:
                warnings.warn("`gridspec_kws` ignored when using `col_wrap`")

            n_axes = len(col_names)
            axes = np.empty(n_axes, object)
            axes[0] = fig.add_subplot(nrow, ncol, 1, **subplot_kws)
            if sharex:
                subplot_kws["sharex"] = axes[0]
            if sharey:
                subplot_kws["sharey"] = axes[0]
            for i in range(1, n_axes):
                axes[i] = fig.add_subplot(nrow, ncol, i + 1, **subplot_kws)

            axes_dict = dict(zip(col_names, axes))

        # --- Set up the class attributes

        # Attributes that are part of the public API but accessed through
        # a  property so that Sphinx adds them to the auto class doc
        self._figure = fig
        self._axes = axes
        self._axes_dict = axes_dict
        self._legend = None

        # Public attributes that aren't explicitly documented
        # (It's not obvious that having them be public was a good idea)
        self.data = data
        self.row_names = row_names
        self.col_names = col_names
        self.hue_names = hue_names
        self.hue_kws = hue_kws

        # Next the private variables
        self._nrow = nrow
        self._row_var = row
        self._ncol = ncol
        self._col_var = col

        self._margin_titles = margin_titles
        self._margin_titles_texts = []
        self._col_wrap = col_wrap
        self._hue_var = hue_var
        self._colors = colors
        self._legend_out = legend_out
        self._legend_data = {}
        self._x_var = None
        self._y_var = None
        self._sharex = sharex
        self._sharey = sharey
        self._dropna = dropna
        self._not_na = not_na

        # --- Make the axes look good

        self.set_titles()
        self.tight_layout()

        if despine:
            self.despine()

        if sharex in [True, 'col']:
            for ax in self._not_bottom_axes:
                for label in ax.get_xticklabels():
                    label.set_visible(False)
                ax.xaxis.offsetText.set_visible(False)
                ax.xaxis.label.set_visible(False)

        if sharey in [True, 'row']:
            for ax in self._not_left_axes:
                for label in ax.get_yticklabels():
                    label.set_visible(False)
                ax.yaxis.offsetText.set_visible(False)
                ax.yaxis.label.set_visible(False)
        

    # __init__.__doc__ = dedent("""\
    #     Initialize the matplotlib figure and FacetGrid object.

    #     This class maps a dataset onto multiple axes arrayed in a grid of rows
    #     and columns that correspond to *levels* of variables in the dataset.
    #     The plots it produces are often called "lattice", "trellis", or
    #     "small-multiple" graphics.

    #     It can also represent levels of a third variable with the ``hue``
    #     parameter, which plots different subsets of data in different colors.
    #     This uses color to resolve elements on a third dimension, but only
    #     draws subsets on top of each other and will not tailor the ``hue``
    #     parameter for the specific visualization the way that axes-level
    #     functions that accept ``hue`` will.

    #     The basic workflow is to initialize the :class:`FacetGrid` object with
    #     the dataset and the variables that are used to structure the grid. Then
    #     one or more plotting functions can be applied to each subset by calling
    #     :meth:`FacetGrid.map` or :meth:`FacetGrid.map_dataframe`. Finally, the
    #     plot can be tweaked with other methods to do things like change the
    #     axis labels, use different ticks, or add a legend. See the detailed
    #     code examples below for more information.

    #     .. warning::

    #         When using seaborn functions that infer semantic mappings from a
    #         dataset, care must be taken to synchronize those mappings across
    #         facets (e.g., by defining the ``hue`` mapping with a palette dict or
    #         setting the data type of the variables to ``category``). In most cases,
    #         it will be better to use a figure-level function (e.g. :func:`relplot`
    #         or :func:`catplot`) than to use :class:`FacetGrid` directly.

    #     See the :ref:`tutorial <grid_tutorial>` for more information.

    #     Parameters
    #     ----------
    #     {data}
    #     row, col, hue : strings
    #         Variables that define subsets of the data, which will be drawn on
    #         separate facets in the grid. See the ``{{var}}_order`` parameters to
    #         control the order of levels of this variable.
    #     {col_wrap}
    #     {share_xy}
    #     {height}
    #     {aspect}
    #     {palette}
    #     {{row,col,hue}}_order : lists
    #         Order for the levels of the faceting variables. By default, this
    #         will be the order that the levels appear in ``data`` or, if the
    #         variables are pandas categoricals, the category order.
    #     hue_kws : dictionary of param -> list of values mapping
    #         Other keyword arguments to insert into the plotting call to let
    #         other plot attributes vary across levels of the hue variable (e.g.
    #         the markers in a scatterplot).
    #     {legend_out}
    #     despine : boolean
    #         Remove the top and right spines from the plots.
    #     {margin_titles}
    #     {{x, y}}lim: tuples
    #         Limits for each of the axes on each facet (only relevant when
    #         share{{x, y}} is True).
    #     subplot_kws : dict
    #         Dictionary of keyword arguments passed to matplotlib subplot(s)
    #         methods.
    #     gridspec_kws : dict
    #         Dictionary of keyword arguments passed to
    #         :class:`matplotlib.gridspec.GridSpec`
    #         (via :meth:`matplotlib.figure.Figure.subplots`).
    #         Ignored if ``col_wrap`` is not ``None``.

    #     See Also
    #     --------
    #     PairGrid : Subplot grid for plotting pairwise relationships
    #     relplot : Combine a relational plot and a :class:`FacetGrid`
    #     displot : Combine a distribution plot and a :class:`FacetGrid`
    #     catplot : Combine a categorical plot and a :class:`FacetGrid`
    #     lmplot : Combine a regression plot and a :class:`FacetGrid`

    #     Examples
    #     --------

    #     .. note::

    #         These examples use seaborn functions to demonstrate some of the
    #         advanced features of the class, but in most cases you will want
    #         to use figue-level functions (e.g. :func:`displot`, :func:`relplot`)
    #         to make the plots shown here.

    #     .. include:: ../docstrings/FacetGrid.rst

    #     """).format(**_facet_docs)

    def facet_data(self):
        """Generator for name indices and data subsets for each facet.

        Yields
        ------
        (i, j, k), data_ijk : tuple of ints, DataFrame
            The ints provide an index into the {row, col, hue}_names attribute,
            and the dataframe contains a subset of the full data corresponding
            to each facet. The generator yields subsets that correspond with
            the self.axes.flat iterator, or self.axes[i, j] when `col_wrap`
            is None.

        """
        data = self.data

        # Construct masks for the row variable
        if self.row_names:
            row_masks = [data[self._row_var] == n for n in self.row_names]
        else:
            row_masks = [np.repeat(True, len(self.data))]

        # Construct masks for the column variable
        if self.col_names:
            col_masks = [data[self._col_var] == n for n in self.col_names]
        else:
            col_masks = [np.repeat(True, len(self.data))]

        # Construct masks for the hue variable
        if self.hue_names:
            hue_masks = [data[self._hue_var] == n for n in self.hue_names]
        else:
            hue_masks = [np.repeat(True, len(self.data))]

        # Here is the main generator loop
        for (i, row), (j, col), (k, hue) in product(enumerate(row_masks),
                                                    enumerate(col_masks),
                                                    enumerate(hue_masks)):
            data_ijk = data[row & col & hue & self._not_na]
            yield (i, j, k), data_ijk

    def map(self, func, *args, **kwargs):
        """Apply a plotting function to each facet's subset of the data.

        Parameters
        ----------
        func : callable
            A plotting function that takes data and keyword arguments. It
            must plot to the currently active matplotlib Axes and take a
            `color` keyword argument. If faceting on the `hue` dimension,
            it must also take a `label` keyword argument.
        args : strings
            Column names in self.data that identify variables with data to
            plot. The data for each variable is passed to `func` in the
            order the variables are specified in the call.
        kwargs : keyword arguments
            All keyword arguments are passed to the plotting function.

        Returns
        -------
        self : object
            Returns self.

        """
        # If color was a keyword argument, grab it here
        kw_color = kwargs.pop("color", None)

        # How we use the function depends on where it comes from
        func_module = str(getattr(func, "__module__", ""))

        # Check for categorical plots without order information
        if func_module == "seaborn.categorical":
            if "order" not in kwargs:
                warning = ("Using the {} function without specifying "
                           "`order` is likely to produce an incorrect "
                           "plot.".format(func.__name__))
                warnings.warn(warning)
            if len(args) == 3 and "hue_order" not in kwargs:
                warning = ("Using the {} function without specifying "
                           "`hue_order` is likely to produce an incorrect "
                           "plot.".format(func.__name__))
                warnings.warn(warning)

        # Iterate over the data subsets
        for (row_i, col_j, hue_k), data_ijk in self.facet_data():

            # If this subset is null, move on
            if not data_ijk.values.size:
                continue

            # Get the current axis
            modify_state = not func_module.startswith("seaborn")
            ax = self.facet_axis(row_i, col_j, modify_state)

            # Decide what color to plot with
            kwargs["color"] = self._facet_color(hue_k, kw_color)

            # Insert the other hue aesthetics if appropriate
            for kw, val_list in self.hue_kws.items():
                kwargs[kw] = val_list[hue_k]

            # Insert a label in the keyword arguments for the legend
            if self._hue_var is not None:
                kwargs["label"] = utils.to_utf8(self.hue_names[hue_k])

            # Get the actual data we are going to plot with
            plot_data = data_ijk[list(args)]
            if self._dropna:
                plot_data = plot_data.dropna()
            plot_args = [v for k, v in plot_data.items()]

            # Some matplotlib functions don't handle pandas objects correctly
            if func_module.startswith("matplotlib"):
                plot_args = [v.values for v in plot_args]

            # Draw the plot
            self._facet_plot(func, ax, plot_args, kwargs)

        # Finalize the annotations and layout
        self._finalize_grid(args[:2])

        return self

    def map_dataframe(self, func, *args, **kwargs):
        """Like ``.map`` but passes args as strings and inserts data in kwargs.

        This method is suitable for plotting with functions that accept a
        long-form DataFrame as a `data` keyword argument and access the
        data in that DataFrame using string variable names.

        Parameters
        ----------
        func : callable
            A plotting function that takes data and keyword arguments. Unlike
            the `map` method, a function used here must "understand" Pandas
            objects. It also must plot to the currently active matplotlib Axes
            and take a `color` keyword argument. If faceting on the `hue`
            dimension, it must also take a `label` keyword argument.
        args : strings
            Column names in self.data that identify variables with data to
            plot. The data for each variable is passed to `func` in the
            order the variables are specified in the call.
        kwargs : keyword arguments
            All keyword arguments are passed to the plotting function.

        Returns
        -------
        self : object
            Returns self.

        """

        # If color was a keyword argument, grab it here
        kw_color = kwargs.pop("color", None)

        # Iterate over the data subsets
        for (row_i, col_j, hue_k), data_ijk in self.facet_data():

            # If this subset is null, move on
            if not data_ijk.values.size:
                continue

            # Get the current axis
            modify_state = not str(func.__module__).startswith("seaborn")
            ax = self.facet_axis(row_i, col_j, modify_state)

            # Decide what color to plot with
            kwargs["color"] = self._facet_color(hue_k, kw_color)

            # Insert the other hue aesthetics if appropriate
            for kw, val_list in self.hue_kws.items():
                kwargs[kw] = val_list[hue_k]

            # Insert a label in the keyword arguments for the legend
            if self._hue_var is not None:
                kwargs["label"] = self.hue_names[hue_k]

            # Stick the facet dataframe into the kwargs
            if self._dropna:
                data_ijk = data_ijk.dropna()
            kwargs["data"] = data_ijk

            # Draw the plot
            self._facet_plot(func, ax, args, kwargs)

        # For axis labels, prefer to use positional args for backcompat
        # but also extract the x/y kwargs and use if no corresponding arg
        axis_labels = [kwargs.get("x", None), kwargs.get("y", None)]
        for i, val in enumerate(args[:2]):
            axis_labels[i] = val
        self._finalize_grid(axis_labels)
        return self

    def _facet_color(self, hue_index, kw_color):

        color = self._colors[hue_index]
        if kw_color is not None:
            return kw_color
        elif color is not None:
            return color

    def _facet_plot(self, func, ax, plot_args, plot_kwargs):

        # Draw the plot
        if str(func.__module__).startswith("seaborn"):
            plot_kwargs = plot_kwargs.copy()
            semantics = ["x", "y", "hue", "size", "style"]
            for key, val in zip(semantics, plot_args):
                plot_kwargs[key] = val
            plot_args = []
            plot_kwargs["ax"] = ax
        func(*plot_args, **plot_kwargs)

        # Sort out the supporting information
        self._update_legend_data(ax)

    def _finalize_grid(self, axlabels):
        """Finalize the annotations and layout."""
        self.set_axis_labels(*axlabels)
        self.tight_layout()

    def facet_axis(self, row_i, col_j, modify_state=True):
        """Make the axis identified by these indices active and return it."""

        # Calculate the actual indices of the axes to plot on
        if self._col_wrap is not None:
            ax = self.axes.flat[col_j]
        else:
            ax = self.axes[row_i, col_j]

        # Get a reference to the axes object we want, and make it active
        if modify_state:
            plt.sca(ax)
        return ax

    def despine(self, **kwargs):
        """Remove axis spines from the facets."""
        utils.despine(self._figure, **kwargs)
        return self

    def set_axis_labels(self, x_var=None, y_var=None, clear_inner=True, **kwargs):
        """Set axis labels on the left column and bottom row of the grid."""
        if x_var is not None:
            self._x_var = x_var
            self.set_xlabels(x_var, clear_inner=clear_inner, **kwargs)
        if y_var is not None:
            self._y_var = y_var
            self.set_ylabels(y_var, clear_inner=clear_inner, **kwargs)

        return self

    def set_xlabels(self, label=None, clear_inner=True, **kwargs):
        """Label the x axis on the bottom row of the grid."""
        if label is None:
            label = self._x_var
        for ax in self._bottom_axes:
            ax.set_xlabel(label, **kwargs)
        if clear_inner:
            for ax in self._not_bottom_axes:
                ax.set_xlabel("")
        return self

    def set_ylabels(self, label=None, clear_inner=True, **kwargs):
        """Label the y axis on the left column of the grid."""
        if label is None:
            label = self._y_var
        for ax in self._left_axes:
            ax.set_ylabel(label, **kwargs)
        if clear_inner:
            for ax in self._not_left_axes:
                ax.set_ylabel("")
        return self

    def set_xticklabels(self, labels=None, step=None, **kwargs):
        """Set x axis tick labels of the grid."""
        for ax in self.axes.flat:
            curr_ticks = ax.get_xticks()
            ax.set_xticks(curr_ticks)
            if labels is None:
                curr_labels = [label.get_text() for label in ax.get_xticklabels()]
                if step is not None:
                    xticks = ax.get_xticks()[::step]
                    curr_labels = curr_labels[::step]
                    ax.set_xticks(xticks)
                ax.set_xticklabels(curr_labels, **kwargs)
            else:
                ax.set_xticklabels(labels, **kwargs)
        return self

    def set_yticklabels(self, labels=None, **kwargs):
        """Set y axis tick labels on the left column of the grid."""
        for ax in self.axes.flat:
            curr_ticks = ax.get_yticks()
            ax.set_yticks(curr_ticks)
            if labels is None:
                curr_labels = [label.get_text() for label in ax.get_yticklabels()]
                ax.set_yticklabels(curr_labels, **kwargs)
            else:
                ax.set_yticklabels(labels, **kwargs)
        return self

    def set_titles(self, template=None, row_template=None, col_template=None, **kwargs):
        """Draw titles either above each facet or on the grid margins.

        Parameters
        ----------
        template : string
            Template for all titles with the formatting keys {col_var} and
            {col_name} (if using a `col` faceting variable) and/or {row_var}
            and {row_name} (if using a `row` faceting variable).
        row_template:
            Template for the row variable when titles are drawn on the grid
            margins. Must have {row_var} and {row_name} formatting keys.
        col_template:
            Template for the column variable when titles are drawn on the grid
            margins. Must have {col_var} and {col_name} formatting keys.

        Returns
        -------
        self: object
            Returns self.

        """
        args = dict(row_var=self._row_var, col_var=self._col_var)
        kwargs["size"] = kwargs.pop("size", mpl.rcParams["axes.labelsize"])

        # Establish default templates
        if row_template is None:
            row_template = "{row_var} = {row_name}"
        if col_template is None:
            col_template = "{col_var} = {col_name}"
        if template is None:
            if self._row_var is None:
                template = col_template
            elif self._col_var is None:
                template = row_template
            else:
                template = " | ".join([row_template, col_template])

        row_template = utils.to_utf8(row_template)
        col_template = utils.to_utf8(col_template)
        template = utils.to_utf8(template)

        if self._margin_titles:

            # Remove any existing title texts
            for text in self._margin_titles_texts:
                text.remove()
            self._margin_titles_texts = []

            if self.row_names is not None:
                # Draw the row titles on the right edge of the grid
                for i, row_name in enumerate(self.row_names):
                    ax = self.axes[i, -1]
                    args.update(dict(row_name=row_name))
                    title = row_template.format(**args)
                    text = ax.annotate(
                        title, xy=(1.02, .5), xycoords="axes fraction",
                        rotation=270, ha="left", va="center",
                        **kwargs
                    )
                    self._margin_titles_texts.append(text)

            if self.col_names is not None:
                # Draw the column titles  as normal titles
                for j, col_name in enumerate(self.col_names):
                    args.update(dict(col_name=col_name))
                    title = col_template.format(**args)
                    self.axes[0, j].set_title(title, **kwargs)

            return self

        # Otherwise title each facet with all the necessary information
        if (self._row_var is not None) and (self._col_var is not None):
            for i, row_name in enumerate(self.row_names):
                for j, col_name in enumerate(self.col_names):
                    args.update(dict(row_name=row_name, col_name=col_name))
                    title = template.format(**args)
                    self.axes[i, j].set_title(title, **kwargs)
        elif self.row_names is not None and len(self.row_names):
            for i, row_name in enumerate(self.row_names):
                args.update(dict(row_name=row_name))
                title = template.format(**args)
                self.axes[i, 0].set_title(title, **kwargs)
        elif self.col_names is not None and len(self.col_names):
            for i, col_name in enumerate(self.col_names):
                args.update(dict(col_name=col_name))
                title = template.format(**args)
                # Index the flat array so col_wrap works
                self.axes.flat[i].set_title(title, **kwargs)
        return self

    def refline(self, *, x=None, y=None, color='.5', linestyle='--', **line_kws):
        """Add a reference line(s) to each facet.

        Parameters
        ----------
        x, y : numeric
            Value(s) to draw the line(s) at.
        color : :mod:`matplotlib color <matplotlib.colors>`
            Specifies the color of the reference line(s). Pass ``color=None`` to
            use ``hue`` mapping.
        linestyle : str
            Specifies the style of the reference line(s).
        line_kws : key, value mappings
            Other keyword arguments are passed to :meth:`matplotlib.axes.Axes.axvline`
            when ``x`` is not None and :meth:`matplotlib.axes.Axes.axhline` when ``y``
            is not None.

        Returns
        -------
        :class:`FacetGrid` instance
            Returns ``self`` for easy method chaining.

        """
        line_kws['color'] = color
        line_kws['linestyle'] = linestyle

        if x is not None:
            self.map(plt.axvline, x=x, **line_kws)

        if y is not None:
            self.map(plt.axhline, y=y, **line_kws)

        return self

    # ------ Properties that are part of the public API and documented by Sphinx

    @property
    def axes(self):
        """An array of the :class:`matplotlib.axes.Axes` objects in the grid."""
        return self._axes

    @property
    def ax(self):
        """The :class:`matplotlib.axes.Axes` when no faceting variables are assigned."""
        if self.axes.shape == (1, 1):
            return self.axes[0, 0]
        else:
            err = (
                "Use the `.axes` attribute when facet variables are assigned."
            )
            raise AttributeError(err)

    @property
    def axes_dict(self):
        """A mapping of facet names to corresponding :class:`matplotlib.axes.Axes`.

        If only one of ``row`` or ``col`` is assigned, each key is a string
        representing a level of that variable. If both facet dimensions are
        assigned, each key is a ``({row_level}, {col_level})`` tuple.

        """
        return self._axes_dict

    # ------ Private properties, that require some computation to get

    @property
    def _inner_axes(self):
        """Return a flat array of the inner axes."""
        if self._col_wrap is None:
            return self.axes[:-1, 1:].flat
        else:
            axes = []
            n_empty = self._nrow * self._ncol - self._n_facets
            for i, ax in enumerate(self.axes):
                append = (
                    i % self._ncol
                    and i < (self._ncol * (self._nrow - 1))
                    and i < (self._ncol * (self._nrow - 1) - n_empty)
                )
                if append:
                    axes.append(ax)
            return np.array(axes, object).flat

    @property
    def _left_axes(self):
        """Return a flat array of the left column of axes."""
        if self._col_wrap is None:
            return self.axes[:, 0].flat
        else:
            axes = []
            for i, ax in enumerate(self.axes):
                if not i % self._ncol:
                    axes.append(ax)
            return np.array(axes, object).flat

    @property
    def _not_left_axes(self):
        """Return a flat array of axes that aren't on the left column."""
        if self._col_wrap is None:
            return self.axes[:, 1:].flat
        else:
            axes = []
            for i, ax in enumerate(self.axes):
                if i % self._ncol:
                    axes.append(ax)
            return np.array(axes, object).flat

    @property
    def _bottom_axes(self):
        """Return a flat array of the bottom row of axes."""
        if self._col_wrap is None:
            return self.axes[-1, :].flat
        else:
            axes = []
            n_empty = self._nrow * self._ncol - self._n_facets
            for i, ax in enumerate(self.axes):
                append = (
                    i >= (self._ncol * (self._nrow - 1))
                    or i >= (self._ncol * (self._nrow - 1) - n_empty)
                )
                if append:
                    axes.append(ax)
            return np.array(axes, object).flat

    @property
    def _not_bottom_axes(self):
        """Return a flat array of axes that aren't on the bottom row."""
        if self._col_wrap is None:
            return self.axes[:-1, :].flat
        else:
            axes = []
            n_empty = self._nrow * self._ncol - self._n_facets
            for i, ax in enumerate(self.axes):
                append = (
                    i < (self._ncol * (self._nrow - 1))
                    and i < (self._ncol * (self._nrow - 1) - n_empty)
                )
                if append:
                    axes.append(ax)
            return np.array(axes, object).flat


# Code adapted from Seborn relpot
# Ability to pass in a pre-existing figure rather than generate a new one, also calls our version of facegrid rather than seaborns

def relplot(
    data=None, *,
    x=None, y=None, hue=None, size=None, style=None, units=None, weights=None,
    row=None, col=None, col_wrap=None, row_order=None, col_order=None,
    palette=None, hue_order=None, hue_norm=None,
    sizes=None, size_order=None, size_norm=None,
    markers=None, dashes=None, style_order=None,
    legend="auto", kind="scatter", height=5, aspect=1, facet_kws=None, figure: matplotlib.figure.Figure = None,
    **kwargs
):

    if kind == "scatter":

        Plotter = _ScatterPlotter
        func = scatterplot
        markers = True if markers is None else markers

    elif kind == "line":

        Plotter = _LinePlotter
        func = lineplot
        dashes = True if dashes is None else dashes

    else:
        err = f"Plot kind {kind} not recognized"
        raise ValueError(err)

    # Check for attempt to plot onto specific axes and warn
    if "ax" in kwargs:
        msg = (
            "relplot is a figure-level function and does not accept "
            "the `ax` parameter. You may wish to try {}".format(kind + "plot")
        )
        warnings.warn(msg, UserWarning)
        kwargs.pop("ax")

    # Use the full dataset to map the semantics
    variables = dict(x=x, y=y, hue=hue, size=size, style=style)
    if kind == "line":
        variables["units"] = units
        variables["weight"] = weights
    else:
        if units is not None:
            msg = "The `units` parameter has no effect with kind='scatter'."
            warnings.warn(msg, stacklevel=2)
        if weights is not None:
            msg = "The `weights` parameter has no effect with kind='scatter'."
            warnings.warn(msg, stacklevel=2)
    p = Plotter(
        data=data,
        variables=variables,
        legend=legend,
    )
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    p.map_size(sizes=sizes, order=size_order, norm=size_norm)
    p.map_style(markers=markers, dashes=dashes, order=style_order)

    # Extract the semantic mappings
    if "hue" in p.variables:
        palette = p._hue_map.lookup_table
        hue_order = p._hue_map.levels
        hue_norm = p._hue_map.norm
    else:
        palette = hue_order = hue_norm = None

    if "size" in p.variables:
        sizes = p._size_map.lookup_table
        size_order = p._size_map.levels
        size_norm = p._size_map.norm

    if "style" in p.variables:
        style_order = p._style_map.levels
        if markers:
            markers = {k: p._style_map(k, "marker") for k in style_order}
        else:
            markers = None
        if dashes:
            dashes = {k: p._style_map(k, "dashes") for k in style_order}
        else:
            dashes = None
    else:
        markers = dashes = style_order = None

    # Now extract the data that would be used to draw a single plot
    variables = p.variables
    plot_data = p.plot_data

    # Define the common plotting parameters
    plot_kws = dict(
        palette=palette, hue_order=hue_order, hue_norm=hue_norm,
        sizes=sizes, size_order=size_order, size_norm=size_norm,
        markers=markers, dashes=dashes, style_order=style_order,
        legend=False,
    )
    plot_kws.update(kwargs)
    if kind == "scatter":
        plot_kws.pop("dashes")

    # Add the grid semantics onto the plotter
    grid_variables = dict(
        x=x, y=y, row=row, col=col, hue=hue, size=size, style=style,
    )
    if kind == "line":
        grid_variables.update(units=units, weights=weights)
    p.assign_variables(data, grid_variables)

    # Define the named variables for plotting on each facet
    # Rename the variables with a leading underscore to avoid
    # collisions with faceting variable names
    plot_variables = {v: f"_{v}" for v in variables}
    if "weight" in plot_variables:
        plot_variables["weights"] = plot_variables.pop("weight")
    plot_kws.update(plot_variables)

    # Pass the row/col variables to FacetGrid with their original
    # names so that the axes titles render correctly
    for var in ["row", "col"]:
        # Handle faceting variables that lack name information
        if var in p.variables and p.variables[var] is None:
            p.variables[var] = f"_{var}_"
    grid_kws = {v: p.variables.get(v) for v in ["row", "col"]}

    # Rename the columns of the plot_data structure appropriately
    new_cols = plot_variables.copy()
    new_cols.update(grid_kws)
    full_data = p.plot_data.rename(columns=new_cols)

    # Set up the FacetGrid object
    facet_kws = {} if facet_kws is None else facet_kws.copy()
    g = FacetGrid(
        data=full_data.dropna(axis=1, how="all"),
        **grid_kws,
        col_wrap=col_wrap, row_order=row_order, col_order=col_order,
        height=height, aspect=aspect, dropna=False,
        figure=figure, # pass figure into FG
        **facet_kws
    )

    # Draw the plot
    g.map_dataframe(func, **plot_kws)

    # Label the axes, using the original variables
    # Pass "" when the variable name is None to overwrite internal variables
    g.set_axis_labels(variables.get("x") or "", variables.get("y") or "")

    if legend:
        # Replace the original plot data so the legend uses numeric data with
        # the correct type, since we force a categorical mapping above.
        p.plot_data = plot_data

        # Handle the additional non-semantic keyword arguments out here.
        # We're selective because some kwargs may be seaborn function specific
        # and not relevant to the matplotlib artists going into the legend.
        # Ideally, we will have a better solution where we don't need to re-make
        # the legend out here and will have parity with the axes-level functions.
        keys = ["c", "color", "alpha", "m", "marker"]
        if kind == "scatter":
            legend_artist = utils._scatter_legend_artist
            keys += ["s", "facecolor", "fc", "edgecolor", "ec", "linewidth", "lw"]
        else:
            legend_artist = partial(mpl.lines.Line2D, xdata=[], ydata=[])
            keys += [
                "markersize", "ms",
                "markeredgewidth", "mew",
                "markeredgecolor", "mec",
                "linestyle", "ls",
                "linewidth", "lw",
            ]

        common_kws = {k: v for k, v in kwargs.items() if k in keys}
        attrs = {"hue": "color", "style": None}
        if kind == "scatter":
            attrs["size"] = "s"
        elif kind == "line":
            attrs["size"] = "linewidth"
        p.add_legend_data(g.axes.flat[0], legend_artist, common_kws, attrs)
        # if p.legend_data:
        #     g.add_legend(legend_data=p.legend_data,
        #                  label_order=p.legend_order,
        #                  title=p.legend_title,
        #                  adjust_subtitles=True)

    # Rename the columns of the FacetGrid's `data` attribute
    # to match the original column names
    orig_cols = {
        f"_{k}": f"_{k}_" if v is None else v for k, v in variables.items()
    }
    grid_data = g.data.rename(columns=orig_cols)
    if data is not None and (x is not None or y is not None):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        g.data = pd.merge(
            data,
            grid_data[grid_data.columns.difference(data.columns)],
            left_index=True,
            right_index=True,
        )
    else:
        g.data = grid_data

    return g


# relplot.__doc__ = """\
# Figure-level interface for drawing relational plots onto a FacetGrid.

# This function provides access to several different axes-level functions
# that show the relationship between two variables with semantic mappings
# of subsets. The `kind` parameter selects the underlying axes-level
# function to use:

# - :func:`scatterplot` (with `kind="scatter"`; the default)
# - :func:`lineplot` (with `kind="line"`)

# Extra keyword arguments are passed to the underlying function, so you
# should refer to the documentation for each to see kind-specific options.

# {narrative.main_api}

# {narrative.relational_semantic}

# After plotting, the :class:`FacetGrid` with the plot is returned and can
# be used directly to tweak supporting plot details or add other layers.

# Parameters
# ----------
# {params.core.data}
# {params.core.xy}
# hue : vector or key in `data`
#     Grouping variable that will produce elements with different colors.
#     Can be either categorical or numeric, although color mapping will
#     behave differently in latter case.
# size : vector or key in `data`
#     Grouping variable that will produce elements with different sizes.
#     Can be either categorical or numeric, although size mapping will
#     behave differently in latter case.
# style : vector or key in `data`
#     Grouping variable that will produce elements with different styles.
#     Can have a numeric dtype but will always be treated as categorical.
# {params.rel.units}
# weights : vector or key in `data`
#     Data values or column used to compute weighted estimation.
#     Note that use of weights currently limits the choice of statistics
#     to a 'mean' estimator and 'ci' errorbar.
# {params.facets.rowcol}
# {params.facets.col_wrap}
# row_order, col_order : lists of strings
#     Order to organize the rows and/or columns of the grid in, otherwise the
#     orders are inferred from the data objects.
# {params.core.palette}
# {params.core.hue_order}
# {params.core.hue_norm}
# {params.rel.sizes}
# {params.rel.size_order}
# {params.rel.size_norm}
# {params.rel.style_order}
# {params.rel.dashes}
# {params.rel.markers}
# {params.rel.legend}
# kind : string
#     Kind of plot to draw, corresponding to a seaborn relational plot.
#     Options are `"scatter"` or `"line"`.
# {params.facets.height}
# {params.facets.aspect}
# facet_kws : dict
#     Dictionary of other keyword arguments to pass to :class:`FacetGrid`.
# kwargs : key, value pairings
#     Other keyword arguments are passed through to the underlying plotting
#     function.

# Returns
# -------
# {returns.facetgrid}

# Examples
# --------

# .. include:: ../docstrings/relplot.rst

# """.format(
#     narrative=_relational_narrative,
#     params=_param_docs,
#     returns=_core_docs["returns"],
# )