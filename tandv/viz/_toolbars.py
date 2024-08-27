# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
from typing import Dict, List, Union

import ipywidgets as widgets
import pandas as pd

from ..track.common._types import TT as TensorType


# Default Plot Configs...
class _ExponentHistogramToolbar:
    kind: List[str] = ["bar", "kde", "line"]


# class _GlobalScalarHeatmapToolbar: ...


class _ScalarLineToolbar:
    kind: List[str] = ["line", "scatter"]


def get_toolbar(**kwargs) -> Dict[str, widgets.Widget]:
    # to prevent circular import
    from ._interact import NotebookType, what_nb_frontend

    """
        Helper functions for the set of widgets which populate the \
            toolbar in the widget holder.

        Args:
            **kwargs: k,v pairs for the various inputs, i.e. \
                kind =  ['bar','kde','line'], etc., plus (df for \
                    pd.Dataframe if needed)

        Returns:
            Dict[str, widgets.Widget]]: A dictionary of the populated \
                widgets required for the tool bar, with their keys being \
                    arg name that this vis functions expect

    """
    toolbar_components = {}

    # kind
    if "kind" in kwargs.keys():
        kind: Union[List[str], None] = kwargs.get("kind")
        assert (
            type(kind) is list and len(kind) > 0
        ), f"Invalid type provided for kind: {kind}"
        toolbar_components["kind"] = widgets.Dropdown(options=kind,
                                                      value=kind[0])

    # tt
    if "tt" in kwargs.keys():
        tt: Union[str, TensorType, None] = kwargs.get("tt")
        df: Union[pd.DataFrame, None] = kwargs.get("df")
        assert type(tt) in [
            str,
            TensorType,
        ], f"tt must be of type str or TensorType not: {type(tt)}"
        assert (
            type(df) is pd.DataFrame
        ), f"df must be of type pd.DataFrame not: {type(df)}"

        toolbar_components["tt"] = widgets.Dropdown(
            options=df.metadata.tensor_type.unique().tolist(), value=tt
        )

    # scalar metric
    if "scalar_metric" in kwargs.keys():
        scalar_metric: Union[List[str], str, None] = kwargs.get(
            "scalar_metric")
        df: Union[pd.DataFrame, None] = kwargs.get("df")
        assert (
            type(df) is pd.DataFrame
        ), f"df must be of type pd.DataFrame not: {type(df)}"

        # select multiple route (for scalar line)
        if "tt" in kwargs.keys() and "layer" in kwargs.keys():
            assert type(scalar_metric) in [
                list,
                str,
            ], f"scalar_metric must be of type str or \
                list[str] not: {type(scalar_metric)}"

            if type(scalar_metric) is list:
                value = tuple(scalar_metric)
            elif type(scalar_metric) is str:
                value = (scalar_metric,)

            toolbar_components["scalar_metric"] = widgets.SelectMultiple(
                options=df.scalar_stats.columns.to_list(), value=value
            )

        elif "tt" in kwargs.keys():
            assert (
                type(scalar_metric) is str
            ), f"scalar_metric must be of type str not: {type(scalar_metric)}"

            toolbar_components["scalar_metric"] = widgets.Dropdown(
                options=df.scalar_stats.columns.tolist(), value=scalar_metric
            )

        else:
            raise Exception(
                "Invalid Arguments provided to \
                    generate widget for scalar_metric"
            )

    # step
    if "step" in kwargs.keys():
        step: Union[int, None] = kwargs.get("step")
        assert type(step) is int, f"step must be an int not {type(step)}"
        df: Union[pd.DataFrame, None] = kwargs.get("df")
        assert (
            type(df) is pd.DataFrame
        ), f"df must be of type pd.DataFrame not: {type(df)}"

        toolbar_components["step"] = widgets.Dropdown(
            options=df.metadata.step.unique().tolist(), value=step
        )

    # layer
    if "layer" in kwargs.keys():
        layer: Union[List[str], str, None] = kwargs.get("layer")
        assert type(layer) in [
            list,
            str,
        ], f"scalar_metric must be of type str or list[str] not: {type(layer)}"

        df: Union[pd.DataFrame, None] = kwargs.get("df")
        assert (
            type(df) is pd.DataFrame
        ), f"df must be of type pd.DataFrame not: {type(df)}"

        if type(layer) is list:
            value = tuple(layer)
        elif type(layer) is str:
            value = (layer,)

        if what_nb_frontend() != NotebookType.colab:
            toolbar_components["layer"] = widgets.TagsInput(
                allowed_tags=df.metadata.name.unique().tolist(),
                value=value,
            )  # type: ignore
        else:
            toolbar_components["layer"] = widgets.SelectMultiple(
                options=df.metadata.name.unique().tolist(),
                value=value
            )  # type: ignore

    return toolbar_components
