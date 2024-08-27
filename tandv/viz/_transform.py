# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
import pandas as _pd


def _flatten_multi_index(df: _pd.DataFrame) -> _pd.DataFrame:
    """
    Only works with two levels i.e. (Upper, Lower)
    return df with columns that are solely Lower

    Eg:
    ('metadata','name') -> 'name'
    """

    df.columns = [mic[1] for mic in df.columns.to_list()]  # type: ignore

    return df
