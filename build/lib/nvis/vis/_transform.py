from ..log.common._types import LogFrame, TensorType
from ..log.common import _q
import pandas as _pd

# All the pandas ops / queries that are used internally in visualisations

def _flatten_multi_index(df: _pd.DataFrame) -> _pd.DataFrame:
    """
        Only works with two levels i.e. (Upper, Lower)
        return df with columns that are solely Lower        
        
        Eg:
        ('metadata','name') -> 'name'
    """

    df.columns = [mic[1] for mic in df.columns.to_list()]  # type: ignore

    return df


