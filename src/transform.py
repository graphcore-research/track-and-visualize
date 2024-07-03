from .log.common._types import LogFrame, MasterView, TensorType
from .log.common import _q
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


def to_master_view(
        lf: LogFrame,
        tt: TensorType,
        inc: int,
        scalar_metric: str) -> MasterView:
    """Transforms a Logframe to a master view

    Args:
        lf (LogFrame): the logs for the entire training run
        tt (TensorType): the type of tensor for the view to include
        inc (int): the increment between training iterations to include
        scalar_metric (str): which scalar metric to include in view
       

    Returns:
        MasterView: a gloval view of the data, col: its, index: layer_name, \
            values: chosen scalar metric
    """
    # Assertions to do, 
    # max_step % inc == 0
    # assert scalar_metric in column_names
    
    # filter required data
    df = lf._df.query(
        f'@lf._df.metadata.grad == "{tt.name}" & \
            @lf._df.metadata.step % {inc} == 0')
    
    # Columns that matter
    df = df[[_q.NAME, _q.IT, ('general_stats', scalar_metric)]]
    
    # remove multi-index for easy pivot
    df = _flatten_multi_index(df=df)
    # pivot currently changes order (which I don't want)
    return MasterView(df=df.pivot(
        index=_q.NAME[1],
        columns=_q.IT[1],
        values=scalar_metric),
        tt=tt,
        metric=scalar_metric)

