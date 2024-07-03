import matplotlib.pyplot as _plt
import seaborn as _sns

from src._transform import _flatten_multi_index  # type: ignore
from ..log.common import LogFrame, TensorType
from ..log.common import _q
from typing import Tuple, Optional


def _get_fig():
    ...


"""
    Generates a heatmap of some scalar metric (w.r.t to the chosen Tensor \
        Type) for each layer.
        Y-axis:  Layer Name
        X-axis:  Time Step


    Args:
        mv (MasterView): the MV you wish to visualise
        title (str): Optional Plot title, 
        inc (int): the increment between training iterations to include
        figsize (Tuple[int, int]): Tuple of width, height for the size you \
              want the plot to be

    Returns:
        Figure: A Heatmap of the provided MasterView
"""

# def mv_heatmap(mv: MasterView, 
#                title: Optional[str] = None,
#                figsize: Tuple[int, int] = (20, 10)):
#     with _plt.ioff():
#         fig, axs = _plt.subplots(ncols=1, nrows=1, figsize=figsize)
#         _sns.heatmap(mv._df, square=False, ax=axs)
#         if title:
#             fig.suptitle(title)
#         else:
#             fig.suptitle(f'{mv.tt.name} - {mv.metric}')

#         return fig
    

def alt_global_view(
        df: LogFrame,
        tt: TensorType,
        inc: int,  # Only makes sense if x = step (or equivalent)
        scalar_metric: str,
        x=_q.IT,
        y=_q.NAME, **kwargs):

    df = lf._df.query(
        f'@lf._df.metadata.grad == "{tt.name}" & \
            @lf._df.metadata.step % {inc} == 0')

    df = df[[x, y, _q.SCA(scalar_metric)]]

    df = _flatten_multi_index(df=df)

    return _sns.heatmap(
        data=df.pivot(
            index=_q.NAME[1],
            columns=_q.IT[1],
            values=scalar_metric),
            **kwargs)

