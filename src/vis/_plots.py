import matplotlib.pyplot as _plt
import seaborn as _sns  # type: ignore
from ..log.common import MasterView
from typing import Tuple, Optional


def _get_fig():
    ...


def global_view(mv: MasterView, 
                title: Optional[str] = None,
                figsize: Tuple[int, int] = (20, 10)):
    """
    Generates a heatmap of some scalar metric (w.r.t to the chosen Tensor Type) for each layer.
        Y-axis:  Layer Name
        X-axis:  Time Step


    Args:
        mv (MasterView): the MV you wish to visualise
        title (str): Optional Plot title, 
        inc (int): the increment between training iterations to include
        figsize (Tuple[int, int]): Tuple of width, height for the size you want the plot to be
       

    Returns:
        Figure: A Heatmap of the provided MasterView


    """
    with _plt.ioff():
        fig, axs = _plt.subplots(ncols=1, nrows=1, figsize=figsize)
        _sns.heatmap(mv._df, square=False, ax=axs)
        
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f'{mv.tt.name} - {mv.metric}')

        return fig

