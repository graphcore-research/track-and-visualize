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
    
    
    
    """
    with _plt.ioff():
        fig, axs = _plt.subplots(ncols=1, nrows=1, figsize=figsize)
        _sns.heatmap(mv._df, square=False, ax=axs)
        
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f'{mv.tt.name} - {mv.metric}')

        return fig

