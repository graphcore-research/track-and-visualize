# Adapters for pytorch logging -> In progress
from typing import List
from .core import *  # NOQA: F401 F403
from .core import __all__ as coreall, __doc__  # NOQA: F401
from .stash_functions import stash_hist,stash_full_tensor,stash_all_stats_and_hist,stash_scalar_stats

__all__ = ['stash_hist','stash_full_tensor','stash_all_stats_and_hist','stash_scalar_stats',*coreall]