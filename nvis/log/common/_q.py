# Pandas Query utils
# Column Tuples -> so if schema changes 
from typing import Tuple


def _exp(c: int | float) -> Tuple[str, int | float]:
    # if float check if +/- infinity
    return ('exponent_counts', c)


def _scalar(s: str) -> Tuple[str, str]:
    return ('scalar_stats', s)


NAME = ('metadata', 'name')
LTYPE = ('metadata', 'type')
TTYPE = ('metadata', 'tensor_type')
IT = ('metadata', 'step')
DT = ('metadata', 'dtype')
SCA = _scalar
EXP = _exp
