# Pandas Query utils
# Column Tuples -> so if schema changes 
from typing import Tuple


def _exp(c: int | float) -> Tuple[str, int | float]:
    # if float check if +/- infinity
    return ('exponent_count', c)


def _scalar(s: str) -> Tuple[str, str]:
    return ('general_stats', s)


NAME = ('metadata', 'name')
LTYPE = ('metadata', 'type')
TTYPE = ('metadata', 'grad')
IT = ('metadata', 'step')
SCA = _scalar
EXP = _exp