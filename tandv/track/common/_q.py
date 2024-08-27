# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
from typing import Tuple


def _exp(c: int | float) -> Tuple[str, int | float]:
    # if float check if +/- infinity
    return ("exponent_counts", c)


def _scalar(s: str) -> Tuple[str, str]:
    return ("scalar_stats", s)


def _meta(s: str) -> Tuple[str, str]:
    return ("metadata", s)


NAME = ("metadata", "name")
LTYPE = ("metadata", "type")
TTYPE = ("metadata", "tensor_type")
IT = ("metadata", "step")
DT = ("metadata", "dtype")
SCA = _scalar
EXP = _exp
META = _meta
