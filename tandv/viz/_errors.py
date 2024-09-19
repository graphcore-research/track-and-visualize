# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
class FacetException(Exception):
    """The provided arguments to the vis function are invalid, \
        as they'd require an additional facet dimension"""

    ...


class QueryException(Exception):
    """The provided arguments to the vis function are invalid, \
        as the query returned no results"""
