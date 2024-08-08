class FacetException(Exception):
    """The provided arguments to the vis function are invalid, as they'd require an additional facet dimension"""
    ...

class QueryException(Exception):
    """The provided arguments to the vis function are invalid, as the query returned no results"""

