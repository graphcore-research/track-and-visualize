import hashlib
import pandas as pd
from typing import Any
from ._errors import SchemaException
from . import _types
from pandas.api.types import is_string_dtype, is_integer_dtype, is_float_dtype, is_dtype_equal
def _dataframe_migration(df: pd.DataFrame, schema_map: Any) -> pd.DataFrame:
    """
        Takes an arbitrary dataframe of logs and a schema-map and returns a DataFrame where values are stored under the appropriate metadata.
    """

    return df


def _get_df_schema_hash(df: pd.DataFrame):
    """Gets A SHA-256 Hash of the dataframe 'schema' - does not conside attr values"""
    return hashlib.sha256(df.dtypes.__str__().encode()).hexdigest()

def _validate_df_hash(df: pd.DataFrame):
    if not df.attrs['vis-meta']['hash'] == _get_df_schema_hash(df):
        raise SchemaException()




def _validate_schema(df: pd.DataFrame, SCHEMA = _types.LogFrame):
    
    import typing
    # get lf schema as Dict[tuple(tl_col, col) : type]
    comp_fn = {
        str: is_string_dtype,
        int: is_integer_dtype,
        float: is_float_dtype
    }
    schema, wcs = SCHEMA.get_flat_schema()

    def _check_dtype(col, key_map = None) -> bool:
        # uses outer fn varaibles, schema, comp_fn and df
        dtype_opt = schema[col] if key_map == None else wcs[key_map].dtype

        if typing.get_origin(dtype_opt) == typing.Union:
            # iterate over args
            for arg in typing.get_args(dtype_opt):
                if arg != None:
                    # check it is this type
                    if comp_fn[arg](df.dtypes[col]):
                        # if so return True
                        return True
        # must be a single type                
        else:
            # check if Dtype matches
            if (comp_fn[dtype_opt](df.dtypes[col])):
                # if so return true
                return True
        # No Dtype matches .. return false
        return False
    
    def _check_wildcard(col):
        # checks if the column tuple matches a wildcard, if the count is correct,
        # create temporary key mappings -> this assumes that there's only 1 wild card per higherlevel index (which may not be valid)
        # also very inefficient as is done per wildcard check!
        t_col = (col[0], '*')
        tmp_wcs, map_wcs = {},{}
        for k,v in wcs.items():
            new_key = (k[0], str(k[1]()))
            tmp_wcs[new_key] = v
            map_wcs[new_key] = k

        if t_col in tmp_wcs.keys():
            map_key = map_wcs[t_col]
            if _check_dtype(col, map_key):
                # dtype matches -> make sure it doesn't break WC Constraints
                wcs[map_key].count += 1
                return True
        return False

    counter = 0 
    matched_keys = []
    un_matched_cols = []
    # iterate over columns
    for col in df.columns:
        schema_match = False
        # easy match
        if col in schema.keys():
            # check if union
            schema_match = _check_dtype(col=col)
            matched_keys.append(col)
            
        # Check Wildcard Match
        elif _check_wildcard(col):
            schema_match = True

        # no match
        else: 
            # useful for giving meaningful errors
            un_matched_cols.append(col)
            ...
        # if the column matches the schema, then increment counter
        if schema_match:
            counter += 1

    
    # if all columns don't pass validation, Raise Schema Exception


    if counter != len(df.columns):
        print(counter)
        raise SchemaException

    # Make sure WildCards match  
    for key, value in wcs.items():
        if value.count >= key[1].max or value.count < key[1].min:
            raise SchemaException
    

    

    df.attrs['vis-meta'] = {
        'hash': _get_df_schema_hash(df)}
    return df