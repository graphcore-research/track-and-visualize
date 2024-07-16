from typing import Any, Callable, Hashable, Iterable, Literal, Mapping, Sequence

import csv
from pandas._libs import lib

import pandas as _pd
from pandas._typing import (
    CompressionOptions, ReadPickleBuffer, FilePath, StorageOptions, 
    ReadBuffer, DtypeBackend,IndexLabel, UsecolsArgType, ReadCsvBuffer, 
    DtypeArg, CSVEngine)

from ._utils import _validate_schema, _dataframe_migration

def read_pickle(
        filepath_or_buffer: FilePath | ReadPickleBuffer,
        schema_map: Any = None,
        compression: CompressionOptions = "infer",
        storage_options: StorageOptions | None = None) -> _pd.DataFrame:
    """
        Wrapper function for pd.read_pickle. Can only be used to read DataFrames from pickle files \
        unlike the pd version. Asserts the DF to the LogFrame schema...
        Translates to LogFrame Schema if map is provided...

        Args:

        Returns:
    """

    # Read -> df as per usual
    df = _pd.read_pickle(filepath_or_buffer,
                    compression,
                    storage_options)
    
    # if a schema_map is provided migrate DF to LF schema
    if schema_map:
        df = _dataframe_migration(df, schema_map)
    
    # Validate and return
    return _validate_schema(df)




def read_parquet(path: FilePath | ReadBuffer[bytes],
    schema_map: Any = None,
    engine: str = "auto",
    columns: list[str] | None = None,
    storage_options: StorageOptions | None = None,
    use_nullable_dtypes: bool | lib.NoDefault = lib.no_default,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    filesystem: Any = None,
    filters: list[tuple] | list[list[tuple]] | None = None,
    **kwargs,) -> _pd.DataFrame:
    """
        wrapper function for pd.read_parquet.
        Asserts the DF to the LogFrame schema...
        Translates to LogFrame Schema...
    """
    
    # read DF as per usal 
    df  = _pd.read_parquet(
        path,
        engine,
        columns,
        storage_options,
        use_nullable_dtypes,
        dtype_backend,
        filesystem,
        filters,
        **kwargs
    )
    
    # if a schema_map is provided migrate DF to LF schema
    if schema_map:
        df = _dataframe_migration(df, schema_map)
    
    # Validate and return
    return _validate_schema(df)


def read_hdf() -> _pd.DataFrame:
    """
        wrapper function for pd.read_hdf.
        Asserts the DF to the LogFrame schema...
        Translates to LogFrame Schema...
    """
    raise NotImplementedError

    ...


def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *args,
    schema_map: Any = None,
    **kwargs) -> _pd.DataFrame:

    # read DF as per usual.
    df = _pd.read_csv(
        filepath_or_buffer,
        *args,
        **kwargs
    )

    # if a schema_map is provided migrate DF to LF schema
    if schema_map:
        df = _dataframe_migration(df, schema_map)
    
    # Validate and return
    return _validate_schema(df)

