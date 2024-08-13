from typing import Any, Callable, Hashable, Iterable, Literal, Mapping, Sequence

import csv
from pandas._libs import lib

import pandas as _pd
from pandas._typing import (
    CompressionOptions, ReadPickleBuffer, FilePath, StorageOptions, 
    ReadBuffer, DtypeBackend,IndexLabel, UsecolsArgType, ReadCsvBuffer, 
    DtypeArg, CSVEngine)
from typing import TypeVar
from ._utils import _validate_schema, _dataframe_migration
from .wandb_int import download_wandb_artifact

WandbArtifactFullName = str


def read_pickle(
        filepath_or_buffer: FilePath | ReadPickleBuffer | WandbArtifactFullName,
        from_wandb: bool =False,
        schema_map: Any = None,
        compression: CompressionOptions = "infer",
        storage_options: StorageOptions | None = None) -> _pd.DataFrame:
    """
        Wrapper function for pd.read_pickle. Can only be used to read DataFrames from pickle files \
        unlike the pd version. Asserts the DF to the LogFrame schema and attaches metadata to the DataFrame \
        which is required by downstream functions.

        Args:
        filepath_or_buffer (pd.FilePath | ReadPickleBuffer | WandbArtefactFullName): String, path object (implementing os.PathLike[str]), or file-like object implementing a binary readlines() function. Also accepts URL. URL is not limited to S3 and GCS
        from_wandb (bool): Retrieve the logframe artifact(s) from wandb (filepath_or_buffer is where you put the WandB artifact fullname)
        schema_map (): Place holder, currently not used.
        compression (pd.CompressionOptions)
        storage_options ()

        Returns:
            pd.DataFrame
    """

    if from_wandb:
        assert type(filepath_or_buffer) == str, f'When pulling an artifact from wandb, filepath_or_buffer must be a str, not {type(filepath_or_buffer)}'
        df = download_wandb_artifact(artifact_fullname=filepath_or_buffer)


    else:
        # Read -> df as per usual
        df = _pd.read_pickle(filepath_or_buffer,
                        compression,
                        storage_options)
    
    # if a schema_map is provided migrate DF to LF schema
    if schema_map:
        df = _dataframe_migration(df, schema_map)
    
    # Validate and return
    return _validate_schema(df)




def read_parquet(
        path: FilePath | ReadBuffer[bytes],
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
        Wrapper function for pd.read_parquet. Can only be used to read DataFrames from parquet files \
        unlike the pd version. Asserts the DF to the LogFrame schema and attaches metadata to the DataFrame \
        which is required by downstream functions.

        Args:
        path : str, path object or file-like object
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a binary ``read()`` function.
            The string could be a URL. Valid URL schemes include http, ftp, s3,
            gs, and file. For file URLs, a host is expected. A local file could be:
            ``file://localhost/path/to/table.parquet``.
            A file URL can also be a path to a directory that contains multiple
            partitioned parquet files. Both pyarrow and fastparquet support
            paths to directories as well as file URLs. A directory path could be:
            ``file://localhost/path/to/tables`` or ``s3://bucket/partition_dir``.
        engine : {{'auto', 'pyarrow', 'fastparquet'}}, default 'auto'
            Parquet library to use. If 'auto', then the option
            ``io.parquet.engine`` is used. The default ``io.parquet.engine``
            behavior is to try 'pyarrow', falling back to 'fastparquet' if
            'pyarrow' is unavailable.

            When using the ``'pyarrow'`` engine and no storage options are provided
            and a filesystem is implemented by both ``pyarrow.fs`` and ``fsspec``
            (e.g. "s3://"), then the ``pyarrow.fs`` filesystem is attempted first.
            Use the filesystem keyword with an instantiated fsspec filesystem
            if you wish to use its implementation.
        columns : list, default=None
            If not None, only these columns will be read from the file.
        {storage_options}

            .. versionadded:: 1.3.0

        use_nullable_dtypes : bool, default False
            If True, use dtypes that use ``pd.NA`` as missing value indicator
            for the resulting DataFrame. (only applicable for the ``pyarrow``
            engine)
            As new dtypes are added that support ``pd.NA`` in the future, the
            output with this option will change to use those dtypes.
            Note: this is an experimental option, and behaviour (e.g. additional
            support dtypes) may change without notice.

            .. deprecated:: 2.0

        dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
            Back-end data type applied to the resultant :class:`DataFrame`
            (still experimental). Behaviour is as follows:

            * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
            (default).
            * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
            DataFrame.

            .. versionadded:: 2.0

        filesystem : fsspec or pyarrow filesystem, default None
            Filesystem object to use when reading the parquet file. Only implemented
            for ``engine="pyarrow"``.

            .. versionadded:: 2.1.0

        filters : List[Tuple] or List[List[Tuple]], default None
            To filter out data.
            Filter syntax: [[(column, op, val), ...],...]
            where op is [==, =, >, >=, <, <=, !=, in, not in]
            The innermost tuples are transposed into a set of filters applied
            through an `AND` operation.
            The outer list combines these sets of filters through an `OR`
            operation.
            A single list of tuples can also be used, meaning that no `OR`
            operation between set of filters is to be conducted.

            Using this argument will NOT result in row-wise filtering of the final
            partitions unless ``engine="pyarrow"`` is also specified.  For
            other engines, filtering is only performed at the partition level, that is,
            to prevent the loading of some row-groups and/or files.

            .. versionadded:: 2.1.0

        schema_map (): Place holder, currently not used.

        **kwargs
            Any additional kwargs are passed to the engine.


        Returns:
            pd.DataFrame
    """
    
    # read DF as per usal 
    df  = _pd.read_parquet(
        path,
        engine, #type: ignore
        columns,
        storage_options,
        use_nullable_dtypes=use_nullable_dtypes, #type: ignore
        dtype_backend=dtype_backend,
        filesystem=filesystem,
        filters=filters,
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
    """
        
    
    """

    # read DF as per usual.
    df = _pd.read_csv(
        filepath_or_buffer,
        *args,
        **kwargs
    )

    # if a schema_map is provided migrate DF to LF schema
    if schema_map:
        df = _dataframe_migration(df, schema_map) #type: ignore
    
    # Validate and return
    return _validate_schema(df) #type: ignore

