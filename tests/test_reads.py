# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
# import pytest

# from tandv.track.common import LogFrame, _utils, read_pickle
# from tandv.track.common._errors import SchemaException
# from tandv.track.common._types import TT as TensorType

# To Do
# Test for schema validator
# Test for DataFrame Migration

# SchemaMap (post init validators sanity check)


# For generating random DF's in future
# md_cols = list(
# map(lambda e: (e, "metadata"), ["name", "type", "grad", "step", "dtype"])
# )
# gt_cols = list(map(lambda e: (e, "scalar_stats"), [i for i in range(5)]))
# exp_cols = list(map(lambda e: (e, "exponent_counts"), [i for i in range(5)]))

# def test_read_df_from_pickle():
#     df_path = 'test-data/numerics_df_v3.pkl'
#     invalid_df_path = 'test-data/numerics_df.pkl'

#     # Valid DF -> Pass
#     df = read_pickle(df_path)

#     # Will make these more specific SchemaException
#     with pytest.raises(Exception) as e_info:
#         read_pickle(invalid_df_path)

#     with pytest.raises(SchemaException) as e_info:
#         _utils._validate_schema(df.metadata) # type: ignore

#     with pytest.raises(SchemaException) as e_info:
#         _utils._validate_schema(df.drop(columns='exponent_count'),debug=True)

#     with pytest.raises(SchemaException) as e_info:
#         _utils._validate_schema(df.drop(columns='general_stats'))

#     with pytest.raises(SchemaException) as e_info:
#         _utils._validate_schema(df.drop(columns=[('exponent_count',float('inf'))]))
