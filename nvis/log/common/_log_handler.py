from typing import Dict, List, Tuple, Union, Any
import numpy as np
import pandas as pd
from ._types import Stash
from ._q import EXP,SCA,META


"""
    Transforming Stashes to LogFrames

    Needs to handle framework tensors...
"""


def map_hist(value: Dict[str,Any]) -> Dict[Tuple[str,Union[int,float]],int]:
    counts = value['hist'].tolist()
    bins = value['bins'].tolist()
    bins[0] = float('-inf')
    bins[-1] = float('inf')

    return {EXP(bin) : count for bin,count in zip(bins,counts)}

def check_for_prefill(global_dict: Dict):
    # This is more future proofing than anything..
    prefills = []
    for k,v in global_dict.items():
        if 'scalar_stats' in k:
            # not the first iteration
            if len(v) > 1:
                prefills.append(len(v))
    if prefills:
        # need to add an additional check, so an extra zero isn't backfilled
        return np.zeros(np.min(prefills),dtype=int).tolist()
    else:
        return []

def global_stash_to_logframe(global_stash: Dict[int, List[Stash]]) -> pd.DataFrame:
    df_dict: Dict[Tuple[str,Union[str,float,int]],List] = {}

    def _backfill(min_len: int):
        # also future proofing
        for global_key,global_value in df_dict.items():
            if 'scalar_stats' in global_key:
                if len(global_value) < min_len:
                    df_dict[global_key].extend(np.zeros(min_len - len(global_value)).tolist())
        


    for step, stashes in global_stash.items():
        for stash in stashes:
            if META('step') not in df_dict.keys():
                df_dict[META('step')] = [step]
            else:
                df_dict[META('step')].append(step)

            for key, value in stash.__dict__.items():
                if key != 'value':
                    if META(key) not in df_dict.keys():
                        df_dict[META(key)] = [value]
                    else:
                        df_dict[META(key)].append(value)

                else:
                    for value_type, nested_dict in value.items():
                        if value_type == 'exp_hist':
                            hist_list_lens = []
                            for bin_edge, count in map_hist(nested_dict).items():
                                if bin_edge not in df_dict.keys():
                                    # If there histograms are being tracked for variable dtype ranges (i.e. -8 to +8 & -16 to +16)
                                    # Need to check if there are other bin edges in the dict and if there length > 1, prefill with zeros
                                    df_dict[bin_edge] = [*check_for_prefill(df_dict),count]
                                else:
                                    df_dict[bin_edge].append(count)
                                    
                                hist_list_lens.append(len(df_dict[bin_edge]))
                            max_len, min_len = np.max(hist_list_lens),np.min(hist_list_lens)
                            assert max_len == min_len , f'Something has gone wrong with zero prefills, {max_len},{min_len}'
                            # need to backfill here also (i.e. after all the hist counts for the stashs have been added, are any lists shorter)
                            _backfill(min_len=min_len)
                            
                        elif value_type == 'scalar_stats':
                            
                            for stat, stat_value in nested_dict.items():
                                assert stat_value.dim() == 0, f'Scalar Stats must be Scalar Tensors, i.e. dim == 0, not {stat_value.dim()}'
                                if SCA(stat) not in df_dict.keys():
                                    df_dict[SCA(stat)] = [stat_value.item()]
                                else:
                                    df_dict[SCA(stat)].append(stat_value.item())
                        else:
                            assert nested_dict.dim() == 0, f'Scalar Stats must be Scalar Tensors, i.e. dim == 0, not {nested_dict.dim()}'
                            if SCA(value_type) not in df_dict.keys():
                                df_dict[SCA(value_type)] = [nested_dict.item()]
                            else:
                                df_dict[SCA(value_type)].append(nested_dict.item())

    df = pd.DataFrame(df_dict)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df
