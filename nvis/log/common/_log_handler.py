import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
import warnings
import numpy as np
import pandas as pd
from ._types import Stash
from ._q import EXP,SCA,META
from ..._config import _libname


"""
    Transforming Stashes to LogFrames

    Needs to handle framework tensors...
"""


def map_hist(value: Dict[str,Any]) -> Dict[Tuple[str,Union[int,float]],int]:
    """
        Simple fn which replaces the max and min bin edges with +/- infinity and returns a dict of tuple(metadata, bin_edge) : count
    """
    counts = value['hist']
    bins = value['bins']
    bins[0] = float('-inf')
    bins[-1] = float('inf')

    return {EXP(bin) : count for bin,count in zip(bins,counts)}

def check_for_prefill(global_dict: Dict):
    # This is more future proofing than anything..
    # If someone logged different bin edges for different tensors,
    # This would prefill zeros where needed (i.e. the first time the new bin edge is encountered)
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

def global_stash_to_logframe(global_stash: Dict[int, List[Dict]]) -> pd.DataFrame:
    """
        Takes the global stash and converts it a Dataframe (that conforms to the LogFrame Schema)

        Args:
            global_stast (Dict[int,List[Dict]]): the dictionary which contains all the lists of tensors statistic stashes.

        Returns:
            pd.DataFrame
    
    """
    df_dict: Dict[Tuple[str,Union[str,float,int]],List] = {}
    def _backfill(min_len: int):
        # also future proofing
        # add zeros to the counts for bin_edges which weren't included in the most recent stash added
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

            for key, value in stash.items():
                # this is the eqivalent of .first_value
                if type(value) in [tuple,list] and len(value) >= 1:
                    value: Dict = value[0]

                if key != 'value':
                    if META(key) not in df_dict.keys():
                        df_dict[META(key)] = [value]
                    else:
                        df_dict[META(key)].append(value)

                else:
                    assert type(value) == dict,f'{key} needs to be a dict not {type(value)}, {value}'
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
                                # assert stat_value.dim() == 0, f'Scalar Stats must be Scalar Tensors, i.e. dim == 0, not {stat_value.dim()}'
                                if SCA(stat) not in df_dict.keys():
                                    df_dict[SCA(stat)] = [stat_value]
                                else:
                                    df_dict[SCA(stat)].append(stat_value)
                        elif value_type == 'full_tensor':
                            warnings.warn('Full tensors cannot be added to a log frame, therefore it is being skipped')

                        else:
                            # assert nested_dict.dim() == 0, f'Scalar Stats must be Scalar Tensors, i.e. dim == 0, not {nested_dict.dim()}'
                            if SCA(value_type) not in df_dict.keys():
                                df_dict[SCA(value_type)] = [nested_dict]
                            else:
                                df_dict[SCA(value_type)].append(nested_dict)

    df = pd.DataFrame(df_dict)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    # remove framework name from dtype
    df[('metadata','dtype')] = df[('metadata','dtype')].apply(lambda x: x.split('.')[-1])
    return df


def summarise_logframe(df: pd.DataFrame) -> Dict[int,Any]:
    """
        Takes the logframe from step n to step n+m and gets summary stats across each tensor type for each step and returns them in a format suitable for wandb visualisation.

        Args:
            df (pd.DataFrame) the logframe for step n to n+m (m being the offload increment)

        returns:
            Dict[int,Any] a dict of dicts, contains the summary stats for each step.

    """
    # Get Underflow & overflow stats
    df[('scalar_stats','underflow_rate')] = df.exponent_counts.div(df.exponent_counts.sum(axis='columns'),axis=0)[float('-inf')] # type: ignore
    df[('scalar_stats','overflow_rate')] = df.exponent_counts.div(df.exponent_counts.sum(axis='columns'),axis=0)[float('inf')] # type: ignore

    # Grouby tensor_type & step, then get mean for each scalar_stats
    df_dict = df.groupby([('metadata','step'),('metadata','tensor_type')],as_index=False).scalar_stats.mean().to_dict()

    # Put dict in a format suitable for wandb
    steps = df_dict.pop(('metadata', 'step'))
    tt_types = df_dict.pop(('metadata','tensor_type'))
    summary_dict = dict()
    for key in steps.keys():
        if steps[key] not in summary_dict.keys():
            summary_dict[steps[key]] = {}
        for k,v in df_dict.items():
            summary_dict[steps[key]][f'Numerics.{tt_types[key]}/{k[-1]}'] = v[key]

    return summary_dict


def combine_incremental_dfs(name: str) -> str:
    """
        Combines all the intermediate logframes into a single global one and writes it to disk
    """
    p = Path(f"./{_libname}/{name}/lf")
    files = os.listdir(p)
    files = [f for f in files if f.endswith('.pkl')] # ensure only pickle files
    # sort by step (might be brittle doing this!)
    files.sort(key=lambda x: int(x.split('-')[0]))
    # read_pickle should maybe be an argument?
    all_lfs = [pd.read_pickle(p/file) for file in files]

    p2 = Path(f"./{_libname}/{name}/final_logframe.pkl")
    pd.concat(all_lfs,ignore_index=True).to_pickle(p2)

    return str(p2.absolute())
    

def nuke_intermediate_logframes(name: str):
    p = Path(f"./{_libname}/{name}/lf")
    # delete intermediate logs
    shutil.rmtree(p)

    