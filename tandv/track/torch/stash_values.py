# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
from typing import Dict, List, Union

from ... import _config

if _config._TORCH_EXTRA:
    import torch


def exp_histogram(tensor: torch.Tensor,
                  min_exp=-16,
                  max_exp=16) -> Dict[str, List]:
    """
    Gets the exponent histogram for the tensor, any thing over/under \
        max_exp/min_exp will be set to +/-inf

    Args:
        t (torch.Tensor): The tensor you're getting the histogram from
        min_exp (int): the exponent underwhich you consider the \
            values to be underflowing
        max_exp (int): the exponent over which you consider the \
            values to be overflowing

    Returns:
        Dict : A dictionary containing lists for the for \
            histogram counts & bin edges

    """
    tensor = tensor.detach()
    e = torch.frexp(tensor).exponent.flatten()
    # Values exceeding upper and lower exps hit overflow bins
    e = torch.where(e < min_exp, (min_exp - 1) * torch.ones_like(e), e)
    e = torch.where(e > max_exp, (max_exp + 1) * torch.ones_like(e), e)

    bins = torch.Tensor([i for i in range(
        min_exp - 1, max_exp + 2)]).to(e.dtype)
    hist = torch.histc(
        e.to(tensor.dtype),
        bins=(max_exp + 2) - (min_exp - 1),
        min=min_exp - 1,
        max=max_exp + 1,
    ).int()

    return {"hist": hist.cpu().tolist(), "bins": bins.cpu().tolist()}


def stash_scalar_stats(tensor: torch.Tensor) -> Dict[str, Union[int, float]]:
    """
    Default stash value fn for gathering scalar stats, \
        gets the mean, std, abs_mean, abs_min, abs_max, rms (called rm2), \
            rm4 & rm8

    Args:
        t (torch.Tensor): The tensor you're getting the stats for.

    Returns:
        Dict : A dictionary of the containing the scalar \
            values for various statistics
        ```python
        {'stat1' : 0.0,...}
        ```


    """
    tensor = tensor.detach()
    rm2 = tensor.pow(2).mean().pow(1 / 2)
    abs_t = tensor.abs()
    return {
        "mean": tensor.mean().cpu().item(),
        "std": tensor.std().cpu().item(),
        "mean_abs": abs_t.mean().cpu().item(),
        "max_abs": abs_t.max().cpu().item(),
        "min_abs": abs_t.min().cpu().item(),
        "rm2": rm2.cpu().item(),
        "rm4": tensor.div(rm2).pow_(4).mean().pow(1 / 4).mul(rm2).cpu().item(),
        "rm8": tensor.div(rm2).pow_(8).mean().pow(1 / 8).mul(rm2).cpu().item(),
    }


# Should refactor this slightly - as exponent hist is always a \
# nested dict, yet scalar_stats is not (and there's no reason \
# for this behaviour)


def stash_hist(tensor: torch.Tensor, min_exp=-16, max_exp=16) -> Dict:
    """
    Get the exponent histogram for the Tensor, values under \
        min_exp and over max_exp will be set to +/- infitity.

    Args:
        tensor (torch.Tensor): The tensor you're getting the histogram from

    Returns:
        Dict : A dictionary of the containing the scalar \
            statistics and the Exponent Histogram for the tensor.
        ```python
        {'exp_hist' : {'hist' : [...], 'bins' : [... ]}}
        ```

    """
    return {"exp_hist": exp_histogram(tensor, min_exp, max_exp)}


def stash_all_stats_and_hist(tensor: torch.Tensor) -> Dict:
    """
    The default stash value function, and the one which \
        is most compatible with the visualisation library.

    Args:
        tensor (torch.Tensor): The tensor you're \
            getting the the stastics for.

    Returns:
        Dict : A dictionary of the containing the scalar \
            statistics and the Exponent Histogram for the tensor.
        ```python
        {'scalar_stats' {'stat1' : 0.0,...}, \
            'exp_hist' : {'hist' : [...], 'bins' : [... ]}}
        ```
    """

    return {
        "scalar_stats": stash_scalar_stats(tensor=tensor),
        "exp_hist": exp_histogram(tensor=tensor),
    }


def stash_full_tensor(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"full_tensor": tensor.detach().cpu().clone()}
