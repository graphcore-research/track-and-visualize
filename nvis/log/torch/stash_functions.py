from typing import Dict, List
import torch

def exp_histogram(tensor: torch.Tensor, min_exp=-16, max_exp=16) -> Dict[str,List]:
    """
    Gets the exponent histogram for the tensor, any thing over/under max_exp/min_exp will be set to +/-inf

    Args:
        t (torch.Tensor): The tensor you're getting the histogram from
        min_exp (int): the exponent underwhich you consider the values to be underflowing
        max_exp (int): the exponent over which you consider the values to be overflowing

    Returns:
        Dict : A dictionary of the containing the tensors for histogram counts & bin edges

    """
    tensor = tensor.detach()
    e = torch.frexp(tensor).exponent.flatten()
    # Values exceeding upper and lower exps hit overflow bins
    e = torch.where(e < min_exp, (min_exp - 1) * torch.ones_like(e), e)
    e = torch.where(e > max_exp, (max_exp + 1) * torch.ones_like(e), e)
    # bincount wants all values to be >= 0, so shift
    e -= min_exp - 1 #No longer needed
    bins = torch.Tensor([i for i in range(min_exp-1, max_exp+2)]).to(e.dtype)
    hist = torch.histc(e,bins=(max_exp+2)-(min_exp-1),min=0,max=max_exp-min_exp)
    
    return {
        'hist': hist.cpu().tolist(), 
        'bins' : bins.cpu().tolist()
        }

def stash_scalar_stats(tensor: torch.Tensor) -> Dict:
    """
        Default stash fn for gathering scalar stats, gets the mean, std, abs_mean, abs_min, abs_max, rms (called rm2), rm4 & rm8

        Args:
            t (torch.Tensor): The tensor you're getting the histogram from

        Returns:
            Dict : A dictionary of the containing the scalar tensors for various statistics
    
    
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

# Should refactor this slightly - as exponent hist is always a nested dict, yet scalar_stats is not (and there's no reason for this behaviour)

def stash_hist(tensor: torch.Tensor) -> Dict:
    return {'exp_hist': exp_histogram(tensor)}

def stash_all_stats_and_hist(tensor: torch.Tensor) -> Dict:
    
    return {
        'scalar_stats' : stash_scalar_stats(tensor=tensor),
        'exp_hist' : exp_histogram(tensor=tensor)
    }


def stash_full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()

