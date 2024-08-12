import pytest
import torch
import math
import numpy as np

from ..log.torch import stash_hist,stash_scalar_stats, stash_full_tensor

B,N,M = 1,1000,1000

def test_stash_hist_cpu():
    t1,t0,tn = torch.ones((B,N,M),dtype=torch.float32), torch.zeros((B,N,M),dtype=torch.float32), torch.randn((B,N,M),dtype=torch.float32)

    e1 = stash_hist(t1)
    e0 = stash_hist(t0)
    en = stash_hist(tn)


    assert np.sum(e1['exp_hist']['hist']) == B*N*M and e1['exp_hist']['bins'][np.argmax(e1['exp_hist']['hist'])] == 1, f"{e1['exp_hist']['bins'][np.argmax(e1['exp_hist']['hist'])]}"

    assert np.sum(e0['exp_hist']['hist']) == B*N*M and e0['exp_hist']['bins'][np.argmax(e0['exp_hist']['hist'])] == 0, f"{e0['exp_hist']['bins'][np.argmax(e0['exp_hist']['hist'])]}"

    assert np.sum(en['exp_hist']['hist']) == B*N*M, ' '

def test_stash_hist_cuda():
    t1,t0,tn = torch.ones((B,N,M),dtype=torch.float32).to('cuda'), torch.zeros((B,N,M),dtype=torch.float32).to('cuda'), torch.randn((B,N,M),dtype=torch.float32).to('cuda')

    e1 = stash_hist(t1)
    e0 = stash_hist(t0)
    en = stash_hist(tn)

    assert len(e1['exp_hist']['hist']) == len(e1['exp_hist']['bins'])
    assert len(e0['exp_hist']['hist']) == len(e0['exp_hist']['bins'])
    assert len(en['exp_hist']['hist']) == len(en['exp_hist']['bins'])


    assert np.sum(e1['exp_hist']['hist']) == B*N*M and e1['exp_hist']['bins'][np.argmax(e1['exp_hist']['hist'])] == 1, f"{e1['exp_hist']['bins'][np.argmax(e1['exp_hist']['hist'])]}"

    assert np.sum(e0['exp_hist']['hist']) == B*N*M and e0['exp_hist']['bins'][np.argmax(e0['exp_hist']['hist'])] == 0, f"{e0['exp_hist']['bins'][np.argmax(e0['exp_hist']['hist'])]}"

    assert np.sum(en['exp_hist']['hist']) == B*N*M


def test_stash_scalar_stats_cpu():
    t1,t0,tn = torch.ones((B,N,M)), torch.zeros((B,N,M)), torch.randn((B,N,M))

    s1,s0,sn = stash_scalar_stats(t1),stash_scalar_stats(t0),stash_scalar_stats(tn)

    for k,v in s1.items():
        assert v == 1 or v == 0, f'{k} does not have a valid value, value = {v}'

    for k,v in s0.items():
        assert v == 0 or math.isnan(v), f'{k} does not have a valid value, value = {v}'

    for k,v in sn.items():
        assert math.isfinite(v), f'{k} does not have a valid value, value = {v}'


def test_stash_scalar_stats_cuda():
    t1,t0,tn = torch.ones((B,N,M)).to('cuda'), torch.zeros((B,N,M)).to('cuda'), torch.randn((B,N,M)).to('cuda')

    s1,s0,sn = stash_scalar_stats(t1),stash_scalar_stats(t0),stash_scalar_stats(tn)

    for k,v in s1.items():
        assert v == 1 or v == 0, f'{k} does not have a valid value, value = {v}'

    for k,v in s0.items():
        assert v == 0 or math.isnan(v), f'{k} does not have a valid value, value = {v}'

    for k,v in sn.items():
        assert math.isfinite(v), f'{k} does not have a valid value, value = {v}'


def test_stash_tensors_cpu():
    tn = torch.randn((B,N,M))

    assert torch.equal(stash_full_tensor(tn)['full_tensor'],tn)

def test_stash_tensors_cuda():
    tn = torch.randn((B,N,M)).to('cuda')

    assert torch.equal(stash_full_tensor(tn)['full_tensor'],tn.cpu())