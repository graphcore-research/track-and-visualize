# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
import math

import jax
import jax.numpy as jnp
import numpy as np

from tandv.track.jax import stash_full_tensor, stash_hist, stash_scalar_stats
from tandv.track.jax.stash_values import (
    _concretize_as_list,
    _concretize_as_scalar,
    _stash_scalar_stats,
    exp_histogram,
)

B, N, M = 1, 1000, 1000


def gen_unit_gauss(B, N, M):
    key = jax.random.PRNGKey(0)

    # Mean and standard deviation for the Gaussian
    mean = 0.0
    std_dev = 1.0

    # Generate Gaussian samples
    return mean + std_dev * jax.random.normal(key, (B, N, M), dtype=jnp.float32)


def test_stash_hist_cpu():
    t1, t0, tn = (
        jnp.ones((B, N, M), dtype=jnp.float32),
        jnp.zeros((B, N, M), dtype=jnp.float32),
        gen_unit_gauss(B, N, M),
    )

    e1 = stash_hist(t1)
    e0 = stash_hist(t0)
    en = stash_hist(tn)

    assert (
        np.sum(e1["exp_hist"]["hist"]) == B * N * M
        and e1["exp_hist"]["bins"][np.argmax(e1["exp_hist"]["hist"])] == 1
    ), f"{e1['exp_hist']['bins'][np.argmax(e1['exp_hist']['hist'])]}"

    assert (
        np.sum(e0["exp_hist"]["hist"]) == B * N * M
        and e0["exp_hist"]["bins"][np.argmax(e0["exp_hist"]["hist"])] == 0
    ), f"{e0['exp_hist']['bins'][np.argmax(e0['exp_hist']['hist'])]}"

    assert np.sum(en["exp_hist"]["hist"]) == B * N * M, " "


def test_stash_hist_cuda():
    t1, t0, tn = (
        jnp.ones((B, N, M), dtype=jnp.float32),
        jnp.zeros((B, N, M), dtype=jnp.float32),
        gen_unit_gauss(B, N, M),
    )

    fn = jax.jit(exp_histogram)

    e1 = _concretize_as_list(fn(t1))
    e0 = _concretize_as_list(fn(t0))
    en = _concretize_as_list(fn(tn))

    assert len(e1["hist"]) == len(e1["bins"])
    assert len(e0["hist"]) == len(e0["bins"])
    assert len(en["hist"]) == len(en["bins"])

    assert (
        np.sum(e1["hist"]) == B * N * M and e1["bins"][np.argmax(e1["hist"])] == 1
    ), f"{e1['bins'][np.argmax(e1['hist'])]}"

    assert (
        np.sum(e0["hist"]) == B * N * M and e0["bins"][np.argmax(e0["hist"])] == 0
    ), f"{e0['bins'][np.argmax(e0['hist'])]}"

    assert np.sum(en["hist"]) == B * N * M


def test_stash_scalar_stats_cpu():
    t1, t0, tn = jnp.ones((B, N, M)), jnp.zeros((B, N, M)), gen_unit_gauss(B, N, M)

    s1, s0, sn = stash_scalar_stats(t1), stash_scalar_stats(t0), stash_scalar_stats(tn)

    for k, v in s1.items():
        assert (
            v == 1 or v == 0
        ), f"{k} does not \
            have a valid value, value = {v}"

    for k, v in s0.items():
        assert v == 0 or math.isnan(
            v
        ), f"{k} does not \
            have a valid value, value = {v}"

    for k, v in sn.items():
        assert math.isfinite(
            v
        ), f"{k} does not have a \
            valid value, value = {v}"


def test_stash_scalar_stats_cuda():
    t1, t0, tn = jnp.ones((B, N, M)), jnp.zeros((B, N, M)), gen_unit_gauss(B, N, M)

    fn = jax.jit(_stash_scalar_stats)

    s1, s0, sn = (
        _concretize_as_scalar(fn(t1)),
        _concretize_as_scalar(fn(t0)),
        _concretize_as_scalar(fn(tn)),
    )

    for k, v in s1.items():
        assert (
            v == 1 or v == 0
        ), f"{k} does not \
            have a valid value, value = {v}"

    for k, v in s0.items():
        assert v == 0 or math.isnan(
            v
        ), f"{k} does not \
            have a valid value, value = {v}"

    for k, v in sn.items():
        assert math.isfinite(
            v
        ), f"{k} does not \
            have a valid value, value = {v}"


def test_stash_tensors_cpu():
    tn = gen_unit_gauss(B, N, M)

    assert jnp.equal(stash_full_tensor(tn)["full_tensor"], tn).all()


def test_stash_tensors_cuda():
    # Create a random key

    tn = gen_unit_gauss(B, N, M)

    fn = jax.jit(stash_full_tensor)

    assert jnp.equal(fn(tn)["full_tensor"], tn).all
