# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
import logging
import shutil
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ..track.common import read_pickle
from ..track.torch import stash_all_stats_and_hist, track

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


INDIM = 2**8
B_SIZE = 1


class SimpleModel(nn.Module):
    def __init__(self, input_size=32):
        super().__init__()
        self.l1 = nn.Linear(input_size, input_size // 2, bias=False)
        self.l2 = nn.Linear(input_size // 2, 1, bias=False)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


def test_tracker_cpu():

    model = SimpleModel(input_size=INDIM).to("cpu")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # type: ignore

    with track(
        module=model,
        track_gradients=True,
        optimizer=optimizer,
        track_weights=True,
        offload_inc=1,
        stash_value=stash_all_stats_and_hist,
    ) as tracker:
        for i in range(10):
            X = torch.randn((B_SIZE, INDIM))
            Y = torch.randint(0, 2, (B_SIZE, 1)).float()
            optimizer.zero_grad()
            y = model(X)
            loss = loss_fn(y, Y)
            loss.backward()
            optimizer.step()
            tracker.step()

    p_names = [n[0].split(".")[0] for n in model.named_parameters()]

    df = read_pickle(tracker.out_path)  # type: ignore

    for tt in df.metadata.tensor_type.unique().tolist():

        if tt not in ["Activation", "Gradient"]:

            names = (
                df.query(
                    f'@df.metadata.tensor_type == "{tt}"\
                             '
                )
                .metadata.name.unique()
                .tolist()
            )  # type: ignore

            assert set(names) == set(
                p_names
            ), f"{tt} tracked {len(set(names))} but it should have \
                tracked {len(set(p_names))}"

    # clean up
    p = Path(tracker.out_path).parent
    shutil.rmtree(p)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tracker_cuda():
    model = SimpleModel(input_size=INDIM).to("cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # type: ignore

    with track(
        module=model,
        track_gradients=True,
        optimizer=optimizer,
        track_weights=True,
        offload_inc=1,
        stash_value=stash_all_stats_and_hist,
    ) as tracker:
        for i in range(10):
            X = torch.randn((B_SIZE, INDIM)).to("cuda")
            Y = torch.randint(0, 2, (B_SIZE, 1)).float().to("cuda")
            optimizer.zero_grad()
            y = model(X)
            loss = loss_fn(y, Y)
            loss.backward()
            optimizer.step()
            tracker.step()

    p_names = [n[0].split(".")[0] for n in model.named_parameters()]
    logging.warning(p_names)

    if tracker.out_path is not None:
        df = read_pickle(tracker.out_path)

        for tt in df.metadata.tensor_type.unique().tolist():

            if tt not in ["Activation", "Gradient"]:

                names = (
                    df.query(
                        f'@df.metadata.tensor_type == "{tt}"\
                    '
                    )
                    .metadata.name.unique()
                    .tolist()
                )  # type: ignore

                assert set(names) == set(
                    p_names
                ), f"{tt} tracked {len(set(names))} but it should \
                    have tracked {len(set(p_names))}"
    else:
        assert (
            False
        ), "Test failed because the tracker \
            did not output a logframe"

    # clean up
    p = Path(tracker.out_path).parent
    shutil.rmtree(p)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tracker_torch_compile():

    model = SimpleModel(input_size=INDIM).to("cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # type: ignore

    model = torch.compile(model)  # type: ignore
    p_names = [".".join(n[0].split(".")[:-1]) for n in model.named_parameters()]
    with track(
        module=model,  # type: ignore
        track_gradients=True,
        optimizer=optimizer,
        track_weights=True,
        offload_inc=1,
        stash_value=stash_all_stats_and_hist,
    ) as tracker:
        for _ in range(10):
            X = torch.randn((B_SIZE, INDIM)).to("cuda")
            Y = torch.randint(0, 2, (B_SIZE, 1)).float().to("cuda")
            optimizer.zero_grad()
            y = model(X)
            loss = loss_fn(y, Y)
            loss.backward()
            optimizer.step()
            tracker.step()

    logging.warning(p_names)

    if tracker.out_path is not None:
        df = read_pickle(tracker.out_path)

        for tt in df.metadata.tensor_type.unique().tolist():

            if tt not in ["Activation", "Gradient"]:

                names = (
                    df.query(
                        f'@df.metadata.tensor_type == "{tt}"\
                        '
                    )
                    .metadata.name.unique()
                    .tolist()
                )  # type: ignore

                assert set(names) == set(
                    p_names
                ), f"{tt} tracked {len(set(names))} but it should \
                    have tracked {len(set(p_names))}"
    else:
        assert (
            False
        ), "Test failed because the tracker did not \
            output a logframe"

    # clean up
    p = Path(tracker.out_path).parent
    shutil.rmtree(p)
