# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from pathlib import Path

import setuptools

setuptools.setup(
    name="tandv",
    version="0.1",
    install_requires=Path("requirements.txt")
            .read_text()
            .rstrip("\n")
            .split("\n"),
    extras_require={
        'torch': Path("requirements-torch.txt")
            .read_text()
            .rstrip("\n")
            .split("\n")[1:],
        'jax':  Path("requirements-jax.txt")
            .read_text()
            .rstrip("\n")
            .split("\n")[1:],
        'wandb': ['wandb']
    },
    packages=["tandv", "tandv.viz", "tandv.track", "tandv.track.common",
              "tandv.track.jax", "tandv.track.torch"],
)