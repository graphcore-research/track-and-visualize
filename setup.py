# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from pathlib import Path

import setuptools

setuptools.setup(
    name="nvis",
    version="0.1",
    install_requires=Path("requirements.txt").read_text().rstrip("\n").split("\n"),
    extras_require={
        'torch' : Path("requirements-torch.txt").read_text().rstrip("\n").split("\n"),
        'jax' :  Path("requirements-jax.txt").read_text().rstrip("\n").split("\n")
    },
    packages=["nvis", "nvis.vis", "nvis.log", "nvis.log.jax", "nvis.log.torch"],
)


