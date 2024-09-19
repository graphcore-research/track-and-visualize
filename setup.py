# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
import setuptools
import pathlib
import pkg_resources


def load_requirements(filename: str):
    with pathlib.Path(filename).open() as requirements_txt:
        requirements = [
            str(requirement)
            for requirement
            in pkg_resources.parse_requirements(requirements_txt)
        ]
        return requirements


setuptools.setup(
    name="tandv",
    version="0.1",
    install_requires=load_requirements("requirements.txt"),
    extras_require={
        'torch': load_requirements("requirements-torch.txt"),
        'jax':  load_requirements("requirements-jax.txt"),
        'wandb': ['wandb']
    },
    packages=["tandv", "tandv.viz", "tandv.track", "tandv.track.common",
              "tandv.track.jax", "tandv.track.torch"],
    #  Disable zip_safe to allow compatibility with mypy.
    # See: https://mypy.readthedocs.io/en/stable/installed_packages.html#making-pep-561-compatible-packages
    zip_safe=False,
)