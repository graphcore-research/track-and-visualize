# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
import os
from pathlib import Path
from typing import Callable, List

import pandas as pd

from ... import _config

if _config._WANDB_EXTRA:
    import wandb


def download_wandb_artifact(
    artifact_fullname: str, pd_read_fn: Callable = pd.read_pickle
) -> pd.DataFrame:
    """
    This fn, simply downloads the set of files in a artifact locally, \
        reads them to pd.DataFrame's and concatenates them to
    a single DataFrame.
    Args:
        artifact_fullname (str) : The fullname of the artefact account \
            belongs to.

        Can be copied from a URL that matches this:
        `https://wandb.ai/<entity>/<project-name>/artifact_files/nvis-logframe/<artifact-alias>/<artifact-version>/overview`


    Returns:
        str (path to root directory where artifact_files have been downloaded)

    """

    api = wandb.Api()
    artifact = api.artifact(artifact_fullname)

    outdir = artifact.download()
    p = Path(outdir)  # type: ignore
    artifact_files: List[str] = os.listdir(outdir)

    if len(artifact_files) > 1:
        artifact_files.sort(key=lambda x: int(x.split("-")[0]))
        # sort by step (might be brittle doing this!)
        # read_pickle should maybe be an argument?
        all_lfs = [pd_read_fn(p / file) for file in artifact_files]

        return pd.concat(all_lfs, ignore_index=True)

    else:
        return pd_read_fn(p / artifact_files[0])
