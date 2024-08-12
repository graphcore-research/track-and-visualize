
# import wandb.errors
# import wandb.sdk
from typing import Callable, List
from pathlib import Path
import wandb
import pandas as pd
import os
from ._log_handler import combine_incremental_dfs

def download_wandb_artifact(artifact_fullname: str, pd_read_fn: Callable = pd.read_pickle) -> pd.DataFrame:
    """
        Args:
            artifact_fullname (str) : The fullname of the artefact account belongs to.

            Can be copied from a URL that matches this:
            `https://wandb.ai/<entity>/<project-name>/artifacts/nvis-logframe/<artifact-alias>/<artifact-version>/overview`
            

        Returns:
            str (path to root directory where artifacts have been downloaded)
    
    """

    api = wandb.Api()
    artifact = api.artifact(artifact_fullname)

    outdir = artifact.download()
    p = Path(outdir) #type: ignore
    artifacts: List[str] = os.listdir(outdir)


    if len(artifacts) > 1:
        artifacts.sort(key=lambda x: int(x.split('-')[0]))
        # sort by step (might be brittle doing this!)
        # read_pickle should maybe be an argument?
        all_lfs = [pd_read_fn(p/file) for file in artifacts]

        return pd.concat(all_lfs,ignore_index=True)

    else:
        return pd_read_fn(p/artifacts[0])