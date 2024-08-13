import logging
from pathlib import Path
import shutil,sys,os
import wandb


# add local lib to sys path for relative import
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from nvis.log.torch.stash_values import stash_scalar_stats
from nvis.log.torch import track, stash_all_stats_and_hist
from nvis.log import read_pickle
import torch
from nvis._config import _libname
from torch import nn
from torch.nn import functional as F
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


INDIM = 2**8
B_SIZE = 2**4

class SimpleModel(nn.Module):
    def __init__(self, input_size = 32):
        super().__init__()
        self.l1 = nn.Linear(input_size,input_size // 2, bias=False)
        self.l2 = nn.Linear(input_size // 2, 1, bias=False)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))
    

if __name__ == '__main__':
    model = SimpleModel(input_size=INDIM).to('cuda')
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #type: ignore
    use_wand = True
    
    if use_wand:
        wandb.init(project='nvis')


    with track(
        module=model,
        track_gradients=True,
        optimizer=optimizer,
        track_weights=True,
        offload_inc=100,
        use_wandb=use_wand,
        async_offload=True,
        stash_value=stash_all_stats_and_hist) as tracker:
        for i in range(1000): 
            X = torch.randn((B_SIZE,INDIM)).to('cuda')
            Y = torch.randint(0,2,(B_SIZE,1)).float().to('cuda')
            y = model(X)
            loss = loss_fn(y,Y)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0 and use_wand:
                wandb.log(
                    {'train_loss' : loss},
                    step = i
                )
            tracker.step()
            optimizer.zero_grad()


    df = read_pickle(tracker.out_path) #type: ignore

    # clean up
    p = Path(f"./{_libname}")
    shutil.rmtree(p)