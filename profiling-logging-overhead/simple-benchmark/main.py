from contextlib import nullcontext
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import schedule

import os,sys


# add local lib to sys path for relative import
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from nvis.log.torch import (stash_full_tensor, stash_scalar_stats, stash_hist ,stash_all_stats_and_hist)
from nvis.log.torch import track

import pickle
import logging


torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


class SimpleModel(nn.Module):
    def __init__(self, input_size = 32):
        super().__init__()
        self.l1 = nn.Linear(input_size,input_size // 2, bias=False)
        self.l2 = nn.Linear(input_size // 2, 1, bias=False)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


if __name__ == '__main__':

    # from my_logger import track
    stash_values = [stash_scalar_stats,stash_hist,stash_all_stats_and_hist]
    all_perms = [
        tuple(['activations']),
        # ('activations','gradients'),
        ('activations','opt_state'),
        ('weights','activations','opt_state')
        ]
    
    batch_size = [1,2,4,8,16,32,64,128,256][-1:]
    indims = [2**exp for exp in range(10,14)][-1:]
    device = "cuda" 
    num_its = 10

    # Gen X,Y
    _X = torch.randn((batch_size[-1],indims[-1]))
    _Y = torch.randint(0,2,(batch_size[-1],1)).float()

    results = {}

    

    for baseline in [False]:
        for compile in [True,False]:
            for svf in stash_values:
                for which_to_track in all_perms:
                    for bsize in batch_size:
                        for indim in indims:
                            
                            # Create a copy of the tensor subset and move to gpu
                            # X = _X[:bsize,:indim].clone().to(device)
                            # Y = _Y[:bsize,:].clone().to(device)

                            

                            # define model

                            model = SimpleModel(input_size=indim).to(device)
                            loss_fn = torch.nn.CrossEntropyLoss()
                            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                            conf_name = f"{bsize}-{indim}-{'-'.join(which_to_track)}-{svf.__name__}{'-compiled' if compile else ''}" if not baseline else f"{bsize}-{indim}-baseline{'-compiled' if compile else ''}"
                            

                            if baseline:
                                tracker_context = nullcontext()
                            else:
                                tracker_context = track(
                                    module=model,
                                    grad=True if 'gradients' in which_to_track else False, #
                                    optimiser=optimizer if 'opt_state' in which_to_track else None,
                                    track_weights=True if 'weights' in which_to_track else False,
                                    stash_value=svf,
                                )

                            # compile the model
                            if compile:
                                print("compiling the model... (takes a ~minute)")
                                # torch.compiler.reset()
                                unoptimized_model = model
                                # torch._dynamo.config.capture_scalar_outputs = True
                                model = torch.compile(model,
                                                      options={'trace.enabled' : True,'trace.graph_diagram': False}) # 
                            

                            

                                
                            total_time = 0 
                            # Benchmark
                            with tracker_context as tracker:
                                for i in range(3+num_its):
                                    X = torch.randn((bsize,indim)).cuda()
                                    Y = torch.randint(0,2,(bsize,1)).float().cuda()
                                    start = torch.cuda.Event(enable_timing=True)
                                    end = torch.cuda.Event(enable_timing=True)
                                    start.record()
                                    optimizer.zero_grad()
                                    y = model(X)
                                    loss = loss_fn(y,Y)
                                    loss.backward()
                                    optimizer.step()
                                    if not baseline:
                                        tracker.step()
                                    end.record()
                                    torch.cuda.synchronize()
                                    if i > 2:
                                        total_time += start.elapsed_time(end)
                                    logging.warning('-'*10 + conf_name.upper() + '-'*10 + ' ' + str(start.elapsed_time(end)))
                            logging.warning('-'*10 + conf_name.upper() + '-'*10 + ' ' + str(total_time / num_its))
                            results[conf_name]= total_time / num_its
                            if not baseline:
                                logging.warning(tracker._global_stash)

                    if baseline:
                        break     
                if baseline:
                    break   

    with open('results.pkl', 'wb') as f:
        pickle.dump(results,f)                    
