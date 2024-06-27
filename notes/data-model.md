## Tensors Stats
    Activations (Input to forward pass for each nn.Module)
    Gradients 
    Weights (inbetween each optimiser step)
    Optimiser State ()
    Log Shape

## Static Hyper Parameters
Model Config

## Dynamic Hyper Parameters
W.r.t. (Parameter Group, Iteration)
    Learning Rate
    Weight Decay 

## Training Stats
W.r.t Iteration:
    Training Loss
    Validation Loss (although likely at some interval) 



# Data Model:

Need to be able to query by Module & by Iteration (or both),

Use Cases;
    Single experiment run, single layers across intervals of n (need to assure each Iteration has the same Stats) so the plots make sense
    Single iteration run, across all layers, (need to assure each module has the same Stats)
    Querying by module type..
- 

Log:
- name -> str (experiment name)
- modules -> List[Module]
- iterations -> List[Iteration]
- config -> Config
- training_stats -> {
    train_loss -> List[Tuple[it -> int, loss -> float]]
    val_loss -> List[Tuple[it -> int, loss -> float]]
}

Config:
- (architecture details)
- (optimiser params)
...

Module:
- name -> str (name)
- type -> str (torch.nn.*)
- stats -> Stats

Iteration:
- it -> int (iteration number)
- modules_stats -> : Dict[Module.name, Stats] (stats for modules for that iteration)
- state_type -> Scalar | Hist | All (If not logging detail stats every iteration)

Stats:
- activation -> Scalar | Hist | All | None
- weights -> Scalar | Hist | All | None
- gradients -> Scalar | Hist | All | None
- optim_state -> Scalar | Hist | All | None



Scalar:
... (Doug Stats from u-mup)

Hist: 
...

All:
- scalar: Scalar
- hist: Hist