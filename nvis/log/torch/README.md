The torch framework specific utilities for extracting the tensor statistics during training.


One thing which I would imagine being useful is, reusing the torch.compile kernel use for capturing activations for opt_state and weights (which it currently is not)