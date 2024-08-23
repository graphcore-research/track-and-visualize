from pathlib import Path
import shutil

from flax import linen as nn
import jax
from jax import tree_util
import jax.numpy as jnp
import optax
from functools import partial
from ..log.jax import track
from .._config import _libname
from ..log import read_pickle
import logging


INDIM = 2**8
OUTDIM = 10
B_SIZE = 1

class NestedModule(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features)(x)
        return nn.relu(x)

class MainModuleSetup(nn.Module):
    hidden_features: int
    output_features: int

    def setup(self):
        self.layer1 = NestedModule(features=self.hidden_features)
        self.layer2 = NestedModule(features=self.hidden_features)
        self.layer3 = NestedModule(features=self.hidden_features)
        self.layer4 = NestedModule(features=self.output_features)
    
    def __call__(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
class MainModuleCompact(nn.Module):
    hidden_features: int
    output_features: int

    @nn.compact
    def __call__(self, x):
        x = NestedModule(features=self.hidden_features)(x)
        x = NestedModule(features=self.hidden_features)(x)
        x = NestedModule(features=self.hidden_features)(x)
        x = NestedModule(features=self.output_features)(x)
        return x
    
def gen_unit_gauss(B,N,key):
    key = jax.random.PRNGKey(key)


    # Mean and standard deviation for the Gaussian
    mean = 0.0
    std_dev = 1.0

    # Generate Gaussian samples
    return mean + std_dev * jax.random.normal(key, (B,N),dtype=jnp.float32)


def test_flax_cpu():
    key = jax.random.PRNGKey(0)
    x = jnp.ones((B_SIZE, INDIM))
    step_size = 0.001

    model = MainModuleSetup(hidden_features=INDIM,output_features=OUTDIM)
    # Initialize parameters
    model_state = model.init(key, x)

    flat, _ = tree_util.tree_flatten_with_path(model_state)
    p_names = list(set([".".join([key.key for key in  fl[0][1:-1]]) for fl in flat]))


    optimizer = optax.adam(learning_rate=step_size, eps=2**-16)
    opt_state = optimizer.init(model_state)

    def loss_fn(params, x, y):
        y_pred = model.apply(params, x)
        return jnp.mean((y_pred - y) ** 2)


    def update(model_state,opt_state,optimizer, x, y):
        grads = jax.grad(loss_fn)(model_state,x,y)
        updates, opt_state = optimizer.update(grads, opt_state, model_state)
        model_state = optax.apply_updates(model_state, updates)

        return model_state,opt_state,grads


    with track(track_gradients=True,model_state=model_state,optimizer_state=opt_state) as tracker:
        for i in range(10):
            model_state,opt_state,grads = update(model_state,opt_state,optimizer,gen_unit_gauss(B_SIZE,INDIM,i),jnp.ones((1,)))
            tracker.step(model_state,opt_state,grads)

    if tracker.out_path != None:
        df = read_pickle(tracker.out_path)

        for tt in df.metadata.tensor_type.unique().tolist():

            if tt not in ['Activation', 'Gradient']:

                names = df.query(f'@df.metadata.tensor_type == "{tt}"').metadata.name.unique().tolist() #type: ignore
                logging.warning(names)

                assert set(names) == set(p_names), f'{tt} tracked {len(set(names))} but it should have tracked {len(set(p_names))}'
    else:
        assert False, 'Test failed because the tracker did not output a logframe'


    # clean up
    p = Path(f"./{_libname}")
    shutil.rmtree(p)



def test_flax_cuda():
    key = jax.random.PRNGKey(0)
    x = jnp.ones((B_SIZE, INDIM))
    step_size = 0.001

    model = MainModuleSetup(hidden_features=INDIM,output_features=OUTDIM)
    # Initialize parameters
    model_state = model.init(key, x)


    optimizer = optax.adam(learning_rate=step_size, eps=2**-16)
    opt_state = optimizer.init(model_state)

    flat, _ = tree_util.tree_flatten_with_path(model_state)
    p_names = list(set([".".join([key.key for key in  fl[0][1:-1]]) for fl in flat]))

    @jax.jit
    def loss_fn(params, x, y):
        y_pred = model.apply(params, x)
        return jnp.mean((y_pred - y) ** 2)


    def update(model_state,opt_state,optimizer, x, y):
        grads = jax.grad(loss_fn)(model_state,x,y)
        updates, opt_state = optimizer.update(grads, opt_state, model_state)
        model_state = optax.apply_updates(model_state, updates)

        return model_state,opt_state,grads


    with track(track_gradients=True,model_state=model_state,optimizer_state=opt_state) as tracker:
        for i in range(10):
            model_state,opt_state,grads = update(model_state,opt_state,optimizer,gen_unit_gauss(B_SIZE,INDIM,i),jnp.ones((1,)))
            tracker.step(model_state,opt_state,grads)


    if tracker.out_path != None:
        df = read_pickle(tracker.out_path)

        for tt in df.metadata.tensor_type.unique().tolist():

            if tt not in ['Activation', 'Gradient']:

                names = df.query(f'@df.metadata.tensor_type == "{tt}"').metadata.name.unique().tolist() #type: ignore
                logging.warning(names)

                assert set(names) == set(p_names), f'{tt} tracked {len(set(names))} but it should have tracked {len(set(p_names))}'
    else:
        assert False, 'Test failed because the tracker did not output a logframe'


    # clean up
    p = Path(f"./{_libname}")
    shutil.rmtree(p)

def test_flax_compact_cpu():
    key = jax.random.PRNGKey(0)
    x = jnp.ones((B_SIZE, INDIM))
    step_size = 0.001

    model = MainModuleCompact(hidden_features=INDIM,output_features=OUTDIM)
    # Initialize parameters
    model_state = model.init(key, x)

    flat, _ = tree_util.tree_flatten_with_path(model_state)
    p_names = list(set([".".join([key.key for key in  fl[0][1:-1]]) for fl in flat]))


    optimizer = optax.adam(learning_rate=step_size, eps=2**-16)
    opt_state = optimizer.init(model_state)

    def loss_fn(params, x, y):
        y_pred = model.apply(params, x)
        return jnp.mean((y_pred - y) ** 2)


    def update(model_state,opt_state,optimizer, x, y):
        grads = jax.grad(loss_fn)(model_state,x,y)
        updates, opt_state = optimizer.update(grads, opt_state, model_state)
        model_state = optax.apply_updates(model_state, updates)

        return model_state,opt_state,grads


    with track(track_gradients=True,model_state=model_state,optimizer_state=opt_state) as tracker:
        for i in range(10):
            model_state,opt_state,grads = update(model_state,opt_state,optimizer,gen_unit_gauss(B_SIZE,INDIM,i),jnp.ones((1,)))
            tracker.step(model_state,opt_state,grads)

    if tracker.out_path != None:
        df = read_pickle(tracker.out_path)

        for tt in df.metadata.tensor_type.unique().tolist():

            if tt not in ['Activation', 'Gradient']:

                names = df.query(f'@df.metadata.tensor_type == "{tt}"').metadata.name.unique().tolist() #type: ignore
                logging.warning(names)

                assert set(names) == set(p_names), f'{tt} tracked {len(set(names))} but it should have tracked {len(set(p_names))}'
    else:
        assert False, 'Test failed because the tracker did not output a logframe'


    # clean up
    p = Path(f"./{_libname}")
    shutil.rmtree(p)

def test_flax_compact_gpu():
    key = jax.random.PRNGKey(0)
    x = jnp.ones((B_SIZE, INDIM))
    step_size = 0.001

    model = MainModuleCompact(hidden_features=INDIM,output_features=OUTDIM)
    # Initialize parameters
    model_state = model.init(key, x)


    optimizer = optax.adam(learning_rate=step_size, eps=2**-16)
    opt_state = optimizer.init(model_state)

    flat, _ = tree_util.tree_flatten_with_path(model_state)
    p_names = list(set([".".join([key.key for key in  fl[0][1:-1]]) for fl in flat]))

    @jax.jit
    def loss_fn(params, x, y):
        y_pred = model.apply(params, x)
        return jnp.mean((y_pred - y) ** 2)


    def update(model_state,opt_state,optimizer, x, y):
        grads = jax.grad(loss_fn)(model_state,x,y)
        updates, opt_state = optimizer.update(grads, opt_state, model_state)
        model_state = optax.apply_updates(model_state, updates)

        return model_state,opt_state,grads


    with track(track_gradients=True,model_state=model_state,optimizer_state=opt_state) as tracker:
        for i in range(10):
            model_state,opt_state,grads = update(model_state,opt_state,optimizer,gen_unit_gauss(B_SIZE,INDIM,i),jnp.ones((1,)))
            tracker.step(model_state,opt_state,grads)


    if tracker.out_path != None:
        df = read_pickle(tracker.out_path)

        for tt in df.metadata.tensor_type.unique().tolist():

            if tt not in ['Activation', 'Gradient']:

                names = df.query(f'@df.metadata.tensor_type == "{tt}"').metadata.name.unique().tolist() #type: ignore
                logging.warning(names)

                assert set(names) == set(p_names), f'{tt} tracked {len(set(names))} but it should have tracked {len(set(p_names))}'
    else:
        assert False, 'Test failed because the tracker did not output a logframe'


    # clean up
    p = Path(f"./{_libname}")
    shutil.rmtree(p)