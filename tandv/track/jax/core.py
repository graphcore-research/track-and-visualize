# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
import re
from functools import partial
from re import Pattern
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from ... import _config
from ..common._tracker import BaseTracker
from ..common._types import TT, Event, Stash, StashFn
from ..common._write import lf_to_pickle

if _config._JAX_EXTRA:
    import jax
    from flax.linen import intercept_methods
    from jax.tree_util import tree_flatten_with_path
    from optax._src.base import EmptyState, NamedTuple

    from ._utils import backward_callback, forward_callback
    from .stash_values import stash_all_stats_and_hist


StashValueFn = Callable[[jax.Array], Any]
NoneType = type(None)
NamePattern = Union[None, Pattern[str], str]


def is_weights(p: Tuple[jax.tree_util.DictKey]) -> bool:
    """
    Help fn to tell if a leaf node in an arbitrary \
        depth pytree contains weights
    """
    if p[0].key == "params" and p[-1].key == "kernel":
        return True
    return False


def rmap_array(value: Any, fn: Callable[[jax.Array], Any]) -> Any:
    if isinstance(value, (tuple, list)):
        return type(value)(rmap_array(a, fn) for a in value)
    if isinstance(value, dict):
        return {rmap_array(k, fn): rmap_array(a, fn) for k, a in value.items()}
    if isinstance(value, jax.Array):
        return fn(value)


def array_dtype(tensor: jax.Array) -> str:
    return str(tensor.dtype)


def default_stash(event: Event, stash_value: StashValueFn) -> Stash:

    return Stash(
        name=event.name,
        type=event.type,
        tensor_type=event.tensor_type,
        dtype=rmap_array(event.value, array_dtype),
        value=rmap_array(event.value, stash_value),
    )


def get_stash_fn(
    stash_value: Optional[StashValueFn] = None, stash: Optional[StashFn] = None
) -> StashFn:
    if stash_value and stash:
        raise ValueError(
            "Cannot provide StashValueFn and StashFn to get_stash_fn()")
    if stash:
        return stash
    return partial(default_stash,
                   stash_value=stash_value or
                   stash_all_stats_and_hist)


class JaxTracker(BaseTracker):
    def __init__(
        self,
        stash: Callable[[Event], Stash],
        model_state: Union[Dict, None],
        optimizer_state: Union[Tuple, None],
        track_gradients: bool,
        async_offload: bool,
        offload_inc: int,
        offload_fn: Callable,
        include: Union[Pattern[str], str, None],
        exclude: Union[Pattern[str], str, None],
        use_wandb: bool,
        name: str | None = None,
        init_step: int | None = None,
    ):
        super().__init__(
            stash=stash,
            name=name,
            init_step=init_step,
            async_offload=async_offload,
            offload_inc=offload_inc,
            offload_fn=offload_fn,
            use_wandb=use_wandb,
        )
        self.track_gradients = track_gradients
        self.track_weights = False
        self.track_optimizer = False
        self.include = re.compile(include) if \
            isinstance(include, str) else include
        self.exclude = re.compile(exclude) if \
            isinstance(exclude, str) else exclude

        # Capture statistics on initial states of model and optimiser
        if not isinstance(model_state, NoneType):
            self._stash_model_weights(model_state)
            self.track_weights = True

        if not isinstance(optimizer_state, NoneType):
            self._stash_opt_state(opt_state=optimizer_state)
            self.track_optimizer = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any | None,
    ) -> None:
        return super().__exit__(exc_type, exc, traceback)

    def _check_included(self, name: str) -> bool:
        return ((not self.include) or self.include.search(name)) and \
            not (self.exclude and self.exclude.search(name))  # type: ignore

    def create_event_and_stash(
        self,
        value: Any,
        name: str,
        type: str,
        tensor_type: TT,
        stash=True,
        *args,
        **kwargs,
    ) -> None:

        if stash:
            self.stash_event(
                Event(
                    name=name,
                    type=type,
                    tensor_type=tensor_type,
                    value=value,
                    args=args,
                    kwargs=kwargs,
                )
            )

    def _data_interceptor(self, next_fun, args, kwargs, context):
        """
        Captures the output of each linen module for fwd and bwd passes

        Args:
            next_fun: ...
            args : ...
            kwargs: ...
            context: ...

        Returns


        """

        output = next_fun(*args, **kwargs)

        name = ".".join(context.module.path)
        m_type = str(type(context.module))
        # check include and exclude here...
        included = self._check_included(name)

        forward_callback(
            partial(
                self.create_event_and_stash,
                name=name,
                type=m_type,
                tensor_type="Activation",
                stash=included,
                args=(),
                kwargs={},
            ),
            output,
        )

        output = backward_callback(
            partial(
                self.create_event_and_stash,
                name=name,
                type=m_type,
                tensor_type="Gradient",
                stash=self.track_gradients and included,
                args=(),
                kwargs={},
            ),
            output,
        )

        # assert len(output) == 1
        # backward callback returns a tuple of len 1, where as the \
        # module is expecting what's in that tuple (index it outß)
        return output[0]

    def _stash_model_weights(self, params: Dict) -> None:
        """
        Creates a stash for each weight parameter jax.Array in the network.
        Does not capture stats on bias parameters.

        Args
            params (Dict): PyTree which contains the model state

        """
        flat_tree, _ = tree_flatten_with_path(params)
        for path, value in flat_tree:
            if is_weights(path):
                name = ".".join([pk.key for pk in path[1:-1]])
                if self._check_included(name):
                    self.stash_event(Event(
                        name,
                        None,
                        "Weights",
                        value, (), {}))

    def _stash_opt_state(self, opt_state: Tuple) -> None:
        """
        Creates a stash for each jax.Array containing \
            optimizer state in the network.
        Does not capture stats on opt state w.r.t bias

        Args
            opt_state (Tuple): Tuple of optax state object(s)

        """
        for os in opt_state:
            # skip non optstates... or emptystates
            if os.__class__.__orig_bases__[0] is NamedTuple and not isinstance(
                os, EmptyState
            ):
                state_fields: List[str] = [
                    f for f in os._fields if f != "count"]

                for statefield in state_fields:
                    flat_tree, _ = tree_flatten_with_path(
                        getattr(os, statefield))

                    for path, value in flat_tree:
                        if is_weights(path):
                            # check include and exclude here...
                            name = ".".join([pk.key for pk in path[1:-1]])
                            if self._check_included(name):
                                self.stash_event(
                                    Event(
                                        ".".join([
                                            pk.key for pk in path[1:-1]]),
                                        None,
                                        f"Optimiser_State\
                                            .{statefield}",  # type: ignore
                                        value,
                                        (),
                                        {},
                                    )
                                )

    def _stash_gradients(self, grads: Dict) -> None:
        """
        Creates a stash for each jax.Array containing \
            weight gradients in the network.
        Does not capture stats on grads w.r.t bias

        Args
            opt_state (Tuple): Tuple of optax state object(s)

        Returns
            None

        """
        flat_tree, _ = tree_flatten_with_path(grads)
        for path, value in flat_tree:
            if is_weights(path):
                name = ".".join([pk.key for pk in path[1:-1]])
                # check include and exclude here...
                if self._check_included(name):
                    self.stash_event(
                        Event(name, None, "Weight_Gradients", value, (), {})
                    )

    def step(
        self,
        model_state: Optional[Dict] = None,
        optimizer_state: Optional[Tuple] = None,
        gradients: Optional[Dict] = None,
    ) -> None:
        """
        Method to notify the tracker of the completion of a \
            single training step/iteration.
        Takes as optional arguments, the model_state, \
            optimizer_state and gradients to capture stats on them.

        Args
            model_state (Dict | None): PyTree of the model parameters
                Cannot be none if `model_state` arg was provided \
                    to the `track` method when initalisaing the JaxTracker
            optimizer_state (Tuple | None): Tuple containing the \
                optax optimizer state object(s)
                Cannot be none if `model_state` arg was provided to the \
                    `track` method when initalisaing the JaxTracker
            gradients (Dict | None): PyTree of the gradients w.r.t. \
                the model parameters

        Returns
            None

        """

        # capture grads
        if self.track_gradients:
            assert not isinstance(
                gradients, NoneType
            ), "You must provide gradients to the step call, as JaxTracker \
                initialised with `track_gradients == True`"
            self._stash_gradients(grads=gradients)
        # clear stashes to global dict

        self._internal_step()

        # capture model_state
        if self.track_weights:
            assert not isinstance(
                model_state, NoneType
            ), "You must provide model_state to the step call, as JaxTracker \
                initialised with `model_state != None`, therefore it is \
                    configured to track stats on parameter tensors"
            self._stash_model_weights(params=model_state)
        # capture opt_state
        if self.track_optimizer:
            assert not isinstance(
                optimizer_state, NoneType
            ), "You must provide `optimizer_state` to the step call, as \
                JaxTracker initialised with `optimizer_state != None`, \
                    therefore it is configured to track stats on the \
                        optimizer state tensors"
            self._stash_opt_state(opt_state=optimizer_state)

    def intercept(self, f: Callable, *args, **kwargs) -> Any:
        """
        Higher-order functions which facilitates the capturing \
            of intermediate stats for
        the fwd and bwd passes. Simply executes the \
            function and returns its args.

        Example Usage:
        ```python
            # fn
            def update(model,model_state,opt_state,optimizer, x, y):
                def loss_fn(params, x, y):
                    y_pred = model.apply(params, x)
                    return jnp.mean((y_pred - y) ** 2)
                grads = jax.grad(loss_fn)(model_state,x,y)
                # optional, update step does not need to occur within \
                # context of
                # intercept fn
                updates, opt_state = optimizer.update(\
                    grads, opt_state, model_state)
                model_state = optax.apply_updates(model_state, updates)

                return model_state,opt_state,grads

            # pass the update fn and its args into the trackers intercept fn
            model_state,opt_state,grads = tracker.intercept(\
                update,model,model_state,opt_state,optimizer, x, y)
        ```

        """

        with intercept_methods(self._data_interceptor):
            return f(*args, **kwargs)


def track(
    model_state: Union[Dict, None] = None,
    optimizer_state: Union[Tuple, None] = None,  # type: ignore
    track_gradients: bool = False,
    include: NamePattern = None,
    exclude: NamePattern = None,
    stash_value: Optional[StashValueFn] = None,
    async_offload: bool = False,
    offload_inc: int = 10,
    offload_type: Literal[".pkl"] = ".pkl",
    use_wandb: bool = False,
    init_step=None,
) -> JaxTracker:
    """
    Function for initialising the Jax (Flax) Tensor Tracker context manager.

    By default it tracks the stastics for  what we refer to as the Activations\
        (i.e. the outputs of the forward method in the nn module).
    However it can also track gradients, weights and optimiser state.
    At the end of training it will write all the logs to a LogFrame \
        (a `pd.DataFrame` which conforms to the schema outline in our docs)


    ```python

        # pass whichever pytrees you're tracking into track(...)
        with track(...) as tracker:
            #Your training loop goes here
            for i in range(max_its):
                # pass fwd & bwd pass fn into intercept
                _,..,_ = tracker.intercept(...)
                # pass whichever pytrees you're tracking into step(...)
                tracker.step(...) # at the end of your loop

    ```

    Args:
        model_state (Dict | None): PyTree of the model parameters \
            (output from `your_model.init()`)
        optimizer_state (Tuple | None): Tuple containing the optax optimizer \
            state object(s) (output from `your_optimizer.init(model_state)`)
        track_gradients (bool): Whether or not you wish to track the gradients.
        include (None | Pattern[str] | str) : A module or modules (via regex) \
            you wish to track.
        exclude (None | Pattern[str] | str) : A module or modules (via regex) \
            you wish not to track.
        stash_value (StashValueFn): This is the statistics you wish to track, \
            it defaults to `stash_all_stats_and_hist`, you can provide a \
                custom fn here however inspect the other stash_fns to see the \
                    required args/returns values.
        async_offload (bool): If true the set of stashes since last offloaded \
            are serialised and passed to a seperate python process to be \
                converted to a Logframe (currently a very limited reduction \
                in overhead, but working on improving it)
        offload_inc (int): How frequently you wish to the stashes from memory \
            to disk, i.e. offload more frequently to minimise Torch Tracker's \
                memory usage. If using wandb, this value should be the same \
                    (or a multiple) of the increment being used to call \
                        `wandb.log'
        offload_type (Literal['.pkl']): The file format you wish to write the \
            LogFrame(s) to disk as.
        use_wandb (bool): If you wish to push the Logframe(s) as artifacts \
            and get summary numerics statistic in wandb. \
                (`offload_inc` should be the same (or a multiple) of the \
                    increment being used to call `wandb.log')
        init_step (int): The tracker has an internal step property for \
            assigning statistics to the correct iteration, if \
                `init_step == None`, defaults to zero, \
                    else init_step (if you are continuing \
                        from a checkpoint for example)


    Returns:
        JaxTracker (The context manager)

    """

    if _config._JAX_EXTRA:
        offload_fn: Dict[str, Callable] = {".pkl": lf_to_pickle}

        tracker = JaxTracker(
            stash=get_stash_fn(stash_value=stash_value, stash=None),
            async_offload=async_offload,
            offload_inc=offload_inc,
            offload_fn=offload_fn[offload_type],
            track_gradients=track_gradients,
            model_state=model_state,
            optimizer_state=optimizer_state,
            include=include,
            exclude=exclude,
            use_wandb=use_wandb,
            init_step=init_step,
        )

        return tracker

    else:
        raise ImportError(
            f"track requires jax,flax and optax to be installed, \
                please install via `pip install jax,flax,optax or \
                    `pip install {_config._libname}[jax]"
        )


__all__ = ["JaxTracker", "track"]
