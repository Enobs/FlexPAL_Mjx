import jax
import jax.numpy as jnp
import dataclasses
from dataclasses import dataclass
from flax import struct

@dataclass
class State:
    params: jnp.ndarray

@jax.jit
def loss_function(state: State) -> float:
    return jnp.sum(state.params ** 2)


initial_state = State(params=jnp.array([1.0, 2.0, 3.0]))
loss_function(initial_state)


# grad_fn = jax.grad(loss_function)
# grads = grad_fn(initial_state)
# print(grads)
