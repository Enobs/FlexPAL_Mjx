import jax.numpy as jnp
import jax

def linear(W, x):
    return jnp.dot(W, x)

W = -2* jnp.eye(3)
Xs = jnp.arange(9.0).reshape(3,3)
print(jax.vmap(linear, in_axes=(None, 0))(W, Xs))
