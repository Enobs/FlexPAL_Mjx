# metrics.py
import jax.numpy as jnp

def length(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(a - b)

def site_to_site_length(s, site_id_a: int, site_id_b: int) -> jnp.ndarray:
    pa = s.data.site_xpos[site_id_a]
    pb = s.data.site_xpos[site_id_b]
    return jnp.linalg.norm(pa - pb)
