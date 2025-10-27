import jax.numpy as jnp
import jax 

def rotmat_to_quat(R: jnp.ndarray) -> jnp.ndarray:
    """R: (3,3) -> quat (w,x,y,z)"""
    t = jnp.trace(R)

    # Branch 0: If trace > 0 (Standard Case)
    def branch0(R_):
        r = jnp.sqrt(1.0 + t)
        w = 0.5 * r
        x = (R_[2,1] - R_[1,2]) / (2.0 * r)  # Use r, not w, in denominator for clarity
        y = (R_[0,2] - R_[2,0]) / (2.0 * r)
        z = (R_[1,0] - R_[0,1]) / (2.0 * r)
        return jnp.array([w, x, y, z]) # <-- CORRECTED: z instead of second y

    # Branches 1, 2, 3: For small trace (when r would be close to zero)
    def from_x(R_):
        r = jnp.sqrt(1.0 + 2*R_[0,0] - t)
        x = 0.5 * r
        # Use r in denominator
        w = (R_[2,1] - R_[1,2]) / (2.0 * r)
        y = (R_[0,1] + R_[1,0]) / (2.0 * r)
        z = (R_[0,2] + R_[2,0]) / (2.0 * r)
        return jnp.array([w, x, y, z])
    
    def from_y(R_):
        r = jnp.sqrt(1.0 + 2*R_[1,1] - t)
        y = 0.5 * r
        # Use r in denominator
        w = (R_[0,2] - R_[2,0]) / (2.0 * r)
        x = (R_[0,1] + R_[1,0]) / (2.0 * r)
        z = (R_[1,2] + R_[2,1]) / (2.0 * r)
        return jnp.array([w, x, y, z])

    def from_z(R_):
        r = jnp.sqrt(1.0 + 2*R_[2,2] - t)
        z = 0.5 * r
        # Use r in denominator
        w = (R_[1,0] - R_[0,1]) / (2.0 * r)
        x = (R_[0,2] + R_[2,0]) / (2.0 * r)
        y = (R_[1,2] + R_[2,1]) / (2.0 * r)
        return jnp.array([w, x, y, z])

    # Case for small trace (t <= 0)
    i = jnp.argmax(jnp.array([R[0,0], R[1,1], R[2,2]]))
    
    # jax.lax.switch selects a function to execute based on index 'i' (0, 1, or 2)
    # R is passed as the operand to the selected function
    q_low_trace = jax.lax.switch(i, [from_x, from_y, from_z], R)
    
    # jax.lax.cond selects between the two main cases:
    # 1. High trace (branch0)
    # 2. Low trace (q_low_trace)
    q = jax.lax.cond(t > 0.0, 
                     lambda R_: branch0(R_),
                     lambda R_: q_low_trace, 
                     operand=R) # Pass R as the operand

    # The result is already normalized (r is the magnitude), 
    # but an extra normalization step is sometimes included for numerical stability.
    # q = q / jnp.linalg.norm(q) 
    return q