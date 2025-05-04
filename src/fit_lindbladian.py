import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Callable
from .results_plotter import collect_csv_files
jax.config.update("jax_enable_x64", True)


r"""
    Consider the Lindblandian in the following form
    \begin{align}
        \mathcal{L}(\rho)=\frac{i\omega_z}{2}[\sigma_z, \rho] + \gamma_{\text{PD}}(\sigma_z\rho\sigma_z - \rho) + \gamma_{\text{AD}}(\sigma_- \rho \sigma_+ - \frac{1}{2} \{\sigma_+\sigma_-, \rho \})
    \end{align}
    The corresponding left and right eigenoperators are
    \begin{align}
        R_i &= \frac{I+\sigma_z}{\sqrt{2}}, \frac{\sigma_z}{\sqrt{2}}, \sigma_-, \sigma_+ \\
        L_i &= \frac{I}{\sqrt{2}}, \frac{\sigma_z-I}{\sqrt{2}}, \sigma_+, \sigma_-
    \end{align}
"""

I = jnp.eye(2, dtype=jnp.complex128)
sigma_z = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
sigma_x = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
sigma_y = jnp.array([[0.0, -1j], [1j, 0.0]], dtype=jnp.complex128)
sigma_p = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.complex128)
sigma_m = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.complex128)

zero_state = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex128) 
one_state = jnp.array([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.complex128)
plus_state = jnp.array([[1.0, 1.0], [1.0, 1.0]], dtype=jnp.complex128) / 2
minus_state = jnp.array([[1.0, -1.0], [-1.0, 1.0]], dtype=jnp.complex128) / 2
plus_i_state = jnp.array([[1.0, -1j], [1j, 1.0]], dtype=jnp.complex128) / 2
minus_i_state = jnp.array([[1.0, 1j], [-1j, 1.0]], dtype=jnp.complex128) / 2


def get_markov_approx_rho(omega_z, gamma_ad, gamma_pd, init_dm):

    Right_eigop = jnp.array([(I + sigma_z) / jnp.sqrt(2), sigma_z / jnp.sqrt(2), sigma_m, sigma_p]) # (4, 2, 2)
    Left_eigop = jnp.array([I / jnp.sqrt(2), (sigma_z - I) / jnp.sqrt(2), sigma_p, sigma_m])
    mu_i = jnp.trace(Left_eigop @ init_dm, axis1=-2, axis2=-1)
    
    eigen_val = jnp.array([0, -gamma_ad, 1j * omega_z - 2 * gamma_pd - 0.5 * gamma_ad, -1j * omega_z - 2 * gamma_pd - 0.5 * gamma_ad])
    
    def rho(t):
        lambda_t = eigen_val[:, None] * t[None, :]  # (4, len(t))
        return jnp.einsum("i...,ij->j...", mu_i[:, None, None] * Right_eigop, jnp.exp(lambda_t))
    
    return rho

def quantum_relative_entropy_loss(rho1: jax.Array, rho2: jax.Array):
    
    """
        rho1: shape (N, 2, 2)
        rho2: shape (N, 2, 2)
        
        return shape (1,)
    """
    
    def matrix_function(M: jax.Array ,fn: Callable) -> jax.Array:
        eigvals, eigvecs = jnp.linalg.eigh(M)
        return eigvecs @ jnp.stack([jnp.diag(fn(eigval)) for eigval in eigvals], axis=0) @ eigvecs.transpose(0, -1, -2).conj()
    
    log_rho1 = matrix_function(rho1, jnp.log2)
    log_rho2 = matrix_function(rho2, jnp.log2)
    
    relative_entropy = jnp.abs(jnp.trace(rho1 @ (log_rho1 - log_rho2), axis1=-2, axis2=-1))
    
    return jnp.mean(relative_entropy)

if __name__ == "__main__":
    
    files = collect_csv_files(tomography_type="state", method="crosstalk", mode="bloch", backend="ibm_strasbourg", init_state="+,+,+,+", qubit="*", num_points=50, gates_per_point=50, shots=8192)
    
    rho = get_markov_approx_rho(omega_z=4, gamma_ad=0.2, gamma_pd=0.4, init_dm=0.3 * one_state + 0.2 * plus_i_state + 0.5 * minus_i_state)
    t = jnp.linspace(0, 10, 100)
    rho_val = rho(t)
    X = jnp.trace(rho_val @ sigma_x, axis1=-2, axis2=-1)
    Y = jnp.trace(rho_val @ sigma_y, axis1=-2, axis2=-1)
    Z = jnp.trace(rho_val @ sigma_z, axis1=-2, axis2=-1)
    
    plt.plot(t, X, label='X')
    plt.plot(t, Y, label='Y')
    plt.plot(t, Z, label='Z')
    plt.ylim([-1.0, 1.0])
    
    plt.legend()
    plt.show()