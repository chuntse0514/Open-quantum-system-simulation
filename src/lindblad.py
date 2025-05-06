import jax
import jax.numpy as jnp
import optax
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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

# ------------------------------------------------------------------
# Pauli matrices (complex128)
# ------------------------------------------------------------------
I2 = jnp.eye(2, dtype=jnp.complex128)
σx = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
σy = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
σz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
σp = jnp.array([[0, 0], [1, 0]], dtype=jnp.complex128)
σm = jnp.array([[0, 1], [0, 0]], dtype=jnp.complex128)
pauli_stack = jnp.stack([σx, σy, σz], axis=0)          # shape (3,2,2)

# reference states
dm_init = {
    "0": jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
    "1": jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128),
    "+": 0.5 * jnp.array([[1, 1], [1, 1]], dtype=jnp.complex128),
    "-": 0.5 * jnp.array([[1, -1], [-1, 1]], dtype=jnp.complex128),
    "+i": 0.5 * jnp.array([[1, -1j], [1j, 1]], dtype=jnp.complex128),
    "-i": 0.5 * jnp.array([[1, 1j], [-1j, 1]], dtype=jnp.complex128),
}


def rho_markov(t, θ, ρ0):
    ω, γ_pd, γ_ad = θ
    # right and left eigen‑operators
    R = jnp.array([(I2 + σz)/jnp.sqrt(2), σz/jnp.sqrt(2), σm, σp])
    L = jnp.array([I2/jnp.sqrt(2), (σz - I2)/jnp.sqrt(2), σp, σm])

    μ0 = jnp.trace(L @ ρ0, axis1=-2, axis2=-1)[:, None, None]        # shape (4, 1, 1)
    λ = jnp.array([0,
                   -γ_ad,
                   1j*ω - 2*γ_pd - 0.5*γ_ad,
                   -1j*ω - 2*γ_pd - 0.5*γ_ad], dtype=jnp.complex128)
    expλt = jnp.exp(λ[:, None] * t[None, :])            # (4,T)
    return jnp.einsum("i...,ij->j...", μ0 * R, expλt)   # (T, 2, 2)

def mu_markov(t, θ, ρ0):
    ω, γ_pd, γ_ad = θ
    # left eigen‑operators
    L = jnp.array([I2/jnp.sqrt(2), (σz - I2)/jnp.sqrt(2), σp, σm])

    μ0 = jnp.trace(L @ ρ0, axis1=-2, axis2=-1)[:, None]        # shape (4,1)
    λ = jnp.array([0,
                   -γ_ad,
                   1j*ω - 2*γ_pd - 0.5*γ_ad,
                   -1j*ω - 2*γ_pd - 0.5*γ_ad], dtype=jnp.complex128)
    expλt = jnp.exp(λ[:, None] * t[None, :])            # (4,T)
    return μ0 * expλt

def rho_to_mu(ρ_t):
    
    L = jnp.array([I2/jnp.sqrt(2), (σz - I2)/jnp.sqrt(2), σp, σm])
    return jnp.trace(L[:, None, :, :] @ ρ_t[None, :, :, :], axis1=-2, axis2=-1) # (4, T)

def mu_to_rho(μ_t):
    
    # right eigen‑operators
    R = jnp.array([(I2 + σz)/jnp.sqrt(2), σz/jnp.sqrt(2), σm, σp])
    rho = jnp.einsum("ij,i...->j...", μ_t, R)
    return rho

def rho_to_bloch(ρt):
    
    return jnp.trace(pauli_stack[:, None, :, :] @ ρt[None, :, :, :], axis1=-2, axis2=-1) # (3, T)

def fit_lindblad(csv_path: str | Path,
                 lr: float = 1e-2,
                 steps: int = 2000,
                 seed: int = 0,
                 plot_result: bool = False,
                 verbose: bool = False):

    # ------- load experiment -----------------------------------------------
    df     = pd.read_csv(csv_path)
    t_exp  = df.iloc[:, 0].to_numpy()                       # (T,)
    bloch_exp = jnp.array(df[["X_mean","Y_mean","Z_mean"]].to_numpy()).swapaxes(-1, -2) # (3, T)

    # initial state from filename "init=..."
    tag    = csv_path.split("/")[4][4:].split(",")[0]
    ρ0     = dm_init[tag]

    # ------- parameter initialisation (log/soft‑plus) -----------------------
    key    = jax.random.PRNGKey(seed)
    ω_raw, γpd_raw, γad_raw = jnp.abs(jax.random.normal(key, (3,)))


    # ------- loss -----------------------------------------------------------
    def loss_fn(params):
        μ_t = mu_markov(t_exp, params, ρ0)
        ρ_t = mu_to_rho(μ_t)
        bloch = rho_to_bloch(ρ_t)
        return jnp.mean(jnp.abs(bloch - bloch_exp) ** 2)

    # ------- optimiser loop -------------------------------------------------
    opt    = optax.adam(lr)
    params = jnp.array([ω_raw, γpd_raw, γad_raw])
    state  = opt.init(params)

    @jax.jit
    def step(p, s):
        l, g  = jax.value_and_grad(loss_fn)(p)
        u, s  = opt.update(g, s, p)
        p     = optax.apply_updates(p, u)
        return p, s, l

    for k in range(steps):
        params, state, L = step(params, state)
        if k % 10 == 0 and verbose:
            print(f"iter {k:4d}   loss={L:.3e}")

    θ_fit = params
    print("\nFitted parameters for Markov approximation:")
    print(f"ω_z   = {float(θ_fit[0]):.5f} (rad/µs)")
    print(f"γ_PD  = {float(θ_fit[1]):.5f} (µs⁻¹)")
    print(f"γ_AD  = {float(θ_fit[2]):.5f} (µs⁻¹)")
    print(f"Final loss for Markovian approximation = {float(loss_fn(θ_fit))}")
    
    ρ_fit = rho_markov(t_exp, θ_fit, ρ0)
    bloch_fit = rho_to_bloch(ρ_fit)
    
    if plot_result:
        
        colors = ["#ffc93c", "#f07b3f", "#ea5455"]
        markers = ["x", "*", "."]
        
        for i, pauli in enumerate(["X", "Y", "Z"]):
            plt.plot(t_exp, bloch_fit[i], color=colors[i], label=f"Lindblad-${pauli}$")
        for i, pauli in enumerate(["X", "Y", "Z"]):
            plt.plot(t_exp, bloch_exp[i], ls="", marker=markers[i], color=colors[i], label=f"Exp-${pauli}$")
        plt.ylim([-1.0, 1.0])
        plt.xlabel("t $(\\mu s)$")
        plt.ylabel("Bloch vectors")
        plt.legend()
        plt.show()
    
    return θ_fit
    

if __name__ == "__main__":
    
    file = "results/state_tomography/crosstalk/ibm_strasbourg/init-,+,+,+/bloch-q[64, 63, 65, 54]-np50-gpp50-s8192-2025-04-28T23-14-04.csv"
    
    θ = fit_lindblad(file,
                    lr=3e-2,
                    steps=1000,
                    seed=1222432,
                    plot_result=True)
    