import jax
import jax.numpy as jnp
from typing import Callable
from pathlib import Path
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import optax 
from .utils import *
from .lindblad import fit_lindblad, rho_from_markov_parameters

jax.config.update("jax_enable_x64", True)

def xi_pmme_roots(t: jax.Array, a: jax.Array, b: jax.Array, c: jax.Array, λ0: jax.Array, λ1: jax.Array):
    
    ξ = []
    for λ0i, λ1i in zip(λ0, λ1):
        P_poly_coeff = jnp.poly(a + c[0] * (λ0i + λ1i))   # P(s-\lambda_i)
        Q_poly_coeff = jnp.poly(b + c[0] * (λ0i + λ1i))
    
        P_poly_coeff = jnp.atleast_1d(P_poly_coeff)
        s_minus_λ0_Q = jnp.polymul(Q_poly_coeff, jnp.array([1.0, -λ0i], dtype=jnp.complex128))
        D_poly_coeff = jnp.polysub(s_minus_λ0_Q, c[1] * λ1i * P_poly_coeff)
        N_poly_coeff = Q_poly_coeff
        
        ξ.append(inverse_laplace(N_poly_coeff, D_poly_coeff)(t))

    return jnp.stack(ξ, axis=0)

def xi_pmme_coeffs(t: jax.Array, a: jax.Array, b: jax.Array, c: jax.Array, λ0: jax.Array, λ1: jax.Array):
    
    ξ = []
    for λ0i, λ1i in zip(λ0, λ1):
        P_poly_coeff = jnp.roots(a)
        Q_poly_coeff = jnp.roots(b)
        P_poly_coeff = jnp.poly(P_poly_coeff + c[0] * (λ0i + λ1i))
        Q_poly_coeff = jnp.poly(Q_poly_coeff + c[0] * (λ0i + λ1i))
    
        P_poly_coeff = jnp.atleast_1d(P_poly_coeff)
        s_minus_λ0_Q = jnp.polymul(Q_poly_coeff, jnp.array([1.0, -λ0i], dtype=jnp.complex128))
        D_poly_coeff = jnp.polysub(s_minus_λ0_Q, c[1] * λ1i * P_poly_coeff)
        N_poly_coeff = Q_poly_coeff
        
        ξ.append(inverse_laplace(N_poly_coeff, D_poly_coeff)(t))

    return jnp.stack(ξ, axis=0)


def fit_kernel(csv_path: str | Path,
               n_zeros: int,
               n_poles: int,
               lr: float = 1e-2,
               steps: int = 2000,
               mode: str = "coeff",
               seed: int = 0,
               plot_result: bool = False):
    
    assert n_poles > n_zeros, ("Number of poles needs to larger than number of zeros!")
    
    # ---------- get Markov parameters first -------------------------------
    ω0, γpd0, γad0 = fit_lindblad(csv_path, steps=3000, lr=3e-3, seed=seed)

    # ---------- load experiment -------------------------------------------
    df   = pd.read_csv(csv_path)
    t_exp = df.iloc[:,0].to_numpy()
    bloch_exp = jnp.array(df[["X_mean", "Y_mean", "Z_mean"]].to_numpy()).swapaxes(-1, -2)  # (3, T)
    ρ_exp = bloch_to_rho(bloch_exp)
    μ_exp = rho_to_mu(ρ_exp)
    ξ_exp = μ_exp / μ_exp[:, 0:1]

    tag  = csv_path.split("/")[4][4:].split(",")[0]
    ρ0   = dm_init[tag]
    μ0   = rho_to_mu(ρ0[None, :, :])[...,0]   # (4,)
    
    λ0 = jnp.array([0,
                   -γad0,
                   1j*ω0 - 0.5*γad0,
                   -1j*ω0 - 0.5*γad0], dtype=jnp.complex128)
    λ1 = jnp.array([0, 0, -2*γpd0, -2*γpd0])

    if mode == "visualize-mu":
        fig, axes = plt.subplots(2, 2)
        fig.set_figwidth(12)
        fig.set_figheight(8)
        axes = axes.flatten()
        
        for i in range(4):
            axes[i].plot(t_exp, μ_exp[i].real, label="real")
            axes[i].plot(t_exp, μ_exp[i].imag, label="imag")
            axes[i].set_ylim([-1.0, 1.0])
            axes[i].set_xlim([0.0, t_exp[-1]])
            axes[i].legend()
            axes[i].set_title(f"$\\mu_{i}(t)$")
        plt.suptitle("$\\mu_i(t)$ visualization")
        plt.show()

        return 

    if mode == "visualize-xi":
        fig, axes = plt.subplots(2, 2)
        fig.set_figwidth(12)
        fig.set_figheight(8)
        axes = axes.flatten()
        
        for i in range(4):
            axes[i].plot(t_exp, ξ_exp[i].real, label="real")
            axes[i].plot(t_exp, ξ_exp[i].imag, label="imag")
            axes[i].set_ylim([-1.0, 1.0])
            axes[i].set_xlim([0.0, t_exp[-1]])
            axes[i].legend()
            axes[i].set_title(f"$\\xi_{i}(t)$")
        plt.suptitle("$\\xi_i(t)$ visualization")
        plt.show()

        return 
    
    if mode == "visualize-k":
        fig, axes = plt.subplots(1, 2)
        fig.set_figwidth(12)
        fig.set_figheight(4)
        axes = axes.flatten()
        
        for i in range(2, 4, 1):
            k_exp = numerical_memory_kernel_from_mu(μ_exp[i], t_exp, λ0[i], λ1[i])
            
            axes[i-2].plot(t_exp, k_exp.real, label="real")
            axes[i-2].plot(t_exp, k_exp.imag, label="imag")
            axes[i-2].set_ylim([-1.0, 1.0])
            axes[i-2].set_xlim([0.0, t_exp[-1]])
            axes[i-2].legend()
            axes[i-2].set_title(f"$k_{i}(t)$ ")
        plt.suptitle("memory kernel $k(t)$ visualization")
        plt.show()
        
        return
    
    # ---------- parameter vector initialisation ---------------------------
    key  = jax.random.PRNGKey(seed)
    subkeys = jax.random.split(key, num=4)
    if mode == "roots":
        a0 = 0.1 * jax.random.normal(key=subkeys[0], shape=(n_zeros,))
        b0 = 1.0 * jax.random.normal(key=subkeys[1], shape=(n_poles,))
        a0 = jnp.array(-a0 ** 2)
        b0 = jnp.array(-b0 ** 2)
        c0 = 1.0 * jnp.ones(shape=(2,))
    elif mode == "coeff":
        a0 = 0.5 * jax.random.normal(key=subkeys[0], shape=(n_zeros,))
        b0 = 3.0 * jax.random.normal(key=subkeys[1], shape=(n_poles,))
        a0 = jnp.poly(-a0 ** 2)
        b0 = jnp.poly(-b0 ** 2)
        c0 = 1.0 * jnp.ones(shape=(2,))
    
    θ0   = {"a": jnp.atleast_1d(a0), "b": b0, "c": c0}   
    
    def loss_kernel(params): 
        a = params["a"]
        b = params["b"]
        c = params["c"]
        
        if mode == "roots":
            ξ_t = xi_pmme_roots(t_exp, -a**2, -b**2, c, λ0, λ1) # (4, T)
        else:
            ξ_t = xi_pmme_coeffs(t_exp, a, b, c, λ0, λ1)
            
        μ_t = μ0[:, None] * ξ_t
        ρ_t = mu_to_rho(μ_t)
        bloch = rho_to_bloch(ρ_t)

        return jnp.mean(jnp.abs(bloch - bloch_exp) ** 2)
    
    loss  = lambda θ: loss_kernel(θ)

    opt   = optax.adam(lr)
    state = opt.init(θ0)
    θ     = θ0

    def step(θ, state):
        l, g = jax.value_and_grad(loss)(θ)
        upd, state = opt.update(g, state)
        θ = optax.apply_updates(θ, upd)
        return θ, state, l

    for k in range(steps):
        θ, state, L = step(θ, state)
        if k % 10 == 0:
            print(f"it {k:4d}  loss={L:.3e}")

    if mode == "roots":
        print("Solved a = ", -θ["a"] ** 2)
        print("Solved b = ", -θ["b"] ** 2)
        print("Solved c = ", θ["c"])
    elif mode == "coeff":
        print("Solved a = ", θ["a"])
        print("Solved b = ", θ["b"])
        print("Solved c = ", θ["c"])
    print("Final loss:", float(L))
    
    if plot_result:
        a, b, c = θ.values()
        fig, axes = plt.subplots(4, 1)
        
        if mode == "roots":
            ξ_t = xi_pmme_roots(t_exp, -a**2, -b**2, c, λ0, λ1) # (4, T) 
        elif mode == "coeff":
            ξ_t = xi_pmme_coeffs(t_exp, a, b, c, λ0, λ1)
        
        μ_t = μ0[:, None] * ξ_t
        ρ_pmme = mu_to_rho(μ_t)
        bloch_pmme = rho_to_bloch(ρ_pmme)
        
        ρ_lindblad = rho_from_markov_parameters(t_exp, jnp.array([ω0, γpd0, γad0]), ρ0)
        bloch_lindblad = rho_to_bloch(ρ_lindblad)
        
        if mode == "roots":
            N_poly_coeff = jnp.poly(-θ["a"] ** 2)
            D_poly_coeff = jnp.poly(-θ["b"] ** 2)
            memory_kernel = inverse_laplace(N_poly_coeff, D_poly_coeff)
        elif mode == "coeff":
            memory_kernel = inverse_laplace(θ["a"], θ["b"])
        
        colors = ["#ffc93c", "#f07b3f", "#ea5455"]
        
        for i, pauli in enumerate(["X", "Y", "Z"]):
            axes[i].plot(t_exp, bloch_pmme[i], color=colors[i], ls="-", label=f"PMME-{pauli}")
            axes[i].legend()
        for i, pauli in enumerate(["X", "Y", "Z"]):
            axes[i].plot(t_exp, bloch_lindblad[i], color=colors[i], ls="--", label=f"Lindblad-{pauli}")
            axes[i].legend()
        for i, pauli in enumerate(["X", "Y", "Z"]):
            axes[i].plot(t_exp, bloch_exp[i], ls="", color=colors[i], marker="x", label=f"Exp-{pauli}")
            axes[i].legend()
        axes[3].plot(jnp.linspace(0, 140, 1000), memory_kernel(jnp.linspace(0, 140, 1000)))
        plt.show()
    
    return θ
    
if __name__ == "__main__":
    
    # csv_path = "results/state_tomography/crosstalk/ibm_strasbourg/init-,+,+,+/bloch-q[64, 63, 65, 54]-np50-gpp50-s8192-2025-04-28T23-14-04.csv"
    csv_path = "results/state_tomography/crosstalk/ibm_brussels/init-i,+,+,+/bloch-q[49, 48, 50, 55]-np50-gpp50-s8192-2025-05-05T17-23-17.csv"
    θ_opt = fit_kernel(csv_path, n_zeros=1, n_poles=2, steps=1000, lr=0.3, plot_result=True, seed=45, mode="visualize-k")
    

