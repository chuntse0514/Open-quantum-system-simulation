import jax
import jax.numpy as jnp
from typing import Callable
from pathlib import Path
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import optax 
from .lindblad import *

jax.config.update("jax_enable_x64", True)


def inverse_laplace(N_poly_coeff: jax.Array, D_poly_coeff: jax.Array, tol=1e-5):
    r"""
        Calculating the inverse laplace transform by explicitly doing 
        the partial fraction expansion of rational function N(s) / D(s). 
        
        parameters:
            N_poly_coeff: The polynomial coefficient of numerator N(s) (In descending order of degree)
            D_poly_coeff: The polynomial coefficient of denominator D(s) (In descending order of degree)
            
        return:
            f: Callable, The inverse laplace transform of N(s) / D(s)
    """
    
    N_poly_coeff = jnp.array(N_poly_coeff, dtype=jnp.complex128)
    D_poly_coeff = jnp.array(D_poly_coeff, dtype=jnp.complex128)
    
    roots = jnp.roots(D_poly_coeff)
    # sort to make grouping reproducible
    roots_sorted = roots[jnp.argsort(roots.real)]
    
    # cluster roots that are closer than `tol`  →  multiplicities
    def cluster_roots(xs):
        clusters = []
        current = [xs[0]]
        for r in xs[1:]:
            if jnp.abs(r - current[-1]) < tol:
                current.append(r)
            else:
                clusters.append(jnp.array(current))
                current = [r]
        clusters.append(jnp.array(current))
        return clusters
    
    clusters = cluster_roots(list(roots_sorted))
    
    def make_n_order_grads(G: Callable, order=0):
        r"""
        Generate the high order derivatives of G
        return: [d^(order)/dx^(order) G,..., dG/dx, G]
        """
        n_order_grads = [G]
        for _ in range(order):
            n_order_grads.append(jax.grad(n_order_grads[-1], holomorphic=True))
        return n_order_grads[::-1]
    
    # For multiplicity m:
    # A_{l,j} = 1/(m-j)! * d^{m-j}/ds^{m-j} [ (s-r_l)^m * F(s) ] |_{s=r_l}
    poles_residues = []
    
    for index in range(len(clusters)):
        r = jnp.mean(clusters[index])              # representative root
        m = clusters[index].size                   # multiplicity
        
        D_bar_poly_coeff = deepcopy(clusters)
        D_bar_poly_coeff.pop(index)
        if len(D_bar_poly_coeff) > 0:
            D_bar_poly_coeff = jnp.poly(jnp.concat(D_bar_poly_coeff))
            G = lambda s: jnp.polyval(N_poly_coeff, s) / jnp.polyval(D_bar_poly_coeff, s)
        else:
            G = lambda s: jnp.polyval(N_poly_coeff, s)
        
        n_order_grads = make_n_order_grads(G, order=m-1)

        # use automatic diff to get derivatives
        coeffs = []
        for j, G_grad in zip(range(1, m + 1), n_order_grads):
            order = m - j
            deriv = G_grad(r)
            coeffs.append(deriv / jax.scipy.special.factorial(order))
        poles_residues.append((r, jnp.array(coeffs)))        # coeffs[j-1] = A_{j}

    # ---------------- f(t) -------------------------------------------------
    def f(t):
        t = jnp.atleast_1d(t)
        total = jnp.zeros_like(t, dtype=jnp.complex64)
        for r_l, A_lj in poles_residues:
            m = A_lj.size
            # broadcast-friendly time powers
            tj = jnp.stack([(t ** (j) / jax.scipy.special.factorial(j)) for j in range(m)],
                           axis=-1)         # shape (T, m)
            expo = jnp.exp(r_l * t)         # shape (T,)
            total += expo * (tj @ A_lj)     # \sum_j A_{l,j} t^{j-1}/(j-1)! e^{rt}
        return total.squeeze()

    return f                                # differentiable JAX‑Callable


def xi_pmme(t: jax.Array, a: jax.Array, b: jax.Array, c: jax.Array, λ: jax.Array, mode="coeff"):
    r"""
    Assuming that the memory kernel is s domain has the following functional form:
        $\tilde{k}(s) = P(s) / Q(s)$
    where 
        $P(s) = (s-a_0)(s-a_1)\cdots(s-a_{m-1})$
        $Q(s) = (s-b_0)(s-b_1)\cdots(s_b_{n-1})$
    This assumes that $m < n$ and also Re(b_i) < 0 to ensure stability  
    
    This function return a Callable function $\xi_i(t)$ where 
        $\xi_i(t) = Lap^{-1} \frac{ Q(s-\lambda_i) }{ sQ(s-\lambda_i) - \lambda_i P(s-\lambda_i)} $ 
    Define 
        $D(s) = sQ(s-\lambda_i) - \lambda_i P(s-\lambda_i)$
    """

    ξ = []
    for λi in λ:
        # ---------------- P(s-λ_i), Q(s-λ_i) -----------------------------------------
        
        if mode == "roots":
            P_poly_coeff = jnp.poly(a + c[0] * λi)   # P(s-\lambda_i)
            Q_poly_coeff = jnp.poly(b + c[0] * λi)   # Q(s-\lambda_i)
        elif mode == "coeff":
            P_poly_coeff = jnp.roots(a)
            Q_poly_coeff = jnp.roots(b)
            P_poly_coeff = jnp.poly(P_poly_coeff + λi)
            Q_poly_coeff = jnp.poly(Q_poly_coeff + λi)
        
        if len(a) == 0:
            P_poly_coeff = jnp.array([P_poly_coeff])
        
        # D(s) = s Q(s-λ_i) - λ_i P(s-λ_i)
        sQ = jnp.concat([Q_poly_coeff, jnp.zeros(1, Q_poly_coeff.dtype)])
        D_poly_coeff = jnp.polysub(sQ, c[1] * λi * P_poly_coeff)
        N_poly_coeff = Q_poly_coeff
    
        ξ.append(inverse_laplace(N_poly_coeff, D_poly_coeff)(t))
        
    return jnp.stack(ξ, axis=0)

def rho_pmme(μ):
    
    # left eigen‑operators
    R = jnp.array([(I2 + σz)/jnp.sqrt(2), σz/jnp.sqrt(2), σm, σp])

    rho = jnp.einsum("ij,i...->j...", μ, R)
    return rho

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

    tag  = csv_path.split("/")[4][4:].split(",")[0]
    ρ0   = dm_init[tag]
    μ0   = rho_to_mu(ρ0[None, :, :])[...,0]   # (4,)

    # ---------- parameter vector initialisation ---------------------------
    key  = jax.random.PRNGKey(seed)
    subkeys = jax.random.split(key, num=4)
    if mode == "roots":
        a0 = 0.1 * jax.random.normal(key=subkeys[0], shape=(n_zeros,))
        b0 = 5.0 * jax.random.normal(key=subkeys[1], shape=(n_poles,))
        c0 = 1.0 * jnp.ones(shape=(2,))
    elif mode == "coeff":
        a0 = 0.1 * jnp.abs(jax.random.normal(key=subkeys[0], shape=(n_zeros,)))
        b0 = 5.0 * jnp.abs(jax.random.normal(key=subkeys[0], shape=(n_poles,)))
        a0 = jnp.poly(-a0 ** 2)
        b0 = jnp.poly(-b0 ** 2)
        c0 = 1.0 * jnp.ones(shape=(2,))
    
    θ0   = {"a": a0, "b": b0, "c": c0}   
    
    # ξ_t = xi_pmme(t_exp, a0, b0, c0, λ) # (4, T)
    λ = jnp.array([0,
                -γad0,
                1j*ω0 - 2*γpd0 - 0.5*γad0,
                -1j*ω0 - 2*γpd0 - 0.5*γad0], dtype=jnp.complex128)
    
    
    def loss_kernel(params): 
        a = params["a"]
        b = params["b"]
        c = params["c"]
        
        if mode == "roots":
            ξ_t = xi_pmme(t_exp, -a**2, -b**2, c, λ, mode=mode) # (4, T)
        else:
            ξ_t = xi_pmme(t_exp, a, b, c, λ, mode=mode)
            
        μ_t = μ0[:, None] * ξ_t
        ρ_t = rho_pmme(μ_t)
        bloch = rho_to_bloch(ρ_t)

        return jnp.mean(jnp.abs(bloch - bloch_exp)**2)
    
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
        
        if mode == "roots":
            ξ_t = xi_pmme(t_exp, -a**2, -b**2, c, λ, mode=mode) # (4, T) 
        elif mode == "coeff":
            ξ_t = xi_pmme(t_exp, a, b, c, λ, mode=mode)
        μ_t = μ0[:, None] * ξ_t
        rho = rho_pmme(μ_t)
        bloch = rho_to_bloch(rho)
        
        for i, pauli in enumerate(["X", "Y", "Z"]):
            plt.plot(t_exp, bloch[i], label=f"PMME-{pauli}")
        for i, pauli in enumerate(["X", "Y", "Z"]):
            plt.plot(t_exp, bloch_exp[i], ls="", marker="x", label=f"Exp-{pauli}")

        plt.legend()
        plt.show()
    
    return θ
    
if __name__ == "__main__":
    
    csv_path = file = "results/state_tomography/crosstalk/ibm_strasbourg/init-,+,+,+/bloch-q[64, 63, 65, 54]-np50-gpp50-s8192-2025-04-28T23-14-04.csv"
    θ_opt = fit_kernel(csv_path, n_zeros=1, n_poles=4, steps=1000, lr=0.05, plot_result=True, mode="roots")
    

