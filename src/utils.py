import jax.numpy as jnp
import jax
from typing import Callable
from copy import deepcopy

jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray

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

# mu_to_bloch: these are the RIGHT eigen-operators R_i (not left)
def mu_to_bloch(mu):
    R = jnp.array([(I2 + σz)/jnp.sqrt(2), σz/jnp.sqrt(2), σm, σp], dtype=jnp.complex128)
    rho = jnp.einsum("ij,i...->j...", mu, R)
    return jnp.trace(pauli_stack[:, None, :, :] @ rho[None, :, :, :], axis1=-2, axis2=-1)


def bloch_to_mu(bloch):
    rho = 0.5 * (jnp.eye(2, dtype=jnp.complex128)[None, :, :] +
                 jnp.tensordot(bloch, pauli_stack, axes=[[0], [0]]))
    L = jnp.array([I2/jnp.sqrt(2), (σz - I2)/jnp.sqrt(2), σp, σm], dtype=jnp.complex128)
    return jnp.trace(L[:, None, :, :] @ rho[None, :, :, :], axis1=-2, axis2=-1)


def numerical_memory_kernel_from_mu(μt: jnp.ndarray, t: jnp.ndarray, λ0: complex, λ1: complex):
    """
    Compute complex PMME kernel k(t) from a single damped-basis mode μ(t).
    Uses forward_laplace_fft (samples on s=σ+iω_k) and the complex
    inverse above.
    """
   
    grid = make_durbin_grid(T=t[-1] * 1.0, K=16384, gamma=0)

    # # 2) normalize ξ(t)=μ(t)/μ(0) to avoid division blowups in 1/ξ̃(s)
    ξt = μt / μt[0]

    # # 3) forward Laplace samples on the FFT-aligned grid
    ξs = forward_laplace_from_samples_on_grid(t, ξt, grid)   # s_grid = σ + i ω_k 
    
    fs = grid["s"] - λ0 - λ1 - 1 / ξs
    
    import matplotlib.pyplot as plt 
    plt.figure()
    plt.plot(t, ξt.real, label="$\\text{Re}\\xi(t)$")
    plt.plot(t, ξt.imag, label="$\\text{Im}\\xi(t)$")
    plt.legend()
    plt.xlabel("$t$ (µs)")
    plt.ylabel("$\\xi(t)$")
    
    plt.figure()
    plt.plot(grid["omega"], jnp.abs(fs), label="$|\\tilde k(s)|$")
    plt.ylim([0, 10])
    plt.legend()
    plt.show()    
    
    # fs = - (1/ξs) / λ1
    ft = inverse_laplace_from_samples(fs, t, grid)
    
    plt.figure()
    plt.plot(t, ft.real, label="$\\text{Re}k(t)$")
    plt.plot(t, ft.imag, label="$\\text{Im}k(t)$")
    plt.legend()
    plt.xlabel("$t$ (µs)")
    plt.ylabel("$k(t)$")
    plt.legend()
    
    return ft # complex-valued, shape (T,)


# ------------------------------
# 1) Durbin (Bromwich) grid
# ------------------------------
def make_durbin_grid(T: float, K: int, gamma: float):
    """Two-sided Durbin grid for complex inversion."""
    h = jnp.pi / T
    k = jnp.arange(-K, K + 1, dtype=jnp.int32)
    omega = h * k
    w = jnp.ones_like(omega, dtype=jnp.float64)
    w = w.at[0].set(0.5); w = w.at[-1].set(0.5)
    s = (gamma + 1j * omega).astype(jnp.complex128)
    return {"gamma": float(gamma), "T": float(T), "h": h,
            "omega": omega, "w": w, "s": s}

# --------------------------------------------
# 2) Forward Laplace on that s-grid (samples)
# --------------------------------------------
def forward_laplace_from_samples_on_grid(t: Array, f: Array, grid) -> Array:
    """
    Composite trapezoid on [0, max(t)] for the set of s in `grid["s"]`.
    Works for non-uniform t. Returns F_k at all s_k on the Durbin grid.
    """
    t = jnp.asarray(t, dtype=jnp.float64)           # (N,)
    f = jnp.asarray(f, dtype=jnp.complex128)        # (N,)
    s = jnp.asarray(grid["s"], dtype=jnp.complex128)  # (K+1,)

    # trapezoid weights in t (non-uniform-safe)
    dt = jnp.diff(t)
    w_t = jnp.concatenate([dt[:1] / 2.0, (dt[:-1] + dt[1:]) / 2.0, dt[-1:] / 2.0])  # (N,)

    kernel = jnp.exp(-s[:, None] * t[None, :])      # (K+1,N)
    Fk = jnp.sum(kernel * (w_t * f)[None, :], axis=1)   # (K+1,)
    return Fk.astype(jnp.complex128)

# ----------------------------------------------------------
# 3) Inverse Laplace from sampled values on the same grid
# ----------------------------------------------------------
def inverse_laplace_from_samples(Gk: Array, t_eval: Array, grid) -> Array:
    """Two-sided Durbin inversion from samples (complex-valued)."""
    gamma = grid["gamma"]; h = grid["h"]
    omega = jnp.asarray(grid["omega"], dtype=jnp.float64)
    w = jnp.asarray(grid["w"], dtype=jnp.float64)
    Gk = jnp.asarray(Gk, dtype=jnp.complex128)

    t_eval = jnp.atleast_1d(jnp.asarray(t_eval, dtype=jnp.float64))
    E = jnp.exp(1j * omega[:, None] * t_eval[None, :])
    integ = (Gk[:, None] * E) * w[:, None]
    f = jnp.exp(gamma * t_eval) * (h / (2.0 * jnp.pi)) * jnp.sum(integ, axis=0)
    return f  # complex


def inverse_laplace(N_poly_coeff: jax.Array, D_poly_coeff: jax.Array, tol=1e-4):
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
        print(r)
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
        
    print(poles_residues)

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


