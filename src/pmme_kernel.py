import jax
import jax.numpy as jnp
from typing import Callable
from copy import deepcopy
import matplotlib.pyplot as plt
import optax 

def inverse_laplace(N_poly_coeff: jax.Array, D_poly_coeff: jax.Array, tol=1e-7):
    r"""
        Calculating the inverse laplace transform by explicitly doing 
        the partial fraction expansion of rational function N(s) / D(s). 
        
        parameters:
            N_poly_coeff: The polynomial coefficient of numerator N(s) (In descending order of degree)
            D_poly_coeff: The polynomial coefficient of denominator D(s) (In descending order of degree)
            
        return:
            f: Callable, The inverse laplace transform of N(s) / D(s)
    """
    
    roots = jnp.roots(D_poly_coeff)
    print(roots)

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
        D_bar_poly_coeff = jnp.concat(D_bar_poly_coeff)

        G = lambda s: jnp.polyval(N_poly_coeff, s) / jnp.polyval(D_bar_poly_coeff, s)
        
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


def make_xi_i_of_t(a: jax.Array, b:jax.Array, lambda_i: float, tol=1e-7):
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
    
    assert bool(jnp.all(b.real < 0)), (
        "The real part of b must be all less than 0 to ensure the stability!"
    ) 

    # ---------------- P(s-λ_i), Q(s-λ_i) -----------------------------------------
    P_poly_coeff = jnp.poly(a + lambda_i)   # P(s-\lambda_i)
    Q_poly_coeff = jnp.poly(b + lambda_i)   # Q(s-\lambda_i)
    
    # D(s) = s Q(s-λ_i) - λ_i P(s-λ_i)
    sQ = jnp.concat([Q_poly_coeff, jnp.zeros(1, Q_poly_coeff.dtype)])
    D_poly_coeff = jnp.polysub(sQ, lambda_i * P_poly_coeff)
    N_poly_coeff = Q_poly_coeff
    
    return inverse_laplace(N_poly_coeff, D_poly_coeff)

    
    
if __name__ == "__main__":
    
    a = jnp.array([1, 4])
    b = jnp.array([-2, -1, -4])
    xi = make_xi_i_of_t(a, b, 1+2j)
    
    # P_poly_coeff = jnp.poly(a)   # P(s-\lambda_i)
    # Q_poly_coeff = jnp.poly(b)   # Q(s-\lambda_i)
    
    # memory_kernel = inverse_laplace(P_poly_coeff, Q_poly_coeff)
    
    t = jnp.linspace(0, 5, 100)
    xi_val = xi(t)
    # k_val = memory_kernel(t)
    
    plt.plot(t, xi_val.real, label="real")
    plt.plot(t, xi_val.imag, label="imag")
    plt.legend()
    plt.show()
    

