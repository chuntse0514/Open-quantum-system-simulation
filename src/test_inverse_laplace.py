import pytest
import jax.numpy as jnp
import jax
import jax.scipy as jsp
import matplotlib.pyplot as plt 
import numpy as np

from .pmme_kernel import inverse_laplace  # edit path

T = jnp.linspace(0.0, 10.0, 501)
TOL = 1e-10

mse = lambda a, b: jnp.mean(jnp.abs(a - b) ** 2)

# ----------------------------------------------------------------------
# helper: forward Laplace via trapezoid on exp(‑st) f(t)      (s positive)
# ----------------------------------------------------------------------
def laplace_numeric(f_t, t, s):
    dt = t[1] - t[0]
    return jnp.trapezoid(f_t * jnp.exp(-s * t), t, dx=dt)

# ----------------------------------------------------------------------
# 1) Triple pole  (s+2)^{-3}  → ½ t^2 e^{-2t}
# ----------------------------------------------------------------------

def test_triple_pole():
    N = jnp.array([1.0])
    D = jnp.array([1.0, 6.0, 12.0, 8.0])  # (s+2)^3
    f = inverse_laplace(N, D)
    y = f(T)
    y_true = 0.5 * T ** 2 * jnp.exp(-2 * T)
    assert float(mse(y, y_true)) < TOL

# ----------------------------------------------------------------------
# 2) Complex‑conjugate pair  (bi‑exponential cosine) ------------
#    F(s) = 1/[(s+1)^2 + 4]   → ½ e^{-t} sin 2t /2
# ----------------------------------------------------------------------

def test_complex_pair():
    N = jnp.array([1.0])
    D = jnp.array([1.0, 2.0, 5.0])  # (s+1)^2+4
    f = inverse_laplace(N, D)
    y = f(T)
    y_true = 0.5 * jnp.exp(-1 * T) * jnp.sin(2 * T)
    assert float(mse(y, y_true)) < TOL

# ----------------------------------------------------------------------
# 3) Mixed: double real + single complex ------------------------
#    F(s) = (s+3)/[(s+1)^2 (s+2+3i)(s+2-3i)]
#    Verify by forward numeric Laplace
# ----------------------------------------------------------------------

def test_mixed_poles():
    N = jnp.array([1.0, 3.0])                       # s+3
    D = jnp.poly(jnp.array([-1.0, -1.0, -2.0+3j, -2.0-3j]))
    f = inverse_laplace(N, D)
    y = f(T)
    # Forward Laplace at three s‑points and compare to analytic F(s)
    for s in [1.0, 2.5, 5.0]:
        F_num = laplace_numeric(y, T, s)
        F_true = jnp.polyval(N, s) / jnp.polyval(D, s)
        assert jnp.abs(F_num - F_true) < 1e-8

# ----------------------------------------------------------------------
# 4) Random stable rational function  n<=m -----------------------
#    Poles placed in LHP; zeros random.  Compare numeric round‑trip.
# ----------------------------------------------------------------------

def test_random_stable(key=jax.random.PRNGKey(0)):
    poles = -jax.random.uniform(key, (4,)) - 1j * jax.random.uniform(key, (4,))
    zeros = jax.random.uniform(key, (2,)) + 1j * jax.random.uniform(key, (2,))
    N = jnp.poly(zeros)
    D = jnp.poly(poles)
    f = inverse_laplace(N, D)
    y = f(T)
    for s in [0.5, 1.3, 3.7]:
        F_num = laplace_numeric(y, T, s)
        F_true = jnp.polyval(N, s) / jnp.polyval(D, s)
        print("F_num=", F_num)
        print("F_true=", F_true)
        assert jnp.abs(F_num - F_true) < 1e-8

# ----------------------------------------------------------------------
# 5) Identity check: inverse followed by forward Laplace (self‑consistency)
# ----------------------------------------------------------------------

def test_round_trip():
    N = jnp.array([2.0, 1.0])
    D = jnp.array([1.0, 4.0, 5.0])
    f = inverse_laplace(N, D)
    y = f(T)
    F_back = laplace_numeric(y, T, 2.0)
    F_true = jnp.polyval(N, 2.0) / jnp.polyval(D, 2.0)
    assert jnp.abs(F_back - F_true) < 1e-8


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])