import pytest
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# import the function under test
from .pmme_kernel import inverse_laplace  # <- adapt the path

T_GRID = jnp.linspace(0.0, 5.0, 101)  # 0 … 5 inclusive, 101 pts
TOL = 1e-10


def _mse(a, b):
    return jnp.mean(jnp.abs(a - b) ** 2)


def test_single_pole():
    """F(s)=1/(s+1)  →  f(t)=e^{-t}."""
    N = jnp.array([1.0])
    D = jnp.array([1.0, 1.0])               # s+1
    f = inverse_laplace(N, D)
    y_pred = f(T_GRID)
    y_true = jnp.exp(-T_GRID)
    assert float(_mse(y_pred, y_true)) < TOL


def test_repeated_pole():
    """F(s)=1/(s+1)^2  →  f(t)=t e^{-t}."""
    N = jnp.array([1.0])
    D = jnp.array([1.0, 2.0, 1.0])          # (s+1)^2 = s^2+2s+1
    f = inverse_laplace(N, D)
    y_pred = f(T_GRID)
    y_true = T_GRID * jnp.exp(-T_GRID)
    assert float(_mse(y_pred, y_true)) < TOL


def test_damped_cosine():
    """F(s)=(s+1)/((s+1)^2+1)  →  f(t)=e^{-t} cos t."""
    # (s+1)^2+1 = s^2+2s+2  and numerator s+1
    N = jnp.array([1.0, 1.0])               # s+1
    D = jnp.array([1.0, 2.0, 2.0])          # s^2+2s+2
    f = inverse_laplace(N, D)
    y_pred = f(T_GRID)
    y_true = jnp.exp(-T_GRID) * jnp.cos(T_GRID)
    assert float(_mse(y_pred, y_true)) < TOL


if __name__ == "__main__":
    # run tests without pytest runner
    pytest.main([__file__])
    
    
    t = jnp.linspace(0, 10, 400)

    test_cases = {
        r"1/(s+1)":                 (jnp.array([1.0]),             jnp.array([1.0, 1.0])),
        r"1/(s+1)^2":               (jnp.array([1.0]),             jnp.array([1.0, 2.0, 1.0])),
        r"(s+1)/((s+1)^2+1)":       (jnp.array([1.0, 1.0]),        jnp.array([1.0, 2.0, 2.0])),
    }

    fig, axes = plt.subplots(len(test_cases), 1, figsize=(7, 9), sharex=True)
    axes = np.atleast_1d(axes)
    
    for ax, (label, (N, D)) in zip(axes, test_cases.items()):
        f = inverse_laplace(N, D)
        y = f(t)
        ax.plot(t, jnp.real(y), label="Re")
        ax.plot(t, jnp.imag(y), "--", label="Im")
        ax.set_title(f"Inverse Laplace of {label}")
        ax.legend(); ax.grid(alpha=.3)

    axes[-1].set_xlabel("t")
    plt.tight_layout(); plt.show()
