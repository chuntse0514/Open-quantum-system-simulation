import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np

from .pmme_kernel import inverse_laplace

jax.config.update("jax_enable_x64", True)

# ----------------------------------------------------------------------
# helper: forward Laplace via trapezoid on exp(‑st) f(t)      (s positive)
# ----------------------------------------------------------------------
def laplace_numeric(f_t, t, s):
    dt = t[1] - t[0]
    return jnp.trapezoid(f_t * jnp.exp(-s * t), t, dx=dt)


# ---------------------------------------------------------------------
# analytic reference functions
# ---------------------------------------------------------------------

def f_single(t):                      # 1/(s+1)
    return jnp.exp(-t)

def f_double(t):                      # 1/(s+1)^2
    return t * jnp.exp(-t)

def f_triple(t):                      # 1/(s+2)^3
    return 0.5 * t**2 * jnp.exp(-2*t)

def f_damped_cos(t):                  # (s+1)/((s+1)^2+1)
    return jnp.exp(-t) * jnp.cos(t)

def f_mixed_numeric(t):               # will be computed numerically later
    pass


# ---------------------------------------------------------------------
# polynomial coefficients for the five cases
# ---------------------------------------------------------------------
CASES = [
    (r"$\frac{1}{s+1}$",               jnp.array([1.0]),              jnp.array([1.0, 1.0])),
    (r"$\frac{1}{(s+1)^2}$",           jnp.array([1.0]),              jnp.array([1.0, 2.0, 1.0])),
    (r"$\frac{1}{(s+2)^3}$",           jnp.array([1.0]),              jnp.array([1.0, 6.0, 12.0, 8.0])),
    (r"$\frac{(s+1)}{(s+1)^2+1}$",     jnp.array([1.0, 1.0]),         jnp.array([1.0, 2.0, 2.0])),
    (r"$(s+1)^2+4$",                   jnp.array([1.0]),              jnp.array([1.0, 2.0, 5.0])),
    (r"$\frac{s+3}{(s+1)^2(s+2-3j)(s+2+3j)}$", jnp.array([1.0, 3.0]), jnp.poly(jnp.array([-1.0, -1.0, -2.0+3j, -2.0-3j]))),
    (r"$\frac{2s+1}{s^2+4s+5}$",       jnp.array([2.0, 1.0]),         jnp.array([1.0, 4.0, 5.0]))
]

# random stable rational used in pytest (regenerate deterministically)
key = jax.random.PRNGKey(0)
rand_poles = -jax.random.uniform(key, (4,)) - 1j*jax.random.uniform(key, (4,))
rand_zeros = jax.random.uniform(key, (2,)) + 1j*jax.random.uniform(key, (2,))
N_rand = jnp.poly(rand_zeros)
D_rand = jnp.poly(rand_poles)
CASES.append((r"random stable $P/Q$",   N_rand,                        D_rand))

# ---------------------------------------------------------------------
# grid and figure
# ---------------------------------------------------------------------
T = jnp.linspace(0.0, 10.0, 401)
fig, axes = plt.subplots(len(CASES), 1, figsize=(7, 11), sharex=True)
axes = np.atleast_1d(axes)

for ax, (label, N, D) in zip(axes, CASES):
    f_num = inverse_laplace(N, D)(T)
    ax.plot(T, jnp.real(f_num), label="Re νLap", lw=1.8)
    if jnp.max(jnp.abs(jnp.imag(f_num))) > 1e-12:
        ax.plot(T, jnp.imag(f_num), "--", label="Im νLap", lw=1.2)

    ax.set_title(label)
    ax.grid(alpha=.3); ax.legend()

axes[-1].set_xlabel("t")
fig.tight_layout(); plt.show()
