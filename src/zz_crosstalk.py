from pathlib import Path

import jax, jax.numpy as jnp
import optax
import pandas as pd
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
#  Utility : initial‑state parsing
# -----------------------------------------------------------------------------

def _state_to_bloch(label: str):
    """Return (x,y,z) Bloch coords for |+>, |->, |0>, |1>."""
    label = label.strip()
    if label == "+":
        return 1.0, 0.0, 0.0
    if label == "-":
        return -1.0, 0.0, 0.0
    if label == "+i":
        return 0.0, 1.0, 0.0
    if label == "-i":
        return 0.0, -1.0, 0.0
    if label == "0":
        return 0.0, 0.0, 1.0
    if label == "1":
        return 0.0, 0.0, -1.0
    raise ValueError(f"Unknown qubit label '{label}'.")


def parse_initial_state(path: str | Path, n_spec: int):
    """Extract main and spectator Bloch vectors from filename."""
    path = str(path)
    try:
        init_chunk = path.split("init")[1].split("/")[0]     # '+,+,+,+'
    except IndexError as e:
        raise RuntimeError("Filename must contain 'init<prep_string>/' segment") from e

    labels = init_chunk.split(",")
    if len(labels) != n_spec + 1:
        raise ValueError(
            f"Expected {n_spec+1} qubit labels, found {len(labels)} in '{init_chunk}'.")

    # main qubit first
    x0, y0, z0 = _state_to_bloch(labels[0])
    spectator_xyz = jnp.array([_state_to_bloch(l) for l in labels[1:]])  # (N,3)
    z_spec = spectator_xyz[:, 2]                                          # (N,)
    return jnp.asarray([x0, y0, z0], dtype=jnp.float64), z_spec

# -----------------------------------------------------------------------------
#  Analytic Σ₀⁺(t) and Bloch‑vector functions (Eq. C58)
# -----------------------------------------------------------------------------

def sigma_plus_general(t, x0, y0, ω0, Γ0, J_0q, Γq, zq):
    """Analytic Σ₀⁺(t) for arbitrary spectator preparations.

    t : (T,)          time grid
    x0,y0            : main‑qubit Bloch coords in xy plane at t=0
    ω0               : detuning (rad/µs)
    Γ0               : envelope rate (½γ↓₀ + γϕ₀)
    J_0q, Γq, zq : (N,)   : ZZ couplings, spectator decay, spectator z‑coords
    """
    t = t[None, :]                     # broadcast (1,T)
    Σ = 0.5 * (x0 - 1j * y0) * jnp.exp(-(Γ0 - 1j * ω0) * t)

    if J_0q.size:
        Jq = J_0q[:, None]
        Γq = Γq[:, None]
        zq = zq[:, None]
        Ωq  = Γq / 2.0 - 1j * Jq
        eps = 1e-12
        absΩ = jnp.abs(Ωq)
        Cq  = jnp.cosh(Ωq * t)
        Sq  = jnp.sinh(Ωq * t)
        ratio = jnp.where(absΩ > eps,
                  (Γq/2.0 - 1j*zq*Jq)/Ωq,
                  1.0 + 0.0j)         # limit: use series (below) when needed
        bracket = jnp.where(absΩ > eps,
                            Cq + ratio*Sq,
                            1.0 + (Γq/2.0 - 1j*zq*Jq)*t)   # 1st-order series
        Σ *= jnp.exp(-Γq * t / 2.0).prod(axis=0) * bracket.prod(axis=0)

    return Σ.squeeze(0)               # (T,)


def bloch_from_parameters(t, init_config, θ, n_spec):
    """Decode parameter vector θ → Bloch vector array (3,T)."""
    
    bloch_main = init_config["bloch-main"]
    x0, y0, z0 = bloch_main
    z_spec = init_config["z-spec"]
    
    γ_φ  = jnp.abs(θ[:n_spec+1])     # dephasing rate
    γ_down = jnp.abs(θ[n_spec+1])    # amplitude damping rate
    J_0q = θ[n_spec+2: 2*n_spec+2]   # ZZ crosstalk strength
    ω_0 = θ[2*n_spec+2]
    
    Γ_0 = 2 * γ_φ[0] + 0.5 * γ_down
    Γ_q = γ_φ[1:]
    

    Σ  = sigma_plus_general(t, x0, y0, ω_0, Γ_0, J_0q, Γ_q, z_spec)
    vx =  2.0 * Σ.real
    vy = -2.0 * Σ.imag
    vz = 1 + (z0 - 1) * jnp.exp(-γ_down * t)

    return jnp.stack([vx, vy, vz])    # (3,T)

# -----------------------------------------------------------------------------
#  Fitting routine
# -----------------------------------------------------------------------------

def fit_crosstalk(csv_path: str | Path,
                  n_spec: int,
                  lr: float = 1e-2,
                  steps: int = 5000,
                  seed: int = 0,
                  plot: bool = False,
                  verbose: bool = False):
    """Fit ZZ‑crosstalk parameters to a CSV containing Bloch trajectories."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    t_exp = jnp.asarray(df.iloc[:, 0].to_numpy(), dtype=jnp.float64)
    bloch_exp = jnp.asarray(df[["X_mean", "Y_mean", "Z_mean"]].to_numpy().T,
                            dtype=jnp.float64)  # (3,T)

    bloch0_main, z_spec_init = parse_initial_state(csv_path, n_spec)
    init_config = {
        'bloch-main': bloch0_main, # list of bloch vectors 
        'z-spec': z_spec_init,     # list of pauli Z expectation of spectator qubits
    }

    # ---- build initial θ --------------------------------------------------
    key = jax.random.PRNGKey(seed)
    θ   = jax.random.normal(key, (3 + 2 * n_spec,)) * 0.05  # small noise

    # ---- loss -------------------------------------------------------------
    def loss_fn(θ):
        pred = bloch_from_parameters(t_exp, init_config, θ, n_spec)
        mse_loss = jnp.mean((pred - bloch_exp) ** 2)
        return mse_loss

    opt = optax.adam(lr)
    opt_state = opt.init(θ)

    @jax.jit
    def step(θ, opt_state):
        l, g = jax.value_and_grad(loss_fn)(θ)
        updates, opt_state = opt.update(g, opt_state, θ)
        θ = optax.apply_updates(θ, updates)
        return θ, opt_state, l

    for k in range(steps):
        θ, opt_state, L = step(θ, opt_state)
        if verbose and k % 100 == 0:
            print(f"[{k:5d}] loss = {float(L):.5e}")

    # ---- results ----------------------------------------------------------
    names = [
        *[f"γ_φ{q}" for q in range(0, n_spec+1)], 
        "γ_↓",
        *[f"J_0{q}" for q in range(1, n_spec + 1)],
        "ω_0",
    ]

    print("\nFitted parameters (physical values)")
    θ = θ.at[:n_spec+2].set(jnp.abs(θ[:n_spec+2]))
    
    
    for n, v in zip(names, θ):
        print(f"{n:6s} = {float(v):.6g}")

    print(f"Final weighted MSE  = {float(loss_fn(θ)):.4e}\n")

    if plot:
        bloch_fit = bloch_from_parameters(t_exp, init_config, θ, n_spec)
        col = ["#0096c7", "#48cae4", "#ade8f4"]
        plt.figure(figsize=(10, 5))
        for i, pauli in enumerate("XYZ"):
            plt.plot(t_exp, bloch_exp[i], "o", color=col[i], label=f"exp ⟨{pauli}⟩")
            plt.plot(t_exp, bloch_fit[i],   "-", color=col[i], label=f"fit ⟨{pauli}⟩")
        plt.xlabel("t [µs]"); plt.ylabel("Bloch components"); plt.ylim([-1.1, 1.1])
        plt.legend(); plt.tight_layout(); plt.show()

    return θ