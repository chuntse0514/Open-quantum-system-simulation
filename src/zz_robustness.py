"""Shared ZZ fitting and kernel numerics for three equatorial spectators.

Same physical model/XYZ least-squares objective as zz_crosstalk.py, using
bounded-exponential evaluation and an analytic Jacobian. Time is in microseconds.
Spectator Gamma parameters are relaxation rates, not pure-dephasing rates.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from scipy.linalg import eig, expm
from scipy.optimize import least_squares
from scipy.sparse.linalg import expm_multiply


PARAMETERS = ["gamma_phi0", "Gamma_1", "Gamma_2", "Gamma_3",
              "gamma_down0", "J_1", "J_2", "J_3", "omega0"]
INITIAL = {"+": (1., 0., 0.), "-": (-1., 0., 0.),
           "+i": (0., 1., 0.), "-i": (0., -1., 0.)}


@dataclass
class Dataset:
    label: str
    path: Path
    t: np.ndarray
    y: np.ndarray
    se: np.ndarray
    n0: np.ndarray
    shots: int

    @property
    def initial(self):
        return np.array(INITIAL[self.label])


def load_data(paths):
    """Load selected Bloch CSVs; infer preparations and shots from their paths.

    Each file must describe one equatorial target and three equatorial spectators.
    Times come from the CSV, and count/SE checks use the filename's shot count.
    """
    datasets = []
    for path in paths:
        path = Path(path)
        match = re.fullmatch(r"bloch-q\[([\d, ]+)\]-np\d+-gpp\d+-s(\d+)\.csv", path.name)
        if match is None:
            raise ValueError(f"Expected a Bloch CSV filename with qubits and shots: {path}")
        qubits = [int(q) for q in match[1].replace(",", " ").split()]
        if len(qubits) != 4 or len(set(qubits)) != 4:
            raise ValueError(f"Expected one target and three distinct spectator qubits: {path}")
        shots = int(match[2])
        if shots <= 0:
            raise ValueError(f"Shot count must be positive: {path}")
        preparations = path.parent.name.removeprefix("init").split(",")
        if (not path.parent.name.startswith("init") or len(preparations) != 4
                or any(state not in INITIAL for state in preparations)):
            raise ValueError(f"Expected four equatorial preparations (+, -, +i, -i): {path}")
        label = preparations[0]
        if any(data.label == label for data in datasets):
            raise ValueError(f"Select only one run per target preparation: {label}")

        frame = pd.read_csv(path)
        t = frame.t_us.to_numpy()
        y = frame[[f"{p}_mean" for p in "XYZ"]].to_numpy().T
        se = frame[[f"{p}_std" for p in "XYZ"]].to_numpy().T
        if (len(t) == 0 or not np.all(np.isfinite(t))
                or np.any(t < 0) or np.any(np.diff(t) <= 0)):
            raise ValueError(f"Times must be finite, nonnegative, and strictly increasing: {path}")
        if not np.all(np.isfinite(y)) or np.any(np.abs(y) > 1):
            raise ValueError(f"Bloch means must be finite and within [-1, 1]: {path}")
        counts = shots * (1 + y) / 2
        if not np.allclose(counts, np.rint(counts), atol=1e-8, rtol=0):
            raise ValueError(f"Bloch means do not match integer counts at {shots} shots: {path}")
        if not np.allclose(se, np.sqrt((1-y*y)/shots), atol=1e-12, rtol=1e-7):
            raise ValueError(f"Standard errors do not match the count convention: {path}")
        datasets.append(Dataset(label, path, t, y, se, np.rint(counts).astype(int), shots))
    if not datasets:
        raise ValueError("Select at least one Bloch CSV")
    return datasets


def _spectator(t, gamma, coupling):
    """Phi and its two derivatives, including the joint Gamma=J=0 limit."""
    d = gamma - 2j * coupling
    x = d * t
    small = np.abs(x) < 1e-4
    f = np.empty_like(x, dtype=complex)
    df = np.empty_like(x, dtype=complex)
    # f = (1-exp(-d*t))/d; derivative with respect to d.
    f[small] = t[small] * (1-x[small]/2+x[small]**2/6-x[small]**3/24
                         + x[small]**4/120)
    df[small] = t[small]**2 * (-.5+x[small]/3-x[small]**2/8
                             + x[small]**3/30-x[small]**4/144)
    f[~small] = -np.expm1(-x[~small]) / d
    df[~small] = (x[~small]*np.exp(-x[~small])+np.expm1(-x[~small])) / d**2
    e = np.exp(-1j * coupling * t)
    phi = e * (1 + 1j * coupling * f)
    dg = e * 1j * coupling * df
    dj = e * (-1j*t*(1+1j*coupling*f) + 1j*f + 2*coupling*df)
    return phi, dg, dj


def predict_jac(theta, t, initial, model="ZZ"):
    """Return Bloch predictions (3,T) and Jacobian (3,T,P)."""
    t = np.asarray(t, float)
    theta = np.asarray(theta, float)
    x0, y0, z0 = initial
    if model == "ZZ":
        phi0, down, omega = theta[0], theta[4], theta[8]
    else:
        phi0, down, omega = theta
    base = (x0-1j*y0)*np.exp((-2*phi0-down/2+1j*omega)*t)
    deriv = np.zeros((len(t), len(theta)), complex)
    if model == "ZZ":
        pieces = [_spectator(t, theta[q+1], theta[q+5]) for q in range(3)]
        factors = np.array([p[0] for p in pieces])
        coherence = base * np.prod(factors, axis=0)
        for q, (_, dg, dj) in enumerate(pieces):
            other = base * np.prod(np.delete(factors, q, axis=0), axis=0)
            deriv[:, q+1] = other * dg
            deriv[:, q+5] = other * dj
        down_index, omega_index = 4, 8
    else:
        coherence = base
        down_index, omega_index = 1, 2
    deriv[:, 0] = -2*t*coherence
    deriv[:, down_index] = -.5*t*coherence
    deriv[:, omega_index] = 1j*t*coherence
    z = 1 + (z0-1)*np.exp(-down*t)
    jac = np.zeros((3, len(t), len(theta)))
    jac[0], jac[1] = deriv.real, -deriv.imag
    jac[2, :, down_index] = -(z0-1)*t*np.exp(-down*t)
    return np.array([coherence.real, -coherence.imag, z]), jac


def predict(theta, t, initial, model="ZZ"):
    return predict_jac(theta, t, initial, model)[0]


def canonical(theta):
    """Order spectator pairs by |J|; these are ranks, not physical qubit IDs."""
    p = np.array(theta, copy=True)
    order = np.argsort(np.abs(p[5:8]), kind="stable")
    p[1:4], p[5:8] = p[1:4][order], p[5:8][order]
    return p


def starting_points(model="ZZ", count=16, seed=230519):
    """Data-independent starts shared across all windows; no held-out leakage."""
    rng = np.random.default_rng(seed)
    if model == "GKLS":
        return [np.array([.01, .005, w]) for w in np.linspace(-.2, .2, count)]
    starts = []
    for _ in range(count):
        p = np.zeros(9)
        p[0] = rng.uniform(.0002, .015)
        p[1:4] = 10**rng.uniform(-4, -.3, 3)
        p[4] = rng.uniform(.002, .015)
        p[5:8] = rng.normal(0, .06, 3)
        p[8] = rng.uniform(-.15, .15)
        starts.append(p)
    return starts


def fit(t, y, initial, starts, model="ZZ", max_nfev=1500):
    """Nonnegative rates, unbounded signed frequencies; unweighted XYZ loss.

    Every attempted start is retained in diagnostics. A nonconverged best
    solution is returned as such, not replaced silently by a worse solution.
    """
    p = 9 if model == "ZZ" else 3
    n_rates = 5 if model == "ZZ" else 2
    lower = np.r_[np.zeros(n_rates), np.full(p-n_rates, -np.inf)]
    cache = {}

    def evaluate(theta):
        if "theta" not in cache or not np.array_equal(theta, cache["theta"]):
            pred, jac = predict_jac(theta, t, initial, model)
            cache.update(theta=theta.copy(), residual=(pred-y).ravel(),
                         jac=jac.reshape(-1, p))
        return cache

    candidates, diagnostics = [], []
    for index, start in enumerate(starts):
        try:
            res = least_squares(lambda v: evaluate(v)["residual"], start,
                                jac=lambda v: evaluate(v)["jac"],
                                bounds=(lower, np.full(p, np.inf)), x_scale="jac",
                                ftol=1e-9, xtol=1e-9, gtol=1e-9,
                                max_nfev=max_nfev)
            mse = np.mean(res.fun**2)
            diagnostics.append(dict(start=index, mse=mse, success=bool(res.success),
                                    status=int(res.status), nfev=res.nfev,
                                    optimality=float(res.optimality),
                                    **dict(zip(PARAMETERS if model == "ZZ" else
                                               ["gamma_phi0", "gamma_down0", "omega0"],
                                               canonical(res.x) if model == "ZZ" else res.x))))
            candidates.append((mse, res))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            diagnostics.append(dict(start=index, mse=np.nan, success=False,
                                    status=-99, nfev=0, error=str(exc)))
    if not candidates:
        raise RuntimeError(f"All {model} starts failed: {diagnostics}")
    _, best = min(candidates, key=lambda item: item[0])
    theta = canonical(best.x) if model == "ZZ" else best.x
    singular = np.linalg.svd(best.jac, compute_uv=False)
    return dict(theta=theta, mse=float(np.mean(best.fun**2)),
                success=bool(best.success), attempts=diagnostics,
                jacobian_condition=float(singular[0]/max(singular[-1], 1e-300)))


def baseline(theta, model="ZZ"):
    return (-2*theta[0]-.5*theta[4]+1j*theta[8] if model == "ZZ"
            else -2*theta[0]-.5*theta[1]+1j*theta[2])


def coherence_matrix(theta):
    """Parity-space generator for c_S=<sigma^- prod_{q in S} Z_q>.

    Equatorial spectator initial conditions give c_S(0)=delta_{S,empty}.
    Eliminating the seven hidden components gives k_reg=b exp(Ct)c exactly.
    No polynomial roots or distinct-pole assumption is needed.
    """
    matrix = np.eye(8, dtype=complex)*baseline(theta)
    for subset in range(8):
        for q in range(3):
            matrix[subset, subset ^ (1 << q)] -= 1j*theta[5+q]
            if subset & (1 << q):
                matrix[subset, subset] -= theta[1+q]
                matrix[subset, subset ^ (1 << q)] += theta[1+q]
    return matrix


def kernel(theta, t, *, reference_time=None):
    """Exact regular kernel, with a matrix-exponential fallback near degeneracy.

    reference_time selects the dominant-mode diagnostic only; by default it is
    the last evaluation time. It does not change the calculated kernel.
    """
    t = np.asarray(t, float)
    if reference_time is None:
        reference_time = float(t[-1])
    matrix = coherence_matrix(theta)
    b, c, hidden = matrix[0, 1:], matrix[1:, 0], matrix[1:, 1:]
    poles, vectors = eig(hidden)
    condition = float(np.linalg.cond(vectors))
    method = "eigen"
    if condition < 1e8:
        residues = (b @ vectors) * np.linalg.solve(vectors, c)
        values = residues @ np.exp(poles[:, None]*t)
        # k(0)=b.c; cancellation-sensitive checks at two further times.
        check_t = np.array([0., t[len(t)//2], t[-1]])
        direct = np.array([b @ expm(hidden*x) @ c for x in check_t])
        fast = residues @ np.exp(poles[:, None]*check_t)
        good = np.allclose(fast, direct, rtol=1e-7, atol=1e-11)
    else:
        good = False
    if not good:
        method = "expm"
        if len(t) > 1 and np.allclose(np.diff(t), t[1]-t[0]):
            values = expm_multiply(hidden, c, start=t[0], stop=t[-1],
                                   num=len(t), endpoint=True) @ b
        else:
            values = np.array([b @ expm(hidden*x) @ c for x in t])
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Nonfinite kernel: retain failure in diagnostics")
    # Diagnostic descriptor, not an assumption of single-frequency dynamics.
    dominant = int(np.argmax(np.abs(residues*np.exp(poles*reference_time)))) if good else None
    return values, dict(eigenvector_condition=condition, method=method,
                        reference_time=float(reference_time),
                        max_pole_real=float(poles.real.max()),
                        dominant_omega=float(poles[dominant].imag) if good else np.nan,
                        dominant_decay=float(-poles[dominant].real) if good else np.nan)


def score(y, prediction, se):
    residual = prediction-y
    return dict(rmse_xyz=float(np.sqrt(np.mean(residual**2))),
                rmse_xy=float(np.sqrt(np.mean(residual[:2]**2))),
                # Diagnostic only, not reduced chi-square or a formal p-value.
                rms_standardized=float(np.sqrt(np.mean((residual/se)**2))))


def fit_shared_gkls(datasets, cutoff, n_starts=16, *, seed=230519):
    """Fit one GKLS channel to all preparations, as in manuscript Eq. (91)."""
    reference = datasets[0]
    train = reference.t <= cutoff+1e-9
    aligned = []
    for data in datasets:
        np.testing.assert_allclose(data.t, reference.t)
        initial_coherence = data.initial[0] - 1j*data.initial[1]
        if abs(initial_coherence) == 0:
            raise ValueError("Shared GKLS fit requires an equatorial initial state")
        coherence = (data.y[0, train] - 1j*data.y[1, train]) / initial_coherence
        aligned.append(np.array([coherence.real, -coherence.imag, data.y[2, train]]))

    # Phase-aligning the four preparations makes their average sufficient for
    # the shared least-squares optimum; preparation-dependent residuals add
    # only a parameter-independent constant to the objective.
    mean_bloch = np.mean(aligned, axis=0)
    result = fit(reference.t[train], mean_bloch, np.array([1., 0., 0.]),
                 starting_points("GKLS", n_starts, seed=seed), "GKLS")
    residuals = [predict(result["theta"], data.t[train], data.initial, "GKLS")
                 - data.y[:, train] for data in datasets]
    result["mse"] = float(np.mean(np.stack(residuals)**2))
    return result


def fit_windows(datasets, cutoffs, n_starts=16, *, seed=230519):
    """Fit caller-selected time windows with reproducible multistart guesses."""
    fits, rows, attempts = {}, [], []
    for cutoff in cutoffs:
        gkls_result = fit_shared_gkls(datasets, cutoff, n_starts, seed=seed)
        for attempt in gkls_result["attempts"]:
            attempts.append(dict(state="shared", cutoff=cutoff,
                                 model="GKLS", **attempt))
        for data in datasets:
            train = data.t <= cutoff+1e-9
            zz_result = fit(data.t[train], data.y[:, train], data.initial,
                            starting_points("ZZ", n_starts, seed=seed), "ZZ")
            fits[(data.label, cutoff, "ZZ")] = zz_result
            fits[(data.label, cutoff, "GKLS")] = gkls_result
            for attempt in zz_result["attempts"]:
                attempts.append(dict(state=data.label, cutoff=cutoff,
                                     model="ZZ", **attempt))
            for model, result in (("ZZ", zz_result), ("GKLS", gkls_result)):
                pred = predict(result["theta"], data.t, data.initial, model)
                for split, mask in (("train", train), ("held_out", ~train)):
                    if mask.any():
                        rows.append(dict(state=data.label, cutoff=cutoff, model=model,
                                         split=split, n_times=int(mask.sum()),
                                         first_time=data.t[mask][0], last_time=data.t[mask][-1],
                                         converged=result["success"],
                                         **score(data.y[:, mask], pred[:, mask], data.se[:, mask])))
            print(f"{data.label:>2}, cutoff {cutoff:g}: ZZ train MSE "
                  f"{fits[data.label, cutoff, 'ZZ']['mse']:.6g}", flush=True)
    return fits, pd.DataFrame(rows), pd.DataFrame(attempts)


def bootstrap(data, cutoff, zz_fit, grid, count=1000, seed=0,
              max_nfev=1000, *, fit_seed=230519, n_starts=16, reference_time=None):
    """Empirical count bootstrap conditional on this run and model.

    Each replicate uses a warm start, a perturbed warm start, and an independent
    data-free start (selected using bootstrap training loss). This is a local
    basin-aware refit, not a guarantee of global optimum. Failed replicates stay
    NaN and are logged; summaries must disclose their number.

    seed controls count resampling and warm-start perturbations; fit_seed and
    n_starts control the independent starting-point pool. reference_time is
    passed to the kernel's dominant-mode diagnostic.
    """
    rng = np.random.default_rng(seed)
    train = data.t <= cutoff+1e-9
    shape = (count,)
    arrays = dict(theta=np.full(shape+(9,), np.nan),
                  kernel=np.full(shape+(len(grid),), np.nan+1j*np.nan),
                  bloch=np.full(shape+(3, len(grid)), np.nan))
    records = []
    independent = starting_points(count=n_starts, seed=fit_seed)
    start_time = time.monotonic()
    for iteration in range(count):
        sampled = rng.binomial(data.shots, data.n0[:, train]/data.shots)
        y = 2*sampled/data.shots-1
        warm = zz_fit["theta"].copy()
        perturb = warm.copy()
        perturb[:5] = np.maximum(perturb[:5]*np.exp(rng.normal(0, .3, 5)), 1e-7)
        perturb[5:] += rng.normal(0, .005, 4)
        record = dict(replicate=iteration, state=data.label, cutoff=cutoff, seed=seed,
                      zz_success=False, kernel_success=False)
        try:
            result = fit(data.t[train], y, data.initial,
                         [warm, perturb, independent[iteration % len(independent)]],
                         max_nfev=max_nfev)
            record.update(zz_success=result["success"], mse=result["mse"],
                          failed_starts=sum(not a["success"] for a in result["attempts"]))
            if result["success"]:
                theta = result["theta"]
                arrays["theta"][iteration] = theta
                arrays["bloch"][iteration] = predict(theta, grid, data.initial)
                values, diagnostic = kernel(theta, grid, reference_time=reference_time)
                arrays["kernel"][iteration] = values
                record.update(kernel_success=True, **diagnostic)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            record["error"] = str(exc)
        records.append(record)
        if (iteration+1) % 100 == 0:
            print(f"{data.label:>2} {cutoff:g}: {iteration+1}/{count}, "
                  f"{time.monotonic()-start_time:.0f}s", flush=True)
    return arrays, pd.DataFrame(records)


def numerical_checks():
    """Independent checks of the forward model, derivatives and Schur kernel."""
    theta = np.array([.002, .001, .11, .23, .008, .025, -.045, .07, .035])
    t = np.linspace(0, 400, 41)
    initial = np.array([0., 1., 0.])
    pred, jac = predict_jac(theta, t, initial)
    for q in range(9):
        h = 1e-7
        step = np.eye(9)[q]*h
        numerical = (predict(theta+step, t, initial)-predict(theta-step, t, initial))/(2*h)
        np.testing.assert_allclose(jac[:, :, q], numerical, atol=2e-6, rtol=2e-5)
    m = coherence_matrix(theta)
    xi = np.array([expm(m*x)[0, 0] for x in t])
    np.testing.assert_allclose(pred[0]-1j*pred[1], -1j*xi, atol=1e-12)
    b, c, hidden = m[0, 1:], m[1:, 0], m[1:, 1:]
    for s in (.1+.03j, .02+.2j, .5-.1j):
        xi_laplace = np.linalg.inv(s*np.eye(8)-m)[0, 0]
        k_laplace = b @ np.linalg.solve(s*np.eye(7)-hidden, c)
        np.testing.assert_allclose(k_laplace, s-baseline(theta)-1/xi_laplace, atol=1e-12)
    values, _ = kernel(theta, t)
    np.testing.assert_allclose(values, [b @ expm(hidden*x) @ c for x in t], atol=1e-12)
    single = theta.copy()
    single[6:8] = 0
    expected = (-single[5]**2-1j*single[1]*single[5])*np.exp((baseline(single)-single[1])*t)
    np.testing.assert_allclose(kernel(single, t)[0], expected, atol=1e-12)
    zero = theta.copy()
    zero[5:8] = 0
    np.testing.assert_allclose(kernel(zero, t)[0], 0, atol=1e-12)
    np.testing.assert_allclose(_spectator(t, 0., 0.)[0], 1, atol=1e-12)
    # Check cancellation-safe evaluation at rates for which cosh would overflow.
    large = theta.copy()
    large[1:4] = 10
    assert np.isfinite(predict(large, t, initial)).all()
    return "Passed: analytic Jacobian, full matrix dynamics, Laplace identity, direct expm, single/zero spectator limits, large-rate stability."
