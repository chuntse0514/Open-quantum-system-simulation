import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
import pandas as pd
from pathlib import Path
from typing import List, Union, Optional
import numpy as np

DF = Union[pd.DataFrame, List[pd.DataFrame]]


from pathlib import Path
from typing import List, Optional

def collect_csv_files(
    method: str,
    mode: str,
    backend: Optional[str] = None,
    init_state: Optional[str | List[str]] = None,
    qubit: Optional[int | List[int]] = None,
    num_points: Optional[int] = None,
    gates_per_point: Optional[int] = None,
    shots: Optional[int] = None,
) -> List[Path]:
    """
    Collect matching .csv files under results/ based on specified filters.

    Parameters
    ----------
    method: str
        'idle' or 'crosstalk'
    mode : str
        'population' or 'bloch'
    backend : str, optional
        Backend name, e.g., 'ibm_strasbourg'
    init_state : str, optional
        Initial state, e.g., '+', '0', '-i'
    qubit : int, optional
        Qubit index
    num_points : int, optional
        Number of idle time points
    gates_per_point : int, optional
        Number of identity gates per point
    shots : int, optional
        Number of shots per circuit

    Returns
    -------
    List of matching Path objects
    """
    # Always start from the project root
    project_root = Path(__file__).resolve().parent.parent
    results_root = project_root / "results"

    if backend:
        results_root = results_root / method / backend
    if init_state:
        results_root = results_root / f"init{init_state}"

    if not results_root.exists():
        raise FileNotFoundError(f"Path {results_root} does not exist.")

    # Build filename pattern
    pattern = f"{mode}"
    if qubit is not None:
        pattern += f"-q{qubit}"
    if num_points is not None:
        pattern += f"-np{num_points}"
    if gates_per_point is not None:
        pattern += f"-gpp{gates_per_point}"
    if shots is not None:
        pattern += f"-s{shots}"
    pattern += "-*.csv"  # wildcard for the timestamp

    # Collect files
    files = list(results_root.rglob(pattern))
    return sorted(files)


def plot_population(
    file_paths: List[Path],
    init_state: str = "+",
    *,
    sharex: bool = True,
    save: bool = False,
    ncols: int = 2,
) -> None:
    """Plot ρ₀₀ (or Z) vs idle time."""
    dfs = [pd.read_csv(fp) for fp in file_paths]

    n = len(dfs)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows), sharex=sharex)
    axes = np.atleast_1d(axes).ravel()

    for ax, df, fp in zip(axes, dfs, file_paths):
        x = df.iloc[:, 0]
        y = df.iloc[:, -1]
        ax.plot(x, y, "o-")
        ax.set_ylabel(df.columns[-1])
        ax.set_xlabel(df.columns[0])
        ax.set_ylim([0, 1])
        ax.set_title(_prettify_title(fp))
        ax.grid(True)

    fig.suptitle(f"Ground-state Population for $|{init_state}\\rangle$ vs Idle Time", fontsize=16)
    _hide_extra_axes(axes, n)
    _finalise(fig, file_paths, save)


def plot_bloch(
    file_paths: List[Path],
    init_state: str = "+",
    *,
    sharex: bool = True,
    save: bool = False,
    ncols: int = 2,
) -> None:
    """Plot ⟨X⟩, ⟨Y⟩, ⟨Z⟩ vs idle time."""
    dfs = [pd.read_csv(fp) for fp in file_paths]

    n = len(dfs)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4 * nrows), sharex=sharex)
    axes = np.atleast_1d(axes).ravel()

    markers = {"X": "o-", "Y": "s-", "Z": "^-"}
    for ax, df, fp in zip(axes, dfs, file_paths):
        x = df.iloc[:, 0]
        for comp, m in markers.items():
            if comp in df.columns:
                ax.plot(x, df[comp], m, label=f"⟨{comp}⟩")
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel("Bloch vectors")
        ax.set_title(_prettify_title(fp))
        ax.set_ylim([-1.0, 1.0])
        ax.legend()
        ax.grid(True)

    fig.suptitle(f"Bloch Vectors for inital state $|{init_state}\\rangle$ vs Idle Time", fontsize=16)
    _hide_extra_axes(axes, n)
    _finalise(fig, file_paths, save)


def plot_fidelity(
    file_paths: List[Path],
    init_state: str = "+",
    *,
    sharex: bool = False,
    save: bool = False,
    ncols: int = 2,
) -> None:
    """Plot fidelity to initial state vs idle time."""
    state_dict = {
        "0":  np.array([1, 0]),
        "1":  np.array([0, 1]),
        "+":  np.array([1, 1]) / np.sqrt(2),
        "-":  np.array([1, -1]) / np.sqrt(2),
        "+i": np.array([1, 1j]) / np.sqrt(2),
        "-i": np.array([1, -1j]) / np.sqrt(2),
    }
    pauli_dict = {
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
    }
    pauli_vec = np.stack([pauli_dict["X"], pauli_dict["Y"], pauli_dict["Z"]], axis=0)

    if init_state not in state_dict:
        raise ValueError("init_state must be one of 0,1,+,-,+i,-i")
    ket = state_dict[init_state][:, None]
    proj = ket @ ket.conj().T

    dfs = [pd.read_csv(fp) for fp in file_paths]

    n = len(dfs)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4 * nrows), sharex=sharex)
    axes = np.atleast_1d(axes).ravel()

    for ax, df, fp in zip(axes, dfs, file_paths):
        if not {"X", "Y", "Z"}.issubset(df.columns):
            raise ValueError("CSV must contain X, Y, Z columns")

        t = df.iloc[:, 0].to_numpy()
        bloch = df[["X", "Y", "Z"]].to_numpy()

        rho = 0.5 * (np.eye(2) + (bloch @ pauli_vec.reshape(3, 4)).reshape(-1, 2, 2))
        F = np.abs(np.einsum("tij,ji->t", rho, proj))

        ax.plot(t, F, "o-")
        ax.set_ylabel("Fidelity")
        ax.set_xlabel(df.columns[0])
        ax.set_ylim([0.0, 1.0])
        ax.set_title(_prettify_title(fp))
        ax.grid(True)

    fig.suptitle(f"Fidelity for $|{init_state}\\rangle$ vs Idle Time", fontsize=16)
    _hide_extra_axes(axes, n)
    _finalise(fig, file_paths, save)


def plot_purity(
    file_paths: List[Path],
    init_state: str = "+",
    *,
    sharex: bool = False,
    save: bool = False,
    ncols: int = 2,
) -> None:
    """Plot purity vs idle time."""
    pauli_dict = {
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
    }
    pauli_vec = np.stack([pauli_dict["X"], pauli_dict["Y"], pauli_dict["Z"]], axis=0)

    dfs = [pd.read_csv(fp) for fp in file_paths]

    n = len(dfs)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4 * nrows), sharex=sharex)
    axes = np.atleast_1d(axes).ravel()

    for ax, df, fp in zip(axes, dfs, file_paths):
        if not {"X", "Y", "Z"}.issubset(df.columns):
            raise ValueError("CSV must contain X, Y, Z columns")

        t = df.iloc[:, 0].to_numpy()
        bloch = df[["X", "Y", "Z"]].to_numpy()

        rho = 0.5 * (np.eye(2) + (bloch @ pauli_vec.reshape(3, 4)).reshape(-1, 2, 2))
        purity = np.trace(np.einsum("...ij,...jk->...ik", rho, rho), axis1=-2, axis2=-1).real

        ax.plot(t, purity, "o-")
        ax.set_ylabel("Purity")
        ax.set_xlabel(df.columns[0])
        ax.set_ylim([0.0, 1.0])
        ax.set_title(_prettify_title(fp))
        ax.grid(True)

    fig.suptitle(f"Purity for $|{init_state}\\rangle$ vs Idle Time", fontsize=16)
    _hide_extra_axes(axes, n)
    _finalise(fig, file_paths, save)


# ───────────────────────────────────────────────────── helper functions
def _finalise(fig, file_paths: List[Path], save: bool) -> None:
    """Helper: either save all figures individually or show the whole figure."""
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        for fp in file_paths:
            png_path = fp.with_suffix('.png')
            fig.savefig(png_path, dpi=300)
            print(f"✔ Saved: {png_path}")
        plt.close(fig)
    else:
        plt.show()

def _hide_extra_axes(axes, n_valid: int) -> None:
    """Hide unused subplots if number of files is not a multiple of ncols."""
    for ax in axes[n_valid:]:
        ax.set_visible(False)

def _prettify_title(path: Path) -> str:
    """Parse filename and build a clean title with qubit, num-points, gates-per-point, shots."""
    name = path.stem
    fields = {}

    for part in name.split("-"):
        if part.startswith("q"):
            fields["qubit"] = part[1:]
        elif part.startswith("np"):
            fields["num-points"] = part[2:]
        elif part.startswith("gpp"):
            fields["gate-per-points"] = part[3:]
        elif part.startswith("s"):
            fields["shots"] = part[1:]

    title = ", ".join(f"{k}={v}" for k, v in fields.items())
    return title if title else name
