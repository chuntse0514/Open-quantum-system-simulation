import os
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from itertools import product
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

### set your IBM token here ###
# os.environ["IBM_TOKEN"] = "<your-token-here>"

"""
Idle-noise probe for IBM Quantum backends
========================================

Usage
-----
python -m src.noise_probe_crosstalk <mode> [options]

Modes
-----
population   Measure ground-state population decay (T1-style)
bloch        Measure full Bloch vector components vs idle time

Key options
-----------
-b, --backend              specific backend (e.g. ibm_brisbane); if omitted, pick least busy
-q, --qubit                physical qubit index to probe (default: "0")
-np, --num-points          number of idle-time points       (default: 50)
-gpp, --gates-per-point    identity gates per point         (default: 100)
-s,  --shots               shots per circuit                (default: 8192)
-init, --initial-state     0, 1, +, -, +i, -i               (default: "1")
-id, --job-id              download an existing job instead of submitting

Examples
--------
# new population experiment on least-busy device, qubit 0
python -m src.noise_probe_crosstalk population

# Bloch tomography on qubit 1 of ibm_strasbourg
python -m src.noise_probe_crosstalk bloch -b=ibm_strasbourg -q=4,3,5,15 -np=50 -gpp=20 -init=+,+,+,+

# fetch results of a previously submitted job
python -m src.noise_probe_crosstalk population --job-id=abc1234567890

Notes
-----
* Results are stored in
      results/<backend>/init<state>/
      <mode>-q<qubit>-np<num>-gpp<gpp>-s<shots>-<timestamp>.{{csv,png}}
* Set your API token with  ``export IBM_TOKEN=...``  or hard-code it in the script
"""

# --------------------------------------------------------------------------
#  Utility functions
# --------------------------------------------------------------------------
def init_service(token: str, backend_name: str | None = None):
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)

    if backend_name:
        backend = service.backend(backend_name)
        if backend.status().operational is False:
            raise RuntimeError(f"Backend {backend_name} is not operational.")
        print(f"Using user-selected backend: {backend.name}")
    else:
        backend = service.least_busy(simulator=False, operational=True)
        print(f"Using least-busy backend:     {backend.name}")

    return backend


# -------------------------------------------------
#  Single-qubit state preparation helpers
# -------------------------------------------------
state_preps = {
    "0":  lambda qc, index: None,
    "1":  lambda qc, index: qc.x(index),
    "+":  lambda qc, index: qc.h(index),
    "-":  lambda qc, index: (qc.x(index), qc.h(index)),   # H|1⟩ = |−⟩
    "+i": lambda qc, index: (qc.h(index), qc.s(index)),   # S H|0⟩ = |+i⟩
    "-i": lambda qc, index: (qc.h(index), qc.sdg(index)), # S† H|0⟩ = |−i⟩
}


def build_population_circuits(
    qubit: List[int], 
    initial_state: List[str],
    num_points: int,
    gates_per_point: int,
    pm,
) -> List:
    circuits = []
    for n in range(0, num_points):
        qc = QuantumCircuit(len(qubit), 1)
        # prepare chosen state
        for index, state in enumerate(initial_state):
            state_preps[state](qc, index)
        
        # idle for n × gates_per_point cycles
        for _ in range(n * gates_per_point):
            qc.id(0)
        qc.measure(0, 0)
        circuits.append(pm.run(qc))
    return circuits


def build_bloch_circuits(
    qubit: List[int],
    initial_state: List[str],
    num_points: int,
    gates_per_point: int,
    pm,
) -> List:
    basis_rots = {
        "X": lambda qc: qc.h(0),
        "Y": lambda qc: (qc.sdg(0), qc.h(0)),
        "Z": lambda qc: None,
    }

    circuits = []
    for n, basis in product(range(0, num_points), ("X", "Y", "Z")):
        qc = QuantumCircuit(len(qubit), 1)
        # prepare chosen state
        for index, state in enumerate(initial_state):
            state_preps[state](qc, index)

        for _ in range(n * gates_per_point):
            qc.id(0)

        # rotate into measurement basis
        basis_rots[basis](qc)
        
        qc.measure(0, 0)
        circuits.append(pm.run(qc))
    return circuits


def run_batches(
    sampler: Sampler,
    circuits: List,
    shots: int,
    max_batch: int,
):
    results = []
    for i in range(0, len(circuits), max_batch):
        batch = circuits[i : i + max_batch]
        print(f"Submitting batch {i // max_batch + 1} ... ({len(batch)} circuits)")
        job = sampler.run(batch, shots=shots)
        results.extend(job.result())
    return results


# --------------------------------------------------------------------------
#  Population-decay analysis
# --------------------------------------------------------------------------
def analyse_population(
    results, shots: int, num_points: int, gates_per_point: int
) -> Tuple[np.ndarray, np.ndarray]:
    rho00 = []
    for res in results:
        counts = res.data.c.get_counts()
        p0 = counts.get("0", 0) / shots
        rho00.append(p0)
    time_steps = np.arange(0, num_points) * gates_per_point
    return time_steps, np.array(rho00)


def save_population_plot(t, rho00, out_png: Path):
    plt.figure()
    plt.plot(t, rho00, "o-")
    plt.xlabel("Identity-gate count")
    plt.ylabel("$\\rho_{00}$")
    plt.title("Ground-state population vs idle time")
    plt.grid(True)
    plt.savefig(out_png, dpi=300)
    plt.close()


# --------------------------------------------------------------------------
#  Bloch-vector analysis
# --------------------------------------------------------------------------
def analyse_bloch(
    results, shots: int, num_points: int, gates_per_point: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bx, by, bz = [], [], []
    print(len(results))
    for i in range(num_points):
        bloch: Dict[str, float] = {}
        for j, basis in enumerate(("X", "Y", "Z")):
            counts = results[i * 3 + j].data.c.get_counts()
            p0 = counts.get("0", 0)
            p1 = counts.get("1", 0)
            bloch[basis] = (p0 - p1) / shots
        bx.append(bloch["X"])
        by.append(bloch["Y"])
        bz.append(bloch["Z"])
    t = np.arange(0, num_points) * gates_per_point
    return t, np.array(bx), np.array(by), np.array(bz)


def save_bloch_plot(t, bx, by, bz, out_png: Path):
    plt.figure(figsize=(10, 5))
    plt.plot(t, bx, "o-", label="$⟨X⟩$")
    plt.plot(t, by, "s-", label="$⟨Y⟩$")
    plt.plot(t, bz, "^-", label="$⟨Z⟩$")
    plt.xlabel("Identity-gate count")
    plt.ylabel("Bloch component")
    plt.title("Bloch vector vs idle time")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_png, dpi=300)
    plt.close()


# --------------------------------------------------------------------------
#  CSV helpers
# --------------------------------------------------------------------------
def write_csv(path: Path, header: List[str], rows: List[Tuple]):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
        

def make_paths(
    backend: str,
    init_state: str,
    mode: str,
    qubit: int,
    num_points: int,
    gpp: int,
    shots: int,
    ext: str
) -> Path:
    """
    →  results/<backend>/init<state>/
        <mode>-np<num_points>-gpp<gpp>-YYYY-MM-DDThh-mm-ss.<ext>
    """
    stamp  = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    folder = Path("results/crosstalk") / backend / f"init{",".join(init_state)}"
    folder.mkdir(parents=True, exist_ok=True)
    name   = f"{mode}-q{qubit}-np{num_points}-gpp{gpp}-s{shots}-{stamp}{ext}"
    return folder / name


# --------------------------------------------------------------------------
#  Main CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Idle-noise probe on IBMQ backend")
    sub = parser.add_subparsers(dest="mode", required=True)

    common = {
        "backend":         ("-b",   "--backend",         str, None),
        "qubit":           ("-q",   "--qubit",           str, "0"), 
        "num_points":      ("-np",  "--num-points",      int, 50),
        "gates_per_point": ("-gpp", "--gates-per-point", int, 100),
        "shots":           ("-s",   "--shots",           int, 8192),
    }

    pop = sub.add_parser("population", help="T1-style population decay")
    for dest, (short, long, typ, default) in common.items():
        pop.add_argument(short, long, dest=dest, type=typ, default=default)
    pop.add_argument(
        "-init", "--initial-state",
        type=str, 
        default="1",
        help="'0' | '1' | '+' | '-' | '+i' | '-i'",
    )
    pop.add_argument(
        "-id", "--job-id",
        type=str,
        help="Download results of an existing job instead of submitting a new one",
    )


    bloch = sub.add_parser("bloch", help="Full Bloch tomography vs time")
    for dest, (short, long, typ, default) in common.items():
        bloch.add_argument(short, long, dest=dest, type=typ, default=default)
    bloch.add_argument(
        "-init", "--initial-state",
        type=str,
        default="1",
        help="'0' | '1' | '+' | '-' | '+i' | '-i'",
    )
    bloch.add_argument(
        "-id", "--job-id",
        type=str,
        help="Download results of an existing job instead of submitting a new one",
    )

    args = parser.parse_args()
    if isinstance(args.qubit, str):
        args.qubit = [int(q) for q in args.qubit.replace(",", " ").split()]
    if isinstance(args.initial_state, str):
        args.initial_state = args.initial_state.replace(",", " ").split()
        
    if len(args.initial_state) == 1 and len(args.qubit) > 1:
        args.initial_state = args.initial_state * len(args.qubit)
        
    assert len(args.initial_state) == len(args.qubit), (
        f"Mismatch size: {len(args.initial_state)} initial states vs {len(args.qubit)} qubits"
    )
        
    token = os.getenv("IBM_TOKEN")
    if not token:
        raise RuntimeError("Please set environment variable IBM_TOKEN")
    
    # ------------------------------------------------------------------
    # Job retrieval branch
    # ------------------------------------------------------------------
    if args.job_id:
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        job = service.job(args.job_id)
        backend_name = job.backend().name
        png_path = make_paths(
            backend_name,
            args.initial_state,
            args.mode,             # "population" or "bloch"
            args.qubit,
            args.num_points,
            args.gates_per_point,
            args.shots,
            ".png",
        )
        csv_path = png_path.with_suffix(".csv")
        print(f"Fetching remote job '{job.job_id()}' from backend {backend_name} (status={job.status()})")

        # Wait a reasonable amount of time if the job is still running
        results = job.result()

        # Infer num_points when only results are available
        if args.mode == "population":
            args.num_points = len(results)
        else:  # bloch : 3 results per time step
            args.num_points = len(results) // 3
            
        if args.mode == "population":
            t, rho00 = analyse_population(
                results, args.shots, args.num_points, args.gates_per_point
            )
            save_population_plot(t, rho00, png_path)
            write_csv(
                csv_path, 
                ["id_gates", "rho00"], 
                zip(t, rho00),
            )
            print(f"Saved → {png_path}  and  {csv_path}")
            
        elif args.mode == "bloch":
            t, bx, by, bz = analyse_bloch(
                results, args.shots, args.num_points, args.gates_per_point
            )
            save_bloch_plot(t, bx, by, bz, png_path)
            write_csv(
                csv_path,
                ["id_gates", "X", "Y", "Z"],
                zip(t, bx, by, bz),
            )
            print(f"Saved → {png_path}  and  {csv_path}")
    
    # ------------------------------------------------------------------
    # Fresh submission branch
    # ------------------------------------------------------------------
    else:
        backend = init_service(token, args.backend)
        pm = generate_preset_pass_manager(
            target=backend.target, 
            initial_layout=args.qubit,
            optimization_level=0
        )
        sampler = Sampler(backend)
        max_batch = backend.configuration().max_experiments
        png_path = make_paths(
            backend.name,
            args.initial_state,
            args.mode,             # "population" or "bloch"
            args.qubit,
            args.num_points,
            args.gates_per_point,
            args.shots,
            ".png",
        )
        csv_path = png_path.with_suffix(".csv")
        
        if args.mode == "population":
            circuits = build_population_circuits(
                args.qubit, args.initial_state, args.num_points, args.gates_per_point, pm,
            )
            results = run_batches(sampler, circuits, args.shots, max_batch)
            t, rho00 = analyse_population(
                results, args.shots, args.num_points, args.gates_per_point
            )
            save_population_plot(t, rho00, png_path)
            write_csv(csv_path, ["id_gates", "rho00"], zip(t, rho00))
            print(f"Saved → {png_path}  and  {csv_path}")

        elif args.mode == "bloch":
            circuits = build_bloch_circuits(
                args.qubit, args.initial_state, args.num_points, args.gates_per_point, pm,
            )
            results = run_batches(sampler, circuits, args.shots, max_batch)
            t, bx, by, bz = analyse_bloch(
                results, args.shots, args.num_points, args.gates_per_point
            )
            save_bloch_plot(t, bx, by, bz, png_path)
            write_csv(csv_path, ["id_gates", "X", "Y", "Z"], zip(t, bx, by, bz))
            print(f"Saved → {png_path}  and  {csv_path}")
        

if __name__ == "__main__":
    main()