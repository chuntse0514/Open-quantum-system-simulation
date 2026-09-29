import os
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from itertools import product
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


"""
Single-qubit tomography on IBM Quantum backends
===============================================

The first physical qubit is measured. Any additional qubits are prepared as
spectators, allowing idle and crosstalk experiments to use the same workflow.

Examples
--------
python -m src.single_qubit_tomography bloch -b ibm_strasbourg -q 4 -init +
python -m src.single_qubit_tomography bloch -b ibm_strasbourg -q 4,3,5,15 -init +,+,+,+
python -m src.single_qubit_tomography bloch -b ibm_strasbourg -q 4,3,5,15 -init "+,+,+,+;-,+,+,+;+i,+,+,+;-i,+,+,+"
"""


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


state_preps = {
    "0":  lambda qc, index: None,
    "1":  lambda qc, index: qc.x(index),
    "+":  lambda qc, index: qc.h(index),
    "-":  lambda qc, index: (qc.x(index), qc.h(index)),
    "+i": lambda qc, index: (qc.h(index), qc.s(index)),
    "-i": lambda qc, index: (qc.h(index), qc.sdg(index)),
}


def build_population_circuits(
    qubits: List[int],
    initial_state: List[str],
    num_points: int,
    gates_per_point: int,
    pm,
) -> List:
    circuits = []
    for n in range(num_points):
        qc = QuantumCircuit(len(qubits), 1)
        for index, state in enumerate(initial_state):
            state_preps[state](qc, index)

        for _ in range(n * gates_per_point):
            qc.id(0)

        qc.measure(0, 0)
        circuits.append(pm.run(qc))
    return circuits


def build_bloch_circuits(
    qubits: List[int],
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
    for n, basis in product(range(num_points), ("X", "Y", "Z")):
        qc = QuantumCircuit(len(qubits), 1)
        for index, state in enumerate(initial_state):
            state_preps[state](qc, index)

        for _ in range(n * gates_per_point):
            qc.id(0)

        basis_rots[basis](qc)
        qc.measure(0, 0)
        circuits.append(pm.run(qc))
    return circuits


def run_batches(sampler: Sampler, circuits: List, shots: int, max_batch: int):
    results = []
    for i in range(0, len(circuits), max_batch):
        batch = circuits[i:i + max_batch]
        print(f"Submitting batch {i // max_batch + 1} ... ({len(batch)} circuits)")
        job = sampler.run(batch, shots=shots)
        results.extend(job.result())
    return results


def result_counts(result):
    for register_name in ("c", "meas"):
        register = getattr(result.data, register_name, None)
        if register is not None:
            return register.get_counts()
    raise RuntimeError("Result does not contain a supported measurement register")


def analyse_population(
    results, shots: int, num_points: int, gates_per_point: int,
    id_duration_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0_means = []
    p0_stds = []

    for result in results:
        counts = result_counts(result)
        p0_prob = counts.get("0", 0) / shots
        p0_means.append(p0_prob)
        p0_stds.append(np.sqrt(p0_prob * (1 - p0_prob) / shots))

    time_steps_us = (
        np.arange(num_points) * gates_per_point * id_duration_s * 1e6
    )
    return time_steps_us, np.asarray(p0_means), np.asarray(p0_stds)


def analyse_bloch(
    results, shots: int, num_points: int, gates_per_point: int,
    id_duration_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bx, by, bz = [], [], []
    sx, sy, sz = [], [], []

    for i in range(num_points):
        bloch_means: Dict[str, float] = {}
        bloch_stds: Dict[str, float] = {}
        for j, basis in enumerate(("X", "Y", "Z")):
            counts = result_counts(results[i * 3 + j])
            p0 = counts.get("0", 0)
            p1 = counts.get("1", 0)
            total = p0 + p1

            if total == 0:
                bloch_means[basis] = 0
                bloch_stds[basis] = 0
                continue

            expectation = (p0 - p1) / total
            bloch_means[basis] = expectation
            bloch_stds[basis] = np.sqrt((1 - expectation**2) / shots)

        bx.append(bloch_means["X"])
        by.append(bloch_means["Y"])
        bz.append(bloch_means["Z"])
        sx.append(bloch_stds["X"])
        sy.append(bloch_stds["Y"])
        sz.append(bloch_stds["Z"])

    time_steps_us = (
        np.arange(num_points) * gates_per_point * id_duration_s * 1e6
    )
    return (
        time_steps_us,
        np.asarray(bx), np.asarray(by), np.asarray(bz),
        np.asarray(sx), np.asarray(sy), np.asarray(sz),
    )


def save_population_plot(t, rho00, rho00_stds, out_png: Path):
    plt.figure()
    plt.errorbar(t, rho00, yerr=rho00_stds, fmt="o-", capsize=3)
    plt.xlabel("t $(\\mu s)$")
    plt.ylabel("$\\rho_{00}$")
    plt.ylim([0.0, 1.0])
    plt.title("Ground-state population vs idle time")
    plt.grid(True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_bloch_plot(t_us, bx, by, bz, sx, sy, sz, out_png: Path):
    plt.figure(figsize=(10, 5))
    plt.errorbar(t_us, bx, yerr=sx, fmt="o-", capsize=3, label="$\\langle X\\rangle$")
    plt.errorbar(t_us, by, yerr=sy, fmt="s-", capsize=3, label="$\\langle Y\\rangle$")
    plt.errorbar(t_us, bz, yerr=sz, fmt="^-", capsize=3, label="$\\langle Z\\rangle$")
    plt.xlabel("t $(\\mu s)$")
    plt.ylabel("Bloch vectors")
    plt.ylim([-1.0, 1.0])
    plt.title("Bloch vector vs idle time")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def write_csv(path: Path, header: List[str], rows: List[Tuple]):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def make_paths(
    backend: str,
    initial_state: List[str],
    mode: str,
    qubits: List[int],
    num_points: int,
    gates_per_point: int,
    shots: int,
    run_timestamp: str,
    ext: str,
) -> Path:
    folder = (
        Path("results/single_qubit_tomography")
        / backend
        / run_timestamp
        / f"init{','.join(initial_state)}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    qubit_label = qubits[0] if len(qubits) == 1 else qubits
    name = (
        f"{mode}-q{qubit_label}-np{num_points}"
        f"-gpp{gates_per_point}-s{shots}{ext}"
    )
    return folder / name


def main():
    parser = argparse.ArgumentParser(
        description="Single-qubit idle and spectator-noise tomography"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    common = {
        "backend": ("-b", "--backend", str, None),
        "qubit": ("-q", "--qubit", str, "0"),
        "num_points": ("-np", "--num-points", int, 50),
        "gates_per_point": ("-gpp", "--gates-per-point", int, 50),
        "shots": ("-s", "--shots", int, 8192),
    }

    for mode, help_text in (
        ("population", "T1-style population decay"),
        ("bloch", "Full Bloch tomography vs time"),
    ):
        command = subparsers.add_parser(mode, help=help_text)
        for dest, (short, long, typ, default) in common.items():
            command.add_argument(short, long, dest=dest, type=typ, default=default)
        command.add_argument(
            "-init", "--initial-state", type=str, default="1",
            help="Semicolon-separated configurations of comma-separated states",
        )
        command.add_argument(
            "-id", "--job-id", type=str,
            help="Download an existing job instead of submitting a new one",
        )

    args = parser.parse_args()
    run_timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    qubits = [int(q) for q in args.qubit.replace(",", " ").split()]
    initial_state_configs = []
    for config in args.initial_state.split(";"):
        initial_state = config.replace(",", " ").split()
        if not initial_state:
            continue
        if len(initial_state) == 1 and len(qubits) > 1:
            initial_state *= len(qubits)
        if len(initial_state) != len(qubits):
            parser.error(
                f"Mismatch: {len(initial_state)} initial states for "
                f"{len(qubits)} qubits"
            )
        unknown_states = [state for state in initial_state if state not in state_preps]
        if unknown_states:
            parser.error(f"Unknown initial state: {unknown_states[0]}")
        initial_state_configs.append(initial_state)
    if not initial_state_configs:
        parser.error("At least one initial-state configuration is required")
    if args.job_id and len(initial_state_configs) > 1:
        parser.error("Job retrieval accepts one initial-state configuration only")

    token = os.getenv("IBM_TOKEN")
    if not token:
        raise RuntimeError("Please set environment variable IBM_TOKEN")

    if args.job_id:
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        job = service.job(args.job_id)
        backend = service.backend(job.backend().name)
        print(
            f"Fetching remote job '{job.job_id()}' from backend "
            f"{backend.name} (status={job.status()})"
        )
        results = job.result()
        args.num_points = (
            len(results) if args.mode == "population" else len(results) // 3
        )
        results_by_config = [(initial_state_configs[0], results)]
    else:
        backend = init_service(token, args.backend)
        pass_manager = generate_preset_pass_manager(
            target=backend.target,
            initial_layout=qubits,
            optimization_level=0,
        )
        circuits = []
        for initial_state in initial_state_configs:
            print(f"Preparing initial state: {initial_state}")
            if args.mode == "population":
                circuits.extend(build_population_circuits(
                    qubits, initial_state, args.num_points,
                    args.gates_per_point, pass_manager,
                ))
            else:
                circuits.extend(build_bloch_circuits(
                    qubits, initial_state, args.num_points,
                    args.gates_per_point, pass_manager,
                ))
        sampler = Sampler(backend)
        results = run_batches(
            sampler, circuits, args.shots,
            backend.configuration().max_experiments,
        )
        results_per_config = (
            args.num_points if args.mode == "population" else 3 * args.num_points
        )
        results_by_config = [
            (
                initial_state,
                results[
                    index * results_per_config:(index + 1) * results_per_config
                ],
            )
            for index, initial_state in enumerate(initial_state_configs)
        ]

    id_duration_s = backend.target["id"][(qubits[0],)].duration

    for initial_state, config_results in results_by_config:
        png_path = make_paths(
            backend.name, initial_state, args.mode, qubits, args.num_points,
            args.gates_per_point, args.shots, run_timestamp, ".png",
        )
        csv_path = png_path.with_suffix(".csv")

        if args.mode == "population":
            t_us, means, stds = analyse_population(
                config_results, args.shots, args.num_points,
                args.gates_per_point, id_duration_s,
            )
            save_population_plot(t_us, means, stds, png_path)
            write_csv(
                csv_path,
                ["t_us", "rho00_mean", "rho00_std"],
                zip(t_us, means, stds),
            )
        else:
            t_us, bx, by, bz, sx, sy, sz = analyse_bloch(
                config_results, args.shots, args.num_points,
                args.gates_per_point, id_duration_s,
            )
            save_bloch_plot(t_us, bx, by, bz, sx, sy, sz, png_path)
            write_csv(
                csv_path,
                [
                    "t_us", "X_mean", "Y_mean", "Z_mean",
                    "X_std", "Y_std", "Z_std",
                ],
                zip(t_us, bx, by, bz, sx, sy, sz),
            )

        print(f"Saved -> {png_path} and {csv_path}")


if __name__ == "__main__":
    main()
