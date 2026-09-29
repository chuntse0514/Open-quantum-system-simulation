"""Two-qubit state tomography on IBM Quantum backends.

Examples
--------
python -m src.state_tomography_two_qubit -b ibm_brussels -q 49,50 -init +
python -m src.state_tomography_two_qubit -b ibm_brussels -q 49,50 -init "+,+;-,-;+i,+i;-i,-i"
python -m src.state_tomography_two_qubit -f results/path/to/tomography.npz

Pauli labels follow the supplied qubit order: XI measures the first qubit.
Stored coherent vectors are <P>/2, with standard errors of the mean.
"""

import os
import argparse
import json
from pathlib import Path
from typing import List
from itertools import product
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Batch, SamplerV2 as Sampler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def runtime_service(token: str, instance: str = None):
    if token:
        return QiskitRuntimeService(
            channel="ibm_quantum", token=token, instance=instance
        )
    return QiskitRuntimeService(instance=instance)


def init_service(token: str, backend_name: str = None, instance: str = None):
    service = runtime_service(token, instance)
    if backend_name:
        backend = service.backend(backend_name)
        if not backend.status().operational:
            raise RuntimeError(f"Backend {backend_name} is not operational.")
        print(f"Using user-selected backend: {backend.name}")
    else:
        backend = service.least_busy(simulator=False, operational=True)
        print(f"Using least-busy backend: {backend.name}")
    return backend


state_preps = {
    "0":  lambda qc, index: None,
    "1":  lambda qc, index: qc.x(index),
    "+":  lambda qc, index: qc.h(index),
    "-":  lambda qc, index: (qc.x(index), qc.h(index)),   # H|1⟩ = |−⟩
    "+i": lambda qc, index: (qc.h(index), qc.s(index)),   # S H|0⟩ = |+i⟩
    "-i": lambda qc, index: (qc.h(index), qc.sdg(index)), # S† H|0⟩ = |−i⟩
}

pauli_strings = [
    "IX", "IY", "IZ",
    "XI", "XX", "XY", "XZ",
    "YI", "YX", "YY", "YZ",
    "ZI", "ZX", "ZY", "ZZ",
]
count_bitstrings = ("00", "01", "10", "11")

sigma_0 = np.eye(2, dtype=complex)
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

pauli_collection = [sigma_0, sigma_x, sigma_y, sigma_z]

pauli_stack = [
    np.kron(sigma_i, sigma_j)
    for sigma_i, sigma_j in product(pauli_collection, pauli_collection)
][1:]
pauli_stack = np.stack(pauli_stack, axis=0)  # (15, 4, 4)


def build_state_tomography_circuits(
    qubits: List[int],
    initial_state: List[str],
    num_points: int,
    gates_per_point: int,
    pm,
) -> List:
    basis_rots = {
        "I": lambda qc, index: None,
        "X": lambda qc, index: qc.h(index),
        "Y": lambda qc, index: (qc.sdg(index), qc.h(index)),
        "Z": lambda qc, index: None,
    }
    circuits = []
    for n, pauli_str in product(range(num_points), pauli_strings):
        qc = QuantumCircuit(len(qubits), 2)
        for index, state in enumerate(initial_state):
            state_preps[state](qc, index)

        for _ in range(n * gates_per_point):
            for index in range(2):
                qc.id(index)

        for index, pauli in enumerate(pauli_str):
            basis_rots[pauli](qc, index)
        qc.measure([0, 1], [0, 1])
        circuits.append(pm.run(qc))
    return circuits


def make_paths(
    backend: str,
    initial_state: List[str],
    qubits: List[int],
    num_points: int,
    gates_per_point: int,
    shots: int,
    run_timestamp: str,
    ext: str,
) -> Path:
    folder = (
        Path("results/two_qubit_tomography")
        / backend
        / run_timestamp
        / f"init{','.join(initial_state)}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    name = (
        f"q{qubits}-np{num_points}"
        f"-gpp{gates_per_point}-s{shots}{ext}"
    )
    return folder / name


def save_results(t_us, means, stds, raw_counts, out_npz: Path, metadata: dict):
    np.savez_compressed(
        out_npz,
        time_us=t_us,
        coherent_vec_mean=means,
        coherent_vec_std=stds,
        coherent_vec_std_kind="standard_error",
        raw_counts=raw_counts,
        count_bitstrings=np.asarray(count_bitstrings),
        pauli_strings=np.asarray(pauli_strings),
        **metadata,
    )
    print(f"Saved -> {out_npz}")


def save_qmi_plot(t_us, mutual_information, out_pdf: Path):
    plt.figure()
    plt.plot(t_us, mutual_information, color="#ff9999", marker="x", ls="-.")
    plt.ylim([0.0, 2.0])
    plt.xlabel("t $(\\mu s)$")
    plt.ylabel("$I(A:B)_{\\rho}$")
    plt.title("Evolution of quantum mutual information")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_pdf}")


def run_batches(sampler: Sampler, circuits: List, shots: int, max_batch: int):
    """Return jobs so --submit-only can finish before results are available."""
    jobs = []
    for i in range(0, len(circuits), max_batch):
        batch = circuits[i:i + max_batch]
        print(f"Submitting batch {i // max_batch + 1} ... ({len(batch)} circuits)")
        job = sampler.run(batch, shots=shots)
        print(f"Submitted job {job.job_id()}")
        jobs.append(job)
    return jobs


def result_counts(result):
    for register_name in ("c", "meas"):
        register = getattr(result.data, register_name, None)
        if register is not None:
            return register.get_counts()
    raise RuntimeError("Result does not contain a supported measurement register")


def counts_to_vectors(raw_counts):
    """Convert (T, 15, 4) raw counts to <P>/2 and its standard error."""
    raw_counts = np.asarray(raw_counts)
    if raw_counts.ndim != 3 or raw_counts.shape[1:] != (len(pauli_strings), 4):
        raise ValueError("Expected raw counts with shape (T, 15, 4)")
    totals = raw_counts.sum(axis=2)
    if np.any(raw_counts < 0) or np.any(totals <= 0):
        raise ValueError("Each Pauli setting must contain positive total counts")
    # Qiskit count strings are c1 c0; Pauli labels here are q0 q1.
    signs = np.array([
        [(-1) ** sum(p != "I" and b == "1" for p, b in zip(pauli, bits[::-1]))
         for bits in count_bitstrings]
        for pauli in pauli_strings
    ])
    means = np.sum(raw_counts * signs, axis=2) / totals
    stds = np.sqrt(np.maximum(0, 1 - means**2) / totals)
    return means / 2, stds / 2


def analyse_tomography(
    results, num_points: int, gates_per_point: int, id_duration_s: float,
):
    if num_points <= 0:
        raise ValueError("At least one time point is required")
    if len(results) != num_points * len(pauli_strings):
        raise ValueError("Expected 15 tomography results per time point")
    counts = [result_counts(result) for result in results]
    raw_counts = np.array([
        [counts_at_setting.get(bits, 0) for bits in count_bitstrings]
        for counts_at_setting in counts
    ], dtype=int).reshape(num_points, len(pauli_strings), 4)
    means, stds = counts_to_vectors(raw_counts)
    t_us = np.arange(num_points) * gates_per_point * id_duration_s * 1e6
    return t_us, means, stds, raw_counts


def reconstruct_density_matrix(coherent_vec_batch):
    rho = np.eye(4, dtype=complex) / 4
    rho = np.tile(rho[None, :, :], (len(coherent_vec_batch), 1, 1))  # (T, 4, 4)
    rho = rho + np.tensordot(coherent_vec_batch, pauli_stack, axes=[[1,], [0,]]) / 2
    return rho

def matrix_function(rho, fn, eps=1e-12):
    # rho: (B, d, d)
    eigvals, eigvecs = np.linalg.eigh(rho)  # batch‑eigh 
    eigvals = np.clip(eigvals, eps, None)   # avoid log(0)                    # natural or base‑2 log
    # diag‑embed
    D = np.zeros_like(rho)
    idx = np.arange(rho.shape[-1])
    D[..., idx, idx] = fn(eigvals)
    return eigvecs @ D @ eigvecs.conj().transpose(0, 2, 1)

def quantum_relative_entropy(rho1, rho2):
    log_rho1 = matrix_function(rho1, np.log2)
    log_rho2 = matrix_function(rho2, np.log2)

    relative_entropy = np.trace(rho1 @ (log_rho1 - log_rho2), axis1=-2, axis2=-1).real
    return relative_entropy
    
def von_neumann_entropy(rho):
    log_rho = matrix_function(rho, np.log2)
    entropy = -np.trace(rho @ log_rho, axis1=-2, axis2=-1).real
    return entropy
    
def quantum_mutual_information(density_matrix_batch):
    density_matrix_batch = matrix_function(density_matrix_batch, np.abs)
    density_matrix_batch = density_matrix_batch / np.trace(density_matrix_batch, axis1=-2, axis2=-1)[:, None, None]
    rhoA = np.trace(density_matrix_batch.reshape(-1, 2, 2, 2, 2), axis1=2, axis2=4)
    rhoB = np.trace(density_matrix_batch.reshape(-1, 2, 2, 2, 2), axis1=1, axis2=3)
    rhoA_tensor_rhoB = np.stack([np.kron(rho1, rho2) for rho1, rho2 in zip(rhoA, rhoB)], axis=0)
    mutual_information = quantum_relative_entropy(density_matrix_batch, rhoA_tensor_rhoB)
    
    return mutual_information

def main():
    parser = argparse.ArgumentParser(description="Two-qubit idle state tomography")
    common = {
        "backend": ("-b", "--backend", str, None),
        "qubit": ("-q", "--qubit", str, "0,1"),
        "num_points": ("-np", "--num-points", int, 50),
        "gates_per_point": ("-gpp", "--gates-per-point", int, 50),
        "shots": ("-s", "--shots", int, 8192),
    }
    for dest, (short, long, kind, default) in common.items():
        parser.add_argument(short, long, dest=dest, type=kind, default=default)
    parser.add_argument("--instance", type=str, default=None, help="IBM Quantum instance")
    parser.add_argument("--rep-delay", type=float, default=None, help="Delay between shots in seconds")
    parser.add_argument(
        "-init", "--initial-state", type=str, default="+",
        help="Semicolon-separated configurations, e.g. '+,+;-,-;+i,+i;-i,-i'",
    )
    parser.add_argument("-t", "--test", action="store_true", help="Run without saving tomography data or plots")
    retrieval = parser.add_mutually_exclusive_group()
    retrieval.add_argument("-id", "--job-id", type=str, help="Comma-separated job IDs to retrieve")
    retrieval.add_argument("-f", "--file-name", type=Path, help="Plot an existing tomography NPZ file")
    parser.add_argument("--run-timestamp", type=str, default=None, help="Timestamp used for result paths")
    parser.add_argument("--submit-only", action="store_true", help="Submit jobs and save their IDs without waiting")
    args = parser.parse_args()
    if args.submit_only and (args.job_id or args.file_name):
        parser.error("--submit-only is only valid for a new submission")

    if args.file_name:
        with np.load(args.file_name) as data:
            t_us = data["time_us"]
            means = data["coherent_vec_mean"]
        qmi = quantum_mutual_information(reconstruct_density_matrix(means))
        if not args.test:
            save_qmi_plot(t_us, qmi, args.file_name.with_suffix(".pdf"))
        return

    run_timestamp = args.run_timestamp or datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    try:
        qubits = [int(q) for q in args.qubit.replace(",", " ").split()]
    except ValueError:
        parser.error("Qubit indices must be integers")
    if len(qubits) != 2 or len(set(qubits)) != 2 or min(qubits) < 0:
        parser.error("Select exactly two distinct nonnegative qubit indices")
    if args.num_points <= 0 or args.gates_per_point <= 0 or args.shots <= 0:
        parser.error("Number of points, gates per point, and shots must be positive")

    initial_state_configs = []
    for config in args.initial_state.split(";"):
        initial_state = config.replace(",", " ").split()
        if not initial_state:
            continue
        if len(initial_state) == 1:
            initial_state *= len(qubits)
        if len(initial_state) != len(qubits):
            parser.error(
                f"Mismatch: {len(initial_state)} initial states for {len(qubits)} qubits"
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
    jobs_by_config = []
    if args.job_id:
        job_ids = args.job_id.replace(",", " ").split()
        if not job_ids:
            parser.error("At least one job ID is required")
        service = runtime_service(token, args.instance)
        jobs = [service.job(job_id) for job_id in job_ids]
        backend_name = jobs[0].backend().name
        if any(job.backend().name != backend_name for job in jobs):
            parser.error("All retrieved jobs must use the same backend")
        backend = service.backend(backend_name)
        backend_properties = backend.properties(datetime=jobs[0].creation_date)
        jobs_by_config.append((initial_state_configs[0], jobs))
    else:
        backend = init_service(token, args.backend, args.instance)
        backend_properties = backend.properties()

    backend_config = backend.configuration()
    calibration_timestamp = backend_properties.last_update_date.isoformat()
    t1_us = [backend_properties.qubit_property(q)["T1"][0] * 1e6 for q in qubits]
    t2_us = [backend_properties.qubit_property(q)["T2"][0] * 1e6 for q in qubits]
    id_duration_s = max(backend.target["id"][(q,)].duration for q in qubits)
    print(f"Run timestamp: {run_timestamp}")

    if not args.job_id:
        rep_delay_s = args.rep_delay if args.rep_delay is not None else backend_config.default_rep_delay
        if not backend_config.rep_delay_range[0] <= rep_delay_s <= backend_config.rep_delay_range[1]:
            parser.error(f"Repetition delay must be within {backend_config.rep_delay_range}")
        pass_manager = generate_preset_pass_manager(
            target=backend.target,
            initial_layout=qubits,
            optimization_level=0,
        )
        output_dir = Path("results/two_qubit_tomography") / backend.name / run_timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "submission.json"
        manifest = dict(
            backend=backend.name, instance=args.instance, run_timestamp=run_timestamp,
            qubits=qubits, num_points=args.num_points, gates_per_point=args.gates_per_point,
            shots=args.shots, rep_delay_s=rep_delay_s, init_qubits=True,
            calibration_timestamp=calibration_timestamp, states=[],
        )
        batch = Batch(backend)
        sampler = Sampler(
            mode=batch,
            options={"execution": {"init_qubits": True, "rep_delay": rep_delay_s}},
        )
        manifest["batch_id"] = batch.session_id
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        try:
            for initial_state in initial_state_configs:
                print(f"Preparing initial state: {initial_state}")
                circuits = build_state_tomography_circuits(
                    qubits, initial_state, args.num_points,
                    args.gates_per_point, pass_manager,
                )
                jobs = run_batches(
                    sampler, circuits, args.shots, backend_config.max_experiments,
                )
                jobs_by_config.append((initial_state, jobs))
                manifest["states"].append({
                    "initial_state": initial_state,
                    "job_ids": [job.job_id() for job in jobs],
                })
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        finally:
            batch.close()
        print(f"Submission manifest: {manifest_path}")
        if args.submit_only:
            return

    for initial_state, jobs in jobs_by_config:
        results = []
        for job in jobs:
            print(f"Fetching job '{job.job_id()}' from backend {backend.name}")
            results.extend(job.result())
        num_points = len(results) // len(pauli_strings) if args.job_id else args.num_points
        t_us, means, stds, raw_counts = analyse_tomography(
            results, num_points, args.gates_per_point, id_duration_s,
        )
        density_matrices = reconstruct_density_matrix(means)
        qmi = quantum_mutual_information(density_matrices)
        if not args.test:
            metadata = dict(
                backend=backend.name, qubits=qubits, initial_state=initial_state,
                num_points=num_points, gates_per_point=args.gates_per_point,
                shots=args.shots, optimization_level=0, run_timestamp=run_timestamp,
                job_ids=[job.job_id() for job in jobs],
                t1_us=t1_us, t2_us=t2_us, calibration_timestamp=calibration_timestamp,
            )
            if not args.job_id:
                metadata.update(rep_delay_s=rep_delay_s, init_qubits=True)
            npz_path = make_paths(
                backend.name, initial_state, qubits, num_points,
                args.gates_per_point, args.shots, run_timestamp, ".npz",
            )
            save_results(t_us, means, stds, raw_counts, npz_path, metadata)
            save_qmi_plot(t_us, qmi, npz_path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
