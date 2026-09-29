import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Batch, SamplerV2 as Sampler
from qiskit_experiments.library import ProcessTomography
from qiskit.quantum_info import Choi, SuperOp, average_gate_fidelity, Operator

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


"""
Single-qubit process tomography with optional spectator qubits.

Examples
--------
python -m src.process_tomography -b ibm_strasbourg -q 24
python -m src.process_tomography -b ibm_strasbourg -q 4,3,5,15 -init +
"""


def init_service(token: str, backend_name: str):
    service = (
        QiskitRuntimeService(channel="ibm_cloud", token=token)
        if token else QiskitRuntimeService()
    )
    backend = service.backend(backend_name)
    if not backend.status().operational:
        raise RuntimeError(f"Backend {backend_name} is not operational.")
    print(f"Using backend: {backend.name}")
    return backend


state_preps = {
    "0":  lambda qc, index: None,
    "1":  lambda qc, index: qc.x(index),
    "+":  lambda qc, index: qc.h(index),
    "-":  lambda qc, index: (qc.x(index), qc.h(index)),
    "+i": lambda qc, index: (qc.h(index), qc.s(index)),
    "-i": lambda qc, index: (qc.h(index), qc.sdg(index)),
}


def make_paths(
    backend: str,
    qubits: List[int],
    initial_state: str,
    num_points: int,
    gates_per_point: int,
    shots: int,
    run_timestamp: str,
    ext: str,
) -> Path:
    folder = Path("results/process_tomography") / backend / run_timestamp
    if len(qubits) > 1:
        folder /= f"init{initial_state}"
    folder.mkdir(parents=True, exist_ok=True)
    qubit_label = qubits[0] if len(qubits) == 1 else qubits
    name = (
        f"q{qubit_label}-np{num_points}"
        f"-gpp{gates_per_point}-s{shots}{ext}"
    )
    return folder / name


def build_idle_circuits(
    qubits: List[int],
    initial_state: str,
    num_points: int,
    gates_per_point: int,
) -> List:
    circuits = []
    for n in range(num_points):
        qc = QuantumCircuit(len(qubits))
        for index in range(1, len(qubits)):
            state_preps[initial_state](qc, index)

        for _ in range(n * gates_per_point):
            qc.id(0)

        circuits.append(qc)
    return circuits


def run_process_tomography(
    backend,
    sampler,
    num_points: int,
    gates_per_point: int,
    shots: int,
    qubits: List[int],
    initial_state: str,
):
    idle_circuits = build_idle_circuits(
        qubits, initial_state, num_points, gates_per_point
    )
    choi_matrices = []
    channel_fidelities = []
    raw_counts = []
    job_ids = []
    experiment_ids = []
    basis_indices = None
    count_bitstrings = None
    tomography_clbits = None
    sampler.options.default_shots = shots

    experiment_data = []
    for i, idle_circuit in enumerate(idle_circuits):
        print(f"Queuing job {i + 1}/{num_points}")
        tomography = ProcessTomography(
            idle_circuit,
            backend=backend,
            physical_qubits=qubits,
            measurement_indices=[0],
            preparation_indices=[0],
        )
        experiment_data.append(tomography.run(sampler=sampler))

    for data in experiment_data:
        choi = data.analysis_results("state", dataframe=True).iloc[0].value.data
        records = sorted(
            data.data(),
            key=lambda record: (
                record["metadata"]["p_idx"][0],
                record["metadata"]["m_idx"][0],
            ),
        )
        indices = np.array([
            [record["metadata"]["p_idx"][0], record["metadata"]["m_idx"][0]]
            for record in records
        ], dtype=int)
        if basis_indices is None:
            basis_indices = indices
        elif not np.array_equal(indices, basis_indices):
            raise ValueError("Tomography settings differ between time points")
        clbits = np.array([record["metadata"]["clbits"] for record in records], dtype=int)
        if tomography_clbits is None:
            tomography_clbits = clbits
        elif not np.array_equal(clbits, tomography_clbits):
            raise ValueError("Tomography measurement bits differ between time points")
        if count_bitstrings is None:
            # Keep joint outcomes if the input circuit also measured other bits.
            width = len(next(iter(records[0]["counts"])))
            count_bitstrings = [format(i, f"0{width}b") for i in range(2**width)]
        counts = np.array([
            [record["counts"].get(bits, 0) for bits in count_bitstrings]
            for record in records
        ], dtype=np.int64)
        if not np.all(counts.sum(axis=1) == shots):
            raise ValueError("Unexpected shot count in tomography data")
        if len(data.job_ids) != 1:
            raise ValueError("Expected one tomography job per time point")
        fidelity = average_gate_fidelity(
            SuperOp(Choi(choi)), Operator(np.eye(2))
        )
        choi_matrices.append(choi)
        channel_fidelities.append(fidelity)
        raw_counts.append(counts)
        job_ids.append(data.job_ids[0])
        experiment_ids.append(data.experiment_id)

    return (choi_matrices, channel_fidelities, raw_counts, basis_indices,
            count_bitstrings, tomography_clbits, job_ids, experiment_ids)


def save_results(
    time_steps_us: np.ndarray,
    choi_matrices: List[np.ndarray],
    channel_fidelities: List[float],
    raw_counts: List[np.ndarray],
    basis_indices: np.ndarray,
    count_bitstrings: List[str],
    tomography_clbits: np.ndarray,
    job_ids: List[str],
    experiment_ids: List[str],
    args,
):
    path = make_paths(
        args.backend,
        args.qubit,
        args.initial_state,
        args.num_points,
        args.gates_per_point,
        args.shots,
        args.run_timestamp,
        ".npz",
    )
    # raw_counts[t, setting, outcome] includes all measured bits, not just c_tomo.
    np.savez_compressed(
        path,
        time_us=time_steps_us,
        fidelities=channel_fidelities,
        choi=np.stack(choi_matrices, axis=0),
        raw_counts=np.stack(raw_counts, axis=0),
        basis_indices=basis_indices,
        count_bitstrings=np.asarray(count_bitstrings),
        tomography_clbits=tomography_clbits,
        job_ids=np.asarray(job_ids),
        experiment_ids=np.asarray(experiment_ids),
    )
    print(f"All data -> {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Single-qubit process tomography with optional spectators"
    )
    parser.add_argument("-b", "--backend", type=str, default=None, help="IBMQ backend name")
    parser.add_argument("-q", "--qubit", type=str, default="0", help="Target qubit followed by optional spectators")
    parser.add_argument("-np", "--num-points", type=int, default=50, help="Number of time points")
    parser.add_argument("-gpp", "--gates-per-point", type=int, default=50, help="Identity gates between time points")
    parser.add_argument("-s", "--shots", type=int, default=8192, help="Shots per circuit")
    parser.add_argument("-init", "--initial-state", choices=state_preps, default="+", help="Spectator initial state")
    parser.add_argument("-t", "--test", action="store_true")
    args = parser.parse_args()
    args.run_timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    args.qubit = [int(q) for q in args.qubit.replace(",", " ").split()]

    backend = init_service(os.getenv("IBM_TOKEN"), args.backend)
    args.backend = backend.name
    batch = Batch(backend)
    sampler = Sampler(mode=batch)

    id_duration_s = backend.target["id"][(args.qubit[0],)].duration
    time_steps_us = (
        np.arange(args.num_points)
        * args.gates_per_point
        * id_duration_s
        * 1e6
    )

    print(
        f"Running process tomography: qubit={args.qubit}, "
        f"number of points={args.num_points}, "
        f"gates per point={args.gates_per_point}, shots={args.shots}"
    )
    (choi_matrices, channel_fidelities, raw_counts, basis_indices,
     count_bitstrings, tomography_clbits, job_ids, experiment_ids) = run_process_tomography(
        backend,
        sampler,
        args.num_points,
        args.gates_per_point,
        args.shots,
        args.qubit,
        args.initial_state,
    )
    batch.close()

    if not args.test:
        save_results(
            time_steps_us, choi_matrices, channel_fidelities,
            raw_counts, basis_indices, count_bitstrings, tomography_clbits,
            job_ids, experiment_ids, args,
        )


if __name__ == "__main__":
    main()
