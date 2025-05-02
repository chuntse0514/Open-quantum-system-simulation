import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Batch, SamplerV2 as Sampler
from qiskit_experiments.library import ProcessTomography
from qiskit.quantum_info import Choi, SuperOp, average_gate_fidelity, Operator
from typing import List

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
noise_process_tomo.py

Perform single-qubit idle-noise process tomography on an IBM Quantum backend.

Usage:
  python -m src.noise_process_tomo -b=ibm_strasbourg -q=24 -np=50 -gpp=50 -s=8192

Options:
  -b, --backend         IBMQ backend name (e.g. ibm_osaka) [required]
  -q, --qubit           Physical qubit index (default: 0)
  -np, --num-points     Number of time points (default: 50)
  -gpp, --gates-per-point Number of identity gates per point (default: 50)
  -s, --shots           Shots per circuit (default: 8192)

This script builds idle circuits of increasing identity gates, runs process tomography,
extracts the Choi matrices and average gate fidelities, and saves the results in a
compressed NumPy file for further analysis.
"""


def init_service(token: str, backend_name: str):
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    backend = service.backend(backend_name)
    if not backend.status().operational:
        raise RuntimeError(f"Backend {backend_name} is not operational.")
    print(f"Using backend: {backend.name}")
    return backend

def make_paths(
    backend: str,
    qubit: int,
    initial_state: str,
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
    folder = Path("results/process_tomography/crosstalk") / backend / initial_state
    folder.mkdir(parents=True, exist_ok=True)
    name   = f"q{qubit}-np{num_points}-gpp{gpp}-s{shots}-{stamp}{ext}"
    return folder / name


state_preps = {
    "0":  lambda qc, index: None,
    "1":  lambda qc, index: qc.x(index),
    "+":  lambda qc, index: qc.h(index),
    "-":  lambda qc, index: (qc.x(index), qc.h(index)),   # H|1⟩ = |−⟩
    "+i": lambda qc, index: (qc.h(index), qc.s(index)),   # S H|0⟩ = |+i⟩
    "-i": lambda qc, index: (qc.h(index), qc.sdg(index)), # S† H|0⟩ = |−i⟩
}

def build_idle_circuits(
    qubit: List[int],
    initial_state: str,
    num_points: int,
    gates_per_point: int,
) -> List:

    circuits = []
    for n in range(num_points):
        qc = QuantumCircuit(len(qubit))
        
        for index in range(1, len(qubit)):
            state_preps[initial_state](qc, index)
        
        # idle for n × gates_per_point cycles
        for _ in range(n * gates_per_point):
            qc.id(0)

        circuits.append(qc)
    return circuits


def run_process_tomography(
    backend, sampler, num_points: int, gates_per_point: int, shots: int, qubit: int = 0, initial_state: str="+"
):  
    idle_circuits = build_idle_circuits(qubit, initial_state, num_points, gates_per_point)
    choi_matrices = []
    channel_fidelities = []
    sampler.options.default_shots = shots
    
    # 1) queue up all experiments
    exp_data_list = []
    for i, idle_circuit in enumerate(idle_circuits):
        print(f"queuing job {i+1}/{num_points}")
        qpt = ProcessTomography(idle_circuit, backend=backend, physical_qubits=qubit, 
                                measurement_indices=[0], preparation_indices=[0])
        exp_data_list.append(qpt.run(sampler=sampler))   # no block_for_results()

    # 2) now extract your Choi matrices & fidelities
    for exp_data in exp_data_list:
        choi_obj = exp_data.analysis_results("state", dataframe=True).iloc[0].value
        choi_mat = choi_obj.data
        fid      = average_gate_fidelity(SuperOp(Choi(choi_mat)), Operator(np.eye(2)))
        choi_matrices.append(choi_mat)
        channel_fidelities.append(fid)

    return choi_matrices, channel_fidelities


def save_results(time_steps_us: np.ndarray, choi_matrices: List[np.ndarray], channel_fidelities: List[float], args):
    # Determine output path
    # Build the base path (no extension here)
    base = make_paths(
        args.backend,
        args.qubit,
        args.initial_state,
        args.num_points,
        args.gates_per_point,
        shots=args.shots,
        ext=""
    )
    folder = base.parent
    stem   = base.name

    # Stack the Choi matrices into an (N,4,4) array
    choi_stack = np.stack(choi_matrices, axis=0)

    # Save everything in one .npz
    npz_path = folder / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        time_us=time_steps_us,
        fidelities=channel_fidelities,
        choi=choi_stack
    )
    print(f"All data → {npz_path}")
    


def main():
    parser = argparse.ArgumentParser(description="Idle-noise process tomography")
    parser.add_argument("-b",    "--backend",        type=str, default=None, help="IBMQ backend name")
    parser.add_argument("-q",    "--qubit",          type=str, default="0", help="Physical qubit index")
    parser.add_argument("-np",   "--num-points",     type=int, default=50, help="Number of time points")
    parser.add_argument("-gpp",  "--gates-per-point", type=int, default=50, help="Number of identity gates between two time points")
    parser.add_argument("-s",    "--shots",          type=int, default=8192, help="Shots per circuit")
    parser.add_argument("-init", "--initial-state",  type=str, default="+", help="Initial states of spectator qubits")
    parser.add_argument("-t",    "--test",           action='store_true')
    args = parser.parse_args()
    if isinstance(args.qubit, str):
        args.qubit = [int(q) for q in args.qubit.replace(",", " ").split()]

    token = os.getenv("IBM_TOKEN")
    if not token:
        raise RuntimeError("Please set the IBM_TOKEN environment variable.")

    backend = init_service(token, args.backend)
    # create one Batch for all of our jobs
    batch   = Batch(backend)
    # sampler now queues into that batch
    sampler = Sampler(mode=batch)
    
    id_duration_us = backend.target["id"][(0,)].duration
    time_steps_us = np.arange(0, args.num_points) * args.gates_per_point * id_duration_us * 1e6

    print(f"Running process tomography: qubit={args.qubit}, number of points={args.num_points}, gates per point={args.gates_per_point}, shots={args.shots}")
    choi_matrices, channel_fidelities = run_process_tomography(
        backend, sampler, args.num_points, args.gates_per_point, args.shots, args.qubit, args.initial_state
    )
    if not args.test:
        save_results(time_steps_us, choi_matrices, channel_fidelities, args)


if __name__ == "__main__":
    main()
