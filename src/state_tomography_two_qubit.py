import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Batch, SamplerV2 as Sampler
from qiskit.transpiler import generate_preset_pass_manager
from typing import List
from itertools import product
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def init_service(token: str, backend_name: str):
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    backend = service.backend(backend_name)
    if not backend.status().operational:
        raise RuntimeError(f"Backend {backend_name} is not operational.")
    print(f"Using backend: {backend.name}")
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

pauli_strings = [
    "IX", "IY", "IZ",
    "XI", "XX", "XY", "XZ", 
    "YI", "YX", "YY", "YZ",
    "ZI", "ZX", "ZY", "ZZ",
]

sigma_0 = np.eye(2, dtype=complex)
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

pauli_collection = [sigma_0, sigma_x, sigma_y, sigma_z]

pauli_stack = [
    np.kron(sigma_i, sigma_j) for sigma_i, sigma_j in product(pauli_collection, pauli_collection)
][1:]
pauli_stack = np.stack(pauli_stack, axis=0) # (15, 4, 4)


def build_state_tomography_circuits(
        qubit: List[int], 
        initial_state: List[str], 
        num_points: int, 
        gates_per_point: int, 
        pm
) -> List:
    circuits = []
    for n, pauli_str in product(range(num_points), pauli_strings):
        qc = QuantumCircuit(len(qubit))
        
        # state preparation
        for index, state in enumerate(initial_state):
            state_preps[state](qc, index)
        
        # idle qubits
        for _ in range(n * gates_per_point):
            for index in range(len(qubit)):
                qc.id(index)
                
        for index, pauli in enumerate(pauli_str):
            if pauli == "X":
                qc.h(index)
            elif pauli == "Y":
                qc.sdg(index)
                qc.h(index)
                                
        qc.measure_all()
        circuits.append(pm.run(qc))
    return circuits 
    
    
def make_paths(
    backend: str,
    qubit: int,
    init_state: str,
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
    folder = Path("results/state_tomography/two_qubit") / backend / f"init{",".join(init_state)}"
    folder.mkdir(parents=True, exist_ok=True)
    name   = f"q{qubit}-np{num_points}-gpp{gpp}-s{shots}-{stamp}{ext}"
    return folder / name    

def retrieve_from_job_id(job_id_list: List[str]):
    token = os.getenv("IBM_TOKEN")
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    results = []
    for job_id in job_id_list:
        job = service.job(job_id)
        results.extend(job.result())
        backend_name = job.backend().name
        print(f"Fetching remote job '{job.job_id()}' from backend {backend_name} (status={job.status()})")
    return results
    
def retrieve_from_file_name(file: str):
    
    print(f"Loading data from {file} ...")
    
    with np.load(file) as data:
        # List the stored arrays
        print("Keys  :", data.files)      # -> ['time_us', 'coherent_vec_mean', 'coherent_vec_std']
        # shape (T,)
        coherent_vec_mean  = data["coherent_vec_mean"]  # shape (T, 15)
        coherent_vec_std   = data["coherent_vec_std"]   # shape (T, 15)
        
    return coherent_vec_mean, coherent_vec_std
    
def save_results(time_steps_us: np.ndarray, coherent_vec_mean: List[np.ndarray], coherent_vec_std: List[float], args):
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

    # Stack the mean and std of coherent vectors into an (T,15) array
    mean_stack = np.stack(coherent_vec_mean, axis=0)
    std_stack = np.stack(coherent_vec_std, axis=0)

    # Save everything in one .npz
    npz_path = folder / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        time_us=time_steps_us,
        coherent_vec_mean=mean_stack,
        coherent_vec_std=std_stack
    )
    print(f"All data → {npz_path}")

def save_png(time_steps_us, mutual_information, args):
    
    png_path = make_paths(
        args.backend,
        args.qubit,
        args.initial_state,
        args.num_points,
        args.gates_per_point,
        shots=args.shots,
        ext=".png"
    )
    
    plt.plot(time_steps_us, mutual_information, color="#ff9999", marker="x", ls="-.")
    plt.ylim([0.0, 2.0])
    plt.xlabel("t $(\\mu s)$")
    plt.ylabel("$I(A:B)_{\\rho}$")
    plt.title("evolution of mutual information over time")
    plt.savefig(png_path)

def run_state_tomography(
    backend, sampler: Sampler, num_points: int, gates_per_point: int, 
    shots: int, qubit: List[int], initial_state: List[str], pm
):
    
    idle_circuits = build_state_tomography_circuits(
        qubit, initial_state, num_points, gates_per_point, pm
    )
    max_batch = backend.configuration().max_experiments
    results = []
    for i in range(0, len(idle_circuits), max_batch):
        batch = idle_circuits[i : i + max_batch]
        print(f"Queuing batch {i // max_batch + 1} ... ({len(batch)} circuits)")
        job = sampler.run(batch, shots=shots)
        results.extend(job.result())
    return results


def analyse_result(results, shots: int, num_points: int):
    
    def pauli_sign(pstr: str, bitstr: str) -> int:
        """
        Eigen‑value (±1) of a two‑qubit Pauli string `pstr` when computational
        basis result `bitstr` ('00', '01', …) is obtained *after* the usual
        basis‑change rotations.
        I ⟶ always +1
        X,Y,Z ⟶ +1 if the qubit outcome is '0', −1 if '1'
        """
        s = 1
        for p, b in zip(pstr, bitstr[::-1]):
            if p != "I" and b == "1":
                s *= -1
        return s
    
    """Return mean and std arrays, shape (T, 15)."""
    means_all, stds_all = [], []

    for t in range(num_points):
        means, stds = [], []

        for j, P in enumerate(pauli_strings):
            cnts = results[t*15 + j].data.meas.get_counts()

            # convert once to arrays ------------------------------
            bitstrs = np.array(list(cnts.keys()))
            counts  = np.array(list(cnts.values()), dtype=float)
            probs   = counts / shots

            # eigen‑values and statistics -------------------------
            signs = np.array([pauli_sign(P, b) for b in bitstrs], dtype=float)
            exp   = np.sum(probs * signs)
            var   = np.sum(probs * (signs - exp)**2)
            std   = np.sqrt(var)

            means.append(exp)
            stds.append(std)

        # store one time‑slice -----------------------------------
        means_all.append(np.asarray(means) / 2)   # divide by 2 → v_k = ⟨σ⟩/2
        stds_all .append(np.asarray(stds)  / 2)

    return np.asarray(means_all), np.asarray(stds_all)



def reconstruct_density_matrix(coherent_vec_batch):
    rho = np.eye(4, dtype=complex) / 4
    rho = np.tile(rho[None, :, :], (len(coherent_vec_batch), 1, 1))  # (T, 4, 4)
    rho = rho + np.tensordot(coherent_vec_batch, pauli_stack, axes=[[1,], [0,]]) / 2
    return rho

def matrix_function(rho, fn, eps=1e-12):
    # rho: (B, d, d)
    eigvals, eigvecs = np.linalg.eigh(rho)  # batch‑eigh 
    print(eigvals)
    eigvals = np.clip(eigvals, eps, None)   # avoid log(0)                    # natural or base‑2 log
    # diag‑embed
    D = np.zeros_like(rho)
    idx = np.arange(rho.shape[-1])
    D[..., idx, idx] = fn(eigvals)
    return eigvecs @ D @ eigvecs.conj().transpose(0, 2, 1)

def quantum_relative_entropy(rho1, rho2):
    log_rho1 = matrix_function(rho1, np.log2)
    log_rho2 = matrix_function(rho2, np.log2)

    relative_entropy = np.abs(np.trace(rho1 @ (log_rho1 - log_rho2), axis1=-2, axis2=-1))
    return relative_entropy
    
def quantum_mutual_informaion(density_matrix_batch):
    density_matrix_batch = matrix_function(density_matrix_batch, np.abs)
    density_matrix_batch = density_matrix_batch / np.trace(density_matrix_batch, axis1=-2, axis2=-1)
    rhoA = np.trace(density_matrix_batch.reshape(-1, 2, 2, 2, 2), axis1=1, axis2=3)
    rhoB = np.trace(density_matrix_batch.reshape(-1, 2, 2, 2, 2), axis1=2, axis2=4)
    rhoA_tensor_rhoB = np.stack([np.kron(rho1, rho2) for rho1, rho2 in zip(rhoA, rhoB)], axis=0)
    mutual_information = quantum_relative_entropy(density_matrix_batch, rhoA_tensor_rhoB)
    
    return mutual_information

def main():
    parser = argparse.ArgumentParser(description="Idle-noise process tomography")
    parser.add_argument("-b",    "--backend",        type=str, default=None, help="IBMQ backend name")
    parser.add_argument("-q",    "--qubit",          type=str, default="0", help="Physical qubit index")
    parser.add_argument("-np",   "--num-points",     type=int, default=50, help="Number of time points")
    parser.add_argument("-gpp",  "--gates-per-point", type=int, default=50, help="Number of identity gates between two time points")
    parser.add_argument("-s",    "--shots",          type=int, default=8192, help="Shots per circuit")
    parser.add_argument("-init", "--initial-state",  type=str, default="+", help="Initial states of spectator qubits")
    parser.add_argument("-t",    "--test",           action='store_true')
    parser.add_argument("-id",   "--job-id",         type=str, default=None, help="Job-ID for job retreival")
    parser.add_argument("-f",    "--file-name",      type=str, default=None, help="File name for plotting the results")
    args = parser.parse_args()
    if isinstance(args.qubit, str):
        args.qubit = [int(q) for q in args.qubit.replace(",", " ").split()]
    if isinstance(args.initial_state, str):
        args.initial_state = args.initial_state.replace(",", " ").split()
    if isinstance(args.job_id, str):
        if "," in args.job_id:
            args.job_id = args.job_id.replace(",", " ").split()
        else:
            args.job_id = [args.job_id]
        
    token = os.getenv("IBM_TOKEN")
    if not token:
        raise RuntimeError("Please set the IBM_TOKEN environment variable.")
    
    backend = init_service(token, args.backend)
    pm = generate_preset_pass_manager(
        target=backend.target, 
        initial_layout=args.qubit,
        optimization_level=0
    )
    # create one Batch for all of our jobs
    batch   = Batch(backend)
    # sampler now queues into that batch
    sampler = Sampler(mode=batch)
    
    id_duration_us = backend.target["id"][(0,)].duration
    time_steps_us = np.arange(0, args.num_points) * args.gates_per_point * id_duration_us * 1e6

    if not args.file_name and not args.job_id:
        print(f"Running two qubit state tomography: qubit={args.qubit}, number of points={args.num_points}, gates per point={args.gates_per_point}, shots={args.shots}")
        results = run_state_tomography(
            backend, sampler, args.num_points, args.gates_per_point, 
            args.shots, args.qubit, args.initial_state, pm
        )
    elif args.job_id:
        results = retrieve_from_job_id(id=args.job_id)
    elif args.file_name:
        coherent_vec_mean, coherent_vec_std = retrieve_from_file_name(file=args.file_name)
    
    if not args.file_name:
        coherent_vec_mean, coherent_vec_std = analyse_result(
            results, args.shots, args.num_points,
        )
        
    if not args.test and not args.file_name:
        save_results(time_steps_us, coherent_vec_mean, coherent_vec_std, args)
    
    density_matrix_batch = reconstruct_density_matrix(coherent_vec_mean)
    mutual_information = quantum_mutual_informaion(density_matrix_batch)
    
    if not args.test:
        save_png(time_steps_us, mutual_information, args)

if __name__ == "__main__":
    main()