# Open Quantum System Simulation

## Overview
This repository provides a comprehensive suite of tools for simulating open quantum systems and probing non-Markovian qubit noise. It is designed to facilitate experiments and theoretical simulations involving quantum noise, state and process tomography, and the characterization of crosstalk using the Qiskit framework. 

The codebase includes both executable Python scripts for running simulations/experiments on IBM quantum backends and Jupyter notebooks for analyzing and visualizing the results (e.g., CP-divisibility, information backflow, and quantum mutual information).

## Features
* **Noise Probing & Crosstalk Analysis**: Scripts to investigate idle noise, ZZ crosstalk, and general noise characteristics in multi-qubit systems.
* **Quantum Tomography**: Implementations for two-qubit state tomography and process tomography (both with and without crosstalk considerations).
* **Lindblad Master Equation Solvers**: Tools to model quantum dynamics using the Lindblad formalism.
* **Non-Markovianity Metrics**: Evaluate and visualize signatures of non-Markovian dynamics, including CP-divisibility breakdown, information backflow, and trace distance evolution.
* **IBM Quantum Integration**: Utilities to check queue status and retrieve backend information seamlessly.

## Repository Structure

### `src/` - Core Source Code
Contains the main Python scripts for executing simulations and interacting with quantum backends.
* `lindblad.py`: Functions for simulating open quantum system dynamics via the Lindblad master equation.
* `pmme_kernel.py`: Implementation related to the Projection-based Master Equation (PMME) kernel.
* `single_qubit_tomography.py`: Population and Bloch-vector tomography with optional spectator qubits and multiple initial-state configurations per run.
* `process_tomography.py`: Single-qubit process tomography with optional spectator qubits.
* `two_qubit_tomography.py`: Two-qubit state tomography with raw counts, Pauli means, and standard errors.
* `zz_crosstalk.py`: Specific routines for measuring and analyzing ZZ crosstalk between qubits.
* `ibm_backend_info.py` / `ibm_queue_status.py`: Utilities for interacting with IBM Quantum services.

### `notebook/` - Analysis and Visualization
Jupyter notebooks for processing data and generating plots.
* `CP_divisibility_visualize.ipynb`: Analyzes and visualizes Complete Positivity (CP) divisibility.
* `Info_backflow_visualize.ipynb`: Quantifies and plots information backflow as a measure of non-Markovianity.
* `PMME_kernel_visualize.ipynb`: Visualization tools for the PMME kernel.
* `Quantum_mutual_info_visualize.ipynb`: Calculates and plots quantum mutual information, with notebook-local two-qubit ZZ model fitting helpers.
* `ZZ_crosstalk.ipynb`: Interactive analysis of ZZ crosstalk experimental data.
* `Laplace_transform.ipynb`: Analytical or numerical evaluations involving Laplace transforms of the system dynamics.
* `result_visualize.ipynb`: General visualization utilities for simulation and tomography results.

### `images/` - Figures and Plots
Generated visualizations highlighting key physical quantities, such as trace distance, quantum relative entropy, and CP-divisibility parameters.

## Result Layout

Run timestamps are directory names rather than filename suffixes. Idle and
spectator experiments share the same roots because the qubit list in each
filename identifies whether spectators were used:

```text
results/single_qubit_tomography/<backend>/<timestamp>/init<state>/<artifact>
results/two_qubit_tomography/<backend>/<timestamp>/init<state>/<artifact>
results/process_tomography/<backend>/<timestamp>/[init<spectator-state>/]<artifact>
```

Separate initial-state configurations submitted in one command share a timestamp:

```bash
python -m src.single_qubit_tomography bloch -b ibm_brussels -q 49,48,50,55 -init "+,+,+,+;-,+,+,+;+i,+,+,+;-i,+,+,+"
```

Two-qubit tomography uses the same qubit and preparation syntax:

```bash
python -m src.state_tomography_two_qubit -b ibm_brussels -q 49,50 -init "+,+;-,-;+i,+i;-i,-i" -s 8192
```

Exactly two distinct qubits are required. A single preparation label, such as
`-init +`, prepares both qubits in that state. Use `--submit-only` to save job IDs
in `submission.json` and return immediately; retrieve one preparation's jobs with
`-id "job_id_1,job_id_2"` and the original qubit, preparation, shot, and gate settings.

NPZ files store the 15 Pauli coefficients as `coherent_vec_mean = <P>/2`.
New files store their standard errors in `coherent_vec_std`, marked by
`coherent_vec_std_kind="standard_error"`; older files contain outcome standard
deviations instead. Pauli labels follow the supplied qubit order (`XI` acts on
the first qubit). Analysis uses measured counts without readout-error correction:

```bash
python -m src.state_tomography_two_qubit -f "results/path/to/tomography.npz"
```

The file-only plotting command uses the saved time grid and needs no IBM login.
