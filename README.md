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
* `noise_probe_idle.py` / `noise_probe_crosstalk.py`: Scripts to probe and characterize noise under idle and crosstalk conditions.
* `process_tomography_idle.py` / `process_tomography_crosstalk.py`: Process tomography routines for characterizing quantum channels.
* `state_tomography_two_qubit.py`: Two-qubit state tomography implementation.
* `zz_crosstalk.py`: Specific routines for measuring and analyzing ZZ crosstalk between qubits.
* `ibm_backend_info.py` / `ibm_queue_status.py`: Utilities for interacting with IBM Quantum services.
* `utils.py`: Helper functions used across the repository.

### `notebook/` - Analysis and Visualization
Jupyter notebooks for processing data and generating plots.
* `CP_divisibility_visualize.ipynb`: Analyzes and visualizes Complete Positivity (CP) divisibility.
* `Info_backflow_visualize.ipynb`: Quantifies and plots information backflow as a measure of non-Markovianity.
* `PMME_kernel_visualize.ipynb`: Visualization tools for the PMME kernel.
* `Quantum_mutual_info_visualize.ipynb`: Calculates and plots the evolution of quantum mutual information.
* `ZZ_crosstalk.ipynb`: Interactive analysis of ZZ crosstalk experimental data.
* `Laplace_transform.ipynb`: Analytical or numerical evaluations involving Laplace transforms of the system dynamics.
* `result_visualize.ipynb`: General visualization utilities for simulation and tomography results.

### `images/` - Figures and Plots
Generated visualizations highlighting key physical quantities, such as trace distance, quantum relative entropy, and CP-divisibility parameters.