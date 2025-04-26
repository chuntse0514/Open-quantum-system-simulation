from __future__ import annotations

import os
import argparse
from typing import List, Dict

from qiskit_ibm_runtime import QiskitRuntimeService
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
ibm_queue_status.py  ──  Quick IBM Quantum backend load monitor

Prints a real-time table of operational IBM Quantum backends with their
pending-job queue length so you can pick the least-busy device.

Usage examples
--------------
$ python ibm_queue_status.py                  # default: top 10 least busy
$ python ibm_queue_status.py --all           # show *all* operational devices
$ python ibm_queue_status.py -n 5            # top-5 least busy
$ python ibm_queue_status.py --min-q 15      # only >=15-qubit backends
$ python ibm_queue_status.py --show-sims     # include simulators, too

Environment
-----------
IBM Quantum API token must be available in the environment variable
IBM_TOKEN.  Example:

    export IBM_TOKEN=#########
    python ibm_queue_status.py
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_backend_status(service: QiskitRuntimeService, include_sims: bool) -> List[Dict]:
    """Return list with name, n_qubits, pending_jobs for each backend."""
    backends = service.backends(simulator=include_sims, operational=True)
    data = []
    for backend in backends:
        status = backend.status()  # .pending_jobs, .operational, .status_msg
        data.append({
            "name": backend.name,
            "n_qubits": backend.num_qubits,
            "pending": status.pending_jobs,
            "sim": backend.simulator,
        })
    return data


def print_table(rows: List[Dict], top_n: int | None):
    """Pretty-print using tabulate if available, else fallback."""
    if top_n:
        rows = rows[:top_n]

    headers = ["backend", "qubits", "pending"]
    table = [[r["name"], r["n_qubits"], r["pending"]] for r in rows]

    try:
        from tabulate import tabulate
        print(tabulate(table, headers=headers, tablefmt="github"))
    except ImportError:
        # simple fallback
        print(f"{'backend':20} | {'qubits':6} | pending")
        print("-" * 40)
        for name, nq, pend in table:
            print(f"{name:20} | {nq:6} | {pend}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Show IBM Quantum backend queue lengths")
    ap.add_argument("-n", "--top", type=int, metavar="N", help="Show N least-busy backends")
    ap.add_argument("--all", action="store_true", help="Show all operational backends (overrides -n)")
    ap.add_argument("--min-q", type=int, default=0, help="Only include backends with >= this many qubits")
    ap.add_argument("--show-sims", action="store_true", help="Include simulator backends as well")
    args = ap.parse_args()

    token = os.getenv("IBM_TOKEN")
    if not token:
        ap.error("Environment variable IBM_TOKEN not set")

    service = QiskitRuntimeService(channel="ibm_quantum", token=token)

    rows = fetch_backend_status(service, include_sims=args.show_sims)

    # Filter & sort
    rows = [r for r in rows if r["n_qubits"] >= args.min_q]
    rows.sort(key=lambda r: r["pending"])  # ascending

    top_n = None if args.all else args.top if args.top else 10
    print_table(rows, top_n)


if __name__ == "__main__":
    main()