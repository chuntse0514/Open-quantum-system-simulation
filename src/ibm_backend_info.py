from __future__ import annotations

import os
import argparse
import statistics as st
import csv
from pathlib import Path
from typing import Dict, List

from qiskit_ibm_runtime import QiskitRuntimeService

try:
    from tabulate import tabulate
except ImportError:  # graceful fallback
    tabulate = None
    
    
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""ibm_backend_info.py ── Timing / coherence summary for IBM Quantum backends

Usage examples
--------------
# single backend (least busy by default)
python -m src.ibm_backend_info                    
python -m src.ibm_backend_info -b ibm_brisbane    

# view *all* operational real devices, sorted by queue length
python -m src.ibm_backend_info --all

Flags
-----
-b, --backend NAME      choose a specific device
-a, --all               list every operational real backend
--show-sims             include simulators in --all mode
--sort {pending,wait,t1} choose sort column when --all (default=pending)
--csv PATH              dump the table to CSV as well
IBM_TOKEN environment variable must be set.
"""


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_service() -> QiskitRuntimeService:
    token = os.getenv("IBM_TOKEN")
    if not token:
        raise RuntimeError("Environment variable IBM_TOKEN not set")
    return QiskitRuntimeService(channel="ibm_quantum", token=token)


def get_dt(backend):
    return backend.target.dt


def identity_duration(backend):
    try:
        return backend.target["id"][(0,)].duration
    except Exception:
        return get_dt(backend)


def coherence_stats(backend) -> Dict[str, tuple[float, float, float]]:
    """Return {"T1": (min, med, max), "T2": (...)} in µs."""
    t1s, t2s = [], []
    for q in range(backend.num_qubits):
        qp = backend.qubit_properties(q)
        if qp.t1: t1s.append(qp.t1)
        if qp.t2: t2s.append(qp.t2)

    def _stats(lst):
        return (min(lst), st.median(lst), max(lst)) if lst else (0, 0, 0)

    µ = 1e6
    return {
        "T1": tuple(x * µ for x in _stats(t1s)),
        "T2": tuple(x * µ for x in _stats(t2s)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# printing routines
# ──────────────────────────────────────────────────────────────────────────────

def fmt_table(rows: List[List], headers: List[str]):
    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        header_line = " | ".join(headers)
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print(" | ".join(map(str, row)))
            
def _fmt(triplet):
    return f"{triplet[0]:.0f}/{triplet[1]:.0f}/{triplet[2]:.0f}"


def print_single(backend):
    status = backend.status()
    wait   = getattr(status, "estimated_wait_time", None)
    dt     = get_dt(backend)
    id_dur = identity_duration(backend)
    coh    = coherence_stats(backend)          # now returns tuples

    print(f"=== Backend: {backend.name} ===")
    print(f"qubits          : {backend.num_qubits}")
    print(f"pending jobs    : {status.pending_jobs}")
    if wait is not None:
        print(f"est. wait (min) : {wait/60:.1f}")
    print("--- timing ---")
    print(f"dt              : {dt*1e9:.1f} ns")
    print(f"identity length : {id_dur*1e9:.1f} ns")
    print("--- coherence (µs) ---")
    print(f"T1 min/med/max  : {_fmt(coh['T1'])}")
    print(f"T2 min/med/max  : {_fmt(coh['T2'])}")


def print_many(rows: List[Dict], sort_key: str, csv_path: str | None):
    # default sort by pending unless user chose another numeric key
    rows.sort(key=lambda r: r[sort_key])

    table = [
        [
            r["name"],
            r["nq"],
            r["pending"],
            f"{r['dt']:.1f}",
            f"{r['id']:.1f}",
            _fmt(r["t1"]),
            _fmt(r["t2"]),
        ]
        for r in rows
    ]

    headers = [
        "backend",
        "qubits",
        "pending",
        "dt (ns)",
        "id (ns)",
        "T1 µs (min/med/max)",
        "T2 µs (min/med/max)",
    ]
    fmt_table(table, headers)

    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(table)
        print("CSV saved →", csv_path)


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Backend timing & coherence info")
    mx = ap.add_mutually_exclusive_group()
    mx.add_argument("-b", "--backend")
    mx.add_argument("-a", "--all", action="store_true", help="list every operational backend")
    ap.add_argument("--show-sims", action="store_true", help="include simulators when --all")
    ap.add_argument("--sort", choices=["pending", "wait", "t1"], default="pending")
    ap.add_argument("--csv", metavar="PATH", help="also dump table to CSV (only with --all)")
    args = ap.parse_args()

    service = get_service()

    if args.all:
        bks = service.backends(simulator=args.show_sims, operational=True)
        rows = []
        for bk in bks:
            status = bk.status()
            wait   = getattr(status, "estimated_wait_time", 0)
            coh    = coherence_stats(bk)
            rows.append({
                "name":    bk.name,
                "nq":      bk.num_qubits,
                "pending": status.pending_jobs,
                "dt":      get_dt(bk) * 1e9,           # ns
                "id":      identity_duration(bk) * 1e9,  # ns
                "t1":      coh["T1"],      # tuple (min, med, max)
                "t2":      coh["T2"],      # tuple (min, med, max)
            })

        print_many(rows, args.sort, args.csv)
        return

    # single backend
    if args.backend:
        backend = service.backend(args.backend)
    else:
        backend = service.least_busy(simulator=False, operational=True)
    print_single(backend)


if __name__ == "__main__":
    main()
