#!/usr/bin/env python
"""Run ncpoleon benchmarks locally with one fresh process per case.

Running several benchmarks in a single process is not safe for comparison: heap and allocator
state left behind by earlier cases changes the cost of later ones. Measured on max-cut level 2
with 20 variables, the median moves from 220ms (fresh process) to 410ms (after the smaller cases
in the same process). That 75% swing dwarfs anything worth optimising, so every case here gets
its own interpreter.

Usage:
    scripts/benchmark_local.py --list                # show the available cases
    scripts/benchmark_local.py                       # run every case
    scripts/benchmark_local.py max_cut-L2-n20        # run selected cases
    scripts/benchmark_local.py --baseline            # measure the noise floor

Report `min` alongside `median`: for CPU-bound deterministic work the measurement noise is
additive and one-sided, which makes the minimum the least-biased estimator and the mean useless.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CASES = [
    *(f"max_cut-L{level}-n{n}" for level in (1, 2) for n in (5, 10, 15, 20, 25)),
    "i3322-L4",
    "chsh-L11",
]

# The measured additive noise floor is around 0.8ms, so a case needs to last a good deal longer
# than that before a percentage change means anything. 20ms puts the noise at roughly 4%.
RESOLUTION_FLOOR = 0.020


def _build_case(name):
    """Return a zero-argument callable running the benchmark body for ``name``."""
    sys.path.insert(0, str(REPO_ROOT / "python"))
    from ncpoleon import get_relaxation

    if name.startswith("max_cut-"):
        _, level, n_vars = name.split("-")
        from tests.test_benchmark_max_cut import _max_cut_params

        level = int(level.removeprefix("L"))
        variables, objective, substitutions = _max_cut_params(int(n_vars.removeprefix("n")))
        return lambda: get_relaxation(variables, level, objective=objective, substitutions=substitutions)

    if name.startswith("chsh-"):
        from tests.test_chsh_uniform import _chsh_variables

        level = int(name.removeprefix("chsh-L"))
        # The uniformity moment constraints are deliberately dropped: this case measures the
        # plain CHSH relaxation.
        alice, bob, substitutions, objective, _ = _chsh_variables()
        return lambda: get_relaxation(alice + bob, level, objective, substitutions=substitutions)

    if name.startswith("i3322-"):
        from tests.test_i3322 import _i3322_params

        level = int(name.removeprefix("i3322-L"))
        variables, objective, substitutions = _i3322_params()
        return lambda: get_relaxation(variables, level, objective, substitutions=substitutions)

    raise SystemExit(f"unknown case: {name}")


def _measure(name, rounds, warmup):
    """Time a single case in the current process and return its statistics."""
    # The relaxation emits a warning per substituted variable, which would otherwise dominate
    # the measurement.
    logging.disable(logging.WARNING)
    run = _build_case(name)

    for _ in range(warmup):
        run()

    timings = []
    for _ in range(rounds):
        start = time.perf_counter()
        run()
        timings.append(time.perf_counter() - start)

    return {
        "case": name,
        "median": statistics.median(timings),
        "min": min(timings),
        "max": max(timings),
        "stdev": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "rounds": rounds,
    }


def _run_isolated(name, rounds, warmup, pin):
    """Measure a case in a fresh interpreter so it cannot inherit another case's heap state."""
    command = [sys.executable, str(Path(__file__).resolve()), "--measure", name, "--rounds", str(rounds)]
    command += ["--warmup", str(warmup)]

    if pin is not None:
        if shutil.which("taskset") is None:
            raise SystemExit("--pin requires taskset, which was not found on PATH")
        command = ["taskset", "-c", str(pin), *command]

    result = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)

    if result.returncode != 0:
        raise SystemExit(f"case {name} failed:\n{result.stderr}")

    return json.loads(result.stdout)


def _print_table(rows):
    print(f"{'case':<22} {'median':>10} {'min':>10} {'stdev':>10} {'rel':>7}")
    for row in rows:
        rel = row["stdev"] / row["median"] * 100 if row["median"] else 0.0
        print(
            f"{row['case']:<22} {row['median'] * 1e3:9.3f}ms {row['min'] * 1e3:9.3f}ms "
            f"{row['stdev'] * 1e3:9.3f}ms {rel:6.1f}%"
        )


def _print_baseline(first, second):
    """Compare a build against itself. The spread per case is its minimum detectable effect.

    The floor is reported per case rather than as a single number. Measurement noise is roughly
    a fixed number of milliseconds, so a sub-millisecond case can wobble by 40% while a case
    lasting half a second is stable to 1%. Collapsing those into one figure would tell you to
    discard real wins on the slow cases.
    """
    print(f"{'case':<22} {'run A':>10} {'run B':>10} {'delta':>9}   verdict")
    measurable = []

    for a, b in zip(first, second):
        delta = (b["median"] - a["median"]) / a["median"] * 100
        # Below this the case is comparable to the additive noise itself and cannot resolve
        # anything useful, whatever its percentage says.
        if min(a["median"], b["median"]) < RESOLUTION_FLOOR:
            verdict = "below resolution"
        else:
            verdict = f"floor {abs(delta):.1f}%"
            measurable.append((a["case"], abs(delta)))

        print(f"{a['case']:<22} {a['median'] * 1e3:9.3f}ms {b['median'] * 1e3:9.3f}ms {delta:+8.1f}%   {verdict}")

    if not measurable:
        print(f"\nEvery case ran faster than {RESOLUTION_FLOOR * 1e3:.0f}ms; none of them can resolve a change.")
        return

    worst = max(delta for _, delta in measurable)
    print(
        f"\nUse each case's own floor. Across the {len(measurable)} measurable cases the worst is "
        f"{worst:.1f}%;\ncases marked 'below resolution' are too fast to compare and should be ignored."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="*", default=None, help="cases to run (default: all)")
    parser.add_argument("--rounds", type=int, default=15, help="measured rounds per case")
    parser.add_argument("--warmup", type=int, default=3, help="discarded warmup rounds per case")
    parser.add_argument("--pin", type=int, default=None, help="pin each case to this CPU via taskset")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="run the selection twice on the current build to measure the noise floor",
    )
    parser.add_argument("--list", action="store_true", help="list the available cases and exit")
    parser.add_argument("--measure", metavar="CASE", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    if args.list:
        for name in CASES:
            print(name)
        return

    # Child process: measure one case and hand the numbers back as JSON.
    if args.measure:
        print(json.dumps(_measure(args.measure, args.rounds, args.warmup)))
        return

    selection = args.cases or CASES
    for name in selection:
        if name not in CASES:
            raise SystemExit(f"unknown case: {name}\navailable: {', '.join(CASES)}")

    first = [_run_isolated(name, args.rounds, args.warmup, args.pin) for name in selection]

    if not args.baseline:
        _print_table(first)
        return

    second = [_run_isolated(name, args.rounds, args.warmup, args.pin) for name in selection]
    _print_baseline(first, second)


if __name__ == "__main__":
    main()
