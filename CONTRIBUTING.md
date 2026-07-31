# Contributing

If you want to contribute to `ncpoleon`, first of all, thanks for considering it! This file will guide you through the steps required to do so.

## Setup

In order to build the library, you will have to have [Rust](https://rust-lang.org/tools/install/) and [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed on your system. You will also need the nightly toolchain, along with its `rustfmt` component, since formatting checks run under nightly. You can install both by running

```shell
rustup toolchain install nightly --component rustfmt
```

Once that's done, you can run

```shell
uv sync --dev --all-extras --config-setting 'build-args=--profile=dev'
```

to build the library and install the required dependencies. Note that this will install MOSEK regardless of whether you have a license or not, but it won't be used if no valid license is detected.

## Pull request checks

After having made your changes, there are several checks that your code must pass in order to be merged:

 - Clippy should pass using
 ```shell
 cargo clippy --all-targets --all-features
 ```
 - Rustfmt should pass using
 ```shell
 cargo +nightly fmt --check
 ```
 - Ruff should pass using
 ```shell
 ruff check
 ```
 and
 ```shell
 ruff format --check
 ```
 - The Rust-side tests should pass using
 ```shell
 cargo test
 ```
 - The Python-side tests should pass using
 ```shell
 uv run --dev --all-extras --config-setting 'build-args=--profile=dev' pytest
 ```
 - Finally, you can benchmark your code by running
 ```shell
 uv run --dev --all-extras --config-setting 'build-args=--profile=release' pytest --codspeed
 ```
 Specifically, we will test whether your changes introduces a regression in performance before merging.

Remember also to update the documentation and the `.pyi` stub files, and to add tests to cover your changes if applicable.

## Benchmarking locally

`scripts/benchmark_local.py` answers one question: *did my change make `get_relaxation` faster or
slower?* Use it when you have touched something on the hot path and want an answer before pushing.

Running `pytest --codspeed` outside CI will not tell you: it prints `no performance measurement
will be made since it's running in an unknown environment` and just runs the benchmarks.

The script needs nothing beyond the setup above — it is plain wall-clock timing, with no Valgrind
and no CodSpeed instrumentation involved.

### How to use it

The script always measures whatever is currently built, so comparing two versions means building
each in turn. On a machine with nothing else running:

```shell
# 1. Time the same build twice, to see how much this machine wobbles on its own.
#    Your change has to beat that wobble to count as a real result.
uv run scripts/benchmark_local.py --baseline

# 2. Measure the code without your change
git stash
uv sync --dev --all-extras --config-setting 'build-args=--profile=release'
uv run scripts/benchmark_local.py

# 3. Measure it with your change
git stash pop
uv sync --dev --all-extras --config-setting 'build-args=--profile=release'
uv run scripts/benchmark_local.py
```

Compare the `min` columns from steps 2 and 3. **If the difference is smaller than the floor step 1
gave you for that case, you have not measured anything** — the machine moved more than your code
did. When a result matters, repeat steps 2 and 3 once more; load drifts over minutes, and a single
before/after pair quietly blames your change for that drift.

Run every case, which is the default. The whole sweep takes about 40 seconds, and a change that
helps one problem size can easily hurt another: the level 1 and level 2 max-cut cases stress the
moment matrix fill by very different amounts, while `chsh-L11` and `i3322-L4` are non-commutative
and exercise code the max-cut cases never touch. Measuring a single case hides all of that.

Naming one case — `--list` shows them all — is for tightening the feedback loop while you are
actively tuning something, where 40 seconds a round starts to hurt. `max_cut-L2-n25` (~680ms) and
`chsh-L11` (~520ms) are the ones to pick, being the largest commutative and non-commutative cases
respectively. Come back to the full sweep before you conclude anything.

### Reading the output

`--baseline` runs the same build twice, so every difference it reports is noise. It prints a floor
per case rather than one overall number, because the noise is a roughly fixed number of
milliseconds: `max_cut-L2-n25` (~680ms) reproduces to 0.5%, while `max_cut-L1-n5` (~0.1ms) swings
by over 100% and is marked `below resolution`. Ignore anything under about 20ms.

Read the `min` rather than the mean. The noise is additive and one-sided — interference only ever
makes a run slower — so the fastest round is the closest to the truth. The `median` is a useful
second opinion; the mean is not.

### If the numbers look unstable

 - Close anything running in the background.
 - Pin the CPU with `--pin <cpu>`, which uses `taskset`, and on Linux set the governor to
   `performance`; `powersave` leaves frequency scaling live.
 - Each case already runs in its own interpreter, because benchmarks sharing a process contaminate
   each other: `max_cut-L2-n20` measures 220ms alone but 410ms when it follows the smaller cases.

To find out *why* something is slow, profile it instead. `perf record` shows where the time went,
whereas a timer only gives one number per run.

## Continuous integration and the MOSEK license

Some of our checks, namely the MOSEK-backed test suite and the CodSpeed performance benchmarks, need a commercial MOSEK license to run. That license is stored as a repository secret, and for security reasons GitHub does not expose repository secrets to workflows triggered by pull requests coming from a fork. This means these specific checks can't run directly against an external contributor's pull request the way our other checks do.

### Where to open your pull request

All incoming pull requests should be made against the `buffer` branch. Here's what happens when you do so.

1. **Fast checks run immediately, on every PR, from anyone.** Linting, the
   Rust test suite, and the parts of the Python test suite that don't need
   MOSEK all run automatically — no approval required.
2. **MOSEK-backed checks wait for a maintainer's approval.** Because these
   checks require the license secret, a maintainer has to manually approve
   the run once they've reviewed your changes. This isn't a judgment on your
   contribution, but a deliberate step to make sure nothing in a PR can
   exfiltrate or otherwise misuse the license before a human has looked at
   the diff. You may see these checks sit in a "waiting" state for a bit
   until a maintainer gets to it.
3. **Once a maintainer merges your PR into `buffer`, a second pull request
   opens automatically**, from `buffer` into `main`. You don't need to do
   anything for this — it happens on its own. Since `buffer` lives in this
   repository rather than a fork, this second PR gets the full test suite,
   coverage, and the CodSpeed performance comparison, including CodSpeed's
   check that can block the merge if it detects a regression. This is the
   real final gate before your change reaches `main`; a maintainer will
   merge it once everything is green.
4. On every push on `main`, including the merging of the PR originating 
   from `buffer`, `buffer` is force-pushed into the state of `main`. This
   ensures that both branches stay in sync.

We've adopted this framework because the CodSpeed action, which we use to test for performance regression, doesn't support the `pull_request_target` event. As a result, merging your code onto `buffer` means that your code should pass all the checks required to be merged on `main`. The second PR then checks that no performance regression happens when using a valid MOSEK license, in which case it is then merged onto `main`.

Once the CodSpeed action supports the `pull_request_target`, we will make `main` the default branch once again and simplify this process to run all MOSEK-related workflows after a human verification to avoid any risk of pwn request.
