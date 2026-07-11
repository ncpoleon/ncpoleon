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
   ensures that both branches stay in sync. For this reason, you should
   **never** fork the `buffer` branch, always fork the `main` one.

We've adopted this framework because the CodSpeed action, which we use to test for performance regression, doesn't support the `pull_request_target` event yet. As a result, merging your code onto `buffer` means that your code should pass all the checks required to be merged on `main`. The second PR then checks that no performance regression happens when using a valid MOSEK license, in which case it is then merged onto `main`.

Once the CodSpeed action supports the `pull_request_target`, we will make `main` the default branch once again and simplify this process to run all MOSEK-related workflows after a human verification to avoid any risk of pwn request.