# Ncpoleon

Ncpoleon approximates polynomial optimization problems of (non)commutative variables through semidefinite programming (SDP) relaxations. It is intended as a followup to [ncpol2sdpa](https://gitlab.com/peterwittek/ncpol2sdpa/), focused on improved performance, functionality and long-term improvements. Currently the package can relax polynomial optimization problems in both commutative and noncommutative variables, i.e., we implement
 - Lasserre's hierarchy: if all polynomials involved in the optimization problem are made of commutative variables, then the generated hierarchy is [Lasserre's](https://epubs.siam.org/doi/10.1137/S1052623400366802).
 - Noncommutative moment matrix hierarchy: if all polynomials involved in the optimization problem are made of noncommutative variables, then the generated hierarchy is [Pironio, Navascués and Acín's](https://arxiv.org/abs/0903.4368).

Ncpoleon is able to handle operator constraints, moment constraints and substitutions rules, following the design ideas of ncpol2sdpa.

## Installation
Ncpoleon can be installed using `pip` via
```bash
pip install ncpoleon
```
Ncpoleon includes optional dependencies to solve the generated SDP problem, namely:
 - `picos`, to export the generated SDP to a [Picos](https://gitlab.com/picos-api/picos) problem, or to solve the generated problem using Picos and its default solver CVXOPT.
 - `mosek`, to export the generated SDP to a MOSEK Python Fusion problem, or to solve the generated problem using the MOSEK Python Fusion API. Note that this requires a MOSEK license.

When installing Ncpoleon, you can specify optional dependencies like so
```bash
pip install ncpoleon[picos,mosek]
```

## Example
Let us consider the example from [Tavakoli, Pozas-Kerstjens, Brown and Araújo](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.96.045006). This optimization problem is stated as

$$
    \begin{aligned}
        \max &\quad \mathop{\mathrm{tr}}\left[\rho\left(X_2^2-\frac12\,X_1X_2-\frac12\,X_2X_1-X_2\right)\right]\\
        \text{s.t.} &\quad X_1-X_1^2\succeq0\\,,\\
        &\quad X_2-X_2^2\succeq0\\,,\\
        &\quad\mathop{\mathrm{tr}}\left[\rho\right]=1\\,,\\
        &\quad\rho\succeq0
    \end{aligned}
$$

with $X_1$ and $X_2$ being Hermitian operators.

We can generate and solve the associated SDP relaxation using the following Python code. Note that this requires either Picos or MOSEK to be installed to solve the relaxation.
```python
from ncpoleon import generate_noncommutative_variables, get_relaxation, solve

# Level of relaxation
level = 1

# Define the variables
x1, x2 = generate_noncommutative_variables("X", 2, starting_index=1, hermitian=True)

# Define the objective
obj = x2**2 - x1 * x2 / 2 - x2 * x1 / 2 - x2

# Operator constraints
operator_constraints = [x1 - x1**2 >= 0, x2 - x2**2 >= 0]

# Generate the SDP relaxation
sdp = get_relaxation([x1, x2], level, obj, operator_constraints=operator_constraints)

# Solve the exported problem. MOSEK is preferably used if available, otherwise it defaults to Picos.
solution = solve(sdp, "max")

# Print the solution
print(solution.value)
```
