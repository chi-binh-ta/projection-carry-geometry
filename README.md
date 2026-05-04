# Projection--Carry Geometry
## Status

Research prototype v0.1.0.

- Tests: 36 passed
- Paper draft included in `paper/`
- Reproducibility notebooks included in `notebooks/`
- Scaling figures and tables included in `paper/figures/` and `paper/tables/`

Base-\(B\) multiplication decomposes into bilinear diagonal projection followed by nonlinear carry flow. The complexity of carry is shaped by digit-support sumset geometry, while digit amplitudes and incoming carry determine the realized flow.

A small Python research project for studying multiplication in a positional
base `B` through digit geometry, anti-diagonal projection, and carry
normalization.

All digit lists are low-to-high:

```text
123 in base 10 -> [3, 2, 1]
```

For

```text
X = sum_i a_i B^i
Y = sum_j b_j B^j
```

the framework uses:

```text
M_{ij} = a_i b_j
c_k = sum_{i+j=k} M_{ij}
d = C_B(c)
XY = sum_k d_k B^k
```

## Quickstart

```bash
pip install -r requirements.txt
python src/experiments.py
```

Open the main notebook:

```bash
jupyter notebook notebooks/01_123_times_321_demo.ipynb
```

## Demo: 123 x 321

The base-10 demo gives:

```text
raw diagonal sums: [3, 8, 14, 8, 3]
carry result digits: [3, 8, 4, 9, 3]
result: 39483
```

Multiplication is decomposed into:

1. Low-to-high digit representation.
2. Outer-product interaction matrix.
3. Anti-diagonal projection under `(i, j) -> i + j`.
4. Carry normalization, the nonlinear step that creates valid final digits.

## Sprint 2: Carry Flow

Carry flow interprets raw diagonal coefficients `c_k` as mass on the degree
axis. At degree `k`, normalization combines raw mass with incoming carry,
keeps one valid base-`B` digit, and transports the outgoing carry to degree
`k + 1`.

The recurrence is:

```text
r_0 = 0
s_k = c_k + r_k
d_k = s_k mod B
r_{k+1} = floor(s_k / B)
```

Meaning of the carry-flow table columns:

```text
raw_c           raw coefficient c_k before carry
incoming_carry  incoming carry r_k from degree k-1
digit           normalized digit d_k kept at degree k
outgoing_carry  transported mass r_{k+1} from k to k+1
```

Commands:

```bash
pytest
python src/experiments.py
jupyter notebook notebooks/06_carry_as_mass_flow.ipynb
```

The main demo writes:

```text
results/123_times_321_carry_flow.csv
figures/carry_profiles/123_times_321_carry_flow.png
```

## Sprint 3: Carry Complexity

Carry complexity connects diagonal mass concentration to the nonlinear cost of
carry normalization. The new metrics include raw diagonal mass entropy,
diagonal mass concentration, carry amplification ratio, carry mass, carry
count, carry depth, and carry entropy.

Commands:

```bash
pytest
python src/experiments.py
jupyter notebook notebooks/07_carry_complexity_experiments.ipynb
```

The main experiment writes:

```text
results/carry_complexity_metrics.csv
```

## Sprint 4: Local Carry Laws

Local carry activation follows the threshold law:

```text
r_{k+1} = floor((c_k + r_k) / B)
carry_active iff c_k + r_k >= B
```

Sprint 4 adds local carry-law datasets, carry cascade length, and
value-weighted carry flow. The key empirical point is that incoming carry
matters, so `raw_c` alone is not enough to determine carry activation.

Commands:

```bash
pytest
python src/experiments.py
jupyter notebook notebooks/08_local_carry_laws.ipynb
```

The main experiment writes:

```text
results/local_carry_laws.csv
```

## Sprint 5: Support-Sumset Geometry

Support geometry studies where raw multiplication mass can appear before carry:

```text
S_raw = S_X + S_Y
rho_k = # {(i, j): i + j = k}
additive energy = sum_k rho_k^2
```

Here `rho_k` is the representation count on anti-diagonal `k`. Additive energy
measures concentration of these representation counts, so it acts as a support
level predictor for carry complexity.

Commands:

```bash
pytest
python src/experiments.py
jupyter notebook notebooks/09_support_sumset_geometry.ipynb
```

The main experiment writes:

```text
results/support_geometry_carry.csv
```

## Sprint 6: Paper-Ready Scaling Experiments

Sprint 6 turns the prototype into a paper-ready experimental package. It adds a
scaling grid over bases, digit lengths, and support densities, plus paper-ready
figures for the central hypothesis that support-sumset concentration predicts
carry complexity.

Commands:

```bash
pytest
python src/experiments.py
jupyter notebook notebooks/10_empirical_scaling_laws.ipynb
```

Generated outputs:

```text
results/scaling_grid.csv
paper/figures/*.png
```

## Sprint 7: Paper Freeze

Sprint 7 freezes the codebase into a coherent paper draft. It adds normalized
correlation metrics, normalized paper figures, paper result tables, a
limitations section, and a conclusion.

Commands:

```bash
pytest
python src/experiments.py
jupyter notebook notebooks/11_paper_result_consolidation.ipynb
```

Generated outputs:

```text
paper/tables/correlation_table.csv
paper/tables/normalized_correlation_table.csv
paper/tables/grouped_scaling_summary.csv
paper/figures/normalized_*.png
```

## Directory Guide

```text
src/                 Python modules for pipeline, metrics, flow, visualization, experiments
notebooks/           Demo and experiment notebooks
figures/             Generated figures from demos and experiments
results/             Generated CSV result tables
paper/               LaTeX paper skeleton
tests/               Pytest correctness checks
```

## Research Directions

- carry complexity
- sparse digit support
- binary multiplication
- tensor version for multiplying more than two numbers
- carry as nonlinear mass flow on degree axis
