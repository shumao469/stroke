# Bounded numerical audit and implementation plan

This plan was written before the new implementation and its new-fixture tests.
It does not authorize rerunning the nine exposed NIST evaluation tasks. The
frozen `tactic_natural_numerics_20261026_v1` campaign files are not modified.

## Fixed implementation choices

- Use only the Python standard library and exact `Fraction` arithmetic. Parse
  finite decimal/rational inputs explicitly; reject malformed dimensions,
  non-finite numbers, boolean numbers, resource-excessive numeric tokens,
  exponents or numerator/denominator bit lengths, zero columns and non-full-column-rank
  least-squares designs. Exact positive LDL pivots validate the scaled Gram
  matrix, including the uniqueness assumption missing from a stationarity-only
  verifier.
- Build the Gram matrix symmetrically. Bind the original design, response and
  optional source identifier by SHA-256 of a canonical rational representation.
  A separate statistics digest detects accidental changes to prepared data; it
  is an integrity checksum, not authentication against a malicious data owner.
- Solve the exact normal equations using row-wise denominator clearing followed
  by integer Bareiss elimination and rational back substitution. Retain a
  separate ordinary Fraction Gaussian reference solver for fresh test oracles.
  This is a mature fraction-free algorithm, not a new method.
- Preserve the old scalar infinity-norm certificate and the fixed `1e-8`
  coefficient/RSS tolerance. For a candidate, calculate the residual and RSS
  directly from the original design/response, not from its claimed RSS or the
  solver's normal-equation calculations.
- For the approximate-inverse certificate, let `M=|I-BH|`, `v=|Br|` and
  `q=||M||inf`. Require `q<1`. Use exactly **three** Neumann terms
  `s=v+Mv+M^2v`, then bound the remaining series by
  `u=s+||M^3v||inf/(1-q) * 1`. Retain the componentwise minimum with the old
  scalar bound. Independently require `v+M*u <= u` in exact arithmetic before
  using any componentwise bounds. No learned parameter, tuned iteration count,
  relaxed tolerance or exposed-task result enters this rule.
- Convert each coordinate bound through its own positive column scale. Bound
  the objective gap by `sum_i u_i*abs(r_i)`, preserving the old scalar bound as
  a reported comparator. The denominator is a conservative lower bound on the
  exact solution's norm/objective, not the candidate's unknown true error. The
  reported old RSS bound retains the original `p*scalar*max(abs(r))` gap rule.
- Do not publish the old greedy controller. Its invalid failed-E-state action
  is outside this numerical module; this release establishes no controller
  superiority.

For a full-rank exact Gram problem, `H e=r` and
`e=B r+(I-BH)e`. Hence `|e| <= (I-M)^-1 v`. The fixed Neumann construction
majorizes the tail because `M>=0` and `||M||inf=q<1`. The additional exact
supersolution check is a machine-checkable safeguard. This is a proof for the
declared exact rational problem, not a guarantee over arbitrary floating-point
programs, uncertain source measurements, rank-deficient fits, or external
software's internal computations. QR-based/componentwise validated numerics,
Neumann bounds and Bareiss elimination have substantial prior literature.

## Frozen-result audit (read-only)

The 66 frozen files were rehashed and size-checked without rerunning evaluation;
all matched. Freeze SHA-256:
`4a39cd32931b2c8e22e54927526a43deb69ec1990ea19d4654ff6a1922ab32de`.

- The raw diagnostic denominator is **18**, not 8: six inaccurate outputs from
  three sources were refused; four accurate outputs were conservatively refused.
  These are numerical accuracy failures, not six new software bugs or clinical
  errors.
- All five policies completed the same exact-solve-plus-verification route on
  all 45 timing episodes (nine tasks, five repeats each), so no full solve was
  avoided. The `0.9834454944` geometric mean of per-task median wall ratios is
  not evidence of algorithmic speedup and misses the locked `<=0.8` criterion.
- Stored mean complete wall times were 0.294972 s for direct and 0.295462 s for
  TACTIC. The means and geometric-median summary answer different timing
  questions; neither demonstrates a decision-mechanism advantage. About 94-96%
  of wall time was outside the post-import task body. A stronger direct baseline
  should skip unused SciPy imports and condition-number calculations.
- The old scalar certificate algebra is valid under exact full-rank inputs.
  Its scalar coordinate conversion and objective-gap bound can be conservative.
  Stationarity alone must not certify unique coefficients for singular designs.
- The old greedy selector could request action 5 after the exact route failed;
  this untriggered boundary bug did not explain the frozen timing or accuracy
  result. The old greedy implementation is not included here.

## New validation boundary

Test only fresh small hand-constructed full-rank, scaled, singular, malformed,
accurate and inaccurate rational cases. Compare both solvers, independent
residual/stationarity, componentwise enclosure and scalar-bound contraction.
Such tests establish executable invariants, not universal empirical safety or
NC-level novelty. Any future performance claim needs a new locked workload,
equally optimized baselines and both cold and warm actual-cost reporting.

## Implementation validation record

The implemented module passed 15 fresh-fixture unit tests and seven independently
authored determinant/Cramer-oracle tests. The latter include 48 nonsingular 2x2
systems, 16 original-space least-squares comparisons and 48 certificate enclosure
checks across different contraction factors, plus 12 nondiagonal signed
contraction cases. They do not import the production
solver to construct their reference coefficients. Known-inaccurate candidates,
rank-deficient problems, missing contractions, malformed dimensions, non-finite
numbers, source/statistics changes and resource-excessive exponent tokens are
rejected. The fixed tolerance remains `1e-8`.

All validation used new hand-generated rational inputs and the existing Python
runtime with bytecode disabled. No prior NIST evaluation, patient input,
protected label, training run or frozen campaign was executed or changed. These
are automated code/oracle checks, not human scientific admission or clinical
validation. The old greedy selector is not shipped, so its failed-exact-route
action bug cannot enter this numerical module.

For transparency, strict verification reconstructs sufficient statistics and
checks their exact LDL pivots, not merely their checksum. This has real cost;
the benchmark must include it for every method. No claim of 20% end-to-end
acceleration, reduced clinical harm or controller superiority follows from
passing this validation.
